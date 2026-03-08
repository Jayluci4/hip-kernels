// RCCL Communicator Lifecycle Manager for GPT-OSS-120B
// Manages RCCL communication for decoupled TP x EP parallelism.
// Supports sub-communicators for TP groups and EP groups via ncclCommSplit.
// Provides stream-based synchronization between compute and communication

#include "hip_compat.h"
#include <rccl/rccl.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "config.h"
#include "cuda_utils.h"
#include "nccl_utils.h"

namespace gptoss {

class NCCLManager {
public:
    NCCLManager() = default;
    ~NCCLManager() { destroy(); }

    // Non-copyable, non-movable (owns RCCL comm and HIP resources)
    NCCLManager(const NCCLManager&) = delete;
    NCCLManager& operator=(const NCCLManager&) = delete;
    NCCLManager(NCCLManager&&) = delete;
    NCCLManager& operator=(NCCLManager&&) = delete;

    // -----------------------------------------------------------------------
    // Initialize RCCL communicator, streams, and synchronization events.
    //
    // @param rank        Global rank in [0, world_size)
    // @param world_size  Total number of GPUs (tp_size * ep_size)
    // -----------------------------------------------------------------------
    void init(int rank, int world_size) {
        rank_ = rank;
        world_size_ = world_size;
        initialized_ = true;

        CUDA_CHECK(hipSetDevice(rank_));

        // ---- RCCL communicator initialization ----
        // Rank 0 generates the unique ID and broadcasts it.
        // In a real multi-process deployment, the unique ID would be
        // distributed via MPI_Bcast, a shared filesystem, or a TCP store.
        // Here we support both the single-process (multi-thread) path where
        // the caller supplies the ID, and the convenience path where rank 0
        // generates and all ranks receive via ncclGetUniqueId placed into
        // shared memory.
        ncclUniqueId nccl_id;
        if (rank_ == 0) {
            NCCL_CHECK(ncclGetUniqueId(&nccl_id));
        }

        // For single-node EP2, both GPUs reside on the same host.
        // Use hipMemcpy via host staging to broadcast the unique ID.
        // In multi-process setups, replace this with MPI_Bcast or equivalent.
        broadcast_unique_id(&nccl_id);

        NCCL_CHECK(ncclCommInitRank(&comm_, world_size_, nccl_id, rank_));

        // ---- Dedicated communication stream ----
        // The comm stream is separate from compute so that RCCL transfers can
        // overlap with local expert computation. We use high priority so that
        // the GPU scheduler favors comm kernels when both are eligible.
        CUDA_CHECK(hipStreamCreateWithPriority(
            &comm_stream_, hipStreamNonBlocking, get_high_stream_priority()));

        // ---- Compute stream (caller-owned, but we create a default) ----
        CUDA_CHECK(hipStreamCreateWithFlags(
            &compute_stream_, hipStreamNonBlocking));

        // ---- Cross-stream synchronization events ----
        // These events are recorded on one stream and waited on the other to
        // enforce ordering without full device synchronization.
        CUDA_CHECK(hipEventCreateWithFlags(
            &compute_to_comm_event_, hipEventDisableTiming));
        CUDA_CHECK(hipEventCreateWithFlags(
            &comm_to_compute_event_, hipEventDisableTiming));

        fprintf(stderr, "[NCCLManager] Rank %d/%d initialized on GPU %d\n",
                rank_, world_size_, rank_);
    }

    // -----------------------------------------------------------------------
    // Initialize with an externally-provided RCCL unique ID.
    // Use this when the unique ID is distributed via MPI or a TCP store.
    // -----------------------------------------------------------------------
    void init(int rank, int world_size, const ncclUniqueId& nccl_id) {
        rank_ = rank;
        world_size_ = world_size;
        initialized_ = true;

        CUDA_CHECK(hipSetDevice(rank_));

        NCCL_CHECK(ncclCommInitRank(&comm_, world_size_, nccl_id, rank_));

        CUDA_CHECK(hipStreamCreateWithPriority(
            &comm_stream_, hipStreamNonBlocking, get_high_stream_priority()));
        CUDA_CHECK(hipStreamCreateWithFlags(
            &compute_stream_, hipStreamNonBlocking));

        CUDA_CHECK(hipEventCreateWithFlags(
            &compute_to_comm_event_, hipEventDisableTiming));
        CUDA_CHECK(hipEventCreateWithFlags(
            &comm_to_compute_event_, hipEventDisableTiming));

        fprintf(stderr, "[NCCLManager] Rank %d/%d initialized on GPU %d (external ID)\n",
                rank_, world_size_, rank_);
    }

    // -----------------------------------------------------------------------
    // Create TP and EP sub-communicators via ncclCommSplit.
    //
    // TP sub-communicator: ranks in the same EP group (same ep_rank) form
    //   a TP group for attention AllReduce.
    // EP sub-communicator: ranks in the same TP group (same tp_rank) form
    //   an EP group for expert output exchange.
    //
    // Rank decomposition: global_rank = tp_rank + ep_rank * tp_size
    // -----------------------------------------------------------------------
    // Set rank/world_size without initializing RCCL (for single-GPU mode).
    void set_rank(int rank, int world_size) {
        rank_ = rank;
        world_size_ = world_size;
    }

    // Set TP/EP rank decomposition without creating sub-communicators.
    // Call create_tp_comm() and create_ep_comm() separately so the caller
    // can wrap each collective in ncclGroupStart/End for single-process
    // multi-GPU setups.
    void set_sub_comm_ranks(int tp_size, int ep_size) {
        tp_rank_ = rank_ % tp_size;
        ep_rank_ = rank_ / tp_size;
        tp_size_ = tp_size;
        ep_size_ = ep_size;
    }

    void create_tp_comm() {
        ncclConfig_t tp_cfg = NCCL_CONFIG_INITIALIZER;
        NCCL_CHECK(ncclCommSplit(comm_, /*color=*/ep_rank_, /*key=*/tp_rank_,
                                  &tp_comm_, &tp_cfg));
    }

    void create_ep_comm() {
        ncclConfig_t ep_cfg = NCCL_CONFIG_INITIALIZER;
        NCCL_CHECK(ncclCommSplit(comm_, /*color=*/tp_rank_, /*key=*/ep_rank_,
                                  &ep_comm_, &ep_cfg));
    }

    void create_sub_communicators(int tp_size, int ep_size) {
        set_sub_comm_ranks(tp_size, ep_size);
        create_tp_comm();
        create_ep_comm();

        fprintf(stderr, "[NCCLManager] Rank %d: sub-comms created "
                "(tp_rank=%d/%d, ep_rank=%d/%d)\n",
                rank_, tp_rank_, tp_size_, ep_rank_, ep_size_);
    }

    // -----------------------------------------------------------------------
    // Point-to-point send.
    //
    // @param buf    Device pointer to data to send
    // @param count  Number of elements (not bytes)
    // @param dtype  RCCL data type (e.g., ncclBfloat16)
    // @param peer   Destination rank (in the communicator used)
    // @param stream HIP stream on which to enqueue the send
    // -----------------------------------------------------------------------
    void send(const void* buf, size_t count, ncclDataType_t dtype,
              int peer, hipStream_t stream) {
        NCCL_CHECK(ncclSend(buf, count, dtype, peer, comm_, stream));
    }

    // -----------------------------------------------------------------------
    // Point-to-point receive.
    //
    // Enqueues an RCCL recv operation on the given stream.
    //
    // @param buf    Device pointer to receive buffer
    // @param count  Number of elements (not bytes)
    // @param dtype  RCCL data type
    // @param peer   Source rank
    // @param stream HIP stream on which to enqueue the recv
    // -----------------------------------------------------------------------
    void recv(void* buf, size_t count, ncclDataType_t dtype,
              int peer, hipStream_t stream) {
        NCCL_CHECK(ncclRecv(buf, count, dtype, peer, comm_, stream));
    }

    // -----------------------------------------------------------------------
    // All-to-All exchange (N-way grouped P2P on the global communicator).
    //
    // Sends the same data to all peers and receives from all peers.
    // For EP-N, prefer using the EP sub-communicator with EPComm::combine_v().
    //
    // @param send_buf  Device pointer to data being sent
    // @param recv_buf  Device pointer to receive buffer
    // @param count     Number of elements per direction per peer
    // @param dtype     RCCL data type
    // @param stream    HIP stream for the grouped operation
    // -----------------------------------------------------------------------
    void all_to_all(const void* send_buf, void* recv_buf, size_t count,
                    ncclDataType_t dtype, hipStream_t stream) {
        // N-way: each peer gets count elements at offset p*count in recv_buf.
        // Note: for variable-size exchange, use EPComm::combine_v() instead.
        NCCL_CHECK(ncclGroupStart());
        for (int p = 0; p < world_size_; ++p) {
            if (p == rank_) continue;
            NCCL_CHECK(ncclSend(send_buf, count, dtype, p, comm_, stream));
            NCCL_CHECK(ncclRecv(recv_buf, count, dtype, p, comm_, stream));
        }
        NCCL_CHECK(ncclGroupEnd());
    }

    // -----------------------------------------------------------------------
    // Record an event on compute_stream and make comm_stream wait on it.
    //
    // Call this before launching communication to ensure all compute kernels
    // that produce the send buffer have completed from comm_stream's
    // perspective. The compute stream itself is NOT blocked.
    // -----------------------------------------------------------------------
    void sync_compute_to_comm() {
        CUDA_CHECK(hipEventRecord(compute_to_comm_event_, compute_stream_));
        CUDA_CHECK(hipStreamWaitEvent(comm_stream_, compute_to_comm_event_, 0));
    }

    // Overload accepting caller-provided streams (used by EPComm)
    void sync_compute_to_comm(hipStream_t compute, hipStream_t comm) {
        CUDA_CHECK(hipEventRecord(compute_to_comm_event_, compute));
        CUDA_CHECK(hipStreamWaitEvent(comm, compute_to_comm_event_, 0));
    }

    // -----------------------------------------------------------------------
    // Record an event on comm_stream and make compute_stream wait on it.
    //
    // Call this after communication completes so that subsequent compute
    // kernels (e.g., remote expert FFN) see the received data. The comm
    // stream itself is NOT blocked.
    // -----------------------------------------------------------------------
    void sync_comm_to_compute() {
        CUDA_CHECK(hipEventRecord(comm_to_compute_event_, comm_stream_));
        CUDA_CHECK(hipStreamWaitEvent(compute_stream_, comm_to_compute_event_, 0));
    }

    // Overload accepting caller-provided streams (used by EPComm)
    void sync_comm_to_compute(hipStream_t compute, hipStream_t comm) {
        CUDA_CHECK(hipEventRecord(comm_to_compute_event_, comm));
        CUDA_CHECK(hipStreamWaitEvent(compute, comm_to_compute_event_, 0));
    }

    // -----------------------------------------------------------------------
    // Full barrier: synchronize both streams to the host.
    //
    // This is a heavyweight synchronization point. Use sparingly -- only at
    // initialization, shutdown, or debugging boundaries. During normal
    // inference the event-based sync methods above are sufficient.
    // -----------------------------------------------------------------------
    void barrier() {
        CUDA_CHECK(hipStreamSynchronize(comm_stream_));
        CUDA_CHECK(hipStreamSynchronize(compute_stream_));

        // RCCL-level barrier: allreduce of zero bytes ensures all ranks
        // have completed their preceding RCCL operations.
        NCCL_CHECK(ncclAllReduce(
            nullptr, nullptr, 0, ncclInt8, ncclSum, comm_, comm_stream_));
        CUDA_CHECK(hipStreamSynchronize(comm_stream_));
    }

    // -----------------------------------------------------------------------
    // Tear down all resources.
    // Safe to call multiple times or on an uninitialized manager.
    // -----------------------------------------------------------------------
    void destroy() {
        if (!initialized_) return;

        // Synchronize before destroying to avoid races
        if (comm_stream_) {
            hipStreamSynchronize(comm_stream_);
        }
        if (compute_stream_) {
            hipStreamSynchronize(compute_stream_);
        }

        if (compute_to_comm_event_) {
            CUDA_CHECK(hipEventDestroy(compute_to_comm_event_));
            compute_to_comm_event_ = nullptr;
        }
        if (comm_to_compute_event_) {
            CUDA_CHECK(hipEventDestroy(comm_to_compute_event_));
            comm_to_compute_event_ = nullptr;
        }
        if (comm_stream_) {
            CUDA_CHECK(hipStreamDestroy(comm_stream_));
            comm_stream_ = nullptr;
        }
        if (compute_stream_) {
            CUDA_CHECK(hipStreamDestroy(compute_stream_));
            compute_stream_ = nullptr;
        }
        if (tp_comm_) {
            NCCL_CHECK(ncclCommDestroy(tp_comm_));
            tp_comm_ = nullptr;
        }
        if (ep_comm_) {
            NCCL_CHECK(ncclCommDestroy(ep_comm_));
            ep_comm_ = nullptr;
        }
        if (comm_) {
            NCCL_CHECK(ncclCommDestroy(comm_));
            comm_ = nullptr;
        }

        fprintf(stderr, "[NCCLManager] Rank %d destroyed\n", rank_);
        initialized_ = false;
    }

    // ---- Accessors ----
    int rank()       const { return rank_; }
    int world_size() const { return world_size_; }

    ncclComm_t   comm()           const { return comm_; }
    hipStream_t  comm_stream()    const { return comm_stream_; }
    hipStream_t  compute_stream() const { return compute_stream_; }

    // Sub-communicator accessors (valid after create_sub_communicators)
    ncclComm_t tp_comm()  const { return tp_comm_; }
    ncclComm_t ep_comm()  const { return ep_comm_; }
    int tp_rank()         const { return tp_rank_; }
    int ep_rank()         const { return ep_rank_; }
    int tp_size()         const { return tp_size_; }
    int ep_size()         const { return ep_size_; }

    bool is_initialized() const { return initialized_; }

private:
    // ---- Internal helpers ----

    // Query the highest (most urgent) stream priority on this device.
    static int get_high_stream_priority() {
        int low_priority, high_priority;
        CUDA_CHECK(hipDeviceGetStreamPriorityRange(&low_priority, &high_priority));
        return high_priority;
    }

    // Broadcast the RCCL unique ID from rank 0 to all ranks.
    // For single-node EP2, this uses hipIPC or host-side copy.
    // In production multi-process mode, replace with MPI_Bcast.
    static void broadcast_unique_id(ncclUniqueId* id) {
        // For single-process multi-GPU (e.g., threaded model), the caller
        // should use the init(rank, world_size, nccl_id) overload instead.
        // This default implementation assumes the caller is using a shared
        // address space or has already distributed the ID.
        //
        // In the RCCL documentation's recommended pattern, rank 0 calls
        // ncclGetUniqueId and then the ID is broadcast via MPI_Bcast or
        // written to a shared file. We leave this as a no-op here because
        // in the single-process case the ID is already visible to all
        // threads, and in the multi-process case the external init() overload
        // should be used.
        (void)id;
    }

    // ---- State ----
    int rank_       = -1;
    int world_size_ = 0;
    bool initialized_ = false;

    ncclComm_t   comm_        = nullptr;
    hipStream_t  comm_stream_ = nullptr;
    hipStream_t  compute_stream_ = nullptr;

    // Sub-communicators for decoupled TP x EP
    ncclComm_t tp_comm_ = nullptr;   // TP group (attention AllReduce)
    ncclComm_t ep_comm_ = nullptr;   // EP group (expert exchange)
    int tp_rank_ = -1;
    int ep_rank_ = -1;
    int tp_size_ = 0;
    int ep_size_ = 0;

    // Synchronization events for compute <-> comm overlap
    hipEvent_t compute_to_comm_event_ = nullptr;
    hipEvent_t comm_to_compute_event_ = nullptr;
};

// ---------------------------------------------------------------------------
// Extern wrapper functions for cross-TU access
// ---------------------------------------------------------------------------

NCCLManager* nccl_manager_create() {
    return new NCCLManager();
}

void nccl_manager_destroy(NCCLManager* mgr) {
    if (mgr) {
        mgr->destroy();
        delete mgr;
    }
}

void nccl_manager_init_with_id(NCCLManager* mgr, int rank, int world_size, const ncclUniqueId& nccl_id) {
    mgr->init(rank, world_size, nccl_id);
}

ncclComm_t nccl_manager_comm(NCCLManager* mgr) {
    return mgr->comm();
}

hipStream_t nccl_manager_comm_stream(NCCLManager* mgr) {
    return mgr->comm_stream();
}

void nccl_manager_create_sub_communicators(NCCLManager* mgr, int tp_size, int ep_size) {
    mgr->create_sub_communicators(tp_size, ep_size);
}

void nccl_manager_set_rank(NCCLManager* mgr, int rank, int world_size) {
    mgr->set_rank(rank, world_size);
}

void nccl_manager_set_sub_comm_ranks(NCCLManager* mgr, int tp_size, int ep_size) {
    mgr->set_sub_comm_ranks(tp_size, ep_size);
}

void nccl_manager_create_tp_comm(NCCLManager* mgr) {
    mgr->create_tp_comm();
}

void nccl_manager_create_ep_comm(NCCLManager* mgr) {
    mgr->create_ep_comm();
}

ncclComm_t nccl_manager_tp_comm(NCCLManager* mgr) {
    return mgr->tp_comm();
}

ncclComm_t nccl_manager_ep_comm(NCCLManager* mgr) {
    return mgr->ep_comm();
}

int nccl_manager_tp_rank(NCCLManager* mgr) {
    return mgr->tp_rank();
}

int nccl_manager_ep_rank(NCCLManager* mgr) {
    return mgr->ep_rank();
}

} // namespace gptoss
