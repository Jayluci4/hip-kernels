// Expert Parallelism Combine for GPT-OSS-120B
// Implements the expert output exchange protocol for EP-N MoE layers.
//
// In the dispatch-free architecture, all GPUs in the same EP group hold
// identical hidden states (from TP AllReduce), so no token dispatch is needed.
// Only expert OUTPUTS are exchanged via combine_v() (N-way grouped P2P).
//
// combine_v(): N-way exchange using ncclGroupStart + N-1 Send/Recv pairs.
// combine():   Legacy EP2 wrapper delegating to combine_v().
//
// Data format: BF16, hidden_size=2880 elements per token.

#include "hip_compat.h"
#include <rccl/rccl.h>
#include <cstdio>
#include <cstdint>
#include <cstring>

#include "config.h"
#include "cuda_utils.h"
#include "nccl_utils.h"

namespace gptoss {

class EPComm {
public:
    EPComm() = default;
    ~EPComm() { destroy(); }

    // Non-copyable
    EPComm(const EPComm&) = delete;
    EPComm& operator=(const EPComm&) = delete;

    // -------------------------------------------------------------------
    // Initialization for N-way EP.
    //
    // @param device_id  GPU device to allocate on
    // @param ep_rank    This GPU's rank within the EP group [0, ep_size)
    // @param ep_size    Number of GPUs in the EP group
    // @param tp_rank    This GPU's rank within the TP group (for reference)
    // @param tp_size    Number of GPUs in the TP group (for reference)
    // @param ep_comm    RCCL sub-communicator for the EP group
    // -------------------------------------------------------------------
    void init(int device_id, int ep_rank, int ep_size,
              int tp_rank, int tp_size, ncclComm_t ep_comm) {
        device_id_ = device_id;
        ep_rank_   = ep_rank;
        ep_size_   = ep_size;
        tp_rank_   = tp_rank;
        tp_size_   = tp_size;
        ep_comm_   = ep_comm;

        CUDA_CHECK(hipSetDevice(device_id_));

        // Synchronization events for combine phase (timing disabled for lower overhead).
        CUDA_CHECK(hipEventCreateWithFlags(
            &combine_compute_done_, hipEventDisableTiming));
        CUDA_CHECK(hipEventCreateWithFlags(
            &combine_comm_done_, hipEventDisableTiming));

        initialized_ = true;
    }

    // -------------------------------------------------------------------
    // combine_v() -- N-way expert output exchange (grouped P2P)
    //
    // Each GPU sends its local expert outputs to all EP peers and receives
    // their expert outputs into a contiguous recv buffer.
    //
    // The recv buffer is packed: [peer_0 | peer_1 | ... | peer_{N-1}]
    // where the self-slot is empty (zero-length).
    //
    // @param send_buf           Local expert outputs [send_count, hidden_size]
    // @param send_count         Number of tokens this GPU computed locally
    // @param recv_buf           Contiguous recv for all peers
    // @param peer_recv_counts   [ep_size] tokens expected from each peer
    // @param peer_recv_offsets  [ep_size] token offset per peer in recv_buf
    // @param compute_stream     Compute stream
    // @param comm_stream        Communication stream
    // -------------------------------------------------------------------
    void combine_v(
        const void*   send_buf,
        int           send_count,
        void*         recv_buf,
        const int*    peer_recv_counts,
        const int*    peer_recv_offsets,
        hipStream_t   compute_stream,
        hipStream_t   comm_stream)
    {
        // Single GPU: no peers to exchange with — nothing to do.
        if (ep_size_ <= 1) return;

        constexpr int hidden = ModelConfig::hidden_size;

        // --- Step 1: Synchronize compute -> comm ---
        // Use a fresh event per call. The persistent combine_compute_done_
        // event was shared across all 36 MoE layers. When the GPU processes
        // a later layer's hipStreamWaitEvent before it processes the matching
        // hipEventRecord (possible because record and wait are on different
        // streams), it sees a stale "signaled" state from a previous layer
        // and proceeds without waiting. A fresh event starts unsignaled,
        // eliminating the race.
        hipEvent_t compute_done;
        CUDA_CHECK(hipEventCreateWithFlags(&compute_done, hipEventDisableTiming));
        CUDA_CHECK(hipEventRecord(compute_done, compute_stream));
        CUDA_CHECK(hipStreamWaitEvent(comm_stream, compute_done, 0));
        CUDA_CHECK(hipEventDestroy(compute_done));

        // --- Step 2: N-way grouped P2P exchange ---
        __hip_bfloat16* recv_bf16 = static_cast<__hip_bfloat16*>(recv_buf);
        const size_t send_elems = static_cast<size_t>(send_count) * hidden;

        NCCL_CHECK(ncclGroupStart());
        for (int p = 0; p < ep_size_; ++p) {
            if (p == ep_rank_) continue;

            // Send our local expert outputs to peer p
            if (send_elems > 0) {
                NCCL_CHECK(ncclSend(send_buf, send_elems, ncclBfloat16,
                                    p, ep_comm_, comm_stream));
            }

            // Receive peer p's expert outputs into our recv buffer
            size_t recv_elems = static_cast<size_t>(peer_recv_counts[p]) * hidden;
            if (recv_elems > 0) {
                __hip_bfloat16* recv_ptr = recv_bf16 +
                    static_cast<size_t>(peer_recv_offsets[p]) * hidden;
                NCCL_CHECK(ncclRecv(recv_ptr, recv_elems, ncclBfloat16,
                                    p, ep_comm_, comm_stream));
            }
        }
        NCCL_CHECK(ncclGroupEnd());

        // Step 3 removed: comm completion is recorded by the caller
        // with a fresh event (see moe_layer.cu).
    }

    // -------------------------------------------------------------------
    // combine() -- Legacy EP2 wrapper
    //
    // For backward compatibility with EP2 code paths (ep_size==2 only).
    // Distributes recv_count evenly across all non-self peers.
    // For N-way EP, use combine_v() directly with per-peer counts.
    // -------------------------------------------------------------------
    void combine(
        const void*   send_buf,
        int           send_count,
        void*         recv_buf,
        int           recv_count,
        hipStream_t   compute_stream,
        hipStream_t   comm_stream)
    {
        // Build per-peer arrays: distribute recv_count across peers
        int peer_recv_counts[ModelConfig::max_ep_size] = {};
        int peer_recv_offsets[ModelConfig::max_ep_size] = {};
        int offset = 0;
        for (int p = 0; p < ep_size_; ++p) {
            peer_recv_offsets[p] = offset;
            if (p != ep_rank_) {
                // For ep_size==2, this gives the single peer all recv_count
                // For ep_size>2, caller should use combine_v() instead
                peer_recv_counts[p] = recv_count / (ep_size_ - 1);
                offset += peer_recv_counts[p];
            }
        }

        combine_v(send_buf, send_count, recv_buf,
                  peer_recv_counts, peer_recv_offsets,
                  compute_stream, comm_stream);
    }

    // -------------------------------------------------------------------
    // Wait for the combine communication to finish on a given stream.
    // -------------------------------------------------------------------
    void wait_combine_complete(hipStream_t stream) {
        if (ep_size_ <= 1) return;  // no exchange happened
        CUDA_CHECK(hipStreamWaitEvent(stream, combine_comm_done_, 0));
    }

    // -------------------------------------------------------------------
    // Tear down all resources.
    // -------------------------------------------------------------------
    void destroy() {
        if (!initialized_) return;

        CUDA_CHECK(hipSetDevice(device_id_));

        if (combine_compute_done_)  { CUDA_CHECK(hipEventDestroy(combine_compute_done_));  combine_compute_done_  = nullptr; }
        if (combine_comm_done_)     { CUDA_CHECK(hipEventDestroy(combine_comm_done_));     combine_comm_done_     = nullptr; }

        initialized_ = false;
    }

    // ---- Accessors ----
    int ep_rank()   const { return ep_rank_; }
    int ep_size()   const { return ep_size_; }
    int device_id() const { return device_id_; }

private:
    int device_id_   = -1;
    int ep_rank_     = -1;
    int ep_size_     = 0;
    int tp_rank_     = -1;
    int tp_size_     = 0;
    bool initialized_ = false;

    ncclComm_t ep_comm_ = nullptr;  // EP sub-communicator (borrowed, not owned)

    // Synchronization events for combine phase (compute <-> comm overlap).
    //
    // Combine phase timeline:
    //   compute: [expert FFN ...] --record(combine_compute_done_)--> --wait(combine_comm_done_)--> [unpermute]
    //   comm:    --wait(combine_compute_done_)--> [N-way RCCL send+recv] --record(combine_comm_done_)-->
    hipEvent_t combine_compute_done_  = nullptr;
    hipEvent_t combine_comm_done_     = nullptr;
};

// ---------------------------------------------------------------------------
// Extern wrapper functions
// ---------------------------------------------------------------------------

EPComm* ep_comm_create() {
    return new EPComm();
}

void ep_comm_destroy(EPComm* ep) {
    if (ep) delete ep;
}

void ep_comm_init(EPComm* ep, int device_id, int ep_rank, int ep_size,
                  int tp_rank, int tp_size, ncclComm_t ep_comm) {
    ep->init(device_id, ep_rank, ep_size, tp_rank, tp_size, ep_comm);
}

void ep_comm_combine_v(EPComm* ep,
                       const void* send_buf, int send_count,
                       void* recv_buf,
                       const int* peer_recv_counts,
                       const int* peer_recv_offsets,
                       hipStream_t compute_stream,
                       hipStream_t comm_stream) {
    ep->combine_v(send_buf, send_count, recv_buf,
                  peer_recv_counts, peer_recv_offsets,
                  compute_stream, comm_stream);
}

void ep_comm_wait_combine_complete(EPComm* ep, hipStream_t stream) {
    ep->wait_combine_complete(stream);
}

} // namespace gptoss
