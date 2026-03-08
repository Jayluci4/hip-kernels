// Kernel Profiler Implementation
// HIP event-based GPU timing. Exports Chrome Trace Format (for Perfetto)
// and aggregated summary JSON.
//
// Design: each (device, stream) slot tracks an open region. Calling
// profiler_mark() auto-closes the previous region on that slot by recording
// an end event. No synchronization in the hot path -- elapsed times are
// computed lazily at save time via hipEventElapsedTime.

#include "hip_compat.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <string>
#include <mutex>
#include <algorithm>

#include "profiler.h"
#include "cuda_utils.h"

#ifdef GPTOSS_PROFILE

namespace gptoss {

// ---------------------------------------------------------------------------
// Record: one completed profiled region (events not yet resolved)
// ---------------------------------------------------------------------------
struct ProfileRecord {
    const char* name;       // string literal, not owned
    const char* category;   // string literal, not owned
    int device_id;
    int layer_id;
    hipEvent_t start_event;
    hipEvent_t end_event;
    int64_t wall_us;        // approximate wall-clock timestamp (for trace ordering)
};

// ---------------------------------------------------------------------------
// Per-(device, stream) state for tracking the currently open region
// ---------------------------------------------------------------------------
struct SlotState {
    hipEvent_t start_event = nullptr;
    const char* name     = nullptr;
    const char* category = nullptr;
    int device_id = 0;
    int layer_id  = -1;
    bool open     = false;
    int64_t start_wall_us = 0;
};

// ---------------------------------------------------------------------------
// Global profiler state
// ---------------------------------------------------------------------------
static constexpr int MAX_DEVICES = 8;

// Slot key: (device_id, stream)
struct SlotKey {
    int device_id;
    hipStream_t stream;
    bool operator==(const SlotKey& o) const {
        return device_id == o.device_id && stream == o.stream;
    }
};
struct SlotKeyHash {
    size_t operator()(const SlotKey& k) const {
        return std::hash<int>()(k.device_id) ^ (std::hash<void*>()((void*)k.stream) << 16);
    }
};

static std::mutex g_profiler_mutex;
static std::vector<ProfileRecord> g_records;
static std::unordered_map<SlotKey, SlotState, SlotKeyHash> g_slots;
static thread_local int tl_layer_id = -1;

static int64_t wall_clock_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1000000 + ts.tv_nsec / 1000;
}

// Close an open slot: record end event, push to records (no sync)
static void close_slot(SlotState& slot, hipStream_t stream) {
    if (!slot.open) return;

    // Create end event on correct device and record it
    hipEvent_t end_event;
    CUDA_CHECK(hipSetDevice(slot.device_id));
    CUDA_CHECK(hipEventCreate(&end_event));
    CUDA_CHECK(hipEventRecord(end_event, stream));

    g_records.push_back({
        slot.name, slot.category, slot.device_id,
        slot.layer_id, slot.start_event, end_event, slot.start_wall_us
    });

    slot.start_event = nullptr;
    slot.open = false;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void profiler_mark(const char* name, const char* category,
                   int device_id, hipStream_t stream) {
    std::lock_guard<std::mutex> lock(g_profiler_mutex);

    SlotKey key{device_id, stream};
    SlotState& slot = g_slots[key];

    // Close previous region on this slot (no sync, just record end event)
    if (slot.open) {
        close_slot(slot, stream);
    }

    // Open new region -- create start event on correct device
    CUDA_CHECK(hipSetDevice(device_id));
    hipEvent_t start_event;
    CUDA_CHECK(hipEventCreate(&start_event));
    CUDA_CHECK(hipEventRecord(start_event, stream));

    slot.start_event   = start_event;
    slot.name          = name;
    slot.category      = category;
    slot.device_id     = device_id;
    slot.layer_id      = tl_layer_id;
    slot.open          = true;
    slot.start_wall_us = wall_clock_us();
}

void profiler_end(int device_id, hipStream_t stream) {
    std::lock_guard<std::mutex> lock(g_profiler_mutex);

    SlotKey key{device_id, stream};
    auto it = g_slots.find(key);
    if (it == g_slots.end() || !it->second.open) return;

    close_slot(it->second, stream);
}

void profiler_set_layer(int layer_id) {
    tl_layer_id = layer_id;
}

bool profiler_enabled() { return true; }

void profiler_reset() {
    std::lock_guard<std::mutex> lock(g_profiler_mutex);
    // Destroy all event handles
    for (auto& r : g_records) {
        if (r.start_event) hipEventDestroy(r.start_event);
        if (r.end_event)   hipEventDestroy(r.end_event);
    }
    g_records.clear();
    // Close any open slots
    for (auto& kv : g_slots) {
        if (kv.second.open) {
            if (kv.second.start_event) {
                hipEventDestroy(kv.second.start_event);
                kv.second.start_event = nullptr;
            }
            kv.second.open = false;
        }
    }
    g_slots.clear();
}

// ---------------------------------------------------------------------------
// Resolve all event pairs: synchronize and compute elapsed times
// Called once at save time, not in the hot path.
// ---------------------------------------------------------------------------
struct ResolvedRecord {
    const char* name;
    const char* category;
    int device_id;
    int layer_id;
    float duration_ms;
    int64_t wall_us;
};

static std::vector<ResolvedRecord> resolve_records() {
    // Close any still-open slots (need a default stream per device)
    for (auto& kv : g_slots) {
        if (kv.second.open) {
            close_slot(kv.second, kv.first.stream);
        }
    }

    std::vector<ResolvedRecord> resolved;
    resolved.reserve(g_records.size());

    for (auto& r : g_records) {
        // Synchronize end event to ensure GPU work is complete
        hipError_t err = hipEventSynchronize(r.end_event);
        if (err != hipSuccess) continue;

        float ms = 0.0f;
        err = hipEventElapsedTime(&ms, r.start_event, r.end_event);
        if (err == hipSuccess && ms >= 0.0f) {
            resolved.push_back({
                r.name, r.category, r.device_id,
                r.layer_id, ms, r.wall_us
            });
        }
    }
    return resolved;
}

// ---------------------------------------------------------------------------
// Chrome Trace Format export
// https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU
// ---------------------------------------------------------------------------

void profiler_save_chrome_trace(const char* path) {
    std::lock_guard<std::mutex> lock(g_profiler_mutex);

    auto resolved = resolve_records();

    FILE* f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "profiler: cannot open %s for writing\n", path);
        return;
    }

    // Find base timestamp for relative offsets
    int64_t base_us = INT64_MAX;
    for (const auto& r : resolved) {
        if (r.wall_us < base_us) base_us = r.wall_us;
    }
    if (resolved.empty()) base_us = 0;

    fprintf(f, "{\"traceEvents\":[\n");

    // Metadata: thread names = GPU devices
    for (int d = 0; d < MAX_DEVICES; d++) {
        bool has_device = false;
        for (const auto& r : resolved) {
            if (r.device_id == d) { has_device = true; break; }
        }
        if (!has_device) continue;
        fprintf(f, "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":0,\"tid\":%d,"
                    "\"args\":{\"name\":\"GPU %d\"}},\n", d, d);
    }

    for (size_t i = 0; i < resolved.size(); i++) {
        const auto& r = resolved[i];
        int64_t ts_us = r.wall_us - base_us;
        int64_t dur_us = static_cast<int64_t>(r.duration_ms * 1000.0f);
        if (dur_us < 1) dur_us = 1;

        // Build name with layer info
        char name_buf[256];
        if (r.layer_id >= 0) {
            snprintf(name_buf, sizeof(name_buf), "L%d/%s", r.layer_id, r.name);
        } else {
            snprintf(name_buf, sizeof(name_buf), "%s", r.name);
        }

        fprintf(f, "{\"name\":\"%s\",\"cat\":\"%s\",\"ph\":\"X\","
                    "\"ts\":%lld,\"dur\":%lld,\"pid\":0,\"tid\":%d,"
                    "\"args\":{\"layer\":%d,\"dur_ms\":%.4f}}",
                name_buf, r.category,
                (long long)ts_us, (long long)dur_us,
                r.device_id, r.layer_id, r.duration_ms);

        if (i + 1 < resolved.size()) fprintf(f, ",");
        fprintf(f, "\n");
    }

    fprintf(f, "]}\n");
    fclose(f);
    fprintf(stderr, "profiler: wrote %zu events to %s\n", resolved.size(), path);
}

// ---------------------------------------------------------------------------
// Summary JSON: aggregated stats by (category, name)
// ---------------------------------------------------------------------------

void profiler_save_summary_json(const char* path) {
    std::lock_guard<std::mutex> lock(g_profiler_mutex);

    auto resolved = resolve_records();

    // Aggregate by (category, name)
    struct AggKey {
        std::string name;
        std::string category;
        bool operator==(const AggKey& o) const {
            return name == o.name && category == o.category;
        }
    };
    struct AggKeyHash {
        size_t operator()(const AggKey& k) const {
            return std::hash<std::string>()(k.name) ^ (std::hash<std::string>()(k.category) << 32);
        }
    };
    struct AggStats {
        int count = 0;
        float total_ms = 0;
        float min_ms = 1e9f;
        float max_ms = 0;
    };

    std::unordered_map<AggKey, AggStats, AggKeyHash> agg;
    float total_all_ms = 0;

    for (const auto& r : resolved) {
        AggKey key{r.name, r.category};
        auto& s = agg[key];
        s.count++;
        s.total_ms += r.duration_ms;
        s.min_ms = std::min(s.min_ms, r.duration_ms);
        s.max_ms = std::max(s.max_ms, r.duration_ms);
        total_all_ms += r.duration_ms;
    }

    // Sort by total_ms descending
    struct SortEntry {
        AggKey key;
        AggStats stats;
    };
    std::vector<SortEntry> sorted;
    for (auto& kv : agg) {
        sorted.push_back({kv.first, kv.second});
    }
    std::sort(sorted.begin(), sorted.end(),
              [](const SortEntry& a, const SortEntry& b) {
                  return a.stats.total_ms > b.stats.total_ms;
              });

    FILE* f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "profiler: cannot open %s for writing\n", path);
        return;
    }

    fprintf(f, "{\n  \"total_events\": %zu,\n  \"total_ms\": %.4f,\n  \"kernels\": [\n",
            resolved.size(), total_all_ms);

    for (size_t i = 0; i < sorted.size(); i++) {
        const auto& e = sorted[i];
        float avg_ms = e.stats.total_ms / e.stats.count;
        float pct = (total_all_ms > 0) ? (e.stats.total_ms / total_all_ms * 100.0f) : 0;

        fprintf(f, "    {\"name\":\"%s\",\"category\":\"%s\","
                    "\"count\":%d,\"total_ms\":%.4f,\"avg_ms\":%.4f,"
                    "\"min_ms\":%.4f,\"max_ms\":%.4f,\"pct\":%.2f}",
                e.key.name.c_str(), e.key.category.c_str(),
                e.stats.count, e.stats.total_ms, avg_ms,
                e.stats.min_ms, e.stats.max_ms, pct);
        if (i + 1 < sorted.size()) fprintf(f, ",");
        fprintf(f, "\n");
    }

    fprintf(f, "  ]\n}\n");
    fclose(f);
    fprintf(stderr, "profiler: wrote summary (%zu kernels) to %s\n", sorted.size(), path);
}

} // namespace gptoss

#endif // GPTOSS_PROFILE
