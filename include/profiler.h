#pragma once
// Kernel profiler: HIP event-based GPU timing with Chrome Trace Format export.

#include <hip/hip_runtime.h>

namespace gptoss {

#ifdef GPTOSS_PROFILE

void profiler_mark(const char* name, const char* category,
                   int device_id, hipStream_t stream);
void profiler_end(int device_id, hipStream_t stream);
void profiler_set_layer(int layer_id);
bool profiler_enabled();

void profiler_save_chrome_trace(const char* path);
void profiler_save_summary_json(const char* path);
void profiler_reset();

#define PROF(name, cat, dev, stream) ::gptoss::profiler_mark(name, cat, dev, stream)
#define PROF_END(dev, stream)        ::gptoss::profiler_end(dev, stream)
#define PROF_LAYER(layer)            ::gptoss::profiler_set_layer(layer)

#else

#define PROF(name, cat, dev, stream) ((void)0)
#define PROF_END(dev, stream)        ((void)0)
#define PROF_LAYER(layer)            ((void)0)

#endif

} // namespace gptoss
