#pragma once

#include <cstdio>
#include <cstdlib>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

namespace gptoss {

#define NCCL_CHECK(expr)                                                       \
    do {                                                                        \
        ncclResult_t res = (expr);                                              \
        if (res != ncclSuccess) {                                               \
            fprintf(stderr, "RCCL error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    ncclGetErrorString(res));                                    \
            abort();                                                            \
        }                                                                       \
    } while (0)

} // namespace gptoss
