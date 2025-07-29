#ifndef FLASH_SLIM_ALIGNED_ALLOCATOR_H_
#define FLASH_SLIM_ALIGNED_ALLOCATOR_H_

#include <cstddef>
#include <cstdlib>

#include "tflite/util.h"

namespace ai_edge_torch {
namespace mem {

// TF Lite requires all buffers (including external buffers used for KV cache
// here) be `tflite::kDefaultTensorAlignment` aligned. To ensure that, we use
// this custom allocator. Please use with caution as different platforms may
// have different alignment requirements.
template <typename T>
class AlignedAllocator {
public:
    using value_type = T;

    T *allocate(std::size_t n) {
        void *ptr;
        std::size_t size = n * sizeof(T);
        std::size_t padding = tflite::kDefaultTensorAlignment -
                              (size % tflite::kDefaultTensorAlignment);
        size += padding;
        int ret = posix_memalign(&ptr, tflite::kDefaultTensorAlignment, size);
        if (ret != 0) {
            return nullptr;
        }
        return static_cast<T *>(ptr);
    }

    void deallocate(T *ptr, std::size_t n) { free(ptr); }
};

} // namespace examples
} // namespace ai_edge_torch

#endif // FLASH_SLIM_ALIGNED_ALLOCATOR_H_
