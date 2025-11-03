/* Copyright 2025 The AI Edge Torch Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef FLASH_SLIM_UTILS_H_
#define FLASH_SLIM_UTILS_H_

#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"

#include "tflite/interpreter.h"
#include "tflite/model_builder.h"
#include "tflite/schema/schema_generated.h"
#include "tflite/signature_runner.h"


#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <errno.h>
#include <string.h>
#endif


// A minimal check macro.
#ifndef MINIMAL_CHECK
#define MINIMAL_CHECK(x)                                         \
    if (!(x))                                                    \
    {                                                            \
        fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
        exit(1);                                                 \
    }
#endif // MINIMAL_CHECK

namespace flash_slim::util
{

    // Detect which CPU cores the current process is allowed to run on (Linux-only)
    void detect_active_cores(std::vector<int> &cores);

    // Set the calling thread's affinity to the given CPU cores. Linux-only.
    void set_affinity_to_cores(const std::vector<int> &cores);

    // Assign cores for inference and io
    void set_cores_for_inference_and_io(const std::vector<int> &active_cores, std::vector<int> &cores_to_use_inference, std::vector<int> &cores_to_use_io, int requested_threads);

    // Run a function in a new std::thread while setting that thread's CPU
    // affinity to `cores` (Linux-only). Blocks until the function completes.
    // If `cores` is empty, the function runs on a plain thread without affinity.
    void run_thread_with_affinity_and_join(const std::function<void()> &fn,
                                           const std::vector<int> &cores);

    // Print current page cache size in kB (Linux-only)    
    void print_current_page_cache_kb();

    // Drop page cache (Linux-only)
    int drop_page_cache();

    int DetectKVCacheSequenceDimension(TfLiteTensor *kv_cache_tensor);

    int CountTotalNodes(tflite::Interpreter *interpreter);
    
} // namespace flash_slim::util

#endif // FLASH_SLIM_UTILS_H_
