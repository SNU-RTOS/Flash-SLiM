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

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <functional>
#include <thread>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tflite/interpreter.h"
#include "tflite/signature_runner.h"
#include "tflite/util.h"

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

    //DEPRECATED
    // // Get current timestamp in nanoseconds since epoch
    // int64_t getCurrentTimestampNs();

    // // Format timestamp as JSON object with seconds and nanoseconds
    // std::string formatTimestampJson(int64_t timestamp_ns, const std::string &event_type,
    //                                 const std::string &component, int stage_idx = -1);

    // // Log an event with timestamp in JSON format
    // void logTimestampedEvent(const std::string &event_type, const std::string &component,
    //                          int stage_idx = -1, std::ostream &out = std::cout);

    // // Structured logging helper for detailed event information
    // void logJsonEvent(const std::string &event_type, const std::string &component,
    //                   const std::map<std::string, std::string> &attributes,
    //                   int stage_idx = -1, std::ostream &out = std::cout);

    // const char *TfLiteTypeToString(TfLiteType type);
    // void PrintTensorInfo(const TfLiteTensor *tensor, const char *tensor_name);
    // void PrintSignatureRunnersInfo(tflite::Interpreter *interpreter);

    //
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

    
} // namespace custom::util

#endif // FLASH_SLIM_UTILS_H_
