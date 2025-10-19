# Copyright 2024 OS-Support-for-On-Device-LLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

load("@litert//tflite:build_def.bzl", "tflite_linkopts")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

licenses(["notice"])

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@litert//tflite:framework",
        "@litert//tflite:util",
        "@litert//tflite/schema:schema_fbs",
    ],
)

cc_library(
    name = "sampler",
    srcs = ["sampler.cc"],
    hdrs = ["sampler.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@litert//tflite:framework",
        "@litert//tflite:util",
        "@litert//tflite/schema:schema_fbs",
    ],
)

cc_library(
    name = "profiler",
    srcs = ["profiler.cc"],
    hdrs = ["profiler.h"],
    deps = [
        ":utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:str_format",
        "@litert//tflite:framework",
        "@litert//tflite:util",
        "@litert//tflite/delegates/xnnpack:xnnpack_delegate",
        "@litert//tflite/experimental/genai:genai_ops",
        "@litert//tflite/kernels:builtin_ops",
    ],
)

cc_library(
    name = "aligned_allocator",
    srcs = ["aligned_allocator.h"],
    deps = [
        "@litert//tflite:framework",
        "@litert//tflite:util",
    ],
)

cc_library(
    name = "lora_adapter",
    srcs = [
        "lora_adapter.cc",
    ],
    hdrs = [
        "lora_adapter.h",
    ],
    deps = [
        ":aligned_allocator",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings:str_format",
        "@litert//tflite:framework",
        "@litert//tflite:util",
        "@litert//tflite/delegates/xnnpack:xnnpack_delegate",
        "@litert//tflite/experimental/genai:genai_ops",
        "@litert//tflite/kernels:builtin_ops",
    ],
)
cc_library(
    name="prefetch_planner_util",
    srcs = ["prefetch_planner_util.cc"],
    hdrs = ["prefetch_planner_util.h"],
    deps = [
        ":utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@litert//tflite:framework",
        "@litert//tflite:util",
        "@litert//tflite/schema:schema_fbs",
        "@litert//tflite/delegates/xnnpack:xnnpack_delegate",
        "//flash-slim/nlohmann:nlohmann_json",
    ],
)

cc_library(
    name = "common",
    srcs = ["common.cc"],
    hdrs = ["common.h"],
    deps = [
        ":aligned_allocator",
        ":lora_adapter",
        ":profiler",
        ":sampler",
        ":utils",
        ":prefetch_planner_util",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings",
        "@com_google_sentencepiece//:sentencepiece_processor",
        "@litert//tflite:framework",
        "@litert//tflite:util",
        "@litert//tflite/delegates/xnnpack:xnnpack_delegate",
        "@litert//tflite/experimental/genai:genai_ops",
        "@litert//tflite/kernels:builtin_ops",
        "@litert//tflite/profiling:profile_summarizer",
        "@litert//tflite/profiling:profiler",
        "@litert//tflite/tools/benchmark:benchmark_model_lib",
        "@litert//tflite/delegates/xnnpack:weight_cache",
        "@litert//tflite/delegates/xnnpack:streaming_weight_cache",
    ],
)
cc_binary(
    name = "text_generator_main",
    srcs = [
        "text_generator_main.cc",
    ],
    defines = [
        "TF_LITE_TENSORFLOW_PROFILER",
    ],
    linkopts = tflite_linkopts(),
    deps = [
        ":common",
    ],
)


cc_binary(
    name = "prefetch_planner",
    srcs = [
        "prefetch_planner.cc",
        "prefetch_planner.h",
    ],
    defines = [
        "TF_LITE_TENSORFLOW_PROFILER",
    ],
    linkopts = tflite_linkopts(),
    deps = [
        ":common",
    ],
)