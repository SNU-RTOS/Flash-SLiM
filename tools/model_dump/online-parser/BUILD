load("@litert//tflite:build_def.bzl", "tflite_linkopts")
load("@rules_cc//cc:defs.bzl", "cc_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["@litert//tflite:license"],
    default_visibility = ["//visibility:public"],    
)

cc_binary(
    name = "dump_model_xnnpack",
    srcs = [
        "dump_model_xnnpack.cc",
    ],
    linkopts = tflite_linkopts() + select({
        "//conditions:default": [
            "-lm",  # math library
        ],
    }),
    deps = [
        "@litert//tflite:framework",
        "@litert//tflite/core:cc_api_stable",
        "@litert//tflite/kernels:builtin_ops",
        "@litert//tflite/delegates/xnnpack:xnnpack_delegate",
        "@litert//tflite/profiling:profiler",

    ],
)



cc_binary(
    name = "dump_llm",
    srcs = [
        "dump_llm.cc",
    ],
    linkopts = tflite_linkopts() + select({
        "//conditions:default": [
            "-lm",  # math library
        ],
    }),
    deps = [
        "@litert//tflite:framework",
        "@litert//tflite/core:cc_api_stable",
        "@litert//tflite/kernels:builtin_ops",
        "@litert//tflite/experimental/genai:genai_ops",
        "@litert//tflite/delegates/xnnpack:xnnpack_delegate",
        "@litert//tflite/profiling:profiler",
        "@litert//tflite:util",
        "@litert//tflite/schema:schema_fbs",
        "@litert//tflite/tools/benchmark:benchmark_model_lib",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings",
    ],
)