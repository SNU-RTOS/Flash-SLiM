#ifndef FLASH_SLIM_COMMON_H
#define FLASH_SLIM_COMMON_H
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <ios>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <random>
#include <thread>
#include <iostream>
#include <unordered_set>

#include <mutex>
#include <condition_variable>

// abseil
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/string_view.h"
#include "absl/strings/match.h"

// Sentencepiece
#include "src/sentencepiece_processor.h"

// LiteRT Core
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"
#include "tflite/experimental/genai/genai_ops.h"
#include "tflite/interpreter.h"
#include "tflite/interpreter_builder.h"
#include "tflite/kernels/register.h"
#include "tflite/model_builder.h"
#include "tflite/signature_runner.h"

// LiteRT Profiling
#include "tflite/profiling/profile_summarizer.h"
#include "tflite/profiling/buffered_profiler.h"
#include "tflite/profiling/profile_summary_formatter.h"
#include "tflite/tools/benchmark/benchmark_model.h"
#include "tflite/tools/benchmark/benchmark_params.h"
#include "tflite/tools/logging.h"

// Custom
#include "utils.h"
#include "sampler.h"
#include "profiler.h"
#include "aligned_allocator.h"
#include "lora_adapter.h"
#include "weight_chunk_prefetcher.h"


namespace tflite::profiling
{
    struct ProfilerOutput
    {
        std::shared_ptr<ProfileSummaryFormatter> formatter;
        std::shared_ptr<ProfileSummarizer> init_summarizer;
        std::shared_ptr<ProfileSummarizer> run_summarizer;
        std::string output_type; // "log" or "csv"
        std::string output_path; // empty for log, file path for csv
    };
}

using ai_edge_torch::examples::LoRA;
using ai_edge_torch::mem::AlignedAllocator;

using tflite::profiling::BufferedProfiler;
using tflite::profiling::ProfilerOutput;
using tflite::profiling::ProfileSummarizer;
using tflite::profiling::ProfileSummaryCSVFormatter;
using tflite::profiling::ProfileSummaryFormatter;

using flash_slim::profiling::GenAIMetrics;



#ifdef USE_WEIGHT_STREAMING
#include "tflite/delegates/xnnpack/streaming_weight_cache.h"
using tflite::xnnpack::StreamingWeightCacheProvider;
using flash_slim::streaming::WeightChunkPrefetcher;
#include "flash-slim/weight_chunk_controller.h"
using flash_slim::streaming::WeightChunkController;
void ApplyXNNPACKWithWeightCachingProvider(tflite::Interpreter *interpreter, StreamingWeightCacheProvider *provider);
#else
#include "tflite/delegates/xnnpack/weight_cache.h"
using tflite::xnnpack::MMapWeightCacheProvider;
void ApplyXNNPACKWithWeightCachingProvider(tflite::Interpreter *interpreter);
#endif

std::map<std::string, std::vector<float, AlignedAllocator<float>>>
AllocateKVCache(tflite::Interpreter *interpreter);

void PrepareRunner(tflite::SignatureRunner *runner,
                   std::map<std::string,
                            std::vector<float, AlignedAllocator<float>>> &
                       kv_cache);

tflite::SignatureRunner *GetPrefillRunner(
    tflite::Interpreter *interpreter,
    std::size_t num_input_tokens,
    std::map<std::string, std::vector<float, AlignedAllocator<float>>> &kv_cache,
    const LoRA *lora);

tflite::SignatureRunner *GetDecodeRunner(
    tflite::Interpreter *interpreter,
    std::map<std::string, std::vector<float, AlignedAllocator<float>>> &kv_cache,
    LoRA *lora);

std::unique_ptr<sentencepiece::SentencePieceProcessor> LoadSentencePieceProcessor();


int DetectKVCacheSequenceDimension(TfLiteTensor *kv_cache_tensor);

int CountTotalNodes(tflite::Interpreter *interpreter);



#endif // FLASH_SLIM_COMMON_H