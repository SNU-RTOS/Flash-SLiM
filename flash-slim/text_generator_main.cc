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

// USDT Probes for eBPF tracing with stage index
// Stage indices: 0=Load_Model, 1=Build_Interpreter, 2=Load_Tokenizer, 3=Build_KV_Cache, 4=Prepare_Prompt, 5=Prepare_Signature_Runners, 6=Prefill, 7=Decode_Generation
#ifdef EBPF_TRACE_ENABLED
#include <sys/sdt.h>
#define TRACE_LOGIC_START(stage_idx) DTRACE_PROBE1(tflite_gen, logic_start, stage_idx)
#define TRACE_LOGIC_END(stage_idx) DTRACE_PROBE1(tflite_gen, logic_end, stage_idx)
#define TRACE_IO_START(stage_idx) DTRACE_PROBE1(tflite_gen, io_start, stage_idx)
#define TRACE_IO_END(stage_idx) DTRACE_PROBE1(tflite_gen, io_end, stage_idx)
#else
#define TRACE_LOGIC_START(stage_idx)
#define TRACE_LOGIC_END(stage_idx)
#define TRACE_IO_START(stage_idx)
#define TRACE_IO_END(stage_idx)
#endif

// AI EDGE TORCH & LiteRT
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/string_view.h"
#include "absl/strings/match.h"
#include "src/sentencepiece_processor.h"
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"
#include "tflite/experimental/genai/genai_ops.h"
#include "tflite/interpreter.h"
#include "tflite/interpreter_builder.h"
#include "tflite/kernels/register.h"
#include "tflite/model_builder.h"
#include "tflite/signature_runner.h"

#include "utils.h"
#include "sampler.h"
#include "profiler.h"

// ----------------------
// absl::FLAGS definition
// ----------------------
ABSL_FLAG(std::string, tflite_model, "",
          "Two-signature tflite model for text generation using ODML tools.");
ABSL_FLAG(std::string, sentencepiece_model, "", "Path to the SentencePiece model file.");
ABSL_FLAG(std::string, prompt, "Write an email:", "Input prompt for the model.");
ABSL_FLAG(int, max_decode_steps, -1,
          "Number of tokens to generate. Defaults to the KV cache limit.");
ABSL_FLAG(std::string, start_token, "",
          "Optional start token appended to the beginning of the input prompt.");
ABSL_FLAG(std::string, stop_token, "",
          "Optional stop token that stops the decoding loop if encountered.");
ABSL_FLAG(int, num_threads, 4, "Number of threads to use. Defaults to 4.");
ABSL_FLAG(std::string, weight_cache_path, "",
          "Path for XNNPACK weight caching, e.g., /tmp/model.xnnpack_cache.");
ABSL_FLAG(std::string, lora_path, "", "Optional path to a LoRA artifact.");
ABSL_FLAG(float, temperature, 0.8f, "Temperature for sampling. Higher values make output more random. Defaults to 0.8");
ABSL_FLAG(int, top_k, 40, "Top-k sampling parameter. Only consider the top k tokens. Defaults to 40.");
ABSL_FLAG(float, top_p, 0.9f, "Top-p (nucleus) sampling parameter. Only consider tokens with cumulative probability <= top_p. Defaults to 0.9.");
ABSL_FLAG(float, repetition_penalty, 1.2f, "Repetition penalty for sampling. Higher values reduce repetition. Defaults to 1.2.");
ABSL_FLAG(bool, enable_repetition_penalty, false, "Enable repetition penalty. Defaults to false.");

namespace
{
    using ai_edge_torch::examples::AlignedAllocator;
    using ai_edge_torch::examples::LoRA;

    // --------------------------------------------------------------------------
    // Utility for applying XNNPACK weight caching
    // --------------------------------------------------------------------------
    void ApplyXNNPACKWeightCaching(tflite::Interpreter *interpreter)
    {
        auto delegate_options = TfLiteXNNPackDelegateOptionsDefault();
        std::string weight_cache_path = absl::GetFlag(FLAGS_weight_cache_path);
        delegate_options.weight_cache_file_path = weight_cache_path.c_str();
        delegate_options.num_threads = absl::GetFlag(FLAGS_num_threads);
        // delegate_options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_SUBGRAPH_RESHAPING;
        // delegate_options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;

        MINIMAL_CHECK(interpreter->ModifyGraphWithDelegate(
                          tflite::Interpreter::TfLiteDelegatePtr(
                              TfLiteXNNPackDelegateCreate(&delegate_options),
                              [](TfLiteDelegate *delegate)
                              { TfLiteXNNPackDelegateDelete(delegate); })) == kTfLiteOk);
    }

    // --------------------------------------------------------------------------
    // Loads the TFLite model
    // --------------------------------------------------------------------------
    std::unique_ptr<tflite::FlatBufferModel> LoadModel()
    {
        std::unique_ptr<tflite::FlatBufferModel> model =
            tflite::FlatBufferModel::BuildFromFile(absl::GetFlag(FLAGS_tflite_model).c_str());
        MINIMAL_CHECK(model != nullptr);
        return model;
    }

    // --------------------------------------------------------------------------
    // Builds a TFLite interpreter from the model and applies XNNPACK if requested
    // --------------------------------------------------------------------------
    std::unique_ptr<tflite::Interpreter>
    BuildInterpreter(tflite::FlatBufferModel *model, int num_threads)
    {
        tflite::ops::builtin::BuiltinOpResolver resolver;
        // Register GenAI custom ops
        tflite::ops::custom::GenAIOpsRegisterer(&resolver);

        tflite::InterpreterBuilder builder(*model, resolver);
        MINIMAL_CHECK(builder.SetNumThreads(num_threads) == kTfLiteOk);

        std::unique_ptr<tflite::Interpreter> interpreter;
        builder(&interpreter);
        MINIMAL_CHECK(interpreter != nullptr);

        if (!absl::GetFlag(FLAGS_weight_cache_path).empty())
        {
            ApplyXNNPACKWeightCaching(interpreter.get());
        }
        return interpreter;
    }

    // --------------------------------------------------------------------------
    // Constructs KV cache input structures for decode, based on the decode signature
    // --------------------------------------------------------------------------
    std::map<std::string, std::vector<float, AlignedAllocator<float>>>
    BuildKVCache(tflite::Interpreter *interpreter)
    {
        tflite::SignatureRunner *runner = interpreter->GetSignatureRunner("decode");
        if (runner == nullptr)
        {
            return {};
        }

        // Expect runner->input_size() = tokens, input_pos, plus 2*(num_layers)
        size_t num_layers = (runner->input_size() - 2) / 2;
        if (num_layers == 0)
        {
            return {};
        }

        std::map<std::string, std::vector<float, AlignedAllocator<float>>> kv_cache;
        for (int i = 0; i < num_layers; ++i)
        {
            std::string k_cache_name = "kv_cache_k_" + std::to_string(i);
            std::string v_cache_name = "kv_cache_v_" + std::to_string(i);

            TfLiteTensor *tensor = runner->input_tensor(k_cache_name.c_str());
            size_t count = tensor->bytes / sizeof(float);

            kv_cache.emplace(k_cache_name,
                             std::vector<float, AlignedAllocator<float>>(count, 0.0f));
            kv_cache.emplace(v_cache_name,
                             std::vector<float, AlignedAllocator<float>>(count, 0.0f));
        }
        return kv_cache;
    }

    // --------------------------------------------------------------------------
    // Sets custom memory allocations for the KV cache on the given runner
    // --------------------------------------------------------------------------
    void PrepareRunner(tflite::SignatureRunner *runner,
                       std::map<std::string, std::vector<float, AlignedAllocator<float>>> &kv_cache)
    {
        for (auto &[name, cache] : kv_cache)
        {
            TfLiteCustomAllocation allocation{
                .data = static_cast<void *>(cache.data()),
                .bytes = cache.size() * sizeof(float)};

            MINIMAL_CHECK(runner->SetCustomAllocationForInputTensor(name.c_str(), allocation) == kTfLiteOk);
            MINIMAL_CHECK(runner->SetCustomAllocationForOutputTensor(name.c_str(), allocation) == kTfLiteOk);
        }
        MINIMAL_CHECK(runner->AllocateTensors() == kTfLiteOk);
    }

    // --------------------------------------------------------------------------
    // Finds the appropriate "prefill" runner for the given number of tokens.
    // If LoRA is used, it defers to LoRA's specialized runner selection.
    // --------------------------------------------------------------------------
    tflite::SignatureRunner *GetPrefillRunner(
        tflite::Interpreter *interpreter,
        std::size_t num_input_tokens,
        std::map<std::string, std::vector<float, AlignedAllocator<float>>> &kv_cache,
        const ai_edge_torch::examples::LoRA *lora)
    {
        tflite::SignatureRunner *runner = nullptr;
        int best_seq_size = -1;
        int delta = std::numeric_limits<int>::max();

        for (const std::string *key : interpreter->signature_keys())
        {
            if (!absl::StrContains(*key, "prefill") || absl::StrContains(*key, "lora"))
            {
                continue;
            }
            TfLiteTensor *input_pos =
                interpreter->GetSignatureRunner(key->c_str())->input_tensor("input_pos");
            int seq_size = input_pos->dims->data[0];

            // Choose the runner where seq_size >= num_input_tokens and
            // (seq_size - num_input_tokens) is minimized
            if (num_input_tokens <= static_cast<size_t>(seq_size) &&
                seq_size - static_cast<int>(num_input_tokens) < delta)
            {
                if (lora == nullptr)
                {
                    runner = interpreter->GetSignatureRunner(key->c_str());
                }
                best_seq_size = seq_size;
                delta = seq_size - static_cast<int>(num_input_tokens);
            }
        }

        // If LoRA is enabled, use the LoRA-specific prefill runner
        if (lora != nullptr)
        {
            runner = lora->GetPrefillRunner(interpreter, best_seq_size);
        }
        MINIMAL_CHECK(runner != nullptr);

        // Prepare KV memory allocations
        PrepareRunner(runner, kv_cache);
        return runner;
    }

    // --------------------------------------------------------------------------
    // Retrieves the decode runner (LoRA-based if needed) and prepares it
    // --------------------------------------------------------------------------
    tflite::SignatureRunner *GetDecodeRunner(
        tflite::Interpreter *interpreter,
        std::map<std::string, std::vector<float, AlignedAllocator<float>>> &kv_cache,
        ai_edge_torch::examples::LoRA *lora)
    {
        tflite::SignatureRunner *runner =
            (lora == nullptr)
                ? interpreter->GetSignatureRunner("decode")
                : lora->GetDecodeRunner(interpreter);
        MINIMAL_CHECK(runner != nullptr);

        PrepareRunner(runner, kv_cache);
        return runner;
    }

    // --------------------------------------------------------------------------
    // Loads the SentencePiece model from file
    // --------------------------------------------------------------------------
    std::unique_ptr<sentencepiece::SentencePieceProcessor> LoadSentencePieceProcessor()
    {
        std::ifstream input(absl::GetFlag(FLAGS_sentencepiece_model), std::ios::binary);
        std::string serialized_proto((std::istreambuf_iterator<char>(input)),
                                     std::istreambuf_iterator<char>());

        auto processor = std::make_unique<sentencepiece::SentencePieceProcessor>();
        MINIMAL_CHECK(processor->LoadFromSerializedProto(serialized_proto).ok());
        return processor;
    }

    // --------------------------------------------------------------------------
    // Helper function to detect the sequence dimension in KV cache tensor
    // --------------------------------------------------------------------------
    int DetectKVCacheSequenceDimension(TfLiteTensor *kv_cache_tensor)
    {
        if (kv_cache_tensor == nullptr || kv_cache_tensor->dims == nullptr)
        {
            return -1;
        }

        int num_dims = kv_cache_tensor->dims->size;
        if (num_dims < 2)
        {
            return -1;
        }

        // Print tensor dimensions for debugging
        std::cout << "[INFO] KV Cache tensor dims: [";
        for (int i = 0; i < num_dims; ++i)
        {
            std::cout << kv_cache_tensor->dims->data[i] << (i == num_dims - 1 ? "" : ", ");
        }
        std::cout << "]\n";

        // Check different known patterns
        if (num_dims == 4)
        {
            // Pattern 1: [batch, seq_len, num_heads, head_dim] - e.g., [1, 1280, 3, 64]
            if (kv_cache_tensor->dims->data[1] > 100 && kv_cache_tensor->dims->data[2] < 20)
            {
                std::cout << "[INFO] Detected pattern [batch, seq_len, num_heads, head_dim]\n";
                return 1; // sequence dimension is at index 1
            }
            // Pattern 2: [batch, batch, seq_len, hidden_dim,] - e.g., [1, 1, 1280, 256,]
            else if (kv_cache_tensor->dims->data[1] == 1 && kv_cache_tensor->dims->data[2] > 100)
            {
                std::cout << "[INFO] Detected pattern [batch, batch, seq_len, hidden_dim,]\n";
                return 2; // sequence dimension is at index 2
            }
        }

        // Default fallback: assume sequence dimension is at index 1
        std::cout << "[INFO] Using default: sequence dimension at index 1\n";
        return 1;
    }

} // end anonymous namespace

void __set_affinity_to_cores(const std::vector<int> &cores)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int core : cores)
    {
        CPU_SET(core, &cpuset);
    }
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0)
    {
        perror("Failed to set affinity");
    }
}

void __run_main(custom::profiler::PhaseContext &phase_ctx,
                custom::profiler::GenerationMetrics &generation_metrics)
{

    // Global variables
    std::vector<int> prompt_tokens;
    std::string prompt, start_token, stop_token;
    int stop_token_id = -1;
    std::unordered_set<int> previously_generated_tokens;

    // 1. Load Model
    std::unique_ptr<tflite::FlatBufferModel> model;
    TRACE_LOGIC_START(0); // Load_Model
    TRACE_IO_START(0);
    {
        custom::profiler::ScopeEventPrefetcher prefetcher(phase_ctx, "Load_Model");

        model = LoadModel();
    }
    TRACE_IO_END(0);
    TRACE_LOGIC_END(0);

    // 2. Build Interpreter
    TRACE_LOGIC_START(1); // Build_Interpreter
    TRACE_IO_START(1);
    std::unique_ptr<tflite::Interpreter> interpreter;
    {
        custom::profiler::ScopeEventPrefetcher prefetcher(phase_ctx, "Build_Interpreter");
        interpreter = BuildInterpreter(model.get(), absl::GetFlag(FLAGS_num_threads));
    }
    TRACE_IO_END(1);
    TRACE_LOGIC_END(1);

    // Print signature runner info
    // PrintSignatureRunnersInfo(interpreter.get());

    // 3. Load SentencePiece
    TRACE_LOGIC_START(2); // Load_Tokenizer
    TRACE_IO_START(2);
    std::unique_ptr<sentencepiece::SentencePieceProcessor> sp_processor;
    {
        custom::profiler::ScopeEventPrefetcher prefetcher(phase_ctx, "Load_Tokenizer");
        sp_processor = LoadSentencePieceProcessor();
    }
    TRACE_IO_END(2);
    TRACE_LOGIC_END(2);

    // 4. Build KV Cache
    TRACE_LOGIC_START(3); // Build_KV_Cache
    TRACE_IO_START(3);
    std::map<std::string, std::vector<float, AlignedAllocator<float>>> kv_cache;
    {
        custom::profiler::ScopeEventPrefetcher prefetcher(phase_ctx, "Build_KV_Cache");
        kv_cache = BuildKVCache(interpreter.get());
        MINIMAL_CHECK(!kv_cache.empty());
    }
    TRACE_IO_END(3);
    TRACE_LOGIC_END(3);

    // 5. Optionally load LoRA
    // std::unique_ptr<ai_edge_torch::examples::LoRA> lora = nullptr;
    // {
    //     custom::profiler::ScopeTimer timer("LoRA Loading");
    //     if (!absl::GetFlag(FLAGS_lora_path).empty())
    //     {
    //         lora = ai_edge_torch::examples::LoRA::FromFile(absl::GetFlag(FLAGS_lora_path));
    //         MINIMAL_CHECK(lora != nullptr);
    //     }
    // }

    // 6. Prepare Input Prompt
    TRACE_LOGIC_START(4); // Prepare_Prompt
    TRACE_IO_START(4);
    {
        custom::profiler::ScopeEventPrefetcher prefetcher(phase_ctx, "Prepare_Prompt");
        prompt = absl::GetFlag(FLAGS_prompt);
        MINIMAL_CHECK(sp_processor->Encode(prompt, &prompt_tokens).ok());

        start_token = absl::GetFlag(FLAGS_start_token);
        if (!start_token.empty())
        {
            prompt_tokens.insert(prompt_tokens.begin(), sp_processor->PieceToId(start_token));
        }

        stop_token = absl::GetFlag(FLAGS_stop_token);
        if (!stop_token.empty())
        {
            stop_token_id = sp_processor->PieceToId(stop_token);
            std::cout << "[INFO] Stop token ID: " << stop_token_id << " for token: " << stop_token << "\n";
        }

        // Initialize previously generated tokens with prompt tokens for repetition penalty
        if (absl::GetFlag(FLAGS_enable_repetition_penalty))
        {
            for (int token : prompt_tokens)
            {
                previously_generated_tokens.insert(token);
            }
        }
    }
    TRACE_IO_END(4);
    TRACE_LOGIC_END(4);

    // 7. Prepare Signature Runners
    TRACE_LOGIC_START(5); // Prepare_Signature_Runners
    TRACE_IO_START(5);
    tflite::SignatureRunner *prefill_runner = nullptr;
    tflite::SignatureRunner *decode_runner = nullptr;
    std::size_t effective_prefill_token_size = (prompt_tokens.size() > 0) ? (prompt_tokens.size() - 1) : 0;
    {
        custom::profiler::ScopeEventPrefetcher prefetcher(phase_ctx, "Prepare_Signature_Runners");
        prefill_runner = GetPrefillRunner(interpreter.get(), effective_prefill_token_size, kv_cache, nullptr);
        MINIMAL_CHECK(prefill_runner != nullptr);

        decode_runner = GetDecodeRunner(interpreter.get(), kv_cache, nullptr);
        MINIMAL_CHECK(decode_runner != nullptr);
    }
    TRACE_IO_END(5);
    TRACE_LOGIC_END(5);

    TfLiteTensor *prefill_input = nullptr;
    TfLiteTensor *prefill_input_pos = nullptr;
    TfLiteTensor *decode_input = nullptr;
    TfLiteTensor *decode_input_pos = nullptr;
    TfLiteTensor *kv_cache_k_0 = nullptr;
    int max_seq_size = 0;
    int kv_cache_max_size = 0;
    int prefill_seq_size = 0;

    // 8. Access Tensors
    // Get the input tensors for prefill and decode
    {
        custom::profiler::ScopeEventPrefetcher prefetcher(phase_ctx, "Prepare_Input_Tensor");

        prefill_input = prefill_runner->input_tensor("tokens");
        prefill_input_pos = prefill_runner->input_tensor("input_pos");
        decode_input = decode_runner->input_tensor("tokens");
        decode_input_pos = decode_runner->input_tensor("input_pos");
        kv_cache_k_0 = decode_runner->input_tensor("kv_cache_k_0");
        max_seq_size = prefill_input->dims->data[1];

        // Detect KV cache sequence dimension and set max size accordingly
        int seq_dim_index = DetectKVCacheSequenceDimension(kv_cache_k_0);
        if (seq_dim_index >= 0 && seq_dim_index < kv_cache_k_0->dims->size)
        {
            kv_cache_max_size = kv_cache_k_0->dims->data[seq_dim_index];
        }
        else
        {
            // Fallback to default behavior
            kv_cache_max_size = kv_cache_k_0->dims->data[1];
        }

        std::cout << "[INFO] KV Cache Max Size: " << kv_cache_max_size << " (from dimension index " << seq_dim_index << ")\n";

        prefill_seq_size = std::min<int>(prompt_tokens.size(), max_seq_size);

        // Zero out the input tensors
        std::memset(prefill_input->data.i32, 0, prefill_input->bytes);
        std::memset(prefill_input_pos->data.i32, 0, prefill_input_pos->bytes);

        // Prefill uses all but the last token from the prompt
        for (int i = 0; i < prefill_seq_size - 1; ++i)
        {
            prefill_input->data.i32[i] = prompt_tokens[i];
            prefill_input_pos->data.i32[i] = i;
        }
    }

    // 9. Prefill Stage
    TRACE_LOGIC_START(6); // Prefill
    TRACE_IO_START(6);
    double prefill_time_ms = 0.0;
    {
        custom::profiler::ScopeTimer prefill_timer(prefill_time_ms);
        custom::profiler::ScopeEventPrefetcher prefetcher(phase_ctx, "Prefill");
        MINIMAL_CHECK(prefill_runner->Invoke() == kTfLiteOk); // Execute the prefill runner
    }
    generation_metrics.RecordPrefillTime(prefill_time_ms);
    TRACE_IO_END(6);
    TRACE_LOGIC_END(6);

    // 10. Decoding Stage with separate metrics for inference and sampling

    // Determine how many tokens to generate
    int max_decode_steps = (absl::GetFlag(FLAGS_max_decode_steps) == -1)
                               ? kv_cache_max_size
                               : absl::GetFlag(FLAGS_max_decode_steps);

    int next_token_id = prompt_tokens[prefill_seq_size - 1];
    int next_position = prefill_seq_size - 1;
    int decode_steps = std::min<int>(max_decode_steps, kv_cache_max_size - prefill_seq_size);
    double inference_time_ms = 0.0;
    double sampling_time_ms = 0.0;
    double detok_time_ms = 0.0;

    std::cout << "[INFO] Tokens in Prompt: " << prompt_tokens.size() << "\n";
    std::cout << "[INFO] Tokens to Generate: " << decode_steps << "\n";
    std::cout << "[INFO] Limits of Tokens to Generate: " << kv_cache_max_size << "\n";
    std::cout << "\nPrompt:\n"
              << prompt
              << "\n\nOutput Text:\n"
              << std::endl;

    MINIMAL_CHECK(decode_steps > 0);

    // Decoding loop
    for (int i = 0; i < decode_steps; ++i)
    {
        std::string stage_name = "Decode_" + std::to_string(i);

        TRACE_LOGIC_START(7); // Decode_Generation
        TRACE_IO_START(7);
        std::string single_decoded_text;
        {
            custom::profiler::ScopeEventPrefetcher prefetcher(phase_ctx, stage_name);

            // 1) Model Inference
            {
                custom::profiler::ScopeTimer timer(inference_time_ms);

                decode_input->data.i32[0] = next_token_id;
                decode_input_pos->data.i32[0] = next_position;
                MINIMAL_CHECK(decode_runner->Invoke() == kTfLiteOk);
            }

            // 2) Token Sampling
            {
                custom::profiler::ScopeTimer timer(sampling_time_ms);
                if (absl::GetFlag(FLAGS_enable_repetition_penalty))
                {
                    next_token_id = ai_edge_torch::custom::sampler::temperature_top_k_top_p_repetition_sampler(
                        decode_runner->output_tensor("logits"),
                        absl::GetFlag(FLAGS_temperature),
                        absl::GetFlag(FLAGS_top_k),
                        absl::GetFlag(FLAGS_top_p),
                        previously_generated_tokens,
                        absl::GetFlag(FLAGS_repetition_penalty));
                }
                else
                {
                    next_token_id = ai_edge_torch::custom::sampler::temperature_top_k_top_p_sampler(
                        decode_runner->output_tensor("logits"),
                        absl::GetFlag(FLAGS_temperature),
                        absl::GetFlag(FLAGS_top_k),
                        absl::GetFlag(FLAGS_top_p));
                }
            }

            // 3) Token Detokenization
            {
                custom::profiler::ScopeTimer timer(detok_time_ms);
                std::vector<int> next_token = {next_token_id};
                MINIMAL_CHECK(sp_processor->Decode(next_token, &single_decoded_text).ok());
            }
        }
        TRACE_IO_END(7);
        TRACE_LOGIC_END(7);

        generation_metrics.RecordDecodingTime(inference_time_ms,
                                              sampling_time_ms,
                                              detok_time_ms);

        // Check if the next token is a stop token
        if (next_token_id == stop_token_id)
            break;

        // Add the generated token to previously generated tokens for repetition penalty
        if (absl::GetFlag(FLAGS_enable_repetition_penalty))
        {
            previously_generated_tokens.insert(next_token_id);
        }

        std::cout << single_decoded_text << std::flush;
        next_position++;
    }
    std::cout << "\n\n\n";
    std::cout << "[INFO] Decoded " << decode_steps << " tokens.\n";
    std::cout << "\n[INFO] Decoding stage completed" << std::endl;
}

// =======================================================================
// main() entry
// =======================================================================
int main(int argc, char *argv[])
{
    // set precision
    std::cout.precision(5);
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout << std::boolalpha;
    std::cout << "\n[INFO] Text Generation App on LiteRT Interperter\n";

#ifdef EBPF_TRACE_ENABLED
    std::cout << "[INFO] eBPF tracing is enabled. USDT probes will be used.\n";
#endif

    // 0. Parse flags
    std::cout << "[INFO] Preparing Required Components\n";
    absl::ParseCommandLine(argc, argv);

    // Check which cores we're actually running on
    std::vector<int> active_cores;
    custom::profiler::detect_active_cores(active_cores);

    if (active_cores.size() < 2)
    {
        std::cerr << "[ERROR] At least 2 cores are required.\n";
        return -1;
    }

    // 메인 스레드 코어 설정
    std::vector<int> monitor_core = {active_cores[0]};
    std::vector<int> worker_cores(active_cores.begin() + 1, active_cores.end());
    std::cout << "[INFO] Core used for logging and monitoring: " << monitor_core[0] << "\n";
    std::cout << "[INFO] Cores used for text generation: ";
    for (const auto &core : worker_cores)
    {
        std::cout << core << " ";
    }
    std::cout << "[INFO] Start Generating Text" << std::endl;
    // Just monitor the cores we're allowed to run on (should be only core 0 with taskset)

    custom::profiler::PhaseContext profile_ctx;
    std::vector<custom::profiler::RUsageRecord> rusage_records;
    custom::profiler::GenerationMetrics generation_metrics;
    std::thread monitor_thread([&]()
                               {
        __set_affinity_to_cores(monitor_core);
        custom::profiler::ScopeEventListener listener(profile_ctx, false, &rusage_records);
        listener.Run(); });

    std::thread generate_thread([&]()
                                {
        __set_affinity_to_cores(worker_cores);
        __run_main(profile_ctx, generation_metrics); });

    generate_thread.join(); // Wait for the generate thread to finish
    profile_ctx.generation_done.store(true);
    profile_ctx.signal_cv.notify_all();
    monitor_thread.join(); // Wait for the monitor thread to finish
    
    // Print RUsage results
    custom::profiler::print_rusage_records(rusage_records);

    // Print genenration metrics (inference vs. sampling)
    generation_metrics.PrintMetrics();

    std::cout << "\n[INFO] Text Generation App completed successfully.\n";
    std::cout << "---------------------------------------------------\n\n";

    return 0;
}