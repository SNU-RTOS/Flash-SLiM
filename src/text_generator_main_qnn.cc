// ============================================================================
// Text generation example (LiteRT) – QNN Delegate version
// ============================================================================
//  * Default: QNN CPU / HTA / DSP delegate (depends on SoC & SDK)
//  * Fallback: XNNPACK CPU delegate  (optional weight cache)
// ----------------------------------------------------------------------------
// 2025‑05‑22 – adapted from original GPU/XNNPACK example
// ----------------------------------------------------------------------------

// ----------------- C++ standard library -----------------
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

// ----------------- AI_EDGE_TORCH -----------------
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/string_view.h"
#include "absl/strings/match.h"
#include "src/sentencepiece_processor.h"
// ----------------- LiteRT and Delegates -----------------
// QNN (Qualcomm® Neural Network) delegate
#include "TFLiteDelegate/QnnTFLiteDelegate.h" // for QNN delegate
// XNNPACK for CPU fallback / weight‑cache path
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/experimental/genai/genai_ops.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/signature_runner.h"

// ------------- Third‑party / project headers -------------
#include "utils.h"
#include "sampler.h"
#include "profiler.h"

// --------------------------------------------------------------------------
// ABSEIL Flags
// --------------------------------------------------------------------------
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

namespace
{
    using ai_edge_torch::examples::AlignedAllocator;
    using ai_edge_torch::examples::LoRA;


    // --------------------------------------------------------------------------
    // QNN delegate helpers
    // --------------------------------------------------------------------------
    TfLiteDelegate* CreateQnnDelegate() {
        // Build basic delegate options
        TfLiteQnnDelegateOptions options = TfLiteQnnDelegateOptionsDefault();
        
        // Backend selection
        options.backend_type = kHtpBackend; //	Qualcomm Hexagon Tensor Processor (HTP), 고성능 NPU backend
        // options.backend_type = kGpuBackend; // GPU backend 
        // options.backend_type = kDspBackend; // Hexagon DSP backend (HTP보다 일반적 DSP 오프로드용)

        return TfLiteQnnDelegateCreate(&options);
    }

    void ApplyQnnDelegate(tflite::Interpreter* interp) {
        auto* delegate = CreateQnnDelegate();
        MINIMAL_CHECK(interp->ModifyGraphWithDelegate(
            tflite::Interpreter::TfLiteDelegatePtr(
                delegate, [](TfLiteDelegate* d){ TfLiteQnnDelegateDelete(d); })) == kTfLiteOk);
    }


    // --------------------------------------------------------------------------
    // Utility for applying XNNPACK weight caching
    // --------------------------------------------------------------------------
    void ApplyXNNPACKWeightCaching(tflite::Interpreter *interpreter)
    {
        auto delegate_options = TfLiteXNNPackDelegateOptionsDefault();
        std::string weight_cache_path = absl::GetFlag(FLAGS_weight_cache_path);
        delegate_options.weight_cache_file_path = weight_cache_path.c_str();
        delegate_options.num_threads = absl::GetFlag(FLAGS_num_threads);
        delegate_options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_SUBGRAPH_RESHAPING;
        delegate_options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;

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
    // Builds a TFLite interpreter from the model and applies Delegates if requested
    // --------------------------------------------------------------------------
    std::unique_ptr<tflite::Interpreter> BuildInterpreter(tflite::FlatBufferModel* model, int num_threads) {
        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::ops::custom::GenAIOpsRegisterer(&resolver);

        tflite::InterpreterBuilder builder(*model, resolver);
        builder.SetNumThreads(num_threads);

        std::unique_ptr<tflite::Interpreter> interpreter;
        builder(&interpreter);
        MINIMAL_CHECK(interpreter != nullptr);

        
        ApplyQnnDelegate(interpreter.get());              // ✅ QNM

        if (!absl::GetFlag(FLAGS_weight_cache_path).empty()) {
            ApplyXNNPACKWeightCaching(interpreter.get());      // (fallback) CPU-XNNPACK
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

} // end anonymous namespace

// =======================================================================
// main() entry
// =======================================================================
int main(int argc, char *argv[])
{
    //set precision
    std::cout.precision(3);
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout << std::boolalpha;
    std::cout << "[INFO] Text Generation App on LiteRT Interperter\n";

    // 0. Parse flags
    std::cout << "[INFO] Preparing Required Components\n";
    absl::ParseCommandLine(argc, argv);

    // Global variables
    std::vector<int> prompt_tokens;
    std::string prompt, start_token, stop_token;
    int stop_token_id = -1;

    // 0-1. Perf monitor initialziation
    // Check which cores we're actually running on
    std::vector<int> active_cores;
    ai_edge_torch::custom::profiler::detect_active_cores(active_cores);

    // Just monitor the cores we're allowed to run on (should be only core 0 with taskset)
    ai_edge_torch::custom::profiler::PerformanceMonitor perf_monitor(active_cores);
    ai_edge_torch::custom::profiler::PerformanceMetrics metrics;

    // 0-2. Variable for CPU time only
    rusage usage_start, usage_end;

    // 1. Load Model
    std::unique_ptr<tflite::FlatBufferModel> model;
    {
        ai_edge_torch::custom::profiler::ScopeLogger logger("Model_Loading", perf_monitor, metrics, usage_start, usage_end);
        model = LoadModel();
    }

    // 2. Build Interpreter
    std::unique_ptr<tflite::Interpreter> interpreter;
    {
        ai_edge_torch::custom::profiler::ScopeLogger logger("Interpreter_Building", perf_monitor, metrics, usage_start, usage_end);
        interpreter = BuildInterpreter(model.get(), absl::GetFlag(FLAGS_num_threads));
    }

    // Tensor upload before prefill
    /*
    {
        ai_edge_torch::custom::profiler::ScopeTimer timer("Tensor Uploading", perf_monitor, metrics, usage_start, usage_end);
        
        // Uploading Here
        // upload_tensors_for_all_subgraphs(interpreter.get());
       
    }
    */
    
    // 3. Load SentencePiece
    std::unique_ptr<sentencepiece::SentencePieceProcessor> sp_processor;
    {
        ai_edge_torch::custom::profiler::ScopeLogger logger("SentencePiece_Loading", perf_monitor, metrics, usage_start, usage_end);
        sp_processor = LoadSentencePieceProcessor();
    }

    // 4. Build KV Cache
    std::map<std::string, std::vector<float, AlignedAllocator<float>>> kv_cache;
    {
        ai_edge_torch::custom::profiler::ScopeLogger logger("KV_Cache_Building", perf_monitor, metrics, usage_start, usage_end);
        kv_cache = BuildKVCache(interpreter.get());
        MINIMAL_CHECK(!kv_cache.empty());
    }

    // 5. Optionally load LoRA
    // std::unique_ptr<ai_edge_torch::examples::LoRA> lora = nullptr;
    // {
    //     ai_edge_torch::custom::profiler::ScopeTimer timer("LoRA Loading");
    //     if (!absl::GetFlag(FLAGS_lora_path).empty())
    //     {
    //         lora = ai_edge_torch::examples::LoRA::FromFile(absl::GetFlag(FLAGS_lora_path));
    //         MINIMAL_CHECK(lora != nullptr);
    //     }
    // }

    /* *********************
     * PREFILL PREPROCESS START
     * *********************/
    // 6. Prepare Input Prompt
    {
        ai_edge_torch::custom::profiler::ScopeLogger logger("Prompt_Preparation", perf_monitor, metrics, usage_start, usage_end);
        
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
        }
        
    }
    
    // 7. Prepare Signature Runners
    tflite::SignatureRunner *prefill_runner = nullptr;
    tflite::SignatureRunner *decode_runner = nullptr;
    std::size_t effective_prefill_token_size = (prompt_tokens.size() > 0) ? (prompt_tokens.size() - 1) : 0;
    {
        ai_edge_torch::custom::profiler::ScopeLogger logger("Signature_Runners_Preparation", perf_monitor, metrics, usage_start, usage_end);

        prefill_runner = GetPrefillRunner(interpreter.get(), effective_prefill_token_size, kv_cache, nullptr);
        MINIMAL_CHECK(prefill_runner != nullptr);

        decode_runner = GetDecodeRunner(interpreter.get(), kv_cache, nullptr);
        MINIMAL_CHECK(decode_runner != nullptr);

    }

    TfLiteTensor *prefill_input = nullptr;
    TfLiteTensor *prefill_input_pos = nullptr;
    TfLiteTensor *decode_input = nullptr;
    TfLiteTensor *decode_input_pos = nullptr;
    TfLiteTensor *kv_cache_k_0 = nullptr;
    int max_seq_size = 0;
    int kv_cache_max_size = 0;

    // 8. Access Tensors
    {
        ai_edge_torch::custom::profiler::ScopeLogger logger("Prefill_Input_Tensors_Access", perf_monitor, metrics, usage_start, usage_end);
        
        // Get the input tensors for prefill and decode
        prefill_input = prefill_runner->input_tensor("tokens");
        prefill_input_pos = prefill_runner->input_tensor("input_pos");
        decode_input = decode_runner->input_tensor("tokens");
        decode_input_pos = decode_runner->input_tensor("input_pos");
        kv_cache_k_0 = decode_runner->input_tensor("kv_cache_k_0");
        max_seq_size = prefill_input->dims->data[1];
        kv_cache_max_size = kv_cache_k_0->dims->data[1];
        int prefill_seq_size = std::min<int>(prompt_tokens.size(), max_seq_size);

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
    /* *********************
     * PREFILL PREPROCESS END
     * *********************/

    /* *********************
     * PREFILL START
     * *********************/
    double prefill_time_ms = 0.0;
    // 9. Prefill Stage
    {
        ai_edge_torch::custom::profiler::ScopeLogger logger("Prefill", perf_monitor, metrics, usage_start, usage_end,
            true, nullptr, &prefill_time_ms);

        // Execute the prefill runner
        MINIMAL_CHECK(prefill_runner->Invoke() == kTfLiteOk);
    }
    /* *********************
     * PREFILL END
     * *********************/

    /* *********************
     * DECODE START
     * *********************/
    // 10. Decoding Stage with separate metrics for inference and sampling
    std::cout << "\nPrompt:\n" << prompt << "\n\nOutput Text:\n";
    
    // Determine how many tokens to generate
    int max_decode_steps = (absl::GetFlag(FLAGS_max_decode_steps) == -1)
        ? kv_cache_max_size
        : absl::GetFlag(FLAGS_max_decode_steps);
    
    int prefill_seq_size = std::min<int>(prompt_tokens.size(), max_seq_size);
    int next_token_id = prompt_tokens[prefill_seq_size - 1];
    int next_position = prefill_seq_size - 1;
    int decode_steps = std::min<int>(max_decode_steps, kv_cache_max_size - prefill_seq_size);
    MINIMAL_CHECK(decode_steps > 0);

    // Metrics object
    std::vector<ai_edge_torch::custom::profiler::RUsageRecord> decode_rusage_records;
    ai_edge_torch::custom::profiler::DecodingMetrics decoding_metrics;
    double inference_time_ms = 0.0;
    double sampling_time_ms = 0.0;

    // Decoding loop
    for (int i = 0; i < decode_steps; ++i) {
        {
            ai_edge_torch::custom::profiler::ScopeLogger logger("Decode_" + std::to_string(i), 
                perf_monitor, metrics, usage_start, usage_end, false, &decode_rusage_records);
        
            // 1) Model Inference
            {
                ai_edge_torch::custom::profiler::ScopeTimer timer(inference_time_ms);

                decode_input->data.i32[0] = next_token_id;
                decode_input_pos->data.i32[0] = next_position;
                MINIMAL_CHECK(decode_runner->Invoke() == kTfLiteOk);
            }

            // 2) Token Sampling
            {
                ai_edge_torch::custom::profiler::ScopeTimer timer(sampling_time_ms);

                next_token_id = ai_edge_torch::custom::sampler::temperature_top_k_top_p_sampler(
                    decode_runner->output_tensor("logits"), 0.9f, 85, 0.9f);
            }

            next_position++;
            decoding_metrics.RecordTimes(inference_time_ms, sampling_time_ms);
        }

        // Check if the next token is a stop token
        if (next_token_id == stop_token_id) {
            break;
        }

        // Detokenize the single token to text
        {
            std::vector<int> next_token = {next_token_id};
            std::string single_decoded_text;
            MINIMAL_CHECK(sp_processor->Decode(next_token, &single_decoded_text).ok());
            std::cout << single_decoded_text << std::flush;
        }
    }
    /* *********************
     * DECODE END
     * *********************/
     
    std::cout << "[INFO] Decoding stage completed\n";

    // 12. Print RUsage results
    std::cout << "\n================================\n";
    ai_edge_torch::custom::profiler::print_rusage_records(decode_rusage_records, "Decode");
     
    // 13. Print Perf results
    std::cout << "\n================================\n";
    metrics.PrintStats();

    // 14. Print decoding metrics (inference vs. sampling)
    std::cout << "\n================================\n";
    decoding_metrics.PrintMetrics(prefill_time_ms);
    std::cout << "\n================================\n";

    return 0;
}
