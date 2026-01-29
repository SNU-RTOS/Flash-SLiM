// =======================================================================
// Text Generation Application using LiteRT Interpreter with Flash-SLiM
// =======================================================================

#include "common.h"
#include "utils.h"

// ----------------------
// absl::FLAGS definition
// ----------------------
ABSL_FLAG(std::string, tflite_model, "", "Two-signature tflite model for text generation using ODML tools.");
ABSL_FLAG(std::string, sentencepiece_model, "", "Path to the SentencePiece model file.");
ABSL_FLAG(std::string, prompt, "Write an email:", "Input prompt for the model.");
ABSL_FLAG(int, max_decode_steps, -1, "Number of tokens to generate. Defaults to the KV cache limit.");
ABSL_FLAG(std::string, start_token, "", "Optional start token appended to the beginning of the input prompt.");
ABSL_FLAG(std::string, stop_token, "", "Optional stop token that stops the decoding loop if encountered.");
ABSL_FLAG(int, num_threads, 4, "Number of threads to use. Defaults to 4.");
ABSL_FLAG(std::string, weight_cache_path, "", "Path for XNNPACK weight caching, e.g., /tmp/model.xnnpack_cache.");
ABSL_FLAG(std::string, lora_path, "", "Optional path to a LoRA artifact.");
ABSL_FLAG(float, temperature, 0.8f, "Temperature for sampling. Higher values make output more random. Defaults to 0.8");
ABSL_FLAG(int, top_k, 40, "Top-k sampling parameter. Only consider the top k tokens. Defaults to 40.");
ABSL_FLAG(float, top_p, 0.9f, "Top-p (nucleus) sampling parameter. Only consider tokens with cumulative probability <= top_p. Defaults to 0.9.");
ABSL_FLAG(float, repetition_penalty, 1.2f, "Repetition penalty for sampling. Higher values reduce repetition. Defaults to 1.2.");
ABSL_FLAG(bool, enable_repetition_penalty, false, "Enable repetition penalty. Defaults to false.");
ABSL_FLAG(std::string, csv_profile_output_path, "", "Path to save the profiling results in CSV format. If empty, no CSV output is generated.");
ABSL_FLAG(std::string, prefetch_plan_path, "prefetch_plan.json", "Path to the weight chunk prefetch plan JSON file.");
// TODO : add io_engine selection logic
ABSL_FLAG(std::string, io_engine, "io_uring", "IO engine to use for weight chunk prefetching. Options: 'io_uring', 'pread'.");
ABSL_FLAG(int, io_ring_depth, 64, "io_uring ring depth for weight chunk prefetching. Defaults to 64.");
ABSL_FLAG(int, io_subread_bytes, 524288, "io_uring subread size in bytes for weight chunk prefetching. Defaults to 512KB (524288).");
ABSL_FLAG(int, io_min_block_size, 524288, "Minimum block size in bytes for parallel pread. Defaults to 512KB (524288).");
ABSL_FLAG(int, io_max_threads, 4, "Maximum number of threads for parallel pread. Defaults to 4.");


#ifdef USE_WEIGHT_STREAMING    
void __run_main(GenAIMetrics &genai_metrics,
    std::unique_ptr<BufferedProfiler> &op_profiler,
    const std::vector<ProfilerOutput> &op_profiler_outputs,
    std::unique_ptr<StreamingWeightCacheProvider> &weight_cache_provider,
    std::unique_ptr<WeightChunkController> &weight_chunk_controller
);
#else
void __run_main(GenAIMetrics &genai_metrics,
    std::unique_ptr<BufferedProfiler> &op_profiler,
    const std::vector<ProfilerOutput> &op_profiler_outputs
);
#endif // End USE_WEIGHT_STREAMING


// =======================================================================
// main() entry
// =======================================================================
int main(int argc, char *argv[])
{
    // Set precision
    std::cout.precision(5);
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout << std::boolalpha;
    std::cout << "\n[INFO] Text Generation App on LiteRT Interpreter\n";

#ifdef EBPF_TRACE_ENABLED
    std::cout << "\n[INFO] eBPF tracing is enabled. USDT probes will be used.\n";
#endif

    // Parse flags
    std::cout << "\n[INFO] Preparing Required Components" << std::endl;
    absl::ParseCommandLine(argc, argv);

    // Check which cores we're actually running on
    std::vector<int> active_cores;
    flash_slim::util::detect_active_cores(active_cores);

    // Determine cores to use for inference and I/O
    std::vector<int> cores_to_use_inference;
    std::vector<int> cores_to_use_io;
    flash_slim::util::set_cores_for_inference_and_io(active_cores, cores_to_use_inference, cores_to_use_io, absl::GetFlag(FLAGS_num_threads));

    // Init Custom GenAI Metrics Profiler
    GenAIMetrics genai_metrics;

    // Init Tflite Internal Op-level Profiler
    std::unique_ptr<BufferedProfiler> op_profiler; // Create op_profiler pointer

    // Init Profiler Output configurations
    std::vector<ProfilerOutput> op_profiler_outputs;

    // Use CSV Formatter
    std::string csv_path = absl::GetFlag(FLAGS_csv_profile_output_path);
    auto csv_formatter = std::make_shared<ProfileSummaryCSVFormatter>();
    ProfilerOutput pf_out_csv;
    pf_out_csv.formatter = csv_formatter;
    pf_out_csv.init_summarizer = std::make_shared<ProfileSummarizer>(csv_formatter);
    pf_out_csv.run_summarizer = std::make_shared<ProfileSummarizer>(csv_formatter);
    pf_out_csv.output_type = "csv";
    pf_out_csv.output_path = csv_path.empty() ? "" : csv_path;
    op_profiler_outputs.emplace_back(pf_out_csv);

#ifdef USE_WEIGHT_STREAMING
    const std::string prefetch_plan_path = absl::GetFlag(FLAGS_prefetch_plan_path);

    std::unique_ptr<StreamingWeightCacheProvider> weight_cache_provider = std::make_unique<StreamingWeightCacheProvider>();
    std::unique_ptr<WeightChunkController> weight_chunk_controller = std::make_unique<WeightChunkController>(weight_cache_provider.get());
    std::unique_ptr<WeightChunkPrefetcher> weight_chunk_prefetcher = std::make_unique<WeightChunkPrefetcher>();

    // Configure IO engine parameters from flags
    weight_chunk_prefetcher->ConfigureIOEngine(
        static_cast<std::string>(absl::GetFlag(FLAGS_io_engine)),
        static_cast<unsigned>(absl::GetFlag(FLAGS_io_ring_depth)),
        static_cast<size_t>(absl::GetFlag(FLAGS_io_subread_bytes)),
        static_cast<size_t>(absl::GetFlag(FLAGS_io_min_block_size)),
        static_cast<size_t>(absl::GetFlag(FLAGS_io_max_threads))
    );

    weight_cache_provider->OpenDirectIOFileDescriptor(absl::GetFlag(FLAGS_weight_cache_path));
    weight_chunk_controller->UpdateProviderMode(StreamingWeightCacheProvider::ProviderMode::RUNTIME);
    weight_chunk_controller->AttachPrefetcher(std::move(weight_chunk_prefetcher), cores_to_use_io);

    MINIMAL_CHECK(weight_chunk_controller->LoadPrefetchPlan(prefetch_plan_path));
#endif // USE_WEIGHT_STREAMING

    //* ============ Generate Token ============ */
    std::cout << "\n[INFO] Start Generating Text" << std::endl;

    // Run __run_main in a separate thread while optionally setting that
    // thread's CPU affinity to the first FLAGS_num_threads entries of
    // active_cores. This keeps the main() body simpler by delegating
    // affinity+threading behavior to util helpers.

#ifdef USE_WEIGHT_STREAMING
    flash_slim::util::run_thread_with_affinity_and_join([&]()
                                                        { __run_main(genai_metrics, op_profiler, op_profiler_outputs, weight_cache_provider, weight_chunk_controller);}, cores_to_use_inference);
#else
    flash_slim::util::run_thread_with_affinity_and_join([&]()
                                                        { __run_main(genai_metrics, op_profiler, op_profiler_outputs);}, cores_to_use_inference);
#endif // USE_WEIGHT_STREAMING


    //* ============ Print Results ============ */

    // Print genai metrics (inference vs. sampling)
    genai_metrics.Print();

    // Print Op-level profiling results
    std::cout << "\n[INFO] Generating Ops-level profiling (log)" << std::endl;

    pf_out_csv.formatter->HandleOutput(pf_out_csv.init_summarizer->GetOutputString(),
                                       pf_out_csv.run_summarizer->GetOutputString(), pf_out_csv.output_path);

    std::cout << "\n[INFO] Text Generation App completed successfully.\n";
    std::cout << "---------------------------------------------------\n\n";

    return 0;
}

#ifdef USE_WEIGHT_STREAMING    
void __run_main(GenAIMetrics &genai_metrics,
                std::unique_ptr<BufferedProfiler> &op_profiler,
                const std::vector<ProfilerOutput> &op_profiler_outputs,
                std::unique_ptr<StreamingWeightCacheProvider> &weight_cache_provider,
                std::unique_ptr<WeightChunkController> &weight_chunk_controller
            )
#else
void __run_main(GenAIMetrics &genai_metrics,
                std::unique_ptr<BufferedProfiler> &op_profiler,
                const std::vector<ProfilerOutput> &op_profiler_outputs)
#endif // USE_WEIGHT_STREAMING
{
    // Declare local variables
    std::vector<int> prompt_tokens;
    std::string prompt, start_token, stop_token;
    std::unordered_set<int> previously_generated_tokens;
    TfLiteStatus status = kTfLiteOk;
    int stop_token_id = -1;

    //* ============ [Phase] 1. Load Model ============ */
    std::unique_ptr<tflite::FlatBufferModel> model;
    {
        flash_slim::profiling::ScopeEventHandler handler("Load_Model");
        model = tflite::FlatBufferModel::BuildFromFile(absl::GetFlag(FLAGS_tflite_model).c_str());
    }
    MINIMAL_CHECK(model != nullptr);

    //* ============ [Phase] 2. Build Interpreter ============ */

    std::unique_ptr<tflite::Interpreter> interpreter;
    {
        flash_slim::profiling::ScopeEventHandler handler("Build_Interpreter");
        // Register Ops
        tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
        tflite::ops::custom::GenAIOpsRegisterer(&resolver); // Register GenAI custom ops

        // Build the interpreter
        tflite::InterpreterBuilder builder(*model, resolver);
        builder.SetNumThreads(absl::GetFlag(FLAGS_num_threads));
        builder(&interpreter);
    }
    MINIMAL_CHECK(interpreter != nullptr);

    // Create profiler if profiling is enabled
    constexpr int kProfilingBufferHeadrooms = 512;
    int total_nodes = flash_slim::util::CountTotalNodes(interpreter.get());
    if (total_nodes > kProfilingBufferHeadrooms)
        total_nodes += kProfilingBufferHeadrooms;
    op_profiler = std::make_unique<BufferedProfiler>(total_nodes, true);

    // Set profiler to interpreter
    interpreter->SetProfiler(op_profiler.get());

    //* ============ [Phase] 3. Apply Delegate ============ */

    {
        flash_slim::profiling::ScopeEventHandler handler("Apply_Delegate");

        if (!absl::GetFlag(FLAGS_weight_cache_path).empty())
        {
#ifdef USE_WEIGHT_STREAMING
            ApplyXNNPACKWithWeightCachingProvider(interpreter.get(), weight_cache_provider.get());
            std::cout << "[INFO] Applied XNNPACK Delegate with Weight Caching at: "
                      << absl::GetFlag(FLAGS_weight_cache_path) << std::endl;
#else
            ApplyXNNPACKWithWeightCachingProvider(interpreter.get());
            std::cout << "[WARN] Weight streaming is not enabled. Skipping weight caching provider application.\n";
#endif
        }
    }

    //* ============ [Phase] 4. Load Tokenizer ============ */
    std::unique_ptr<sentencepiece::SentencePieceProcessor> sp_processor;
    {
        flash_slim::profiling::ScopeEventHandler handler("Load_Tokenizer");
        sp_processor = LoadSentencePieceProcessor();
    }

    //* ============ [Phase] 5. Allocate KV Cache ============ */
    std::map<std::string, std::vector<float, AlignedAllocator<float>>> kv_cache;
    {
        flash_slim::profiling::ScopeEventHandler handler("Allocate_KV_Cache_Memory");
        kv_cache = AllocateKVCache(interpreter.get());
    }
    MINIMAL_CHECK(!kv_cache.empty());

    // 5. Optionally load LoRA
    /*
    std::unique_ptr<ai_edge_torch::examples::LoRA> lora = nullptr;
    {
        custom::profiler::ScopeTimer timer("LoRA Loading");
        if (!absl::GetFlag(FLAGS_lora_path).empty())
        {
            lora = ai_edge_torch::examples::LoRA::FromFile(absl::GetFlag(FLAGS_lora_path));
            MINIMAL_CHECK(lora != nullptr);
        }
    }
    */

    //* ============ [Phase] 6. Prepare Prompt ============ */
    {
        flash_slim::profiling::ScopeEventHandler handler("Prepare_Prompt");
        prompt = absl::GetFlag(FLAGS_prompt);
        MINIMAL_CHECK(sp_processor->Encode(prompt, &prompt_tokens).ok());
        // Sanity check
        std::string roundtrip;
        MINIMAL_CHECK(sp_processor->Decode(prompt_tokens, &roundtrip).ok());
        std::cout << "[DEBUG] prompt roundtrip: " << roundtrip << "\n";

        // Initialize start and stop tokens
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

        // Initialize previously generated tokens with prompt tokens for repetition penalty
        if (absl::GetFlag(FLAGS_enable_repetition_penalty))
        {
            for (int token : prompt_tokens)
            {
                previously_generated_tokens.insert(token);
            }
        }
    }
    std::cout << "[INFO] Start token IDs: " << (start_token.empty() ? "N/A" : std::to_string(prompt_tokens[0])) << " for token: " << start_token << std::endl;
    std::cout << "[INFO] Stop token ID: " << stop_token_id << " for token: " << stop_token << std::endl;

    //* ============ [Phase] 7. Prepare Signature Runners ============ */
    tflite::SignatureRunner *prefill_runner = nullptr;
    tflite::SignatureRunner *decode_runner = nullptr;
    {
        flash_slim::profiling::ScopeEventHandler handler("Prepare_Signature_Runners");
        std::size_t effective_prefill_token_size = (prompt_tokens.size() > 0) ? (prompt_tokens.size() - 1) : 0;

#ifdef USE_WEIGHT_STREAMING
        weight_chunk_controller->UpdatePrefetcherMode(WeightChunkPrefetcher::PrefetchMode::PREFILL);
#endif
        prefill_runner = GetPrefillRunner(interpreter.get(), effective_prefill_token_size, kv_cache, nullptr);
#ifdef USE_WEIGHT_STREAMING
        weight_chunk_controller->UpdatePrefetcherMode(WeightChunkPrefetcher::PrefetchMode::DECODE);
#endif
        decode_runner = GetDecodeRunner(interpreter.get(), kv_cache, nullptr);
    }
    MINIMAL_CHECK(prefill_runner != nullptr || decode_runner != nullptr);

    //* ============ [Phase] 8. Prepare Input Tensors ============ */
    auto dump_tensor = [](const char* name, const TfLiteTensor* t){
        std::cout << "[DBG] " << name
                    << " type=" << t->type
                    << " bytes=" << t->bytes
                    << " dims=";
        for (int i=0;i<t->dims->size;i++) std::cout << t->dims->data[i] << " ";
        std::cout << "\n";
    };
    
    TfLiteTensor *prefill_input = nullptr;
    TfLiteTensor *prefill_input_pos = nullptr;
    TfLiteTensor *decode_input = nullptr;
    TfLiteTensor *decode_input_pos = nullptr;
    TfLiteTensor *kv_cache_k_0 = nullptr;
    int max_seq_size = 0;
    int kv_cache_max_size = 0;
    int prefill_seq_size = 0;
    int seq_dim_index = 0;
    {
        flash_slim::profiling::ScopeEventHandler handler("Prepare_Input_Tensor");

        prefill_input = prefill_runner->input_tensor("tokens");
        prefill_input_pos = prefill_runner->input_tensor("input_pos");
        decode_input = decode_runner->input_tensor("tokens");
        decode_input_pos = decode_runner->input_tensor("input_pos");
        kv_cache_k_0 = decode_runner->input_tensor("kv_cache_k_0");
        max_seq_size = prefill_input->dims->data[1];

        // Detect KV cache sequence dimension and set max size accordingly
        seq_dim_index = flash_slim::util::DetectKVCacheSequenceDimension(kv_cache_k_0);
        if (seq_dim_index >= 0 && seq_dim_index < kv_cache_k_0->dims->size)
        {
            kv_cache_max_size = kv_cache_k_0->dims->data[seq_dim_index];
        }
        else // Fallback to default behavior
        {
            kv_cache_max_size = kv_cache_k_0->dims->data[1];
        }

        prefill_seq_size = std::min<int>(prompt_tokens.size(), max_seq_size);
        std::cout << "[INFO] Prefill sequence size: " << prefill_seq_size << " (max: " << max_seq_size << ")\n";

        // Zero out the input tensors
        std::memset(prefill_input->data.i32, 0, prefill_seq_size);
        std::memset(prefill_input_pos->data.i32, 0, prefill_seq_size);
        // const int eos_id = 2; // or sp_processor->PieceToId("</s>")

        // // Fill tokens with eos_id (int32 elements)
        // std::fill_n(prefill_input->data.i32, max_seq_size, eos_id);

        // // Fill positions with monotonic indices
        // for (int i = 0; i < max_seq_size; ++i) {
        //     prefill_input_pos->data.i32[i] = i;
        // }

        // Prefill uses all but the last token from the prompt
        for (int i = 0; i < prefill_seq_size - 1; ++i)
        {
            prefill_input->data.i32[i] = prompt_tokens[i];
            prefill_input_pos->data.i32[i] = i;
        }
    }
    std::cout << "[INFO] KV Cache Max Size: " << kv_cache_max_size << " (from dimension index " << seq_dim_index << ")" << std::endl;
    dump_tensor("prefill_input", prefill_input);
    dump_tensor("prefill_input_pos", prefill_input_pos);
    dump_tensor("decode_input", decode_input);
    dump_tensor("decode_input_pos", decode_input_pos);
    dump_tensor("kv_cache_k_0", kv_cache_k_0);

    auto dump_kv_tensors = [&](tflite::SignatureRunner* r, const char* tag) {
        std::cout << "[DBG] ---- " << tag << " KV tensors ----\n";

        // inputs
        for (const std::string& name : r->input_names()) {
            if (name.find("kv_cache") != std::string::npos) {
                dump_tensor(name.c_str(), r->input_tensor(name.c_str()));
            }
        }

        // outputs
        for (const std::string& name : r->output_names()) {
            if (name.find("kv_cache") != std::string::npos) {
                dump_tensor(name.c_str(), r->output_tensor(name.c_str()));
            }
        }
    };

    dump_kv_tensors(prefill_runner, "prefill");
    dump_kv_tensors(decode_runner, "decode");

    //* ============ [Phase] 9. Prefill Phase ============ */
    double prefill_time_ms = 0.0;
    std::cout << "[INFO] Prefill Phase started" << std::endl;
#ifdef USE_WEIGHT_STREAMING
    weight_chunk_controller->UpdatePrefetcherMode(WeightChunkPrefetcher::PrefetchMode::PREFILL);
#endif
    // Start op-level profiling
    op_profiler->Reset();
    op_profiler->StartProfiling();
    {
        flash_slim::profiling::ScopeTimer prefill_timer(prefill_time_ms);
        flash_slim::profiling::ScopeEventHandler handler("Prefill");
        status = prefill_runner->Invoke(); // Execute the prefill runner
    }
    op_profiler->StopProfiling();
    genai_metrics.RecordPrefillTime(prefill_time_ms);
    for (auto &out : op_profiler_outputs)
        out.run_summarizer->ProcessProfiles(op_profiler->GetProfileEvents(), *interpreter);

    MINIMAL_CHECK(status == kTfLiteOk);
    std::cout << "[INFO] Prefill Phase completed" << std::endl;

    //* ============ [Phase] 10. Decoding Phase ============ */
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

    std::cout << "Piece(0)=" << sp_processor->IdToPiece(0) << "\n";
    std::cout << "Piece(1)=" << sp_processor->IdToPiece(1) << "\n";
    std::cout << "Piece(2)=" << sp_processor->IdToPiece(2) << "\n";
    std::cout << "VocabSize=" << sp_processor->GetPieceSize() << "\n";

#ifdef USE_WEIGHT_STREAMING
    weight_chunk_controller->UpdatePrefetcherMode(WeightChunkPrefetcher::PrefetchMode::DECODE);
#endif
    // Decoding loop
    for (int i = 0; i < decode_steps; ++i)
    {
        std::string phase_name = "Decode_" + std::to_string(i);

        op_profiler->Reset();
        op_profiler->StartProfiling();

        std::string single_decoded_text;
        {
            flash_slim::profiling::ScopeEventHandler handler(phase_name);

            // 1) Model Inference
            {
                flash_slim::profiling::ScopeTimer timer(inference_time_ms);

                decode_input->data.i32[0] = next_token_id;
                decode_input_pos->data.i32[0] = next_position;
                MINIMAL_CHECK(decode_runner->Invoke() == kTfLiteOk);
                dump_tensor("Decode output", decode_runner->output_tensor("logits"));

                const TfLiteTensor* t = decode_runner->output_tensor("logits");
                int vocab = t->dims->data[t->dims->size - 1];
                const float* logits = t->data.f; // rows==1 here

                std::vector<std::pair<float,int>> v;
                v.reserve(vocab);
                for (int id=0; id<vocab; ++id) v.push_back({logits[id], id});
                std::partial_sort(v.begin(), v.begin()+5, v.end(),
                                [](auto&a, auto&b){ return a.first > b.first; });

                for (int i=0;i<5;i++){
                int id = v[i].second;
                std::cout << "[TOP] id=" << id
                            << " logit=" << v[i].first
                            << " piece=" << sp_processor->IdToPiece(id) << "\n";
                }
            }

            // 2) Token Sampling
            {
                flash_slim::profiling::ScopeTimer timer(sampling_time_ms);
                if (absl::GetFlag(FLAGS_enable_repetition_penalty))
                {
                    next_token_id = flash_slim::sampler::temperature_top_k_top_p_repetition_sampler(
                        decode_runner->output_tensor("logits"),
                        absl::GetFlag(FLAGS_temperature),
                        absl::GetFlag(FLAGS_top_k),
                        absl::GetFlag(FLAGS_top_p),
                        previously_generated_tokens,
                        absl::GetFlag(FLAGS_repetition_penalty));
                }
                else
                {
                    next_token_id = flash_slim::sampler::temperature_top_k_top_p_sampler(
                        decode_runner->output_tensor("logits"),
                        absl::GetFlag(FLAGS_temperature),
                        absl::GetFlag(FLAGS_top_k),
                        absl::GetFlag(FLAGS_top_p));
                }
            }

            // 3) Token Detokenization
            {
                flash_slim::profiling::ScopeTimer timer(detok_time_ms);
                std::vector<int> next_token = {next_token_id};
                MINIMAL_CHECK(sp_processor->Decode(next_token, &single_decoded_text).ok());
            }
        }

        genai_metrics.RecordDecodingTime(inference_time_ms, sampling_time_ms, detok_time_ms);
        op_profiler->StopProfiling();
        for (auto &out : op_profiler_outputs)
            out.run_summarizer->ProcessProfiles(op_profiler->GetProfileEvents(), *interpreter);

        // Check if the next token is a stop token
        if (next_token_id == stop_token_id)
            break;

        // Add the generated token to previously generated tokens for repetition penalty
        if (absl::GetFlag(FLAGS_enable_repetition_penalty))
            previously_generated_tokens.insert(next_token_id);

        std::cout << single_decoded_text << std::flush;
        next_position++;
    }

#ifdef USE_WEIGHT_STREAMING
    if (weight_cache_provider)
    {
        weight_cache_provider->CloseDirectIOFileDescriptor();
        weight_cache_provider->Release();
    }

#endif
    std::cout << "\n\n\n";
    std::cout << "[INFO] Decoded " << decode_steps << " tokens." << std::endl;
    std::cout << "[INFO] Decoding Phase completed" << std::endl;
}