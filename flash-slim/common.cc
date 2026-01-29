#include "common.h"

ABSL_DECLARE_FLAG(int, num_threads);
ABSL_DECLARE_FLAG(std::string, weight_cache_path);
ABSL_DECLARE_FLAG(std::string, sentencepiece_model);


    // --------------------------------------------------------------------------
    // Utility for applying XNNPACK weight caching
    // --------------------------------------------------------------------------
    // provider는 delegate보다 오래 살아야 함

#ifdef USE_WEIGHT_STREAMING
    void ApplyXNNPACKWithWeightCachingProvider(tflite::Interpreter *interpreter, StreamingWeightCacheProvider *provider)
    {

        auto delegate_options = TfLiteXNNPackDelegateOptionsDefault();
        delegate_options.num_threads = absl::GetFlag(FLAGS_num_threads);

        // set file path of weight cache
        std::string weight_cache_path = absl::GetFlag(FLAGS_weight_cache_path);
        delegate_options.weight_cache_file_path = weight_cache_path.c_str();

        // set provider
        delegate_options.weight_cache_provider = provider;

        // create and apply delegate
        MINIMAL_CHECK(interpreter->ModifyGraphWithDelegate(
                          tflite::Interpreter::TfLiteDelegatePtr(
                              TfLiteXNNPackDelegateCreate(&delegate_options),
                              [](TfLiteDelegate *d)
                              { TfLiteXNNPackDelegateDelete(d); })) == kTfLiteOk);
    }
#else
    void ApplyXNNPACKWithWeightCachingProvider(tflite::Interpreter *interpreter)
    {

        auto delegate_options = TfLiteXNNPackDelegateOptionsDefault();
        delegate_options.num_threads = absl::GetFlag(FLAGS_num_threads);

        // set file path of weight cache
        std::string weight_cache_path = absl::GetFlag(FLAGS_weight_cache_path);
        if(weight_cache_path.empty())
        {
            delegate_options.weight_cache_file_path = nullptr;
        } else {
            delegate_options.weight_cache_file_path = weight_cache_path.c_str();
        }
        
        // create and apply delegate
        MINIMAL_CHECK(interpreter->ModifyGraphWithDelegate(
                          tflite::Interpreter::TfLiteDelegatePtr(
                              TfLiteXNNPackDelegateCreate(&delegate_options),
                              [](TfLiteDelegate *d)
                              { TfLiteXNNPackDelegateDelete(d); })) == kTfLiteOk);
        std::cout << "[INFO] Applied XNNPACK Delegate with Weight Caching at: "
                  << weight_cache_path << std::endl;
    }
#endif

    // --------------------------------------------------------------------------
    // Allocates KV cache memory structures for decode, based on the decode signature
    // --------------------------------------------------------------------------
    std::map<std::string, std::vector<float, AlignedAllocator<float>>>
    AllocateKVCache(tflite::Interpreter *interpreter)
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
        size_t total_bytes = 0;
        std::map<std::string, std::vector<float, AlignedAllocator<float>>> kv_cache;
        for (int i = 0; i < num_layers; ++i)
        {
            std::string k_cache_name = "kv_cache_k_" + std::to_string(i);
            std::string v_cache_name = "kv_cache_v_" + std::to_string(i);
            std::cout << "[INFO] Allocating KV Cache for layer " << i
                      << ": " << k_cache_name << ", " << v_cache_name << std::endl;

            TfLiteTensor *tensor = runner->input_tensor(k_cache_name.c_str());
            size_t count = tensor->bytes / sizeof(float);
            total_bytes += tensor->bytes * 2; // for k and v

            kv_cache.emplace(k_cache_name,
                             std::vector<float, AlignedAllocator<float>>(count, 0.0f));
            kv_cache.emplace(v_cache_name,
                             std::vector<float, AlignedAllocator<float>>(count, 0.0f));
        }

        std::cout << "[INFO] Allocated KV Cache Size: "
                  << static_cast<float>(total_bytes) / (1024.0f * 1024.0f)
                  << " MiB\n";

        return kv_cache;
    }

    // --------------------------------------------------------------------------
    // Sets custom memory allocations for the KV cache on the given runner
    // --------------------------------------------------------------------------
    void PrepareRunner(tflite::SignatureRunner *runner,
                       std::map<std::string,
                                std::vector<float, AlignedAllocator<float>>> &kv_cache)
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
        const LoRA *lora)
    {
        tflite::SignatureRunner *runner = nullptr;
        int best_seq_size = -1;
        int delta = std::numeric_limits<int>::max();
        std::cout << "[INFO] Selecting prefill runner for "
                  << num_input_tokens << " input tokens.\n";

        for (const std::string *key : interpreter->signature_keys())
        {
            if (!absl::StrContains(*key, "prefill") || absl::StrContains(*key, "lora"))
            {
                continue;
            }
            TfLiteTensor *input_pos =
                interpreter->GetSignatureRunner(key->c_str())->input_tensor("input_pos");
            int seq_size = input_pos->dims->data[0];
            std::cout << "[INFO] Found prefill runner: " << *key
                      << " with seq_size: " << seq_size << std::endl;
            // Choose the runner where seq_size >= num_input_tokens and
            // (seq_size - num_input_tokens) is minimized
            if (num_input_tokens <= static_cast<size_t>(seq_size) &&
                seq_size - static_cast<int>(num_input_tokens) < delta)
            {
                if (lora == nullptr)
                {
                    runner = interpreter->GetSignatureRunner(key->c_str());
                    std::cout << "[INFO] Selected prefill runner: " << *key << std::endl;
                    MINIMAL_CHECK(runner != nullptr);
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
        LoRA *lora)
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

    