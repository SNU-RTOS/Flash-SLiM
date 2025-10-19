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
        delegate_options.weight_cache_file_path = weight_cache_path.c_str();

        // create and apply delegate
        MINIMAL_CHECK(interpreter->ModifyGraphWithDelegate(
                          tflite::Interpreter::TfLiteDelegatePtr(
                              TfLiteXNNPackDelegateCreate(&delegate_options),
                              [](TfLiteDelegate *d)
                              { TfLiteXNNPackDelegateDelete(d); })) == kTfLiteOk);
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


    //* ============= Utility Functions ============ */
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

    // --------------------------------------------------------------------------
    // Counts the total number of nodes across all subgraphs in the interpreter
    // --------------------------------------------------------------------------
    int CountTotalNodes(tflite::Interpreter *interpreter)
    {
        int total_nodes = 0;
        if (!interpreter)
            return 0;

        for (int i = 0; i < interpreter->subgraphs_size(); ++i)
        {
            total_nodes += static_cast<int>(interpreter->subgraph(i)->nodes_size());
            // std::cout << "[INFO] Subgraph " << i
            //           << " has " << interpreter->subgraph(i)->nodes_size() << " nodes." << std::endl;
        }
        return total_nodes;
    }

    // --------------------------------------------------------------------------
    // Utility to get current page cache size from /proc/meminfo (Linux only)
    // --------------------------------------------------------------------------
    void PrintCurrentPageCacheKB()
    {
        std::ifstream meminfo("/proc/meminfo");
        if (!meminfo.is_open())
        {
            std::cerr << "Failed to open /proc/meminfo\n";
            return;
        }

        std::string key, unit;
        size_t value = 0;
        bool found = false;

        while (meminfo >> key >> value >> unit)
        {
            if (key == "Cached:")
            {
                found = true;
                break;
            }
        }

        if (found)
        {
            std::cout << "[INFO] Current Page Cache: " << value << " kB" << std::endl;
        }
        else
        {
            std::cout << "[INFO] Current Page Cache: unknown" << std::endl;
        }
    }