#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <cstdint>

#include "tflite/schema/schema_generated.h"

// abseil
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/string_view.h"
#include "absl/strings/match.h"

// LiteRT Core
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"
#include "tflite/experimental/genai/genai_ops.h"
#include "tflite/interpreter.h"
#include "tflite/interpreter_builder.h"
#include "tflite/kernels/register.h"
#include "tflite/model_builder.h"
#include "tflite/profiling/profiler.h"

#include "tflite/delegates/xnnpack/weight_cache.h"
#include "tflite/delegates/xnnpack/streaming_weight_cache.h"

// ----------------------
// absl::FLAGS definition
// ----------------------
ABSL_FLAG(std::string, weight_cache_path, "", "Path for XNNPACK weight caching, e.g., /tmp/model.xnnpack_cache.");
ABSL_FLAG(std::string, tflite_model, "", "Two-signature tflite model for text generation using ODML tools.");
ABSL_FLAG(std::string, dump_file_path, "", "Path to save the log file. If empty, no log file is generated.");
ABSL_FLAG(bool, dump_tensor_details, false, "Whether to dump detailed tensor information for each node.");
ABSL_FLAG(bool, op_tensor_byte_stats, false, "Whether to append per-operator aggregated tensor bytes (Mmap/Arena) on each operator line.");

namespace
{
#ifdef USE_WEIGHT_STREAMING
    using WeightCacheProviderT = tflite::xnnpack::StreamingWeightCacheProvider;
#else
    using WeightCacheProviderT = tflite::xnnpack::MMapWeightCacheProvider;
#endif

    static WeightCacheProviderT g_weight_cache_provider;

    // Small helper to map: subgraph -> [tensor_idx -> buffer_id]
    // Maps a subgraph-local tensor index to the FlatBuffer buffer identifier.
    class TensorToBufferIdMap
    {
    public:
        void BuildFromFlatBufferModel(const tflite::FlatBufferModel *fb_model_holder)
        {
            map_.clear();
            if (!fb_model_holder)
                return;
            const ::tflite::Model *fb_model = fb_model_holder->GetModel();
            if (!fb_model)
                return;
            auto subgraphs = fb_model->subgraphs();
            if (!subgraphs)
                return;
            map_.resize(subgraphs->size());
            for (size_t i = 0; i < subgraphs->size(); ++i)
            {
                const ::tflite::SubGraph *subgraph = subgraphs->Get(i);
                auto tensors = subgraph ? subgraph->tensors() : nullptr;
                size_t n = tensors ? tensors->size() : 0;
                auto &vec = map_[i];
                vec.resize(n, -1);
                for (size_t i = 0; i < n; ++i)
                {
                    const ::tflite::Tensor *tensor = tensors->Get(i);
                    vec[i] = tensor ? static_cast<int>(tensor->buffer()) : -1;
                }
            }
        }

        int GetBufferId(int subgraph_idx, int tensor_idx) const
        {
            if (subgraph_idx < 0 || subgraph_idx >= static_cast<int>(map_.size()))
                return -1;
            const auto &v = map_[subgraph_idx];
            if (tensor_idx < 0 || tensor_idx >= static_cast<int>(v.size()))
                return -1;
            return v[tensor_idx];
        }

        // Number of subgraphs represented in the map (0 if empty)
        size_t NumSubgraphs() const { return map_.size(); }

    private:
        std::vector<std::vector<int>> map_;
    };

    // Human-readable byte formatting helper
    std::string FormatBytes(size_t bytes)
    {
        const double kb = 1024.0;
        const double mb = kb * 1024.0;
        const double gb = mb * 1024.0;
        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss << std::setprecision(2);
        if (bytes >= (size_t)gb)
        {
            oss << (bytes / gb) << "GB";
        }
        else if (bytes >= (size_t)mb)
        {
            oss << (bytes / mb) << "MB";
        }
        else if (bytes >= (size_t)kb)
        {
            oss << (bytes / kb) << "KB";
        }
        else
        {
            oss.unsetf(std::ios::floatfield);
            oss << bytes << "B";
        }
        return oss.str();
    }
    // --------------------------------------------------------------------------
    // Print detailed tensor information (migrated from text_generator_main.cc)
    // --------------------------------------------------------------------------
    void PrintTensorDetail(int tensor_idx, TfLiteTensor *tensor, std::ostream *output = nullptr)
    {
        std::ostream &out = (output != nullptr) ? *output : std::cout;

        if (!tensor)
        {
            out << "Tensor " << tensor_idx << " is NULL\n";
            return;
        }

        void *tensor_data_address = tensor->data.raw;
        out << "Data Address: " << tensor_data_address << " ";

        // Tensor Type
        const char *type_name = TfLiteTypeGetName(tensor->type);
        out << "Type: " << (type_name ? type_name : "Unknown") << " ";

        // Tensor Allocation Type
        out << "Allocation Type: ";
        switch (tensor->allocation_type)
        {
        case kTfLiteArenaRw:
            out << "Arena RW " << "Bytes: " << tensor->bytes << " ";
            break;
        case kTfLiteArenaRwPersistent:
            out << "Arena Persistent " << "Bytes: " << tensor->bytes << " ";
            break;
        case kTfLiteMmapRo:
            out << "Mmap " << "Bytes: " << tensor->bytes << " ";
            break;
        case kTfLiteDynamic:
            out << "Dynamic " << "Bytes: " << tensor->bytes << " ";
            break;
        case kTfLiteCustom:
            out << "Custom " << "Bytes: " << tensor->bytes << " ";
            break;
        case kTfLitePersistentRo:
            out << "PersistentRo " << "Bytes: " << tensor->bytes << " ";
            break;
        case kTfLiteVariantObject:
            out << "Variant " << "Bytes: " << tensor->bytes << " ";
            break;
        case kTfLiteMemNone:
            out << "MemNone " << "Bytes: 0 ";
            break;
        default:
            out << "Unknown " << "Bytes: 0 ";
            break;
        }

        // Tensor Shape
        out << "Shape: [";
        if (tensor->dims && tensor->dims->size > 0)
        {
            for (int dim_idx = 0; dim_idx < tensor->dims->size; ++dim_idx)
            {
                out << tensor->dims->data[dim_idx];
                if (dim_idx < tensor->dims->size - 1)
                    out << ", ";
            }
        }
        out << "]\n";
    }

    void InspectExecutionPlan(tflite::Interpreter *interpreter, int subgraph_idx, const TensorToBufferIdMap &tbm, std::ostream *output = nullptr, int indent = 0)
    {
        std::ostream &out = (output != nullptr) ? *output : std::cout;
        std::string indent_str(indent, ' ');

        struct State
        {
            int32_t subgraph_index;
            bool subgraph_has_dynamic_output_tensors = false;
        };

        auto *subgraph = interpreter->subgraph(subgraph_idx);
        auto execution_plan = subgraph->execution_plan();

        int cnt_stablehlo_composite = 0;
        int cnt_atomic_ops = 0;     // 일반 atomic operator 개수
        int cnt_delegate_nodes = 0; // DELEGATE 노드 개수
        int cnt_xnn_ops = 0;        // XNNPACK operator 개수

        // Helper function to count XNNPACK operators from delegate output
        auto count_xnnpack_ops = [&](void *delegate_data) -> int
        {
            std::ostringstream temp_stream;
            TfLiteXNNPackDelegateInspect(delegate_data, &temp_stream, "");
            std::string delegate_output = temp_stream.str();

            int xnn_op_count = 0;
            std::istringstream iss(delegate_output);
            std::string line;

            while (std::getline(iss, line))
            {
                // XNNPACK operator lines typically start with "[XXXX]" format
                if (line.find("[") != std::string::npos && line.find("]") != std::string::npos)
                {
                    xnn_op_count++;
                }
            }
            return xnn_op_count;
        };

        // Recursive function to count nodes including nested STABLEHLO_COMPOSITE subgraphs
        std::function<int(int, int)> count_all_nodes = [&](int sg_idx, int depth) -> int
        {
            tflite::Subgraph *sg = interpreter->subgraph(sg_idx);
            auto plan = sg->execution_plan();
            int total = 0; // 0으로 시작하여 명시적으로 카운트

            for (int id : plan)
            {
                const auto *nr = sg->node_and_registration(id);
                auto *n = &nr->first;
                auto *r = &nr->second;

                if (r->builtin_code == tflite::BuiltinOperator_STABLEHLO_COMPOSITE)
                {
                    // count this composite node
                    cnt_stablehlo_composite++;

                    // STABLEHLO_COMPOSITE 노드 자체는 카운트하지 않고, 내부 서브그래프만 카운트
                    if (n->user_data)
                    {
                        auto *op_state = reinterpret_cast<State *>(n->user_data);
                        int child_subgraph_idx = op_state->subgraph_index;
                        total += count_all_nodes(child_subgraph_idx, depth + 1);
                    }
                }
                // DELEGATE 노드는 XNNPACK operators로 치환되므로 delegate 내부의 operator 개수를 세고, delegate 노드 자체는 제외
                else if (r->builtin_code == tflite::BuiltinOperator_DELEGATE)
                {
                    cnt_delegate_nodes++;
                    int xnn_ops = count_xnnpack_ops(n->user_data);
                    cnt_xnn_ops += xnn_ops;
                    total += xnn_ops;
                }
                // 일반 노드들만 카운트 (STABLEHLO_COMPOSITE, DELEGATE 제외)
                else
                {
                    total++;
                    cnt_atomic_ops++;
                }
            }

            return total;
        };

        int total_nodes_including_nested = count_all_nodes(subgraph_idx, 0);

        out << indent_str << "Subgraph " << subgraph_idx << ": " << subgraph->GetName() << " "
            << "(Total Nodes: " << total_nodes_including_nested
            << ", Nodes only in execution_plan: " << execution_plan.size()
            << ", StableHLO Composite Nodes: " << cnt_stablehlo_composite
            << ", DELEGATE Nodes: " << cnt_delegate_nodes
            << ", Atomic Nodes (Default ops): " << cnt_atomic_ops
            << ", XNNPACK Nodes: " << cnt_xnn_ops
            << ")" << std::endl;

        int absolute_op_counter = 0; // 절대 operator 번호 (DELEGATE 제외)
        for (int idx : execution_plan)
        {
            const auto *node_and_reg = subgraph->node_and_registration(idx);
            auto *node = &node_and_reg->first;
            auto *reg = &node_and_reg->second;
            std::string op_name =
                tflite::EnumNameBuiltinOperator(
                    static_cast<tflite::BuiltinOperator>(reg->builtin_code));

            // Optional per-op tensor byte stats
            size_t op_bytes_mmap = 0;
            size_t op_bytes_arena = 0;
            if (absl::GetFlag(FLAGS_op_tensor_byte_stats))
            {
                auto collect = [&](const TfLiteIntArray *arr)
                {
                    if (!arr)
                        return;
                    for (int i = 0; i < arr->size; ++i)
                    {
                        int tidx = arr->data[i];
                        if (tidx < 0)
                            continue;
                        TfLiteTensor *t = subgraph->tensor(tidx);
                        if (!t)
                            continue;
                        switch (t->allocation_type)
                        {
                        case kTfLiteMmapRo:
                            op_bytes_mmap += t->bytes;
                            break;
                        case kTfLiteArenaRw:
                        case kTfLiteArenaRwPersistent:
                            op_bytes_arena += t->bytes;
                            break;
                        default:
                            break;
                        }
                    }
                };
                collect(node->inputs);
                collect(node->outputs);
                collect(node->temporaries);
            }

            out << indent_str << "  " << std::setw(4) << std::setfill(' ') << absolute_op_counter << ": "
                << "[" << std::setw(4) << std::setfill('0') << idx << "] " << op_name;
            if (absl::GetFlag(FLAGS_op_tensor_byte_stats))
            {
                out << " (Mmap=" << FormatBytes(op_bytes_mmap) << ", Arena=" << FormatBytes(op_bytes_arena) << ")";
            }
            absolute_op_counter++;

            // Dump detailed tensor information if requested
            if (absl::GetFlag(FLAGS_dump_tensor_details))
            {
                const auto *node_and_reg = subgraph->node_and_registration(idx);

                std::string tensor_indent_str(indent + 2, ' ');
                if (node_and_reg)
                {
                    const TfLiteNode *node = &node_and_reg->first;

                    // Print input tensors
                    if (node->inputs && node->inputs->size > 0)
                    {
                        out << std::endl
                            << tensor_indent_str << "    Input Tensors:" << std::endl;
                        for (int i = 0; i < node->inputs->size; ++i)
                        {
                            int tensor_idx = node->inputs->data[i];
                            if (tensor_idx >= 0)
                            {
                                auto *tensor = subgraph->tensor(tensor_idx);
                                int buffer_idx = tbm.GetBufferId(subgraph_idx, tensor_idx);
                                out << tensor_indent_str << "      Input " << i << ": " << tensor_idx
                                    << " (buffer " << buffer_idx << ") ";
                                PrintTensorDetail(tensor_idx, tensor, output);
                            }
                        }
                    }

                    // Print output tensors
                    if (node->outputs && node->outputs->size > 0)
                    {
                        out << tensor_indent_str << "    Output Tensors:" << std::endl;
                        for (int i = 0; i < node->outputs->size; ++i)
                        {
                            int tensor_idx = node->outputs->data[i];
                            if (tensor_idx >= 0)
                            {
                                auto *tensor = subgraph->tensor(tensor_idx);
                                int buffer_idx = tbm.GetBufferId(subgraph_idx, tensor_idx);
                                out << tensor_indent_str << "      Output " << i << ": " << tensor_idx
                                    << " (buffer " << buffer_idx << ") ";
                                PrintTensorDetail(tensor_idx, tensor, output);
                            }
                        }
                    }

                    // Print temporary tensors if any
                    if (node->temporaries && node->temporaries->size > 0)
                    {
                        out << tensor_indent_str << "    Temporary Tensors:" << std::endl;
                        for (int i = 0; i < node->temporaries->size; ++i)
                        {
                            int tensor_idx = node->temporaries->data[i];
                            if (tensor_idx >= 0)
                            {
                                auto *tensor = subgraph->tensor(tensor_idx);
                                int buffer_idx = tbm.GetBufferId(subgraph_idx, tensor_idx);
                                out << tensor_indent_str << "      Temporary " << i << ": " << tensor_idx
                                    << " (buffer " << buffer_idx << ") ";
                                PrintTensorDetail(tensor_idx, tensor, output);
                            }
                        }
                    }
                }
            }

            std::string delegate_indent_str(indent + 6, ' ');
            if (reg->builtin_code == tflite::BuiltinOperator_DELEGATE)
            {
                TfLiteXNNPackDelegateInspect(node->user_data, output, delegate_indent_str.c_str());
            }
            else if (reg->builtin_code == tflite::BuiltinOperator_STABLEHLO_COMPOSITE)
            {
                auto *op_state = reinterpret_cast<State *>(node->user_data);
                int subgraph_idx_child = op_state->subgraph_index;
                auto decomposition_subgraph = interpreter->subgraph(subgraph_idx_child);
                auto tmp_name = decomposition_subgraph->GetName();
                out << " (→ Subgraph " << subgraph_idx_child << " :" << tmp_name << ")" << std::endl;
                out << delegate_indent_str << "-------------------" << std::endl;
                // increase indent by 2 spaces for nested subgraph
                InspectExecutionPlan(interpreter, subgraph_idx_child, tbm, output, indent + 4);
                out << delegate_indent_str << "-------------------" << std::endl;
            }
            else
            {
                out << std::endl;
            }
        }
    }

    void InspectSignatureExecutionPlan(tflite::Interpreter *interpreter, const std::string &signature_key, const TensorToBufferIdMap &tbm, std::ostream *output = nullptr, int indent = 0)
    {
        std::ostream &out = (output != nullptr) ? *output : std::cout;

        // Inspect the execution plan with signature context
        int subgraph_idx = interpreter->GetSubgraphIndexFromSignature(signature_key.c_str());
        if (subgraph_idx == -1)
        {
            out << "Failed to get subgraph index for: " << signature_key << std::endl;
            return;
        }
        out << std::endl;
        InspectExecutionPlan(interpreter, subgraph_idx, tbm, output, indent);
        out << std::endl; // spacing after subgraph dump
    }

    void InspectSelectedSignature(tflite::Interpreter *interpreter, int sig_index, const TensorToBufferIdMap &tbm, std::ostream *output = nullptr, int indent = 2)
    {
        std::ostream &out = (output != nullptr) ? *output : std::cout;

        const auto &signature_keys = interpreter->signature_keys();
        if (sig_index < 0 || sig_index >= signature_keys.size())
        {
            out << "Invalid signature index: " << sig_index << std::endl;
            return;
        }

        const std::string &signature_key = *signature_keys[sig_index];
        InspectSignatureExecutionPlan(interpreter, signature_key, tbm, output, indent);
    }

    int GetSignatureIndexFromUser(tflite::Interpreter *interpreter)
    {
        // Print model signature keys (console only)
        std::cout << "\n=== Model Signature ===" << std::endl;
        int sig_index = -1;
        if (interpreter)
        {
            const std::vector<const std::string *> &keys = interpreter->signature_keys();
            std::cout << "The Model contains " << keys.size() << " signature key(s)." << std::endl;
            if (!keys.empty())
            {
                for (int i = 0; i < keys.size(); ++i)
                {
                    const std::string *key = keys[i];
                    std::cout << "  " << std::setfill('0') << i << ": "
                              << *key << std::endl;
                }
            }

            std::cout << "Please select a signature index (0 to " << keys.size() - 1 << "): ";
            std::cin >> sig_index;
            if (sig_index > keys.size() - 1)
            {
                std::cout << "Invalid signature index. Please try again." << std::endl;
                sig_index = -1;
            }
        }
        else
        {
            std::cout << "The Model does not contain any signature keys.";
        }
        std::cout << std::endl;
        return sig_index;
    }

    // --------------------------------------------------------------------------
    // Utility for applying XNNPACK weight caching
    // --------------------------------------------------------------------------
    void ApplyXNNPACKWithWeightCaching(tflite::Interpreter *interpreter)
    {
        auto delegate_options = TfLiteXNNPackDelegateOptionsDefault();
        std::string weight_cache_path = absl::GetFlag(FLAGS_weight_cache_path);
        delegate_options.weight_cache_file_path = weight_cache_path.c_str();
        delegate_options.weight_cache_provider = &g_weight_cache_provider;
        delegate_options.num_threads = 1; // Default num_threads for dump tool

        if (interpreter->ModifyGraphWithDelegate(
                tflite::Interpreter::TfLiteDelegatePtr(
                    TfLiteXNNPackDelegateCreate(&delegate_options),
                    [](TfLiteDelegate *delegate)
                    { TfLiteXNNPackDelegateDelete(delegate); })) != kTfLiteOk)
        {
            std::cerr << "❌ Failed to apply XNNPACK delegate\n";
            exit(1);
        }
    }

    // ----------------------------------------------------------------------------------
    // Validate Weight Cache Mappings 
    // ----------------------------------------------------------------------------------
    void ValidateWeightCacheMappings(tflite::Interpreter *interpreter, const std::string &selected_signature_key, const TensorToBufferIdMap &tbm)
    {
        // 0) Get Subgraph and tensors
        int sg_idx = interpreter->GetSubgraphIndexFromSignature(selected_signature_key.c_str());
        tflite::Subgraph *sg = interpreter->subgraph(sg_idx);
        const size_t num_tensors = static_cast<size_t>(sg->tensors_size());

        // 1) Get TensorBufferAddress → identifier map (provider snapshot)
        auto buffer_address_to_identifier = g_weight_cache_provider.GetBufferAddressToIdentifier();

        // 2) Snapshot provider mappings once and reuse them per tensor.
        auto cache_key_to_offset = g_weight_cache_provider.GetCacheKeyToOffset();

        // Build helper maps used by the validation logic.
        // id_map: tensor_index -> weights_id
        std::unordered_map<size_t, size_t> id_map;
        id_map.reserve(num_tensors);

        for (size_t i = 0; i < num_tensors; ++i)
        {
            TfLiteTensor *t_ptr = sg->tensor(static_cast<int>(i));
            if (!t_ptr)
                continue;
            const TfLiteTensor &t = *t_ptr;

            if ((t.allocation_type != kTfLiteMmapRo && t.allocation_type != kTfLitePersistentRo) || !t.data.data)
                continue;

            // Prefer provider's address -> identifier mapping
            auto pit = buffer_address_to_identifier.find(t.data.data);
            if (pit != buffer_address_to_identifier.end())
            {
                id_map[i] = static_cast<size_t>(pit->second);
            }
            else
            {
                // Fallback to FlatBuffer buffer id
                int fb_id = tbm.GetBufferId(sg_idx, static_cast<int>(i));
                if (fb_id >= 0)
                {
                    id_map[i] = static_cast<size_t>(fb_id);
                }
            }
        }

        // Build reverse index: weights_id -> list of (pack_algorithm_id, bias_id)
        std::unordered_map<size_t, std::vector<std::pair<size_t, size_t>>> weights_to_candidates;
        for (const auto &kv : cache_key_to_offset)
        {
            const auto &pack_id = kv.first; // has pack_algorithm_id, weights_id, bias_id
            const size_t w_id = static_cast<size_t>(pack_id.weights_id);
            const size_t algo = static_cast<size_t>(pack_id.pack_algorithm_id);
            const size_t bias = static_cast<size_t>(pack_id.bias_id);
            weights_to_candidates[w_id].emplace_back(algo, bias);
        }

        // 3) Validate a subset of constant tensors with cache LookUp/OffsetToAddr
        std::ofstream vout("weight_cache_validation.log");
        vout << "\n=== Validation (signature=" << selected_signature_key << ") ===\n";
        vout << "\n";
       // Human-friendly aligned header
       vout << std::left << std::setw(8) << "Tensor" << " | "
           << std::setw(18) << "Ptr" << " | -> | "
           << std::setw(15) << "Pack Algo ID" << " | "
           << std::setw(18) << "Weights(Buffer) ID" << " | "
           << std::setw(11) << "Bias ID" << " | -> | "
           << std::setw(18) << "Offset" << " | -> | "
           << std::setw(18) << "Addr(used)" << "  ||  "
           << std::setw(18) << "MmappedAddr" << "\n";
       vout << std::string(160, '-') << "\n";
        size_t validated = 0, attempted = 0;

        for (size_t i = 0; i < num_tensors; ++i)
        {
            TfLiteTensor *t_ptr = sg->tensor(static_cast<int>(i));
            if (!t_ptr)
                continue;
            const TfLiteTensor &t = *t_ptr;

            if (t.allocation_type != kTfLiteMmapRo && t.allocation_type != kTfLitePersistentRo)
                continue;
            if (!t.data.data)
                continue;

            auto it_id = id_map.find(i);
            if (it_id == id_map.end())
                continue;
            size_t weights_id = it_id->second;

            auto it_cands = weights_to_candidates.find(weights_id);
            if (it_cands == weights_to_candidates.end() || it_cands->second.empty())
                continue;

            // Try candidate algos until a match is found
            bool ok = false;
            for (const auto &cand : it_cands->second)
            {
                const size_t algo = cand.first;
                const size_t bias_id = cand.second;
                size_t offset = g_weight_cache_provider.LookUpByIds(algo, weights_id, bias_id);
                if (offset == SIZE_MAX)
                    continue;

                void *addr = g_weight_cache_provider.OffsetToAddr(offset);
                void *mmaped_addr = g_weight_cache_provider.GetMmappedAddr(offset);
                {
                    std::ostringstream ptrs, useds, mmaps;
                    ptrs << t.data.data;
                    useds << addr;
                    mmaps << mmaped_addr;
                    const std::string bias_id_str = (bias_id == SIZE_MAX) ? std::string("None") : std::to_string(bias_id);
                    vout << std::left << std::setw(8) << i << " | "
                         << std::setw(18) << ptrs.str() << " | -> | "
                         << std::setw(15) << algo << " | "
                         << std::setw(18) << weights_id << " | "
                         << std::setw(11) << bias_id_str << " | -> | "
                         << std::setw(18) << offset << " | -> | "
                         << std::setw(18) << useds.str() << "  ||  "
                         << std::setw(18) << mmaps.str() << "\n";
                }
                ok = true;
                ++validated;
                break;
            }
            ++attempted;
            if (!ok)
            {
                vout << "Tensor[" << i << "] ptr=" << t.data.data
                     << " Buffer id=" << weights_id
                     << " -> cache MISS (no matching algo)\n";
            }
        }
        vout << "\n";
        vout << "Validation summary: validated=" << validated
             << " attempted=" << attempted << "\n";
        vout.close();
    }

    // --------------------------------------------------------------------------
    // Utility to get current page cache size from /proc/meminfo (Linux only)
    // --------------------------------------------------------------------------
    size_t GetPageCacheKB()
    {
        std::ifstream meminfo("/proc/meminfo");
        if (!meminfo.is_open())
        {
            std::cerr << "Failed to open /proc/meminfo\n";
            return 0;
        }

        std::string key;
        size_t value;
        std::string unit;

        while (meminfo >> key >> value >> unit)
        {
            if (key == "Cached:")
            {
                return value; // in KB
            }
        }
        return 0;
    }

} // end anonymous namespace

// =======================================================================
// main() entry
// =======================================================================
int main(int argc, char *argv[])
{
    // Parse flags
    absl::ParseCommandLine(argc, argv);

    std::cout << "====== dump_model_cpu ======" << std::endl;
    std::cout << "TFLite model: " << absl::GetFlag(FLAGS_tflite_model) << std::endl;

    //* ============ Create Model, Interpreter and Profiler ============ */
    std::unique_ptr<tflite::FlatBufferModel> flatbuffer_model;
    flatbuffer_model = tflite::FlatBufferModel::BuildFromFile(absl::GetFlag(FLAGS_tflite_model).c_str());

    // Initialize tensor->buffer map from FlatBuffer model
    TensorToBufferIdMap tensor_buffer_map;
    tensor_buffer_map.BuildFromFlatBufferModel(flatbuffer_model.get());

    std::unique_ptr<tflite::Interpreter> interpreter;
    std::unique_ptr<tflite::profiling::Profiler> op_profiler = std::make_unique<tflite::profiling::Profiler>();
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::ops::custom::GenAIOpsRegisterer(&resolver); // Register GenAI custom ops
    tflite::InterpreterBuilder builder(*flatbuffer_model, resolver);
    builder(&interpreter);
    if (!interpreter)
    {
        std::cerr << "❌ Failed to create interpreter\n";
        return 1;
    }
    interpreter->SetProfiler(op_profiler.get());

    //* ============ Print Model Signature and Select Signature to Dump (console only, as requested) ============ */

    int sig_index = GetSignatureIndexFromUser(interpreter.get());
    if (sig_index < 0)
    {
        std::cerr << "❌ Invalid signature index. Exiting.\n";
        return 1;
    }

    std::ofstream dump_file(absl::GetFlag(FLAGS_dump_file_path));
    if (!dump_file.is_open())
    {
        std::cerr << "❌ Failed to open log file: " << absl::GetFlag(FLAGS_dump_file_path) << std::endl;
        return 1;
    }

    //* ============ Dump Model (Before Delegate) ============ */
    // dump_file << "\n=== Before Applying Delegate ===" << std::endl;
    // InspectSelectedSignature(interpreter.get(), sig_index, tensor_buffer_map, &dump_file);

    //* ============ Apply Delegate ============ */
    // create 400MB buffer for weight caching
    constexpr size_t buf_size = 400 * 1024 * 1024;
    g_weight_cache_provider.InitManagedBuffer(buf_size);
    // g_weight_cache_provider.PrefetchFromFile(absl::GetFlag(FLAGS_tflite_model));

    if (!absl::GetFlag(FLAGS_weight_cache_path).empty())
        ApplyXNNPACKWithWeightCaching(interpreter.get());

    // Use the selected signature key for prefill runner
    const auto &signature_keys = interpreter->signature_keys();
    std::string selected_signature_key = *signature_keys[sig_index];
    tflite::SignatureRunner *prefill_runner = interpreter->GetSignatureRunner(selected_signature_key.c_str());
    tflite::SignatureRunner *decode_runner = interpreter->GetSignatureRunner("decode");

    // Allocate tensors for both runners
    prefill_runner->AllocateTensors();
    decode_runner->AllocateTensors();

    g_weight_cache_provider.DumpWeightCacheStructureToFile("weight_cache_structure.log");
    g_weight_cache_provider.DumpTensorIdentifierMapToFile("weight_cache_tensor_id_map.log");
    ValidateWeightCacheMappings(interpreter.get(), selected_signature_key, tensor_buffer_map);
    
    std::cout << "Verifying Buffer in weight cache" << std::endl;
    // g_weight_cache_provider.VerifyAllBuffers();
    std::cout << "Verification done" << std::endl;

    //* ============ Dump Model (After Delegate) ============ */
    std::cout << "\n\nInvoking prefill ..." << std::endl;
    size_t cached_kb = GetPageCacheKB();
    std::cout << "Current Page Cache: " << cached_kb << " kB" << std::endl;
    prefill_runner->Invoke();
    std::cout << "Prefill Invoke done." << std::endl;
    cached_kb = GetPageCacheKB();
    std::cout << "Current Page Cache: " << cached_kb << " kB" << std::endl;

    std::cout << "\n\nInvoking decode ..." << std::endl;
    cached_kb = GetPageCacheKB();
    std::cout << "Current Page Cache: " << cached_kb << " kB" << std::endl;
    decode_runner->Invoke();
    std::cout << "Decode Invoke done." << std::endl;
    cached_kb = GetPageCacheKB();
    std::cout << "Current Page Cache: " << cached_kb << " kB" << std::endl;

    dump_file << "\n=== After Applying Delegate ===" << std::endl;
    // InspectSelectedSignature(interpreter.get(), sig_index, tensor_buffer_map, &dump_file);

    dump_file.close();

    std::cout << "Model dump result to: " << absl::GetFlag(FLAGS_dump_file_path) << std::endl;
    return 0;
}
