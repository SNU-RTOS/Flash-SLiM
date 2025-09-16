#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <memory>
#include <unordered_set>

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
    // Human-readable byte formatting helper
    std::string FormatBytes(size_t bytes) {
        const double kb = 1024.0;
        const double mb = kb * 1024.0;
        const double gb = mb * 1024.0;
        std::ostringstream oss;
        oss.setf(std::ios::fixed); oss<<std::setprecision(2);
        if (bytes >= (size_t)gb)      { oss << (bytes / gb) << "GB"; }
        else if (bytes >= (size_t)mb) { oss << (bytes / mb) << "MB"; }
        else if (bytes >= (size_t)kb) { oss << (bytes / kb) << "KB"; }
        else                          { oss.unsetf(std::ios::floatfield); oss << bytes << "B"; }
        return oss.str();
    }
    // --------------------------------------------------------------------------
    // Print detailed tensor information (migrated from text_generator_main.cc)
    // --------------------------------------------------------------------------
    void print_tensor_details(int tensor_idx, TfLiteTensor* tensor, std::ostream* output = nullptr) {
        std::ostream& out = (output != nullptr) ? *output : std::cout;
        
        if (!tensor) {
            out << "Tensor " << tensor_idx << " is NULL\n";
            return;
        }
        
        void* tensor_data_address = tensor->data.raw;
        out << "Data Address: " << tensor_data_address << " ";

        // Tensor Type
        const char* type_name = TfLiteTypeGetName(tensor->type);
        out << "Type: " << (type_name ? type_name : "Unknown") << " ";

        // Tensor Allocation Type
        out << "Allocation Type: ";
        switch (tensor->allocation_type) {
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
        if(tensor->dims && tensor->dims->size > 0){
            for (int dim_idx = 0; dim_idx < tensor->dims->size; ++dim_idx) {
                out << tensor->dims->data[dim_idx];
                if (dim_idx < tensor->dims->size - 1) out << ", ";
            }
        }
        out << "]\n";
    }

    // --------------------------------------------------------------------------
    // Utility for applying XNNPACK weight caching
    // --------------------------------------------------------------------------
    void ApplyXNNPACKWithWeightCaching(tflite::Interpreter *interpreter)
    {
        auto delegate_options = TfLiteXNNPackDelegateOptionsDefault();
        std::string weight_cache_path = absl::GetFlag(FLAGS_weight_cache_path);
        delegate_options.weight_cache_file_path = weight_cache_path.c_str();
        delegate_options.num_threads = 4; // Default num_threads for dump tool
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

    void InspectExecutionPlan(tflite::Interpreter *interpreter, int subgraph_idx, std::ostream *output = nullptr, int indent = 0)
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
            if (absl::GetFlag(FLAGS_op_tensor_byte_stats)) {
                auto collect = [&](const TfLiteIntArray* arr){
                    if (!arr) return;
                    for (int i = 0; i < arr->size; ++i) {
                        int tidx = arr->data[i];
                        if (tidx < 0) continue;
                        TfLiteTensor* t = interpreter->tensor(tidx);
                        if (!t) continue;
                        switch (t->allocation_type) {
                            case kTfLiteMmapRo: op_bytes_mmap += t->bytes; break;
                            case kTfLiteArenaRw:
                            case kTfLiteArenaRwPersistent: op_bytes_arena += t->bytes; break;
                            default: break;
                        }
                    }
                };
                collect(node->inputs);
                collect(node->outputs);
                collect(node->temporaries);
            }

            out << indent_str << "  " << std::setw(4) << std::setfill(' ') << absolute_op_counter << ": "
                << "[" << std::setw(4) << std::setfill('0') << idx << "] " << op_name;
            if (absl::GetFlag(FLAGS_op_tensor_byte_stats)) {
                out << " (Mmap=" << FormatBytes(op_bytes_mmap) << ", Arena=" << FormatBytes(op_bytes_arena) << ")";
            }
            absolute_op_counter++;

            // Dump detailed tensor information if requested
            if (absl::GetFlag(FLAGS_dump_tensor_details)) {
                const auto* node_and_reg = subgraph->node_and_registration(idx);

                std::string tensor_indent_str(indent + 2, ' ');
                if (node_and_reg) {
                    const TfLiteNode* node = &node_and_reg->first;
                    
                    // Print input tensors
                    if (node->inputs && node->inputs->size > 0) {
                        out << std::endl << tensor_indent_str << "    Input Tensors:" << std::endl;
                        for (int i = 0; i < node->inputs->size; ++i) {
                            int tensor_idx = node->inputs->data[i];
                            if (tensor_idx >= 0) {
                                out << tensor_indent_str << "      Input " << i << ": " << tensor_idx << " ";
                                auto* tensor = interpreter->tensor(tensor_idx);
                                print_tensor_details(tensor_idx, tensor, output);
                            }
                        }
                    }
                    
                    // Print output tensors
                    if (node->outputs && node->outputs->size > 0) {
                        out << tensor_indent_str << "    Output Tensors:" << std::endl;
                        for (int i = 0; i < node->outputs->size; ++i) {
                            int tensor_idx = node->outputs->data[i];
                            if (tensor_idx >= 0) {
                                out << tensor_indent_str << "      Output " << i << ": " << tensor_idx << " ";
                                auto* tensor = interpreter->tensor(tensor_idx);
                                print_tensor_details(tensor_idx, tensor, output);
                            }
                        }
                    }
                    
                    // Print temporary tensors if any
                    if (node->temporaries && node->temporaries->size > 0) {
                        out << tensor_indent_str << "    Temporary Tensors:" << std::endl;
                        for (int i = 0; i < node->temporaries->size; ++i) {
                            int tensor_idx = node->temporaries->data[i];
                            if (tensor_idx >= 0) {
                                out << tensor_indent_str << "      Temporary " << i << ": " << tensor_idx << " ";
                                auto* tensor = interpreter->tensor(tensor_idx);
                                print_tensor_details(tensor_idx, tensor, output);
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
                int subgraph_idx = op_state->subgraph_index;
                auto decomposition_subgraph = interpreter->subgraph(subgraph_idx);
                auto tmp_name = decomposition_subgraph->GetName();
                out << " (→ Subgraph " << subgraph_idx << " :" << tmp_name << ")" << std::endl;
                out << delegate_indent_str << "-------------------" << std::endl;
                // increase indent by 2 spaces for nested subgraph
                InspectExecutionPlan(interpreter, subgraph_idx, output, indent + 4);
                out << delegate_indent_str << "-------------------" << std::endl;
            }
            else
            {
                out << std::endl;
            }
        }
    }

    void InspectSignatureExecutionPlan(tflite::Interpreter *interpreter, const std::string &signature_key, std::ostream *output = nullptr, int indent = 0)
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
        InspectExecutionPlan(interpreter, subgraph_idx, output, indent);
        out << std::endl; // spacing after subgraph dump
    }

    void InspectSelectedSignature(tflite::Interpreter *interpreter, int sig_index, std::ostream *output = nullptr, int indent = 2)
    {
        std::ostream &out = (output != nullptr) ? *output : std::cout;

        const auto &signature_keys = interpreter->signature_keys();
        if (sig_index < 0 || sig_index >= signature_keys.size())
        {
            out << "Invalid signature index: " << sig_index << std::endl;
            return;
        }

        const std::string &signature_key = *signature_keys[sig_index];
        InspectSignatureExecutionPlan(interpreter, signature_key, output, indent);
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

} // end anonymous namespace

// =======================================================================
// main() entry
// =======================================================================
int main(int argc, char *argv[])
{
    // Parse flags
    absl::ParseCommandLine(argc, argv);

    std::cout << "====== dump_model_cpu ======" << std::endl;
    std::cout << " TFLite model: " << absl::GetFlag(FLAGS_tflite_model) << std::endl;

    //* ============ Create Model, Interpreter and Profiler ============ */
    std::unique_ptr<tflite::FlatBufferModel> model;
    model = tflite::FlatBufferModel::BuildFromFile(absl::GetFlag(FLAGS_tflite_model).c_str());
    std::unique_ptr<tflite::Interpreter> interpreter;
    std::unique_ptr<tflite::profiling::Profiler> op_profiler = std::make_unique<tflite::profiling::Profiler>();
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::ops::custom::GenAIOpsRegisterer(&resolver); // Register GenAI custom ops
    tflite::InterpreterBuilder builder(*model, resolver);
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
    // InspectSelectedSignature(interpreter.get(), sig_index, &dump_file);

    //* ============ Apply Delegate ============ */
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

    //* ============ Dump Model (After Delegate) ============ */

    dump_file << "\n=== After Applying Delegate ===" << std::endl;
    InspectSelectedSignature(interpreter.get(), sig_index, &dump_file);

    dump_file.close();

    std::cout << "Model dump result to: " << absl::GetFlag(FLAGS_dump_file_path) << std::endl;
    return 0;
}
