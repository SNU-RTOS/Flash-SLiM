#include <cstdio>
#include <memory>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <iomanip> // for setw, setfill
#include <fstream> // for ofstream
#include <functional>
#include <sstream> // for stringstream

#include "tflite/interpreter_builder.h"
#include "tflite/kernels/register.h"
#include "tflite/interpreter.h"
#include "tflite/model_builder.h"
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"
#include "tflite/schema/schema_generated.h"
#include "tflite/profiling/profiler.h"
#include "xnnpack/operator.h"

namespace
{
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
        int cnt_atomic_ops = 0;  // 일반 atomic operator 개수
        int cnt_delegate_nodes = 0;  // DELEGATE 노드 개수
        int cnt_xnn_ops = 0;  // XNNPACK operator 개수

        // Helper function to count XNNPACK operators from delegate output
        auto count_xnnpack_ops = [&](void* delegate_data) -> int {
            std::ostringstream temp_stream;
            TfLiteXNNPackDelegateInspect(delegate_data, &temp_stream, "");
            std::string delegate_output = temp_stream.str();
            
            int xnn_op_count = 0;
            std::istringstream iss(delegate_output);
            std::string line;
            
            while (std::getline(iss, line)) {
                // XNNPACK operator lines typically start with "[XXXX]" format
                if (line.find("[") != std::string::npos && line.find("]") != std::string::npos) {
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
            int total = 0;  // 0으로 시작하여 명시적으로 카운트

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
                else if (r->builtin_code == tflite::BuiltinOperator_DELEGATE)
                {
                    // DELEGATE 노드는 XNNPACK operators로 치환되므로 
                    // delegate 내부의 operator 개수를 세고, delegate 노드 자체는 제외
                    cnt_delegate_nodes++;  // DELEGATE 노드 개수 카운트
                    int xnn_ops = count_xnnpack_ops(n->user_data);
                    cnt_xnn_ops += xnn_ops;  // XNNPACK 연산자 개수 누적
                    total += xnn_ops;  // delegate 내부 operators만 카운트
                }
                else
                {
                    // 일반 노드들만 카운트 (STABLEHLO_COMPOSITE, DELEGATE 제외)
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

        for (int idx : execution_plan)
        {
            const auto *node_and_reg = subgraph->node_and_registration(idx);
            auto *node = &node_and_reg->first;
            auto *reg = &node_and_reg->second;
            std::string op_name =
                tflite::EnumNameBuiltinOperator(
                    static_cast<tflite::BuiltinOperator>(reg->builtin_code));

            // print with zero-padded index, prefixed by indent
            out << indent_str << "  [" << std::setw(4) << std::setfill('0') << idx << "] " << op_name;

            std::string delegate_indent_str(indent + 4, ' ');
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
        out << std::endl;
    }

    void InspectAllSignatures(tflite::Interpreter *interpreter, std::ostream *output = nullptr, int indent = 0)
    {
        const auto &signature_keys = interpreter->signature_keys();

        // Inspect execution plan for each signature
        for (const std::string *key : signature_keys)
        {
            InspectSignatureExecutionPlan(interpreter, *key, output, indent);
        }
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

    void GetSignatureIndexFromUser(tflite::Interpreter *interpreter, int &sig_index)
    {
        // Print model signature keys (console only)
        std::cout << "\n=== Model Signature ===" << std::endl;
        sig_index = -1;
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
    }
}

int main(int argc, char *argv[])
{
    setenv("TF_CPP_MIN_LOG_LEVEL", "0", 1);
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <tflite model> <log file>\n";
        return 1;
    }
    const char *model_file_name = argv[1];
    const char *log_file_name = argv[2];

    // Generate log file name
    // std::string log_filename = GenerateLogFileName(filename);
    std::ofstream log_file(log_file_name);
    if (!log_file.is_open())
    {
        std::cerr << "❌ Failed to open log file: " << log_file_name << std::endl;
        return 1;
    }

    std::cout << "====== dump_model_cpu ======" << std::endl;
    std::cout << "🔍 Loading model from: " << model_file_name << std::endl;
    std::cout << "📝 Logging output to: " << log_file_name << std::endl;

    // 1. Load the model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_file_name);
    if (!model)
    {
        std::cerr << "❌ Failed to load model: " << model_file_name << std::endl;
        return 1;
    }

    // 2. Create Op resolver
    tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;

    // 3. Create Interpreter
    std::unique_ptr<tflite::Interpreter> interpreter;

    // 3.5 Create a profiler
    auto profiler = std::make_unique<tflite::profiling::Profiler>();

    // 4. Create InterpreterBuilder and Initialize Interpreter
    tflite::InterpreterBuilder builder(*model, resolver);
    builder(&interpreter);
    if (!interpreter)
    {
        std::cerr << "❌ Failed to create interpreter\n";
        return 1;
    }
    interpreter->SetProfiler(profiler.get());

    // Print model signature (console only, as requested)
    int sig_index = 0;
    GetSignatureIndexFromUser(interpreter.get(), sig_index);
    if (sig_index < 0)
    {
        std::cerr << "❌ Invalid signature index. Exiting.\n";
        return 1;
    }

    // Inspect all signatures before delegate
    log_file << "\n=== Before Applying Delegate ===" << std::endl;

    InspectSelectedSignature(interpreter.get(), sig_index, &log_file);

    // 5. Apply XNNPACK delegate
    TfLiteXNNPackDelegateOptions xnnpack_opts = TfLiteXNNPackDelegateOptionsDefault();
    TfLiteDelegate *xnn_delegate = TfLiteXNNPackDelegateCreate(&xnnpack_opts);
    bool delegate_applied = false;

    if (xnn_delegate && interpreter->ModifyGraphWithDelegate(xnn_delegate) == kTfLiteOk)
        delegate_applied = true;

    // 6. Allocate tensors
    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        std::cerr << "❌ Failed to allocate tensors\n";
        if (delegate_applied)
            TfLiteXNNPackDelegateDelete(xnn_delegate);
        return 1;
    }

    // print model execution plan after applying delegate
    if (delegate_applied)
    {
        log_file << "\n=== After Applying Delegate ===" << std::endl;

        InspectSelectedSignature(interpreter.get(), sig_index, &log_file);
    }

    // 7. Deallocate delegate
    if (delegate_applied)
        TfLiteXNNPackDelegateDelete(xnn_delegate);

    log_file.close();

    std::cout << "\n✔ Parsing complete. Log saved to: " << log_file_name << std::endl;

    return 0;
}