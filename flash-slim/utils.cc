/* Copyright 2025 The AI Edge Torch Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "utils.h"

#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tflite/interpreter.h"
#include "tflite/model_builder.h"
#include "tflite/schema/schema_generated.h"
#include "tflite/signature_runner.h"

namespace ai_edge_torch
{
    namespace examples
    {

        std::unique_ptr<LoRA> LoRA::FromFile(absl::string_view path)
        {
            std::unique_ptr<tflite::FlatBufferModel> model =
                tflite::FlatBufferModel::VerifyAndBuildFromFile(path.data());
            if (model == nullptr)
            {
                return nullptr;
            }

            int rank = -1;
            absl::flat_hash_map<std::string, std::vector<float, AlignedAllocator<float>>>
                tensors;
            for (const auto &tensor :
                 *model->GetModel()->subgraphs()->Get(0)->tensors())
            {
                size_t size = 1;
                for (const int &dim : *tensor->shape())
                {
                    size *= dim;
                }
                std::vector<float, AlignedAllocator<float>> buffer(size);
                const auto *data =
                    model->GetModel()->buffers()->Get(tensor->buffer())->data();
                memcpy(buffer.data(), data->data(), data->size());
                tensors.emplace(*tensor->name(), std::move(buffer));

                if (tensor->name()->str() == "lora_atten_q_a_prime_weight_0")
                {
                    rank = tensor->shape()->Get(1);
                }
            }
            if (rank == -1)
            {
                return nullptr;
            }

            return absl::WrapUnique(new LoRA(rank, std::move(tensors)));
        }

        tflite::SignatureRunner *LoRA::GetPrefillRunner(
            tflite::Interpreter *interpreter, int matched_sequence_length) const
        {
            std::string signature_name =
                absl::StrFormat("prefill_%d_lora_r%d", matched_sequence_length, rank_);
            return GetRunnerHelper(interpreter, signature_name);
        }

        tflite::SignatureRunner *LoRA::GetDecodeRunner(
            tflite::Interpreter *interpreter) const
        {
            std::string signature_name = absl::StrFormat("decode_lora_r%d", rank_);
            return GetRunnerHelper(interpreter, signature_name);
        };

        tflite::SignatureRunner *LoRA::GetRunnerHelper(
            tflite::Interpreter *interpreter, absl::string_view signature_name) const
        {
            tflite::SignatureRunner *runner =
                interpreter->GetSignatureRunner(signature_name.data());
            if (runner == nullptr)
            {
                return nullptr;
            }

            absl::flat_hash_set<std::string> lora_input_tensors;
            lora_input_tensors.reserve(runner->input_size());
            for (const char *input_name : runner->input_names())
            {
                if (absl::StrContains(input_name, "lora"))
                {
                    lora_input_tensors.insert(input_name);
                }
            }

            if (lora_input_tensors.size() < tensors_.size())
            {
                return nullptr;
            }

            for (const auto &[name, buffer] : tensors_)
            {
                TfLiteTensor *tensor = runner->input_tensor(name.c_str());
                if (tensor == nullptr)
                {
                    return nullptr;
                }
                lora_input_tensors.erase(name);
                TfLiteCustomAllocation allocation = {
                    .data = static_cast<void *>(const_cast<float *>(buffer.data())),
                    .bytes = buffer.size() * sizeof(float)};
                if (runner->SetCustomAllocationForInputTensor(name.c_str(), allocation) !=
                    kTfLiteOk)
                {
                    return nullptr;
                }
            };
            if (runner->AllocateTensors() != kTfLiteOk)
            {
                return nullptr;
            }

            for (const auto &name : lora_input_tensors)
            {
                TfLiteTensor *tensor = runner->input_tensor(name.c_str());
                if (tensor == nullptr)
                {
                    return nullptr;
                }
                memset(tensor->data.data, 0, tensor->bytes);
            }

            return runner;
        }
    } // namespace examples

    namespace custom::util
    {
        // --------------------------------------------------------------------------
        // Helper functions to print tensor details.
        // --------------------------------------------------------------------------
        const char *TfLiteTypeToString(TfLiteType type)
        {
            switch (type)
            {
            case kTfLiteNoType:
                return "NoType";
            case kTfLiteFloat32:
                return "Float32";
            case kTfLiteInt32:
                return "Int32";
            case kTfLiteUInt8:
                return "UInt8";
            case kTfLiteInt64:
                return "Int64";
            case kTfLiteString:
                return "String";
            case kTfLiteBool:
                return "Bool";
            case kTfLiteInt16:
                return "Int16";
            case kTfLiteComplex64:
                return "Complex64";
            case kTfLiteInt8:
                return "Int8";
            case kTfLiteFloat16:
                return "Float16";
            case kTfLiteFloat64:
                return "Float64";
            case kTfLiteComplex128:
                return "Complex128";
            case kTfLiteUInt64:
                return "UInt64";
            case kTfLiteResource:
                return "Resource";
            case kTfLiteVariant:
                return "Variant";
            case kTfLiteUInt32:
                return "UInt32";
            default:
                return "Unknown";
            }
        }

        void PrintTensorInfo(const TfLiteTensor *tensor, const char *tensor_name)
        {
            if (tensor == nullptr)
                return;
            std::cout << "    - " << tensor_name
                      << " (Type: " << TfLiteTypeToString(tensor->type)
                      << ", Dims: [";
            for (int i = 0; i < tensor->dims->size; ++i)
            {
                std::cout << tensor->dims->data[i] << (i == tensor->dims->size - 1 ? "" : ", ");
            }
            std::cout << "])\n";
        }

        // --------------------------------------------------------------------------
        // Prints information about all signature runners in the interpreter.
        // --------------------------------------------------------------------------
        void PrintSignatureRunnersInfo(tflite::Interpreter *interpreter)
        {
            std::cout << "\n[INFO] Dumping Signature Runner Information...\n";
            std::cout << "==================================================\n";

            const auto &signature_keys = interpreter->signature_keys();
            for (const auto *key : signature_keys)
            {
                tflite::SignatureRunner *runner = interpreter->GetSignatureRunner(key->c_str());
                if (runner == nullptr)
                    continue;

                std::cout << "Signature: \"" << *key << "\"\n";

                // Print Inputs
                std::cout << "  Inputs:\n";
                const auto &inputs = runner->input_names();
                for (const auto &input_name : inputs)
                {
                    // std::cout << "    - " << input_name << ": "<< std::endl;
                    TfLiteTensor *input_tensor = runner->input_tensor(input_name);
                    PrintTensorInfo(input_tensor, input_name);
                }

                // Print Outputs
                std::cout << "  Outputs:\n";
                const auto &outputs = runner->output_names();
                for (const auto &output_name : outputs)
                {
                    // std::cout << "    - " << output_name << ": "<< std::endl;
                    TfLiteTensor *output_tensor = runner->input_tensor(output_name);
                    PrintTensorInfo(output_tensor, output_name);
                }
                std::cout << "--------------------------------------------------\n";
            }
            std::cout << "==================================================\n\n";
        }

    } // namespace custom::util
} // namespace ai_edge_torch
