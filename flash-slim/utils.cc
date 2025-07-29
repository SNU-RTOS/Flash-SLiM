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
#include <chrono>
#include <map>
#include <sstream>
#include <iomanip>

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

    //
    // 현재 시간을 나노초 단위로 반환
    int64_t getCurrentTimestampNs()
    {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   now.time_since_epoch())
            .count();
    }

    // 타임스탬프를 JSON 형식으로 포맷팅
    std::string formatTimestampJson(int64_t timestamp_ns, const std::string &event_type,
                                    const std::string &component, int stage_idx)
    {
        // 나노초 단위 타임스탬프를 초와 나노초 부분으로 분리
        int64_t seconds = timestamp_ns / 1000000000;
        int64_t nanos = timestamp_ns % 1000000000;

        std::stringstream ss;
        ss << "{";
        ss << "\"timestamp\": {\"seconds\": " << seconds << ", \"nanos\": " << nanos << "}, ";
        ss << "\"event\": \"" << event_type << "\", ";
        ss << "\"component\": \"" << component << "\"";

        if (stage_idx >= 0)
        {
            ss << ", \"stage_idx\": " << stage_idx;
        }

        ss << "}";
        return ss.str();
    }

    // 타임스탬프된 이벤트 로깅
    void logTimestampedEvent(const std::string &event_type, const std::string &component,
                             int stage_idx, std::ostream &out)
    {
        int64_t timestamp = getCurrentTimestampNs();
        out << formatTimestampJson(timestamp, event_type, component, stage_idx) << std::endl;
    }

    // 더 자세한 속성을 가진 JSON 이벤트 로깅
    void logJsonEvent(const std::string &event_type, const std::string &component,
                      const std::map<std::string, std::string> &attributes,
                      int stage_idx, std::ostream &out)
    {
        int64_t timestamp = getCurrentTimestampNs();

        // 나노초 단위 타임스탬프를 초와 나노초 부분으로 분리
        int64_t seconds = timestamp / 1000000000;
        int64_t nanos = timestamp % 1000000000;

        std::stringstream ss;
        ss << "{";
        ss << "\"timestamp\": {\"seconds\": " << seconds << ", \"nanos\": " << nanos << "}, ";
        ss << "\"event\": \"" << event_type << "\", ";
        ss << "\"component\": \"" << component << "\"";

        if (stage_idx >= 0)
        {
            ss << ", \"stage_idx\": " << stage_idx;
        }

        // 추가 속성들 추가
        for (const auto &attr : attributes)
        {
            ss << ", \"" << attr.first << "\": \"" << attr.second << "\"";
        }

        ss << "}";
        out << ss.str() << std::endl;
    }

} // namespace custom::util