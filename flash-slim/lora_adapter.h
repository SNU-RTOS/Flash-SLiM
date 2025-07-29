#ifndef FLASH_SLIM_LORA_ADAPTER_H_
#define FLASH_SLIM_LORA_ADAPTER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tflite/interpreter.h"
#include "tflite/signature_runner.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"

#include "aligned_allocator.h"

namespace ai_edge_torch {
namespace examples {

// An example implementation of LoRA adapters manager for TFLite interpreter.
// The class loads an adapter from a flatbuffers files and provides helper
// methods for finding the right signature and setting the appropriate input
// tensors. Please note the use of CustomAllocator to ensure zero-copy loading
// and potentially hot-swapping between multiple adapters with minimal cost.
class LoRA {
public:
    static std::unique_ptr<LoRA> FromFile(absl::string_view path);

    tflite::SignatureRunner *GetPrefillRunner(tflite::Interpreter *interpreter,
                                              int matched_sequence_length) const;
    tflite::SignatureRunner *GetDecodeRunner(
        tflite::Interpreter *interpreter) const;

    int rank() const { return rank_; };

private:
    explicit LoRA(int rank,
                  absl::flat_hash_map<std::string,
                                      std::vector<float, ai_edge_torch::mem::AlignedAllocator<float>>>
                      tensors)
        : rank_(rank), tensors_(std::move(tensors)) {}

    tflite::SignatureRunner *GetRunnerHelper(
        tflite::Interpreter *interpreter, absl::string_view signature_name) const;

    // The rank of the LoRA adapter.
    const int rank_;
    // A Map of names to LoRA tensors.
    const absl::flat_hash_map<std::string,
                              std::vector<float, ai_edge_torch::mem::AlignedAllocator<float>>>
        tensors_;
};

} // namespace examples
} // namespace ai_edge_torch

#endif // FLASH_SLIM_LORA_ADAPTER_H_
