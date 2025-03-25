#ifndef AI_EDGE_TORCH_GENERATIVE_CUSTOM_SAMPLER_H_
#define AI_EDGE_TORCH_GENERATIVE_CUSTOM_SAMPLER_H_

#include <vector>
#include <random>
#include <algorithm>
#include <limits>
#include <cmath>
#include "tensorflow/lite/interpreter.h"

namespace ai_edge_torch::custom::sampler
{
    // Greedy Sampler
    int greedy_sampler(const TfLiteTensor *logits);

    // Top-K Sampler
    int top_k_sampler(const TfLiteTensor *logits, int k);

    // Top-P (Nucleus) Sampler
    int top_p_sampler(const TfLiteTensor *logits, float p);

    // Temperature + Top-K + Top-P Sampler
    int temperature_top_k_top_p_sampler(const TfLiteTensor *logits, float temperature, int k, float p);

} // namespace ai_edge_torch::custom_util

#endif // AI_EDGE_TORCH_GENERATIVE_CUSTOM_SAMPLER_H_
