#include "sampler.h"

// --------------------------------------------------------------------------
// A class that provides various sampling methods (Greedy, Top-K, Top-P, etc.)
// --------------------------------------------------------------------------
namespace ai_edge_torch::custom::sampler
{
    // ------------------------
    // Greedy Sampler
    // ------------------------
    int greedy_sampler(const TfLiteTensor *logits)
    {
        float max_value = -std::numeric_limits<float>::infinity();
        int max_index = 0;
        int vocab_size = logits->dims->data[2];

        for (int i = 0; i < vocab_size; ++i)
        {
            if (logits->data.f[i] > max_value)
            {
                max_value = logits->data.f[i];
                max_index = i;
            }
        }
        return max_index;
    }

    // ------------------------
    // Top-K Sampler
    // ------------------------
    int top_k_sampler(const TfLiteTensor *logits, int k)
    {
        int vocab_size = logits->dims->data[2];
        std::vector<std::pair<float, int>> sorted_logits;
        sorted_logits.reserve(vocab_size);

        for (int i = 0; i < vocab_size; ++i)
        {
            sorted_logits.emplace_back(logits->data.f[i], i);
        }

        // Partial sort to get the top k elements
        if (k < vocab_size)
        {
            std::partial_sort(sorted_logits.begin(), sorted_logits.begin() + k, sorted_logits.end(),
                              std::greater<std::pair<float, int>>());
            sorted_logits.resize(k);
        }
        else
        {
            // If k >= vocab_size, no need to cut
            std::sort(sorted_logits.begin(), sorted_logits.end(), std::greater<std::pair<float, int>>());
        }

        // Compute normalized probabilities
        float sum_probs = 0.0f;
        for (auto &pair : sorted_logits)
        {
            sum_probs += std::exp(pair.first);
        }
        std::vector<float> probabilities;
        probabilities.reserve(sorted_logits.size());
        for (auto &pair : sorted_logits)
        {
            probabilities.push_back(std::exp(pair.first) / sum_probs);
        }

        // Multinomial sampling
        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());

        return sorted_logits[dist(gen)].second;
    }

    // ------------------------
    // Top-P (Nucleus) Sampler
    // ------------------------
    int top_p_sampler(const TfLiteTensor *logits, float p)
    {
        int vocab_size = logits->dims->data[2];
        std::vector<std::pair<float, int>> sorted_logits;
        sorted_logits.reserve(vocab_size);

        for (int i = 0; i < vocab_size; ++i)
        {
            sorted_logits.emplace_back(logits->data.f[i], i);
        }

        // Sort descending by logit value
        std::sort(sorted_logits.begin(), sorted_logits.end(),
                  std::greater<std::pair<float, int>>());

        // Apply softmax to get probabilities
        std::vector<float> probabilities(vocab_size);
        float sum_exp = 0.0f;
        for (int i = 0; i < vocab_size; ++i)
        {
            float val = std::exp(sorted_logits[i].first);
            probabilities[i] = val;
            sum_exp += val;
        }
        for (int i = 0; i < vocab_size; ++i)
        {
            probabilities[i] /= sum_exp;
        }

        // Find the cutoff index where cumulative probability exceeds p
        float cumulative_prob = 0.0f;
        int cutoff_index = vocab_size - 1;
        for (int i = 0; i < vocab_size; ++i)
        {
            cumulative_prob += probabilities[i];
            if (cumulative_prob > p)
            {
                cutoff_index = i;
                break;
            }
        }

        // Resize vectors to [0..cutoff_index]
        float new_sum = 0.0f;
        for (int i = 0; i <= cutoff_index; ++i)
        {
            new_sum += probabilities[i];
        }
        for (int i = 0; i <= cutoff_index; ++i)
        {
            probabilities[i] /= new_sum;
        }

        probabilities.resize(cutoff_index + 1);
        sorted_logits.resize(cutoff_index + 1);

        // Multinomial sampling
        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
        return sorted_logits[dist(gen)].second;
    }

    // ------------------------
    // Temperature + Top-K + Top-P Sampler
    // ------------------------
    int temperature_top_k_top_p_sampler(const TfLiteTensor *logits,
                                   float temperature, int k, float p)
    {
        int vocab_size = logits->dims->data[2];
        std::vector<std::pair<float, int>> sorted_logits;
        sorted_logits.reserve(vocab_size);

        // 1) Apply Temperature
        std::vector<float> scaled_logits(vocab_size);
        for (int i = 0; i < vocab_size; ++i)
        {
            scaled_logits[i] = logits->data.f[i] / temperature;
        }

        // 2) Softmax over scaled logits
        float max_logit = *std::max_element(scaled_logits.begin(), scaled_logits.end());
        float sum_exp = 0.0f;
        for (int i = 0; i < vocab_size; ++i)
        {
            scaled_logits[i] = std::exp(scaled_logits[i] - max_logit);
            sum_exp += scaled_logits[i];
        }
        for (int i = 0; i < vocab_size; ++i)
        {
            scaled_logits[i] /= sum_exp;
            // Keep index-value pairs for sorting
            sorted_logits.emplace_back(scaled_logits[i], i);
        }

        // 3) Sort descending by probability
        std::sort(sorted_logits.begin(), sorted_logits.end(),
                  std::greater<std::pair<float, int>>());

        // 4) Top-K filter
        int top_k = std::min(k, vocab_size);
        sorted_logits.resize(top_k);

        // 5) Top-P filter within top-k
        float cumulative_prob = 0.0f;
        int cutoff_index = top_k - 1;
        for (int i = 0; i < top_k; ++i)
        {
            cumulative_prob += sorted_logits[i].first;
            if (cumulative_prob > p)
            {
                cutoff_index = i;
                break;
            }
        }
        sorted_logits.resize(cutoff_index + 1);

        // 6) Renormalize final probabilities
        float new_sum = 0.0f;
        for (auto &pair : sorted_logits)
        {
            new_sum += pair.first;
        }

        std::vector<float> final_probs;
        final_probs.reserve(sorted_logits.size());
        for (auto &pair : sorted_logits)
        {
            final_probs.push_back(pair.first / new_sum);
        }

        // 7) Multinomial sampling
        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<> dist(final_probs.begin(), final_probs.end());
        return sorted_logits[dist(gen)].second;
    }

    // ------------------------
    // Temperature + Top-K + Top-P + Repetition Penalty Sampler
    // ------------------------
    int temperature_top_k_top_p_repetition_sampler(const TfLiteTensor *logits,
                                                  float temperature, int k, float p,
                                                  const std::unordered_set<int>& previously_generated_tokens,
                                                  float repetition_penalty)
    {
        int vocab_size = logits->dims->data[2];
        std::vector<std::pair<float, int>> sorted_logits;
        sorted_logits.reserve(vocab_size);

        // 1) Apply Temperature + Repetition Penalty
        std::vector<float> scaled_logits(vocab_size);
        for (int i = 0; i < vocab_size; ++i)
        {
            float logit = logits->data.f[i];

            // Apply repetition penalty
            if (previously_generated_tokens.count(i))
            {
                logit /= repetition_penalty;
            }

            // Apply temperature
            scaled_logits[i] = logit / temperature;
        }

        // 2) Softmax
        float max_logit = *std::max_element(scaled_logits.begin(), scaled_logits.end());
        float sum_exp = 0.0f;
        for (int i = 0; i < vocab_size; ++i)
        {
            scaled_logits[i] = std::exp(scaled_logits[i] - max_logit);
            sum_exp += scaled_logits[i];
        }
        for (int i = 0; i < vocab_size; ++i)
        {
            scaled_logits[i] /= sum_exp;
            sorted_logits.emplace_back(scaled_logits[i], i);
        }

        // 3) Top-K
        int top_k = std::min(k, vocab_size);
        std::sort(sorted_logits.begin(), sorted_logits.end(), std::greater<>());
        sorted_logits.resize(top_k);

        // 4) Top-P
        float cumulative_prob = 0.0f;
        int cutoff_index = top_k - 1;
        for (int i = 0; i < top_k; ++i)
        {
            cumulative_prob += sorted_logits[i].first;
            if (cumulative_prob > p)
            {
                cutoff_index = i;
                break;
            }
        }
        sorted_logits.resize(cutoff_index + 1);

        // 5) Renormalize
        float new_sum = 0.0f;
        for (auto &pair : sorted_logits) new_sum += pair.first;

        std::vector<float> final_probs;
        final_probs.reserve(sorted_logits.size());
        for (auto &pair : sorted_logits)
        {
            final_probs.push_back(pair.first / new_sum);
        }

        // 6) Sample
        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<> dist(final_probs.begin(), final_probs.end());
        return sorted_logits[dist(gen)].second;
    }
};
