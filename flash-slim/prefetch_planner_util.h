// Copyright 2025 Flash-SLiM Research Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
// Prefetch Planner for Weight Streaming
// This module provides tools to record and plan I/O prefetch operations
// for weight chunks during model execution.


#ifndef FLASH_SLIM_PREFETCH_PLANNER_UTIL_H
#define FLASH_SLIM_PREFETCH_PLANNER_UTIL_H

#include <fstream>
#include <string>
#include <memory>
#include <map>
#include "nlohmann/json.hpp"
#include "tflite/delegates/xnnpack/streaming_weight_cache.h"

namespace flash_slim
{

    // JSON-based implementation of WeightChunkInfoWriter
    // Uses nlohmann/json for serialization
    class JsonWeightChunkInfoHandler : public tflite::xnnpack::WeightChunkInfoHandler
    {
    public:
        explicit JsonWeightChunkInfoHandler(const std::string &output_path);
        ~JsonWeightChunkInfoHandler() override;

        // Delete copy constructor and assignment operator
        JsonWeightChunkInfoHandler(const JsonWeightChunkInfoHandler &) = delete;
        JsonWeightChunkInfoHandler &operator=(const JsonWeightChunkInfoHandler &) = delete;

        void WriteChunkInfo(const tflite::xnnpack::StreamingWeightCacheProvider::weight_chunk_info_t &chunk_info,
                            tflite::xnnpack::WeightChunkPrefetcher::PrefetchMode prefetch_mode) override;
        void Finalize() override;

        // Optional: write model info into metadata (e.g., model path)
        void WriteModelInfo(const std::string &model_path) { model_path_ = model_path; }

    private:
        std::string output_path_;
        nlohmann::ordered_json json_root_; // Root JSON object
        bool finalized_;
        size_t max_aligned_size_;
        std::map<std::string, size_t> per_mode_counts_;
        std::map<std::string, size_t> per_mode_total_aligned_size_;
        std::string model_path_;
    };

} // namespace flash_slim

#endif // FLASH_SLIM_PREFETCH_PLANNER_UTIL_H
