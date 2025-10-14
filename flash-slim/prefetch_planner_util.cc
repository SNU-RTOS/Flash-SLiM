// Copyright 2025 Flash-SLiM Research Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
// Prefetch Planner for Weight Streaming
// This module provides tools to record and plan I/O prefetch operations
// for weight chunks during model execution.

#include "prefetch_planner_util.h"
#include <iostream>

namespace flash_slim
{

    JsonWeightChunkInfoHandler::JsonWeightChunkInfoHandler(const std::string &output_path)
        : output_path_(output_path),
          json_root_(nlohmann::ordered_json::object()),
          finalized_(false),
          max_aligned_size_(0)
    {
        // Initialize containers
        json_root_["weight_chunks"] = nlohmann::ordered_json::object();
        std::cout << "[JsonWeightChunkInfoWriter] Initialized: " << output_path_ << std::endl;
    }

    JsonWeightChunkInfoHandler::~JsonWeightChunkInfoHandler()
    {
        if (!finalized_)
        {
            std::cerr << "[JsonWeightChunkInfoWriter] Warning: Destroyed without calling Finalize()" << std::endl;
            Finalize();
        }
    }

    static inline const char *PrefetchModeToString(tflite::xnnpack::WeightChunkPrefetcher::PrefetchMode mode)
    {
        using Mode = tflite::xnnpack::WeightChunkPrefetcher::PrefetchMode;
        switch (mode)
        {
        case Mode::PREFILL:
            return "PREFILL";
        case Mode::DECODE:
            return "DECODE";
        default:
            return "UNKNOWN";
        }
    }

    void JsonWeightChunkInfoHandler::WriteChunkInfo(const tflite::xnnpack::StreamingWeightCacheProvider::weight_chunk_info_t &chunk_info,
                                                    tflite::xnnpack::WeightChunkPrefetcher::PrefetchMode prefetch_mode)
    {
        if (finalized_)
        {
            std::cerr << "[JsonWeightChunkInfoWriter] Error: Cannot write after Finalize()" << std::endl;
            return;
        }

        // std::cout << "[JsonWeightChunkInfoWriter] Writing chunk_index=" << chunk_info.chunk_index
        //           << " aligned_offset=" << chunk_info.aligned_offset << " aligned_size=" << chunk_info.aligned_size
        //           << " buffer_index=" << chunk_info.managed_buffer_index
        //           << " origin_offset=" << chunk_info.origin_offset
        //           << " weights_id=" << chunk_info.weights_id << std::endl;

        nlohmann::ordered_json chunk_json;
        chunk_json["chunk_index"] = chunk_info.chunk_index;
        chunk_json["aligned_offset"] = chunk_info.aligned_offset;
        chunk_json["offset_adjust"] = chunk_info.offset_adjust;
        chunk_json["aligned_size"] = chunk_info.aligned_size;
        chunk_json["origin_offset"] = chunk_info.origin_offset;
        chunk_json["origin_size"] = chunk_info.origin_size;
        chunk_json["managed_buffer_index"] = chunk_info.managed_buffer_index;
        chunk_json["weights_id"] = chunk_info.weights_id;
        chunk_json["prefetch_mode"] = prefetch_mode;

        // Update max aligned size
        if (chunk_info.aligned_size > max_aligned_size_)
        {
            max_aligned_size_ = chunk_info.aligned_size;
        }

        // Group by prefetch mode
        const std::string mode_key = PrefetchModeToString(prefetch_mode);
        if (!json_root_["weight_chunks"].contains(mode_key))
        {
            json_root_["weight_chunks"][mode_key] = nlohmann::ordered_json::array();
        }
        json_root_["weight_chunks"][mode_key].push_back(chunk_json);

        // Counters
        per_mode_counts_[mode_key] += 1;
        per_mode_total_aligned_size_[mode_key] += static_cast<size_t>(chunk_info.aligned_size);
    }

    void JsonWeightChunkInfoHandler::Finalize()
    {
        if (finalized_)
        {
            return;
        }

        std::ofstream output_file(output_path_);
        if (!output_file.is_open())
        {
            std::cerr << "[JsonWeightChunkInfoWriter] Error: Failed to open: " << output_path_ << std::endl;
            finalized_ = true;
            return;
        }

        // Attach metadata before writing
        nlohmann::ordered_json meta;
        meta["version"] = "1.0";
        meta["max_aligned_size"] = max_aligned_size_;
        if (!model_path_.empty())
        {
            meta["model"] = model_path_;
        }
        // Per-mode counts
        nlohmann::ordered_json mode_counts = nlohmann::ordered_json::object();
        for (const auto &kv : per_mode_counts_)
        {
            mode_counts[kv.first] = kv.second;
        }
        meta["chunk_count_by_mode"] = mode_counts;

        // Per-mode total sizes
        nlohmann::ordered_json mode_total_sizes = nlohmann::ordered_json::object();
        for (const auto &kv : per_mode_total_aligned_size_)
        {
            mode_total_sizes[kv.first] = kv.second;
        }
        meta["total_aligned_size_by_mode"] = mode_total_sizes;

        // Build final root with metadata first, then weight_chunks (ordered_json preserves insertion order)
        nlohmann::ordered_json final_root = nlohmann::ordered_json::object();
        final_root["metadata"] = meta;
        if (json_root_.contains("weight_chunks"))
        {
            final_root["weight_chunks"] = std::move(json_root_["weight_chunks"]);
        }
        else
        {
            final_root["weight_chunks"] = nlohmann::ordered_json::object();
        }

        // Write root object with pretty formatting (indent=2)
        output_file << final_root.dump(2) << std::endl;
        output_file.close();

        std::cout << "\n\n"
                  << "[JsonWeightChunkInfoWriter] Wrote chunk metadata to: " << output_path_ << std::endl;

        finalized_ = true;
    }

} // namespace flash_slim
