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

    //* ==================== JsonWeightChunkInfoWriter ==================== */

    JsonWeightChunkInfoWriter::JsonWeightChunkInfoWriter(const std::string &output_path)
        : output_path_(output_path),
          json_root_(nlohmann::ordered_json::object()),
          finalized_(false),
          max_aligned_size_(0)
    {
        // Initialize containers
        json_root_["weight_chunks"] = nlohmann::ordered_json::object();
        std::cout << "[JsonWeightChunkInfoWriter] Initialized: " << output_path_ << std::endl;
    }

    JsonWeightChunkInfoWriter::~JsonWeightChunkInfoWriter()
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

    void JsonWeightChunkInfoWriter::WriteChunkInfo(const tflite::xnnpack::StreamingWeightCacheProvider::weight_chunk_info_t &chunk_info,
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

    void JsonWeightChunkInfoWriter::Finalize()
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

namespace flash_slim
{
    //* ==================== JsonPrefetchPlanLoader ==================== */

    void JsonPrefetchPlanLoader::Clear()
    {
        root_.clear();
        version_.clear();
        model_path_.clear();
        max_aligned_size_ = 0;
        count_by_mode_.clear();
        size_by_mode_.clear();
        groups_.clear();
    }

    std::vector<std::string> JsonPrefetchPlanLoader::KeysOf(const nlohmann::ordered_json &obj)
    {
        std::vector<std::string> keys;
        if (!obj.is_object())
            return keys;
        keys.reserve(obj.size());
        for (auto it = obj.begin(); it != obj.end(); ++it)
        {
            keys.emplace_back(it.key());
        }
        return keys;
    }

    bool JsonPrefetchPlanLoader::LoadFromFile(const std::string &path)
    {
        Clear();

        std::ifstream ifs(path);
        if (!ifs.is_open())
        {
            std::cerr << "[JsonPrefetchPlanLoader] failed to open file: " << path << std::endl;
            return false;
        }

        try
        {
            // ordered_json으로 읽어도 파싱 가능
            ifs >> root_;
        }
        catch (const std::exception &e)
        {
            std::cerr << "[JsonPrefetchPlanLoader] json parse error: " << e.what() << std::endl;
            return false;
        }

        // 기본 구조 검사
        if (!root_.is_object() || !root_.contains("metadata") || !root_.contains("weight_chunks"))
        {
            std::cerr << "[JsonPrefetchPlanLoader] invalid schema: expect { metadata, weight_chunks }" << std::endl;
            return false;
        }

        const auto &meta = root_.at("metadata");
        if (meta.contains("version"))
            version_ = meta.at("version").get<std::string>();
        if (meta.contains("model"))
            model_path_ = meta.at("model").get<std::string>();
        if (meta.contains("max_aligned_size"))
        {
            // 다양한 정수형을 수용
            max_aligned_size_ = meta.at("max_aligned_size").get<uint64_t>();
        }

        // 모드별 카운트/사이즈(옵션)
        if (meta.contains("chunk_count_by_mode") && meta.at("chunk_count_by_mode").is_object())
        {
            for (auto it = meta.at("chunk_count_by_mode").begin(); it != meta.at("chunk_count_by_mode").end(); ++it)
            {
                count_by_mode_[it.key()] = it.value().get<size_t>();
            }
        }
        if (meta.contains("total_aligned_size_by_mode") && meta.at("total_aligned_size_by_mode").is_object())
        {
            for (auto it = meta.at("total_aligned_size_by_mode").begin(); it != meta.at("total_aligned_size_by_mode").end(); ++it)
            {
                size_by_mode_[it.key()] = it.value().get<uint64_t>();
            }
        }

        const auto &groups = root_.at("weight_chunks");
        if (!groups.is_object())
        {
            std::cerr << "[JsonPrefetchPlanLoader] invalid schema: weight_chunks must be an object" << std::endl;
            return false;
        }

        // 각 모드 그룹 파싱
        for (auto git = groups.begin(); git != groups.end(); ++git)
        {
            const std::string mode = git.key();
            const auto &arr = git.value();
            if (!arr.is_array())
                continue;

            auto &vec = groups_[mode];
            vec.reserve(arr.size());
            for (const auto &j : arr)
            {
                Chunk c{}; // alias of weight_chunk_info_t
                if (j.contains("chunk_index"))
                    c.chunk_index = static_cast<size_t>(j.at("chunk_index").get<uint64_t>());
                if (j.contains("aligned_offset"))
                    c.aligned_offset = static_cast<size_t>(j.at("aligned_offset").get<uint64_t>());
                if (j.contains("aligned_size"))
                    c.aligned_size = static_cast<size_t>(j.at("aligned_size").get<uint64_t>());
                if (j.contains("offset_adjust"))
                    c.offset_adjust = static_cast<size_t>(j.at("offset_adjust").get<uint64_t>());
                if (j.contains("origin_offset"))
                    c.origin_offset = static_cast<size_t>(j.at("origin_offset").get<uint64_t>());
                if (j.contains("origin_size"))
                    c.origin_size = static_cast<size_t>(j.at("origin_size").get<uint64_t>());
                if (j.contains("managed_buffer_index"))
                    c.managed_buffer_index = j.at("managed_buffer_index").get<int>();
                if (j.contains("weights_id"))
                    c.weights_id = static_cast<size_t>(j.at("weights_id").get<uint64_t>());
                // Note: prefetch_mode field (if present) is ignored; grouping key is authoritative
                vec.emplace_back(std::move(c));
            }
        }

        return true;
    }

    std::vector<std::string> JsonPrefetchPlanLoader::modes() const
    {
        std::vector<std::string> k = KeysOf(root_.at("weight_chunks"));
        return k;
    }

    const std::vector<JsonPrefetchPlanLoader::Chunk> &
    JsonPrefetchPlanLoader::chunks(const std::string &mode) const
    {
        static const std::vector<Chunk> kEmpty;
        auto it = groups_.find(mode);
        if (it == groups_.end())
            return kEmpty;
        return it->second;
    }

    std::unordered_map<size_t, JsonPrefetchPlanLoader::Chunk>
    JsonPrefetchPlanLoader::BuildOffsetToWeightChunkInfo() const
    {
        std::unordered_map<size_t, Chunk> plan;
        for (const auto& kv : groups_) {
            const auto& vec = kv.second;
            for (const auto& c : vec) {
                const size_t key = c.origin_offset;
                auto [it, inserted] = plan.emplace(key, c);
                if (!inserted) {
                    // 중복 오프셋이 발견되면 마지막 값을 유지하고 경고만 출력
                    std::cerr << "[JsonPrefetchPlanLoader] duplicate origin_offset=" << key
                              << " (mode=" << kv.first << ")\n";
                    it->second = c;
                }
            }
        }
        return plan;
    }

    std::unordered_map<size_t, JsonPrefetchPlanLoader::Chunk>
    JsonPrefetchPlanLoader::BuildOffsetToWeightChunkInfoForMode(const std::string& mode) const
    {
        std::unordered_map<size_t, Chunk> plan;
        auto it = groups_.find(mode);
        if (it == groups_.end()) return plan;
        for (const auto& c : it->second) {
            const size_t key = c.origin_offset;
            auto [pit, inserted] = plan.emplace(key, c);
            if (!inserted) {
                std::cerr << "[JsonPrefetchPlanLoader] duplicate origin_offset=" << key
                          << " in mode=" << mode << "\n";
                pit->second = c;
            }
        }
        return plan;
    }

} // namespace flash_slim
