// Copyright 2025 Flash-SLiM Research Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
// Prefetch Planner for Weight Streaming
// This module provides tools to record and plan I/O prefetch operations
// for weight chunks during model execution.

#include "weight_chunk_controller_utils.h"
#include <algorithm>
#include <iostream>

namespace flash_slim
{
    namespace streaming
    {
        //* ==================== JsonWeightChunkMetaDataWriter ==================== */

        JsonWeightChunkMetaDataWriter::JsonWeightChunkMetaDataWriter(const std::string &output_path)
            : output_path_(output_path),
              json_root_(nlohmann::ordered_json::object()),
              finalized_(false),
              max_aligned_size_(0),
              weight_chunk_buffer_size_(0)
        {
            // Initialize containers
            json_root_["weight_chunks"] = nlohmann::ordered_json::object();
            std::cout << "[JsonWeightChunkMetaDataWriter] Initialized: " << output_path_ << std::endl;
        }

        JsonWeightChunkMetaDataWriter::~JsonWeightChunkMetaDataWriter()
        {
            if (!finalized_)
            {
                std::cerr << "[JsonWeightChunkMetaDataWriter] Warning: Destroyed without calling Finalize()" << std::endl;
                Finalize();
            }
        }

        static inline const char *PrefetchModeToString(
            PrefetchMode mode)
        {
            using Mode = PrefetchMode;
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

        void JsonWeightChunkMetaDataWriter::WriteChunkInfo(
            const weight_chunk_info_t &chunk_info,
            PrefetchMode prefetch_mode)
        {
            if (finalized_)
            {
                std::cerr << "[JsonWeightChunkMetaDataWriter] Error: Cannot write after Finalize()" << std::endl;
                return;
            }

            // std::cout << "[JsonWeightChunkMetaDataWriter] Writing chunk_index=" << chunk_info.chunk_index
            //           << " aligned_offset=" << chunk_info.aligned_offset << " aligned_size=" << chunk_info.aligned_size
            //           << " buffer_index=" << chunk_info.managed_buffer_index
            //           << " origin_offset=" << chunk_info.origin_offset
            //           << " weights_id=" << chunk_info.weights_id << std::endl;

            nlohmann::ordered_json chunk_json;
            chunk_json["chunk_index"] = chunk_info.chunk_index;
            chunk_json["origin_offset"] = chunk_info.origin_offset;
            chunk_json["origin_size"] = chunk_info.origin_size;
            chunk_json["aligned_offset"] = chunk_info.aligned_offset;
            chunk_json["aligned_size"] = chunk_info.aligned_size;
            chunk_json["offset_adjust"] = chunk_info.offset_adjust;
            chunk_json["weights_id"] = chunk_info.weights_id;
            chunk_json["prefetch_mode"] = static_cast<int>(prefetch_mode);
            chunk_json["prefetch_mode_str"] = PrefetchModeToString(prefetch_mode);

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

            weight_chunk_buffer_size_ =
                std::max(weight_chunk_buffer_size_,
                         static_cast<size_t>(chunk_info.aligned_size));
            auto &last_aligned = last_chunk_aligned_size_[mode_key];
            if (last_aligned > 0)
            {
                weight_chunk_buffer_size_ =
                    std::max(weight_chunk_buffer_size_, last_aligned + chunk_info.aligned_size);
            }
            last_aligned = chunk_info.aligned_size;

            // Counters
            per_mode_counts_[mode_key] += 1;
            per_mode_total_aligned_size_[mode_key] += static_cast<size_t>(chunk_info.aligned_size);
        }

        void JsonWeightChunkMetaDataWriter::Finalize()
        {
            if (finalized_)
            {
                return;
            }

            std::ofstream output_file(output_path_);
            if (!output_file.is_open())
            {
                std::cerr << "[JsonWeightChunkMetaDataWriter] Error: Failed to open: " << output_path_ << std::endl;
                finalized_ = true;
                return;
            }

            // Attach metadata before writing
            nlohmann::ordered_json meta;
            meta["version"] = "1.0";
            meta["max_aligned_size"] = max_aligned_size_;
            meta["weight_chunk_buffer_size"] = weight_chunk_buffer_size_;
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
                      << "[JsonWeightChunkMetaDataWriter] Wrote chunk metadata to: " << output_path_ << std::endl;

            finalized_ = true;
        }

        //* ==================== JsonPrefetchPlanLoader ==================== */

        JsonPrefetchPlanLoader::JsonPrefetchPlanLoader()
            : root_(nlohmann::ordered_json::object()),
              version_(),
              model_path_(),
              max_aligned_size_(0),
              count_by_mode_(),
              size_by_mode_(),
              groups_(),
              prefill_chunks_(),
              decode_chunks_()
        {
            std::cout << "[JsonPrefetchPlanLoader] Initialized" << std::endl;
        }
        JsonPrefetchPlanLoader::~JsonPrefetchPlanLoader()
        {
            Clear();
            std::cout << "[JsonPrefetchPlanLoader] Destroyed" << std::endl;
        }

        void JsonPrefetchPlanLoader::Clear()
        {
            root_.clear();
            version_.clear();
            model_path_.clear();
            max_aligned_size_ = 0;
            weight_chunk_buffer_size_ = 0;
            count_by_mode_.clear();
            size_by_mode_.clear();
            groups_.clear();
            prefill_chunks_.clear();
            decode_chunks_.clear();
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
            if (meta.contains("weight_chunk_buffer_size"))
            {
                weight_chunk_buffer_size_ = meta.at("weight_chunk_buffer_size").get<uint64_t>();
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
                    weight_chunk_info_t c{};
                    if (j.contains("chunk_index"))
                        c.chunk_index = static_cast<size_t>(j.at("chunk_index").get<uint64_t>());
                    if (j.contains("origin_offset"))
                        c.origin_offset = static_cast<size_t>(j.at("origin_offset").get<uint64_t>());
                    if (j.contains("origin_size"))
                        c.origin_size = static_cast<size_t>(j.at("origin_size").get<uint64_t>());
                    if (j.contains("aligned_offset"))
                        c.aligned_offset = static_cast<size_t>(j.at("aligned_offset").get<uint64_t>());
                    if (j.contains("aligned_size"))
                        c.aligned_size = static_cast<size_t>(j.at("aligned_size").get<uint64_t>());
                    if (j.contains("offset_adjust"))
                        c.offset_adjust = static_cast<size_t>(j.at("offset_adjust").get<uint64_t>());
                    if (j.contains("weights_id"))
                        c.weights_id = static_cast<size_t>(j.at("weights_id").get<uint64_t>());
                    // Note: prefetch_mode field (if present) is ignored; grouping key is authoritative
                    vec.emplace_back(c);
                    // 전용 벡터에도 저장
                    if (mode == "PREFILL")
                    {
                        prefill_chunks_.emplace_back(c);
                    }
                    else if (mode == "DECODE")
                    {
                        decode_chunks_.emplace_back(c);
                    }
                }
            }

            // If metadata did not provide weight_chunk_buffer_size, derive it from the chunks.
            if (weight_chunk_buffer_size_ == 0)
            {
                for (const auto &kv : groups_)
                {
                    const auto &vec = kv.second;
                    if (vec.empty())
                    {
                        continue;
                    }
                    // Track the largest aligned size (single chunk) to cover degenerate cases.
                    size_t mode_max = 0;
                    for (const auto &chunk : vec)
                    {
                        mode_max = std::max(mode_max, chunk.aligned_size);
                    }
                    size_t mode_pair_max = mode_max;
                    for (size_t i = 0; i + 1 < vec.size(); ++i)
                    {
                        const size_t pair_sum = vec[i].aligned_size + vec[i + 1].aligned_size;
                        mode_pair_max = std::max(mode_pair_max, pair_sum);
                    }
                    weight_chunk_buffer_size_ = std::max<uint64_t>(weight_chunk_buffer_size_, mode_pair_max);
                }
            }
            else
            {
                // Even if the metadata includes the value, confirm it is at least as large as any observed pair.
                uint64_t derived_max = 0;
                for (const auto &kv : groups_)
                {
                    const auto &vec = kv.second;
                    if (vec.empty())
                    {
                        continue;
                    }
                    size_t mode_max = 0;
                    for (const auto &chunk : vec)
                    {
                        mode_max = std::max(mode_max, chunk.aligned_size);
                    }
                    size_t mode_pair_max = mode_max;
                    for (size_t i = 0; i + 1 < vec.size(); ++i)
                    {
                        const size_t pair_sum = vec[i].aligned_size + vec[i + 1].aligned_size;
                        mode_pair_max = std::max(mode_pair_max, pair_sum);
                    }
                    derived_max = std::max<uint64_t>(derived_max, mode_pair_max);
                }
                if (derived_max > weight_chunk_buffer_size_)
                {
                    std::cerr << "[JsonPrefetchPlanLoader] Warning: weight_chunk_buffer_size from metadata ("
                              << weight_chunk_buffer_size_
                              << ") is smaller than derived maximum pair sum (" << derived_max
                              << "). Using derived value.\n";
                    weight_chunk_buffer_size_ = derived_max;
                }
            }

            return true;
        }

        std::vector<std::string> JsonPrefetchPlanLoader::modes() const
        {
            std::vector<std::string> k = KeysOf(root_.at("weight_chunks"));
            return k;
        }

        const std::vector<weight_chunk_info_t> &
        JsonPrefetchPlanLoader::chunks(const std::string &mode) const
        {
            static const std::vector<weight_chunk_info_t> kEmpty;
            auto it = groups_.find(mode);
            if (it == groups_.end())
                return kEmpty;
            return it->second;
        }

        void JsonPrefetchPlanLoader::PrintMetadata(std::ostream &os) const
        {
            os << "[INFO] Prefetch Plan - Metadata" << std::endl;
            os << "  version: " << version_ << std::endl;
            if (!model_path_.empty())
                os << "  model: " << model_path_ << std::endl;
            os << "  max_aligned_size: " << max_aligned_size_ << std::endl;
            os << "  weight_chunk_buffer_size: " << weight_chunk_buffer_size_ << std::endl;

            os << "  chunk_count_by_mode:" << std::endl;
            for (const auto &kv : count_by_mode_)
            {
                os << "    - " << kv.first << ": " << kv.second << std::endl;
            }
            os << "  total_aligned_size_by_mode:" << std::endl;
            for (const auto &kv : size_by_mode_)
            {
                os << "    - " << kv.first << ": " << kv.second << std::endl;
            }
        }

        void JsonPrefetchPlanLoader::PrintMetadata() const
        {
            PrintMetadata(std::cout);
        }

        std::unordered_map<size_t, size_t>
        JsonPrefetchPlanLoader::BuildOffsetToIndexForMode(const std::string &mode) const
        {
            std::unordered_map<size_t, size_t> map;
            auto it = groups_.find(mode);
            if (it == groups_.end())
                return map;
            const auto &vec = it->second;
            map.reserve(vec.size());
            for (size_t i = 0; i < vec.size(); ++i)
            {
                const size_t key = vec[i].origin_offset;
                // first occurrence wins; if duplicates exist, keep first index
                map.emplace(key, i);
            }
            return map;
        }

        std::vector<weight_chunk_info_t>
        JsonPrefetchPlanLoader::BuildIndexToChunkVectorForMode(const std::string &mode) const
        {
            std::vector<weight_chunk_info_t> out;
            auto it = groups_.find(mode);
            if (it == groups_.end())
                return out;
            const auto &vec = it->second;
            out = vec; // 사본 반환
            return out;
        }
    } // namespace streaming
} // namespace flash_slim
