#ifndef FLASH_SLIM_WEIGHT_CHUNK_CONTROLLER_UTILS_H_
#define FLASH_SLIM_WEIGHT_CHUNK_CONTROLLER_UTILS_H_

#include <string>
#include <fstream>
#include <iosfwd>
#include <memory>
#include <map>
#include <unordered_map>
#include "nlohmann/json.hpp"
#include "tflite/delegates/xnnpack/streaming_weight_cache.h"

#include "weight_chunk_prefetcher.h"

namespace flash_slim
{
    namespace streaming
    {

        //* ==================== Interfaces ==================== */
        class WeightChunkMetaDataWriter
        {
        public:
            virtual ~WeightChunkMetaDataWriter() = default;
            virtual void WriteChunkInfo(
                const weight_chunk_info_t &chunk_info,
                PrefetchMode prefetch_mode) = 0;
            virtual void Finalize() = 0;
        };

        class PrefetchPlanLoader
        {
        public:
            virtual ~PrefetchPlanLoader() = default;
            virtual bool LoadFromFile(const std::string &plan_file_path) = 0;
        };

        //* ==================== JsonWeightChunkInfoWriter ==================== */

        // JSON-based implementation of WeightChunkMetaDataWriter
        // Uses nlohmann/json for serialization
        class JsonWeightChunkMetaDataWriter : public flash_slim::streaming::WeightChunkMetaDataWriter
        {
        public:
            explicit JsonWeightChunkMetaDataWriter(const std::string &output_path);
            ~JsonWeightChunkMetaDataWriter() override;

            // Delete copy constructor and assignment operator
            JsonWeightChunkMetaDataWriter(const JsonWeightChunkMetaDataWriter &) = delete;
            JsonWeightChunkMetaDataWriter &operator=(const JsonWeightChunkMetaDataWriter &) = delete;

            void WriteChunkInfo(
                const weight_chunk_info_t &chunk_info,
                flash_slim::streaming::WeightChunkPrefetcher::PrefetchMode prefetch_mode) override;

            void Finalize() override;

            // Optional: write model info into metadata (e.g., model path)
            void WriteModelInfo(const std::string &model_path) { model_path_ = model_path; }

            size_t GetMaxAlignedSize() const { return max_aligned_size_; }

        private:
            std::string output_path_;
            nlohmann::ordered_json json_root_; // Root JSON object
            bool finalized_;
            size_t max_aligned_size_;
            std::map<std::string, size_t> per_mode_counts_;
            std::map<std::string, size_t> per_mode_total_aligned_size_;
            std::string model_path_;
        };

        //* ==================== JsonPrefetchPlanLoader ==================== */
        class JsonPrefetchPlanLoader : public flash_slim::streaming::PrefetchPlanLoader
        {
        public:
            JsonPrefetchPlanLoader();
            ~JsonPrefetchPlanLoader() override;

            // JSON 파일 로드(성공 시 true)
            bool LoadFromFile(const std::string &path) override;

            // 메타데이터 접근자
            const std::string &version() const { return version_; }
            const std::string &model() const { return model_path_; }
            uint64_t max_aligned_size() const { return max_aligned_size_; }

            // 모드 목록("PREFILL", "DECODE", ...)
            std::vector<std::string> modes() const;

            // 모드별 chunk 벡터(모드가 없으면 빈 벡터 반환)
            const std::vector<weight_chunk_info_t> &chunks(const std::string &mode) const;

            // 편의 접근자: PREFILL / DECODE 전용 벡터
            const std::vector<weight_chunk_info_t> &prefill_chunks() const { return prefill_chunks_; }
            const std::vector<weight_chunk_info_t> &decode_chunks() const { return decode_chunks_; }

            // 메타데이터 출력
            void PrintMetadata(std::ostream &os) const;
            // 표준 출력으로 메타데이터 출력 (구현은 .cc에서 std::cout 사용)
            void PrintMetadata() const;

            // 특정 모드: origin_offset -> index 맵(사본) 반환
            std::unordered_map<size_t, size_t> BuildOffsetToIndexForMode(const std::string &mode) const;

            // 특정 모드: index -> weight_chunk_info_t 벡터(사본) 반환
            std::vector<weight_chunk_info_t> BuildIndexToChunkVectorForMode(const std::string &mode) const;

            // 모드별 개수/총 aligned_size
            const std::map<std::string, size_t> &chunk_count_by_mode() const { return count_by_mode_; }
            const std::map<std::string, uint64_t> &total_aligned_size_by_mode() const { return size_by_mode_; }

            // 원본 JSON에 접근이 필요하면 제공
            const nlohmann::ordered_json &raw_json() const { return root_; }

        private:
            void Clear();

            std::string version_;
            nlohmann::ordered_json root_;
            std::string model_path_;
            uint64_t max_aligned_size_ = 0;
            std::map<std::string, size_t> count_by_mode_;
            std::map<std::string, uint64_t> size_by_mode_;
            std::map<std::string, std::vector<weight_chunk_info_t>> groups_;
            std::vector<weight_chunk_info_t> prefill_chunks_;
            std::vector<weight_chunk_info_t> decode_chunks_;
            static std::vector<std::string> KeysOf(const nlohmann::ordered_json &obj);
        };

    } // namespace streaming
} // namespace flash_slim

#endif // FLASH_SLIM_WEIGHT_CHUNK_CONTROLLER_UTILS_H_
