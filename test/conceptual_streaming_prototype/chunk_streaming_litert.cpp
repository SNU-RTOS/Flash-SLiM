// chunk_streaming_litert.cpp (dummy-prefetch thread)
// ------------------------------------------------------------
// Purpose: Simulates the core logic of a prefetch-aware LiteRT runtime.
// This PoC demonstrates how a Chunk Metadata Table (CMT) can guide
// a double-buffered, asynchronous prefetching system.
// It uses a dummy prefetch thread that only logs I/O actions
// instead of performing actual disk reads.
// ------------------------------------------------------------
#include <fstream>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <cassert>
#include <chrono>
#include <stdexcept>

// Assuming use of a lightweight JSON library like jsoncpp
#include <json/json.h>

// ------------------------------------------------------------
// 1. Chunk Metadata (CMT)
// ------------------------------------------------------------
struct ChunkInfo {
    int     chunk_id;
    int     start_idx;
    int     end_idx;
    size_t  weight_size;
};

class ChunkMetadata {
public:
    bool Load(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) {
            std::cerr << "Error: Failed to open CMT file at " << path << "\n";
            return false;
        }
        Json::Value root;
        f >> root;
        for (const auto& j : root) {
            ChunkInfo info{j["chunk_id"].asInt(), j["start_idx"].asInt(), j["end_idx"].asInt(), j["weight_size"].asLargestInt()};
            chunks_.push_back(info);
            for (int i = info.start_idx; i <= info.end_idx; ++i) {
                op2chunk_[i] = info.chunk_id;
            }
        }
        std::cout << "[CMT] Loaded " << chunks_.size() << " chunks from " << path << "\n";
        return true;
    }

    int GetChunkIdForOp(int op_index) const {
        auto it = op2chunk_.find(op_index);
        return it == op2chunk_.end() ? -1 : it->second;
    }

    const ChunkInfo* GetChunkInfo(int chunk_id) const {
        if (chunk_id < 0 || chunk_id >= chunks_.size()) return nullptr;
        return &chunks_[chunk_id];
    }

    int GetTotalChunks() const { return static_cast<int>(chunks_.size()); }

private:
    std::vector<ChunkInfo> chunks_;
    std::unordered_map<int, int> op2chunk_;
};

// ------------------------------------------------------------
// 2. BufferManager (Double Buffer)
// ------------------------------------------------------------
class BufferManager {
public:
    enum class SlotState { EMPTY, PREFETCHING, READY };
    struct Slot {
        int chunk_id = -1;
        SlotState state = SlotState::EMPTY;
    };

    Slot slots_[2];
    int active_slot_ = 0;

    int GetInactiveSlot() const { return 1 - active_slot_; }

    void MarkAsReady(int slot_index, int chunk_id) {
        slots_[slot_index] = {chunk_id, SlotState::READY};
    }

    void MarkAsPrefetching(int slot_index, int chunk_id) {
        slots_[slot_index] = {chunk_id, SlotState::PREFETCHING};
    }

    void Swap() {
        active_slot_ = 1 - active_slot_;
        slots_[GetInactiveSlot()].state = SlotState::EMPTY; // Mark old slot as evictable
    }
};

// ------------------------------------------------------------
// 3. PrefetchThread (Dummy - Logs I/O actions)
// ------------------------------------------------------------
class PrefetchThread {
public:
    PrefetchThread(BufferManager& bm) : bm_(bm), stop_(false) {
        worker_ = std::thread(&PrefetchThread::Run, this);
    }

    ~PrefetchThread() {
        {
            std::unique_lock<std::mutex> lk(mu_);
            stop_ = true;
            cv_.notify_all();
        }
        if (worker_.joinable()) worker_.join();
    }

    void Request(int chunk_id) {
        std::unique_lock<std::mutex> lk(mu_);
        queue_.push(chunk_id);
        cv_.notify_all();
    }

private:
    void Run() {
        while (true) {
            int chunk_id = -1;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return stop_ || !queue_.empty(); });
                if (stop_) break;
                chunk_id = queue_.front();
                queue_.pop();
            }

            int inactive_slot = bm_.GetInactiveSlot();
            bm_.MarkAsPrefetching(inactive_slot, chunk_id);
            std::cout << "[Prefetch] Scheduling chunk " << chunk_id << " into buffer " << inactive_slot << " (dummy I/O)\n";
            
            // Simulate I/O delay
            std::this_thread::sleep_for(std::chrono::milliseconds(50)); 
            
            bm_.MarkAsReady(inactive_slot, chunk_id);
            std::cout << "[Prefetch] Chunk " << chunk_id << " is READY in buffer " << inactive_slot << "\n";
        }
    }

    BufferManager& bm_;
    std::thread worker_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::queue<int> queue_;
    bool stop_ = false;
};

// ------------------------------------------------------------
// 4. ChunkStreamer - Manages events and buffer swaps
// ------------------------------------------------------------
class ChunkStreamer {
public:
    ChunkStreamer(BufferManager& bm, const ChunkMetadata& cmt, PrefetchThread& pf)
        : bm_(bm), cmt_(cmt), pf_(pf) {}

    void OnChunkEnd(int current_chunk_id) {
        int next_chunk_id = current_chunk_id + 1;
        if (next_chunk_id < cmt_.GetTotalChunks()) {
            pf_.Request(next_chunk_id);
        }

        // Wait until the inactive buffer is ready (dummy spin-wait)
        int inactive_slot = bm_.GetInactiveSlot();
        while (bm_.slots_[inactive_slot].state != BufferManager::SlotState::READY) {
            std::this_thread::yield();
        }

        bm_.Swap();
        std::cout << "[Swap] Swapped to buffer " << bm_.active_slot_ 
                  << " (now active with chunk " << bm_.slots_[bm_.active_slot_].chunk_id << ")\n";
    }

private:
    BufferManager& bm_;
    const ChunkMetadata& cmt_;
    PrefetchThread& pf_;
};

// ------------------------------------------------------------
// 5. Dummy Operator Loop (Simulates LiteRT Interpreter)
// ------------------------------------------------------------
void RunDummyInference(const ChunkMetadata& cmt) {
    BufferManager bm;
    PrefetchThread pf(bm);

    // Initial state: Load chunk 0, prefetch chunk 1
    bm.MarkAsReady(0, 0);
    pf.Request(1);

    ChunkStreamer streamer(bm, cmt, pf);
    
    const auto* last_chunk = cmt.GetChunkInfo(cmt.GetTotalChunks() - 1);
    if (!last_chunk) {
        throw std::runtime_error("CMT is empty or invalid.");
    }
    int total_ops = last_chunk->end_idx + 1;

    for (int op_idx = 0; op_idx < total_ops; ++op_idx) {
        int required_chunk_id = cmt.GetChunkIdForOp(op_idx);
        
        // Verification
        assert(bm.slots_[bm.active_slot_].chunk_id == required_chunk_id &&
               bm.slots_[bm.active_slot_].state == BufferManager::SlotState::READY);

        std::cout << "[Exec]   Executing op " << op_idx << " (requires chunk " << required_chunk_id << " from buffer " << bm.active_slot_ << ")\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Simulate compute

        const auto* current_chunk_info = cmt.GetChunkInfo(required_chunk_id);
        if (op_idx == current_chunk_info->end_idx) {
            std::cout << "[Event]  --- End of Chunk " << required_chunk_id << " ---\n";
            if (required_chunk_id < cmt.GetTotalChunks() - 1) {
                streamer.OnChunkEnd(required_chunk_id);
            }
        }
    }
    std::cout << "[Done] Dummy inference run complete.\n";
}

int main(int argc, char** argv) {
    std::string path = "cmt.json";
    if (argc > 1) path = argv[1];
    
    ChunkMetadata cmt;
    if (!cmt.Load(path)) {
        return 1;
    }
    
    try {
        RunDummyInference(cmt);
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
