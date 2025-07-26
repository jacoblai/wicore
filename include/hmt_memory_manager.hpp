// hmt_memory_manager.hpp
#ifndef HMT_MEMORY_MANAGER_HPP
#define HMT_MEMORY_MANAGER_HPP

#include <memory>
#include <unordered_map>
#include <string>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <queue>
#include <chrono>
#include <fstream>

#include <cuda_runtime.h>

namespace wicore {

// 内存层级枚举
enum class MemoryTier {
    GPU,      // GPU显存 - 热数据
    CPU,      // CPU内存 - 温数据  
    STORAGE   // NVMe存储 - 冷数据
};

// 固定块大小
static constexpr size_t BLOCK_SIZE = 4096; // 4KB

// 内存块结构
struct MemoryBlock {
    // 基本信息
    uint64_t block_id;
    MemoryTier current_tier;
    size_t size = BLOCK_SIZE;
    
    // 存储指针
    void* gpu_ptr = nullptr;
    void* cpu_ptr = nullptr;
    std::string storage_path;
    
    // 访问统计
    std::atomic<uint64_t> access_count{0};
    std::atomic<double> attention_score{1.0};
    std::chrono::steady_clock::time_point last_access;
    std::chrono::steady_clock::time_point creation_time;
    
    // 状态标记
    std::atomic<bool> is_dirty{false};
    std::atomic<bool> is_migrating{false};
    std::atomic<bool> is_pinned{false};  // 防止被置换
    
    // 构造函数
    MemoryBlock(uint64_t id) : block_id(id) {
        auto now = std::chrono::steady_clock::now();
        last_access = now;
        creation_time = now;
    }
};

// A²CR策略参数
struct A2CRParams {
    std::atomic<double> time_decay_factor{0.05};
    std::atomic<double> attention_weight{0.4};
    std::atomic<double> frequency_weight{0.3};
    std::atomic<double> recency_weight{0.3};
    std::atomic<double> gpu_pressure_threshold{0.85};
    std::atomic<double> cpu_pressure_threshold{0.90};
};

// 内存统计信息
struct MemoryStats {
    // GPU层统计
    std::atomic<size_t> gpu_total_blocks{0};
    std::atomic<size_t> gpu_used_blocks{0};
    std::atomic<size_t> gpu_free_blocks{0};
    std::atomic<uint64_t> gpu_hit_count{0};
    std::atomic<uint64_t> gpu_miss_count{0};
    
    // CPU层统计
    std::atomic<size_t> cpu_total_blocks{0};
    std::atomic<size_t> cpu_used_blocks{0};
    std::atomic<size_t> cpu_free_blocks{0};
    std::atomic<uint64_t> cpu_hit_count{0};
    std::atomic<uint64_t> cpu_miss_count{0};
    
    // 存储层统计
    std::atomic<size_t> storage_total_blocks{0};
    std::atomic<size_t> storage_used_blocks{0};
    std::atomic<uint64_t> storage_hit_count{0};
    std::atomic<uint64_t> storage_miss_count{0};
    
    // 迁移统计
    std::atomic<uint64_t> gpu_to_cpu_migrations{0};
    std::atomic<uint64_t> cpu_to_gpu_migrations{0};
    std::atomic<uint64_t> cpu_to_storage_migrations{0};
    std::atomic<uint64_t> storage_to_cpu_migrations{0};
    
    // 性能统计
    std::atomic<double> avg_gpu_to_cpu_latency_ms{0.0};
    std::atomic<double> avg_cpu_to_gpu_latency_ms{0.0};
    std::atomic<double> avg_storage_access_latency_ms{0.0};
};

// HMT分层内存管理器
class HMTMemoryManager {
public:
    explicit HMTMemoryManager(size_t gpu_memory_limit_gb,
                             size_t cpu_memory_limit_gb,
                             const std::string& storage_base_path);
    ~HMTMemoryManager();
    
    // 核心接口
    bool initialize();
    void shutdown();
    
    // 内存分配接口
    MemoryBlock* allocate_block(MemoryTier preferred_tier = MemoryTier::GPU);
    void deallocate_block(MemoryBlock* block);
    
    // 访问跟踪
    void record_access(MemoryBlock* block, double attention_score = 1.0);
    void record_access(void* ptr, double attention_score = 1.0);
    
    // 智能迁移接口
    bool promote_to_gpu(MemoryBlock* block);
    bool demote_to_cpu(MemoryBlock* block);
    bool archive_to_storage(MemoryBlock* block);
    bool restore_from_storage(MemoryBlock* block);
    
    // 批量操作
    std::vector<MemoryBlock*> allocate_blocks(size_t count, MemoryTier preferred_tier = MemoryTier::GPU);
    void deallocate_blocks(const std::vector<MemoryBlock*>& blocks);
    
    // 配置管理
    void update_a2cr_params(const A2CRParams& params);
    A2CRParams get_a2cr_params() const;
    
    // 状态查询
    MemoryStats get_memory_stats() const;
    MemoryBlock* find_block(void* ptr) const;
    double get_memory_pressure(MemoryTier tier) const;
    
    // 手动触发优化
    void trigger_gc();
    void trigger_defragmentation();

private:
    // 配置参数
    size_t gpu_memory_limit_;
    size_t cpu_memory_limit_;
    std::string storage_base_path_;
    
    // A²CR策略参数
    A2CRParams a2cr_params_;
    
    // 内存池管理
    struct MemoryPool {
        std::queue<MemoryBlock*> free_blocks;
        std::unordered_map<uint64_t, std::unique_ptr<MemoryBlock>> allocated_blocks;
        std::unordered_map<void*, MemoryBlock*> ptr_to_block;
        mutable std::shared_mutex mutex;
        size_t total_capacity;
        std::atomic<size_t> used_count{0};
    };
    
    MemoryPool gpu_pool_;
    MemoryPool cpu_pool_;
    MemoryPool storage_pool_;
    
    // 统计信息
    mutable MemoryStats stats_;
    
    // 后台线程
    std::thread migration_thread_;
    std::thread gc_thread_;
    std::atomic<bool> shutdown_requested_{false};
    
    // 同步控制
    mutable std::mutex migration_queue_mutex_;
    std::queue<MemoryBlock*> migration_queue_;
    std::condition_variable migration_cv_;
    
    // 内部方法
    bool initialize_gpu_pool();
    bool initialize_cpu_pool();
    bool initialize_storage_pool();
    
    // GPU内存管理
    MemoryBlock* allocate_gpu_block();
    void deallocate_gpu_block(MemoryBlock* block);
    bool evict_gpu_victim();
    
    // CPU内存管理  
    MemoryBlock* allocate_cpu_block();
    void deallocate_cpu_block(MemoryBlock* block);
    bool evict_cpu_victim();
    
    // 存储管理
    MemoryBlock* allocate_storage_block();
    void deallocate_storage_block(MemoryBlock* block);
    bool write_to_storage(MemoryBlock* block);
    bool read_from_storage(MemoryBlock* block);
    std::string generate_storage_path(uint64_t block_id) const;
    
    // A²CR置换算法
    MemoryBlock* select_gpu_victim() const;
    MemoryBlock* select_cpu_victim() const;
    double calculate_victim_score(const MemoryBlock* block) const;
    
    // 迁移引擎
    void migration_worker();
    bool execute_migration(MemoryBlock* block, MemoryTier target_tier);
    void enqueue_migration(MemoryBlock* block, MemoryTier target_tier);
    
    // 垃圾回收
    void gc_worker();
    void cleanup_expired_blocks();
    void defragment_memory();
    
    // 统计更新
    void update_stats();
    void record_migration_latency(MemoryTier from, MemoryTier to, double latency_ms);
    
    // 工具方法
    uint64_t generate_block_id();
    bool is_memory_pressure_high(MemoryTier tier) const;
    std::atomic<uint64_t> next_block_id_{1};
    
    // 禁用拷贝
    HMTMemoryManager(const HMTMemoryManager&) = delete;
    HMTMemoryManager& operator=(const HMTMemoryManager&) = delete;
};

} // namespace wicore

#endif // HMT_MEMORY_MANAGER_HPP 