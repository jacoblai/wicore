// hmt_memory_manager.cpp
#include "../include/hmt_memory_manager.hpp"
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <cmath>

namespace wicore {

HMTMemoryManager::HMTMemoryManager(size_t gpu_memory_limit_gb,
                                 size_t cpu_memory_limit_gb,
                                 const std::string& storage_base_path)
    : gpu_memory_limit_(gpu_memory_limit_gb * 1024 * 1024 * 1024)
    , cpu_memory_limit_(cpu_memory_limit_gb * 1024 * 1024 * 1024)
    , storage_base_path_(storage_base_path) {
    
    // 计算每层的块容量
    gpu_pool_.total_capacity = gpu_memory_limit_ / BLOCK_SIZE;
    cpu_pool_.total_capacity = cpu_memory_limit_ / BLOCK_SIZE;
    storage_pool_.total_capacity = SIZE_MAX; // 存储层容量理论无限
    
    std::cout << "HMT Memory Manager created" << std::endl;
    std::cout << "GPU capacity: " << gpu_pool_.total_capacity << " blocks (" 
              << gpu_memory_limit_gb << "GB)" << std::endl;
    std::cout << "CPU capacity: " << cpu_pool_.total_capacity << " blocks (" 
              << cpu_memory_limit_gb << "GB)" << std::endl;
}

HMTMemoryManager::~HMTMemoryManager() {
    shutdown();
}

bool HMTMemoryManager::initialize() {
    try {
        // 1. 创建存储目录
        std::filesystem::create_directories(storage_base_path_);
        
        // 2. 按顺序初始化各层存储
        if (!initialize_gpu_pool()) {
            std::cerr << "Failed to initialize GPU memory pool" << std::endl;
            return false;
        }
        
        if (!initialize_cpu_pool()) {
            std::cerr << "Failed to initialize CPU memory pool" << std::endl;
            return false;
        }
        
        if (!initialize_storage_pool()) {
            std::cerr << "Failed to initialize storage pool" << std::endl;
            return false;
        }
        
        // 3. 启动后台线程
        shutdown_requested_.store(false);
        migration_thread_ = std::thread(&HMTMemoryManager::migration_worker, this);
        gc_thread_ = std::thread(&HMTMemoryManager::gc_worker, this);
        
        std::cout << "HMT Memory Manager initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during HMT initialization: " << e.what() << std::endl;
        return false;
    }
}

void HMTMemoryManager::shutdown() {
    if (shutdown_requested_.load()) {
        return;
    }
    
    std::cout << "Shutting down HMT Memory Manager..." << std::endl;
    
    // 1. 停止后台线程
    shutdown_requested_.store(true);
    migration_cv_.notify_all();
    
    if (migration_thread_.joinable()) {
        migration_thread_.join();
    }
    
    if (gc_thread_.joinable()) {
        gc_thread_.join();
    }
    
    // 2. 清理内存池
    {
        std::unique_lock<std::shared_mutex> gpu_lock(gpu_pool_.mutex);
        std::unique_lock<std::shared_mutex> cpu_lock(cpu_pool_.mutex);
        std::unique_lock<std::shared_mutex> storage_lock(storage_pool_.mutex);
        
        // 释放GPU内存
        for (auto& [id, block] : gpu_pool_.allocated_blocks) {
            if (block->gpu_ptr) {
                cudaFree(block->gpu_ptr);
            }
        }
        
        // 释放CPU内存
        for (auto& [id, block] : cpu_pool_.allocated_blocks) {
            if (block->cpu_ptr) {
                cudaFreeHost(block->cpu_ptr);
            }
        }
        
        // 清理存储文件
        for (auto& [id, block] : storage_pool_.allocated_blocks) {
            if (!block->storage_path.empty()) {
                std::filesystem::remove(block->storage_path);
            }
        }
    }
    
    std::cout << "HMT Memory Manager shutdown complete" << std::endl;
}

MemoryBlock* HMTMemoryManager::allocate_block(MemoryTier preferred_tier) {
    switch (preferred_tier) {
        case MemoryTier::GPU:
            if (auto block = allocate_gpu_block()) {
                return block;
            }
            // GPU分配失败，降级到CPU
            [[fallthrough]];
            
        case MemoryTier::CPU:
            if (auto block = allocate_cpu_block()) {
                return block;
            }
            // CPU分配失败，降级到存储
            [[fallthrough]];
            
        case MemoryTier::STORAGE:
            return allocate_storage_block();
            
        default:
            return nullptr;
    }
}

void HMTMemoryManager::deallocate_block(MemoryBlock* block) {
    if (!block) return;
    
    switch (block->current_tier) {
        case MemoryTier::GPU:
            deallocate_gpu_block(block);
            break;
        case MemoryTier::CPU:
            deallocate_cpu_block(block);
            break;
        case MemoryTier::STORAGE:
            deallocate_storage_block(block);
            break;
    }
}

void HMTMemoryManager::record_access(MemoryBlock* block, double attention_score) {
    if (!block) return;
    
    block->last_access = std::chrono::steady_clock::now();
    block->access_count.fetch_add(1);
    block->attention_score.store(attention_score);
    
    // 更新命中统计
    switch (block->current_tier) {
        case MemoryTier::GPU:
            stats_.gpu_hit_count.fetch_add(1);
            break;
        case MemoryTier::CPU:
            stats_.cpu_hit_count.fetch_add(1);
            break;
        case MemoryTier::STORAGE:
            stats_.storage_hit_count.fetch_add(1);
            break;
    }
}

void HMTMemoryManager::record_access(void* ptr, double attention_score) {
    if (auto block = find_block(ptr)) {
        record_access(block, attention_score);
    }
}

bool HMTMemoryManager::promote_to_gpu(MemoryBlock* block) {
    if (!block || block->current_tier == MemoryTier::GPU) {
        return true;
    }
    
    if (block->is_migrating.load()) {
        return false;
    }
    
    // 异步迁移
    enqueue_migration(block, MemoryTier::GPU);
    return true;
}

bool HMTMemoryManager::demote_to_cpu(MemoryBlock* block) {
    if (!block || block->current_tier == MemoryTier::CPU) {
        return true;
    }
    
    if (block->is_migrating.load()) {
        return false;
    }
    
    enqueue_migration(block, MemoryTier::CPU);
    return true;
}

bool HMTMemoryManager::archive_to_storage(MemoryBlock* block) {
    if (!block || block->current_tier == MemoryTier::STORAGE) {
        return true;
    }
    
    if (block->is_migrating.load()) {
        return false;
    }
    
    enqueue_migration(block, MemoryTier::STORAGE);
    return true;
}

std::vector<MemoryBlock*> HMTMemoryManager::allocate_blocks(size_t count, MemoryTier preferred_tier) {
    std::vector<MemoryBlock*> blocks;
    blocks.reserve(count);
    
    for (size_t i = 0; i < count; ++i) {
        if (auto block = allocate_block(preferred_tier)) {
            blocks.push_back(block);
        } else {
            // 分配失败，释放已分配的块
            deallocate_blocks(blocks);
            return {};
        }
    }
    
    return blocks;
}

void HMTMemoryManager::deallocate_blocks(const std::vector<MemoryBlock*>& blocks) {
    for (auto block : blocks) {
        deallocate_block(block);
    }
}

void HMTMemoryManager::update_a2cr_params(const A2CRParams& params) {
    a2cr_params_.time_decay_factor.store(params.time_decay_factor.load());
    a2cr_params_.attention_weight.store(params.attention_weight.load());
    a2cr_params_.frequency_weight.store(params.frequency_weight.load());
    a2cr_params_.recency_weight.store(params.recency_weight.load());
    a2cr_params_.gpu_pressure_threshold.store(params.gpu_pressure_threshold.load());
    a2cr_params_.cpu_pressure_threshold.store(params.cpu_pressure_threshold.load());
}

A2CRParams HMTMemoryManager::get_a2cr_params() const {
    return a2cr_params_;
}

MemoryStats HMTMemoryManager::get_memory_stats() const {
    return stats_;
}

MemoryBlock* HMTMemoryManager::find_block(void* ptr) const {
    // 在GPU池中查找
    {
        std::shared_lock<std::shared_mutex> lock(gpu_pool_.mutex);
        auto it = gpu_pool_.ptr_to_block.find(ptr);
        if (it != gpu_pool_.ptr_to_block.end()) {
            return it->second;
        }
    }
    
    // 在CPU池中查找
    {
        std::shared_lock<std::shared_mutex> lock(cpu_pool_.mutex);
        auto it = cpu_pool_.ptr_to_block.find(ptr);
        if (it != cpu_pool_.ptr_to_block.end()) {
            return it->second;
        }
    }
    
    return nullptr;
}

double HMTMemoryManager::get_memory_pressure(MemoryTier tier) const {
    switch (tier) {
        case MemoryTier::GPU:
            return static_cast<double>(gpu_pool_.used_count.load()) / gpu_pool_.total_capacity;
        case MemoryTier::CPU:
            return static_cast<double>(cpu_pool_.used_count.load()) / cpu_pool_.total_capacity;
        case MemoryTier::STORAGE:
            return 0.0; // 存储层假设无限容量
        default:
            return 0.0;
    }
}

void HMTMemoryManager::trigger_gc() {
    // 立即触发垃圾回收
    cleanup_expired_blocks();
}

void HMTMemoryManager::trigger_defragmentation() {
    // 立即触发内存整理
    defragment_memory();
}

// === 私有方法实现 ===

bool HMTMemoryManager::initialize_gpu_pool() {
    std::unique_lock<std::shared_mutex> lock(gpu_pool_.mutex);
    
    // 预分配一定数量的GPU内存块
    size_t initial_blocks = std::min(gpu_pool_.total_capacity / 4, static_cast<size_t>(1024));
    
    for (size_t i = 0; i < initial_blocks; ++i) {
        auto block_id = generate_block_id();
        auto block = std::make_unique<MemoryBlock>(block_id);
        
        cudaError_t result = cudaMalloc(&block->gpu_ptr, BLOCK_SIZE);
        if (result != cudaSuccess) {
            std::cerr << "Failed to allocate GPU memory: " << cudaGetErrorString(result) << std::endl;
            return false;
        }
        
        block->current_tier = MemoryTier::GPU;
        gpu_pool_.ptr_to_block[block->gpu_ptr] = block.get();
        gpu_pool_.free_blocks.push(block.get());
        gpu_pool_.allocated_blocks[block_id] = std::move(block);
    }
    
    stats_.gpu_total_blocks.store(initial_blocks);
    stats_.gpu_free_blocks.store(initial_blocks);
    
    return true;
}

bool HMTMemoryManager::initialize_cpu_pool() {
    std::unique_lock<std::shared_mutex> lock(cpu_pool_.mutex);
    
    // 预分配一定数量的CPU内存块
    size_t initial_blocks = std::min(cpu_pool_.total_capacity / 8, static_cast<size_t>(2048));
    
    for (size_t i = 0; i < initial_blocks; ++i) {
        auto block_id = generate_block_id();
        auto block = std::make_unique<MemoryBlock>(block_id);
        
        cudaError_t result = cudaHostAlloc(&block->cpu_ptr, BLOCK_SIZE, 
                                          cudaHostAllocMapped | cudaHostAllocWriteCombined);
        if (result != cudaSuccess) {
            std::cerr << "Failed to allocate CPU memory: " << cudaGetErrorString(result) << std::endl;
            return false;
        }
        
        // 获取GPU可访问的地址
        cudaHostGetDevicePointer(&block->gpu_ptr, block->cpu_ptr, 0);
        
        block->current_tier = MemoryTier::CPU;
        cpu_pool_.ptr_to_block[block->cpu_ptr] = block.get();
        cpu_pool_.ptr_to_block[block->gpu_ptr] = block.get(); // GPU也能访问
        cpu_pool_.free_blocks.push(block.get());
        cpu_pool_.allocated_blocks[block_id] = std::move(block);
    }
    
    stats_.cpu_total_blocks.store(initial_blocks);
    stats_.cpu_free_blocks.store(initial_blocks);
    
    return true;
}

bool HMTMemoryManager::initialize_storage_pool() {
    std::unique_lock<std::shared_mutex> lock(storage_pool_.mutex);
    
    // 存储池不需要预分配，按需创建
    return true;
}

MemoryBlock* HMTMemoryManager::allocate_gpu_block() {
    std::unique_lock<std::shared_mutex> lock(gpu_pool_.mutex);
    
    // 1. 检查是否有空闲块
    if (!gpu_pool_.free_blocks.empty()) {
        auto block = gpu_pool_.free_blocks.front();
        gpu_pool_.free_blocks.pop();
        gpu_pool_.used_count.fetch_add(1);
        stats_.gpu_used_blocks.fetch_add(1);
        stats_.gpu_free_blocks.fetch_sub(1);
        return block;
    }
    
    // 2. 检查是否可以扩容
    if (gpu_pool_.allocated_blocks.size() < gpu_pool_.total_capacity) {
        auto block_id = generate_block_id();
        auto block = std::make_unique<MemoryBlock>(block_id);
        
        cudaError_t result = cudaMalloc(&block->gpu_ptr, BLOCK_SIZE);
        if (result == cudaSuccess) {
            block->current_tier = MemoryTier::GPU;
            gpu_pool_.ptr_to_block[block->gpu_ptr] = block.get();
            auto* block_ptr = block.get();
            gpu_pool_.allocated_blocks[block_id] = std::move(block);
            
            gpu_pool_.used_count.fetch_add(1);
            stats_.gpu_total_blocks.fetch_add(1);
            stats_.gpu_used_blocks.fetch_add(1);
            
            return block_ptr;
        }
    }
    
    // 3. 尝试置换
    lock.unlock();
    if (evict_gpu_victim()) {
        return allocate_gpu_block(); // 递归重试
    }
    
    stats_.gpu_miss_count.fetch_add(1);
    return nullptr;
}

void HMTMemoryManager::deallocate_gpu_block(MemoryBlock* block) {
    if (!block) return;
    
    std::unique_lock<std::shared_mutex> lock(gpu_pool_.mutex);
    
    // 标记为空闲
    gpu_pool_.free_blocks.push(block);
    gpu_pool_.used_count.fetch_sub(1);
    stats_.gpu_used_blocks.fetch_sub(1);
    stats_.gpu_free_blocks.fetch_add(1);
    
    // 重置访问统计
    block->access_count.store(0);
    block->attention_score.store(1.0);
    block->is_dirty.store(false);
}

bool HMTMemoryManager::evict_gpu_victim() {
    auto victim = select_gpu_victim();
    if (!victim) {
        return false;
    }
    
    // 异步迁移到CPU
    enqueue_migration(victim, MemoryTier::CPU);
    return true;
}

MemoryBlock* HMTMemoryManager::allocate_cpu_block() {
    std::unique_lock<std::shared_mutex> lock(cpu_pool_.mutex);
    
    // 类似GPU分配逻辑
    if (!cpu_pool_.free_blocks.empty()) {
        auto block = cpu_pool_.free_blocks.front();
        cpu_pool_.free_blocks.pop();
        cpu_pool_.used_count.fetch_add(1);
        stats_.cpu_used_blocks.fetch_add(1);
        stats_.cpu_free_blocks.fetch_sub(1);
        return block;
    }
    
    if (cpu_pool_.allocated_blocks.size() < cpu_pool_.total_capacity) {
        auto block_id = generate_block_id();
        auto block = std::make_unique<MemoryBlock>(block_id);
        
        cudaError_t result = cudaHostAlloc(&block->cpu_ptr, BLOCK_SIZE, 
                                          cudaHostAllocMapped | cudaHostAllocWriteCombined);
        if (result == cudaSuccess) {
            cudaHostGetDevicePointer(&block->gpu_ptr, block->cpu_ptr, 0);
            block->current_tier = MemoryTier::CPU;
            cpu_pool_.ptr_to_block[block->cpu_ptr] = block.get();
            cpu_pool_.ptr_to_block[block->gpu_ptr] = block.get();
            auto* block_ptr = block.get();
            cpu_pool_.allocated_blocks[block_id] = std::move(block);
            
            cpu_pool_.used_count.fetch_add(1);
            stats_.cpu_total_blocks.fetch_add(1);
            stats_.cpu_used_blocks.fetch_add(1);
            
            return block_ptr;
        }
    }
    
    lock.unlock();
    if (evict_cpu_victim()) {
        return allocate_cpu_block();
    }
    
    stats_.cpu_miss_count.fetch_add(1);
    return nullptr;
}

void HMTMemoryManager::deallocate_cpu_block(MemoryBlock* block) {
    if (!block) return;
    
    std::unique_lock<std::shared_mutex> lock(cpu_pool_.mutex);
    
    cpu_pool_.free_blocks.push(block);
    cpu_pool_.used_count.fetch_sub(1);
    stats_.cpu_used_blocks.fetch_sub(1);
    stats_.cpu_free_blocks.fetch_add(1);
    
    block->access_count.store(0);
    block->attention_score.store(1.0);
    block->is_dirty.store(false);
}

bool HMTMemoryManager::evict_cpu_victim() {
    auto victim = select_cpu_victim();
    if (!victim) {
        return false;
    }
    
    enqueue_migration(victim, MemoryTier::STORAGE);
    return true;
}

MemoryBlock* HMTMemoryManager::allocate_storage_block() {
    std::unique_lock<std::shared_mutex> lock(storage_pool_.mutex);
    
    auto block_id = generate_block_id();
    auto block = std::make_unique<MemoryBlock>(block_id);
    
    block->current_tier = MemoryTier::STORAGE;
    block->storage_path = generate_storage_path(block_id);
    
    auto* block_ptr = block.get();
    storage_pool_.allocated_blocks[block_id] = std::move(block);
    storage_pool_.used_count.fetch_add(1);
    stats_.storage_total_blocks.fetch_add(1);
    stats_.storage_used_blocks.fetch_add(1);
    
    return block_ptr;
}

void HMTMemoryManager::deallocate_storage_block(MemoryBlock* block) {
    if (!block) return;
    
    std::unique_lock<std::shared_mutex> lock(storage_pool_.mutex);
    
    // 删除存储文件
    if (!block->storage_path.empty()) {
        std::filesystem::remove(block->storage_path);
    }
    
    storage_pool_.allocated_blocks.erase(block->block_id);
    storage_pool_.used_count.fetch_sub(1);
    stats_.storage_used_blocks.fetch_sub(1);
}

MemoryBlock* HMTMemoryManager::select_gpu_victim() const {
    std::shared_lock<std::shared_mutex> lock(gpu_pool_.mutex);
    
    MemoryBlock* best_victim = nullptr;
    double lowest_score = std::numeric_limits<double>::max();
    
    for (const auto& [id, block] : gpu_pool_.allocated_blocks) {
        if (block->is_pinned.load() || block->is_migrating.load()) {
            continue;
        }
        
        double score = calculate_victim_score(block.get());
        if (score < lowest_score) {
            lowest_score = score;
            best_victim = block.get();
        }
    }
    
    return best_victim;
}

MemoryBlock* HMTMemoryManager::select_cpu_victim() const {
    std::shared_lock<std::shared_mutex> lock(cpu_pool_.mutex);
    
    MemoryBlock* best_victim = nullptr;
    double lowest_score = std::numeric_limits<double>::max();
    
    for (const auto& [id, block] : cpu_pool_.allocated_blocks) {
        if (block->is_pinned.load() || block->is_migrating.load()) {
            continue;
        }
        
        double score = calculate_victim_score(block.get());
        if (score < lowest_score) {
            lowest_score = score;
            best_victim = block.get();
        }
    }
    
    return best_victim;
}

double HMTMemoryManager::calculate_victim_score(const MemoryBlock* block) const {
    if (!block) return 0.0;
    
    auto now = std::chrono::steady_clock::now();
    
    // 计算时间衰减
    auto time_diff = std::chrono::duration<double>(now - block->last_access).count();
    double time_decay = std::exp(-a2cr_params_.time_decay_factor.load() * time_diff);
    
    // 归一化访问频率
    double frequency_score = std::log1p(static_cast<double>(block->access_count.load()));
    
    // 归一化新近度
    auto creation_diff = std::chrono::duration<double>(now - block->creation_time).count();
    double recency_score = std::exp(-creation_diff / 3600.0); // 1小时衰减
    
    // 注意力分数
    double attention = block->attention_score.load();
    
    // 加权计算最终分数
    double final_score = 
        a2cr_params_.attention_weight.load() * attention +
        a2cr_params_.frequency_weight.load() * frequency_score * time_decay +
        a2cr_params_.recency_weight.load() * recency_score;
    
    return final_score;
}

void HMTMemoryManager::migration_worker() {
    while (!shutdown_requested_.load()) {
        std::unique_lock<std::mutex> lock(migration_queue_mutex_);
        
        migration_cv_.wait(lock, [this] {
            return !migration_queue_.empty() || shutdown_requested_.load();
        });
        
        if (shutdown_requested_.load()) {
            break;
        }
        
        if (!migration_queue_.empty()) {
            auto block = migration_queue_.front();
            migration_queue_.pop();
            lock.unlock();
            
            // 执行迁移
            // 这里需要根据具体需求实现迁移逻辑
        }
    }
}

void HMTMemoryManager::enqueue_migration(MemoryBlock* block, MemoryTier target_tier) {
    if (!block) return;
    
    std::lock_guard<std::mutex> lock(migration_queue_mutex_);
    migration_queue_.push(block);
    migration_cv_.notify_one();
}

void HMTMemoryManager::gc_worker() {
    while (!shutdown_requested_.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(30)); // 30秒运行一次
        
        if (!shutdown_requested_.load()) {
            cleanup_expired_blocks();
            defragment_memory();
        }
    }
}

void HMTMemoryManager::cleanup_expired_blocks() {
    // 实现过期块清理逻辑
}

void HMTMemoryManager::defragment_memory() {
    // 实现内存碎片整理逻辑
}

std::string HMTMemoryManager::generate_storage_path(uint64_t block_id) const {
    return storage_base_path_ + "/block_" + std::to_string(block_id) + ".cache";
}

uint64_t HMTMemoryManager::generate_block_id() {
    return next_block_id_.fetch_add(1);
}

bool HMTMemoryManager::is_memory_pressure_high(MemoryTier tier) const {
    double pressure = get_memory_pressure(tier);
    
    switch (tier) {
        case MemoryTier::GPU:
            return pressure > a2cr_params_.gpu_pressure_threshold.load();
        case MemoryTier::CPU:
            return pressure > a2cr_params_.cpu_pressure_threshold.load();
        case MemoryTier::STORAGE:
            return false; // 存储层假设无限容量
        default:
            return false;
    }
}

} // namespace wicore 