// batch_scheduler.cpp
#include "../include/batch_scheduler.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <sstream>

namespace wicore {

// === LoadPredictor实现 ===

LoadPredictor::LoadPredictor(int window_size) : window_size_(window_size) {
    recent_durations_.reserve(window_size_);
    recent_arrivals_.reserve(window_size_);
}

void LoadPredictor::record_request(const ScheduledRequest& request) {
    std::lock_guard<std::mutex> lock(predictor_mutex_);
    
    auto now = std::chrono::steady_clock::now();
    recent_arrivals_.push_back(now);
    
    // 保持窗口大小
    if (recent_arrivals_.size() > static_cast<size_t>(window_size_)) {
        recent_arrivals_.pop_front();
    }
}

void LoadPredictor::record_batch_completion(const BatchInfo& batch) {
    std::lock_guard<std::mutex> lock(predictor_mutex_);
    
    if (batch.status == BatchStatus::COMPLETED) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            batch.end_time - batch.start_time).count();
        
        recent_durations_.push_back(static_cast<double>(duration));
        
        // 保持窗口大小
        if (recent_durations_.size() > static_cast<size_t>(window_size_)) {
            recent_durations_.pop_front();
        }
    }
}

double LoadPredictor::predict_batch_duration(const std::vector<std::shared_ptr<ScheduledRequest>>& requests) {
    std::lock_guard<std::mutex> lock(predictor_mutex_);
    
    if (recent_durations_.empty()) {
        // 默认预估：每个请求1秒 + 基础开销100ms
        return 100.0 + requests.size() * 1000.0;
    }
    
    double avg_duration = calculate_moving_average(recent_durations_);
    int total_tokens = 0;
    for (const auto& req : requests) {
        total_tokens += req->estimated_tokens;
    }
    
    // 基于历史数据和token数量预估
    double scale_factor = std::max(1.0, static_cast<double>(total_tokens) / 512.0);
    return avg_duration * scale_factor;
}

int LoadPredictor::predict_optimal_batch_size() {
    std::lock_guard<std::mutex> lock(predictor_mutex_);
    
    double load_factor = get_current_load_factor();
    
    // 根据负载因子调整批大小
    if (load_factor < 0.3) {
        return 16;  // 低负载，使用大批次
    } else if (load_factor < 0.7) {
        return 8;   // 中等负载
    } else {
        return 4;   // 高负载，使用小批次
    }
}

bool LoadPredictor::should_wait_for_more_requests() {
    double request_rate = get_average_request_rate();
    
    // 如果请求率很高（>5req/s），不等待
    // 如果请求率很低（<1req/s），等待
    return request_rate < 1.0;
}

double LoadPredictor::get_current_load_factor() const {
    if (recent_durations_.empty() || recent_arrivals_.empty()) {
        return 0.0;
    }
    
    double avg_duration = calculate_moving_average(recent_durations_);
    double request_rate = calculate_request_rate();
    
    // 负载因子 = 平均处理时间 * 请求率
    return (avg_duration / 1000.0) * request_rate;
}

double LoadPredictor::get_average_request_rate() const {
    std::lock_guard<std::mutex> lock(predictor_mutex_);
    return calculate_request_rate();
}

double LoadPredictor::calculate_moving_average(const std::deque<double>& values) const {
    if (values.empty()) return 0.0;
    
    double sum = 0.0;
    for (double val : values) {
        sum += val;
    }
    return sum / values.size();
}

double LoadPredictor::calculate_request_rate() const {
    if (recent_arrivals_.size() < 2) return 0.0;
    
    auto time_span = recent_arrivals_.back() - recent_arrivals_.front();
    auto seconds = std::chrono::duration<double>(time_span).count();
    
    if (seconds <= 0) return 0.0;
    
    return static_cast<double>(recent_arrivals_.size() - 1) / seconds;
}

// === BatchScheduler实现 ===

BatchScheduler::BatchScheduler(TensorRTInferenceEngine* inference_engine,
                              const SchedulerConfig& config)
    : inference_engine_(inference_engine), config_(config) {
    
    load_predictor_ = std::make_unique<LoadPredictor>(config_.adaptive_window_size);
    
    std::cout << "Batch Scheduler created" << std::endl;
    std::cout << "Max batch size: " << config_.max_batch_size << std::endl;
    std::cout << "Scheduling policy: " << (config_.scheduling_policy == SchedulingPolicy::ADAPTIVE ? "ADAPTIVE" : "OTHER") << std::endl;
    std::cout << "Continuous batching: " << (config_.enable_continuous_batching ? "ENABLED" : "DISABLED") << std::endl;
}

BatchScheduler::~BatchScheduler() {
    shutdown();
}

bool BatchScheduler::initialize() {
    if (initialized_.load()) {
        return true;
    }
    
    if (!inference_engine_) {
        std::cerr << "Inference engine is null" << std::endl;
        return false;
    }
    
    try {
        // 启动工作线程
        shutdown_requested_.store(false);
        paused_.store(false);
        
        scheduler_thread_ = std::thread(&BatchScheduler::scheduler_main_loop, this);
        batch_executor_thread_ = std::thread(&BatchScheduler::batch_executor_main_loop, this);
        monitoring_thread_ = std::thread(&BatchScheduler::monitoring_main_loop, this);
        
        for (int i = 0; i < num_worker_threads_; ++i) {
            worker_threads_.emplace_back(&BatchScheduler::worker_thread_function, this, i);
        }
        
        initialized_.store(true);
        std::cout << "Batch Scheduler initialized successfully" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during Batch Scheduler initialization: " << e.what() << std::endl;
        return false;
    }
}

void BatchScheduler::shutdown() {
    if (!initialized_.load() || shutdown_requested_.load()) {
        return;
    }
    
    std::cout << "Shutting down Batch Scheduler..." << std::endl;
    
    // 停止所有线程
    shutdown_requested_.store(true);
    request_available_cv_.notify_all();
    batch_ready_cv_.notify_all();
    
    if (scheduler_thread_.joinable()) {
        scheduler_thread_.join();
    }
    
    if (batch_executor_thread_.joinable()) {
        batch_executor_thread_.join();
    }
    
    if (monitoring_thread_.joinable()) {
        monitoring_thread_.join();
    }
    
    for (auto& worker : worker_threads_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    
    // 清理剩余请求
    {
        std::unique_lock<std::shared_mutex> lock(requests_mutex_);
        for (auto& queue : priority_queues_) {
            while (!queue.empty()) {
                auto request = queue.front();
                queue.pop();
                
                // 设置取消响应
                auto response = create_error_response(request->request_id, "System shutdown");
                request->promise.set_value(response);
            }
        }
        active_requests_.clear();
    }
    
    initialized_.store(false);
    std::cout << "Batch Scheduler shutdown complete" << std::endl;
}

std::future<InferenceResponse> BatchScheduler::submit_request(
    const ProcessedMultiModal& input,
    RequestPriority priority,
    const std::chrono::milliseconds& timeout) {
    
    if (!is_ready()) {
        auto request_id = generate_request_id();
        std::promise<InferenceResponse> promise;
        auto future = promise.get_future();
        auto response = create_error_response(request_id, "Scheduler not ready");
        promise.set_value(response);
        return future;
    }
    
    // 创建请求
    auto request_id = generate_request_id();
    auto request = std::make_shared<ScheduledRequest>(request_id, input);
    request->priority = priority;
    request->timeout_ms = timeout;
    request->deadline = request->submit_time + timeout;
    
    // 记录到负载预测器
    load_predictor_->record_request(*request);
    
    // 添加到队列
    {
        std::unique_lock<std::shared_mutex> lock(requests_mutex_);
        
        // 检查队列是否已满
        int total_queue_size = 0;
        for (const auto& queue : priority_queues_) {
            total_queue_size += static_cast<int>(queue.size());
        }
        
        if (total_queue_size >= config_.max_queue_size) {
            auto response = create_error_response(request_id, "Queue is full");
            request->promise.set_value(response);
            return request->future;
        }
        
        // 加入优先级队列
        int priority_index = static_cast<int>(priority);
        priority_queues_[priority_index].push(request);
        active_requests_[request_id] = request;
        
        stats_.current_queue_size.fetch_add(1);
        stats_.total_requests.fetch_add(1);
    }
    
    // 通知调度器
    request_available_cv_.notify_one();
    
    return request->future;
}

std::future<InferenceResponse> BatchScheduler::submit_streaming_request(
    const ProcessedMultiModal& input,
    std::function<void(const std::string&)> stream_callback,
    RequestPriority priority,
    const std::chrono::milliseconds& timeout) {
    
    // 创建流式请求
    auto future = submit_request(input, priority, timeout);
    
    // 这里可以扩展实现真正的流式输出
    // 暂时返回普通请求的future
    return future;
}

bool BatchScheduler::cancel_request(const std::string& request_id) {
    std::unique_lock<std::shared_mutex> lock(requests_mutex_);
    
    auto it = active_requests_.find(request_id);
    if (it == active_requests_.end()) {
        return false;
    }
    
    auto request = it->second;
    request->is_cancelled.store(true);
    request->status = RequestStatus::CANCELLED;
    
    // 设置取消响应
    auto response = create_error_response(request_id, "Request cancelled");
    request->promise.set_value(response);
    
    active_requests_.erase(it);
    stats_.cancelled_requests.fetch_add(1);
    
    return true;
}

RequestStatus BatchScheduler::get_request_status(const std::string& request_id) {
    std::shared_lock<std::shared_mutex> lock(requests_mutex_);
    
    auto it = active_requests_.find(request_id);
    if (it == active_requests_.end()) {
        return RequestStatus::COMPLETED; // 或者抛出异常
    }
    
    return it->second->status;
}

int BatchScheduler::get_queue_size() const {
    return stats_.current_queue_size.load();
}

int BatchScheduler::get_active_batch_count() const {
    return stats_.current_batch_count.load();
}

SchedulerStats BatchScheduler::get_stats() const {
    return stats_;
}

void BatchScheduler::pause_scheduling() {
    paused_.store(true);
}

void BatchScheduler::resume_scheduling() {
    paused_.store(false);
    request_available_cv_.notify_all();
}

// === 内部方法实现 ===

void BatchScheduler::scheduler_main_loop() {
    while (!shutdown_requested_.load()) {
        try {
            if (paused_.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            
            // 等待请求或超时
            std::unique_lock<std::shared_mutex> lock(requests_mutex_);
            
            bool has_requests = false;
            for (const auto& queue : priority_queues_) {
                if (!queue.empty()) {
                    has_requests = true;
                    break;
                }
            }
            
            if (!has_requests) {
                request_available_cv_.wait_for(lock, config_.max_wait_time_ms, [this, &has_requests] {
                    for (const auto& queue : priority_queues_) {
                        if (!queue.empty()) {
                            has_requests = true;
                            return true;
                        }
                    }
                    return shutdown_requested_.load();
                });
            }
            
            if (shutdown_requested_.load()) break;
            
            // 检查是否应该创建批次
            if (should_create_batch_now()) {
                auto batch_requests = create_batch();
                if (!batch_requests.empty()) {
                    auto batch = std::make_unique<BatchInfo>(generate_batch_id());
                    batch->requests = batch_requests;
                    batch->current_size = static_cast<int>(batch_requests.size());
                    
                    // 更新请求状态
                    for (auto& req : batch_requests) {
                        req->status = RequestStatus::BATCHED;
                    }
                    
                    lock.unlock();
                    
                    // 提交批次执行
                    {
                        std::lock_guard<std::mutex> batch_lock(batches_mutex_);
                        pending_batches_.push(std::move(batch));
                        stats_.current_batch_count.fetch_add(1);
                    }
                    batch_ready_cv_.notify_one();
                }
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Exception in scheduler main loop: " << e.what() << std::endl;
        }
        
        // 短暂休眠避免过度CPU使用
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void BatchScheduler::batch_executor_main_loop() {
    while (!shutdown_requested_.load()) {
        try {
            std::unique_lock<std::mutex> lock(batches_mutex_);
            
            batch_ready_cv_.wait(lock, [this] {
                return !pending_batches_.empty() || shutdown_requested_.load();
            });
            
            if (shutdown_requested_.load()) break;
            
            if (!pending_batches_.empty()) {
                auto batch = std::move(pending_batches_.front());
                pending_batches_.pop();
                lock.unlock();
                
                // 执行批次
                bool success = execute_batch(std::move(batch));
                
                if (!success) {
                    stats_.failed_batches.fetch_add(1);
                } else {
                    stats_.successful_batches.fetch_add(1);
                }
                
                stats_.current_batch_count.fetch_sub(1);
                stats_.total_batches.fetch_add(1);
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Exception in batch executor: " << e.what() << std::endl;
        }
    }
}

void BatchScheduler::monitoring_main_loop() {
    while (!shutdown_requested_.load()) {
        try {
            // 清理完成的请求
            cleanup_completed_requests();
            
            // 处理超时请求
            handle_timeout_requests();
            
            // 更新负载指标
            update_load_metrics();
            
            // 自适应批大小调整
            if (config_.enable_adaptive_batching) {
                adjust_batch_size_adaptively();
            }
            
            // 更新资源统计
            update_resource_stats();
            
        } catch (const std::exception& e) {
            std::cerr << "Exception in monitoring loop: " << e.what() << std::endl;
        }
        
        // 每秒运行一次
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

std::shared_ptr<ScheduledRequest> BatchScheduler::get_next_request() {
    // 按优先级从高到低检查队列
    for (int i = 3; i >= 0; --i) {
        if (!priority_queues_[i].empty()) {
            auto request = priority_queues_[i].front();
            priority_queues_[i].pop();
            stats_.current_queue_size.fetch_sub(1);
            return request;
        }
    }
    return nullptr;
}

std::vector<std::shared_ptr<ScheduledRequest>> BatchScheduler::create_batch() {
    std::vector<std::shared_ptr<ScheduledRequest>> batch;
    int optimal_size = calculate_optimal_batch_size();
    
    while (batch.size() < static_cast<size_t>(optimal_size)) {
        auto request = get_next_request();
        if (!request) break;
        
        // 检查请求是否已过期
        if (request->is_cancelled.load() || 
            std::chrono::steady_clock::now() > request->deadline) {
            // 处理超时/取消的请求
            continue;
        }
        
        batch.push_back(request);
    }
    
    return batch;
}

bool BatchScheduler::should_create_batch_now() {
    // 检查是否有请求在队列中
    bool has_requests = false;
    for (const auto& queue : priority_queues_) {
        if (!queue.empty()) {
            has_requests = true;
            break;
        }
    }
    
    if (!has_requests) return false;
    
    // 检查是否达到批次超时
    static auto last_batch_time = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_batch_time);
    
    if (elapsed >= config_.batch_timeout_ms) {
        last_batch_time = now;
        return true;
    }
    
    // 检查是否有足够的请求
    int total_requests = 0;
    for (const auto& queue : priority_queues_) {
        total_requests += static_cast<int>(queue.size());
    }
    
    return total_requests >= config_.max_batch_size;
}

int BatchScheduler::calculate_optimal_batch_size() {
    if (config_.enable_adaptive_batching) {
        return load_predictor_->predict_optimal_batch_size();
    }
    return config_.max_batch_size;
}

bool BatchScheduler::execute_batch(std::unique_ptr<BatchInfo> batch) {
    if (!batch || batch->requests.empty()) {
        return false;
    }
    
    try {
        batch->start_time = std::chrono::steady_clock::now();
        batch->status = BatchStatus::EXECUTING;
        
        // 更新请求状态
        for (auto& req : batch->requests) {
            req->status = RequestStatus::EXECUTING;
        }
        
        // 预处理
        if (!preprocess_batch(batch.get())) {
            batch->status = BatchStatus::FAILED;
            return false;
        }
        
        // 执行推理
        if (!run_inference_batch(batch.get())) {
            batch->status = BatchStatus::FAILED;
            return false;
        }
        
        // 后处理
        if (!postprocess_batch(batch.get())) {
            batch->status = BatchStatus::FAILED;
            return false;
        }
        
        batch->end_time = std::chrono::steady_clock::now();
        batch->status = BatchStatus::COMPLETED;
        
        // 记录到负载预测器
        load_predictor_->record_batch_completion(*batch);
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception executing batch: " << e.what() << std::endl;
        batch->status = BatchStatus::FAILED;
        return false;
    }
}

bool BatchScheduler::preprocess_batch(BatchInfo* batch) {
    // 批次预处理逻辑
    // 这里主要是准备推理输入数据
    return true;
}

bool BatchScheduler::run_inference_batch(BatchInfo* batch) {
    try {
        // 串行处理每个请求（简化实现）
        for (auto& request : batch->requests) {
            if (request->is_cancelled.load()) {
                continue;
            }
            
            // 调用推理引擎
            auto response = inference_engine_->infer(request->input_data);
            
            // 设置响应
            request->promise.set_value(response);
            request->status = RequestStatus::COMPLETED;
            
            // 更新统计
            update_request_stats(*request, response.success);
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in batch inference: " << e.what() << std::endl;
        
        // 处理失败的请求
        for (auto& request : batch->requests) {
            if (request->status == RequestStatus::EXECUTING) {
                auto error_response = create_error_response(request->request_id, 
                                                          "Batch inference failed");
                request->promise.set_value(error_response);
                request->status = RequestStatus::FAILED;
                update_request_stats(*request, false);
            }
        }
        
        return false;
    }
}

bool BatchScheduler::postprocess_batch(BatchInfo* batch) {
    // 批次后处理逻辑
    // 清理资源，更新统计等
    
    // 从活跃请求中移除
    {
        std::unique_lock<std::shared_mutex> lock(requests_mutex_);
        for (const auto& request : batch->requests) {
            active_requests_.erase(request->request_id);
        }
    }
    
    return true;
}

void BatchScheduler::cleanup_completed_requests() {
    std::unique_lock<std::shared_mutex> lock(requests_mutex_);
    
    auto it = active_requests_.begin();
    while (it != active_requests_.end()) {
        auto& request = it->second;
        
        if (request->status == RequestStatus::COMPLETED ||
            request->status == RequestStatus::FAILED ||
            request->status == RequestStatus::CANCELLED) {
            it = active_requests_.erase(it);
        } else {
            ++it;
        }
    }
}

void BatchScheduler::handle_timeout_requests() {
    std::unique_lock<std::shared_mutex> lock(requests_mutex_);
    
    auto now = std::chrono::steady_clock::now();
    
    for (auto& [id, request] : active_requests_) {
        if (now > request->deadline && request->status == RequestStatus::QUEUED) {
            request->status = RequestStatus::TIMEOUT;
            
            auto response = create_error_response(id, "Request timeout");
            request->promise.set_value(response);
            
            stats_.timeout_requests.fetch_add(1);
        }
    }
}

void BatchScheduler::update_load_metrics() {
    // 更新吞吐量统计
    static auto last_update = std::chrono::steady_clock::now();
    static uint64_t last_total_requests = 0;
    
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<double>(now - last_update).count();
    
    if (elapsed >= 1.0) { // 每秒更新一次
        uint64_t current_total = stats_.total_requests.load();
        double rps = static_cast<double>(current_total - last_total_requests) / elapsed;
        stats_.requests_per_second.store(rps);
        
        last_update = now;
        last_total_requests = current_total;
    }
}

std::string BatchScheduler::generate_batch_id() {
    return "batch_" + std::to_string(next_batch_id_.fetch_add(1));
}

std::string BatchScheduler::generate_request_id() {
    return "req_" + std::to_string(next_request_id_.fetch_add(1));
}

InferenceResponse BatchScheduler::create_error_response(const std::string& request_id,
                                                       const std::string& error_msg) {
    InferenceResponse response;
    response.request_id = request_id;
    response.success = false;
    response.error_message = error_msg;
    response.inference_time_ms = 0.0;
    response.tokens_generated = 0;
    
    return response;
}

void BatchScheduler::update_request_stats(const ScheduledRequest& request, bool success) {
    if (success) {
        stats_.successful_requests.fetch_add(1);
    } else {
        stats_.failed_requests.fetch_add(1);
    }
    
    // 更新延迟统计（简化实现）
    auto now = std::chrono::steady_clock::now();
    auto latency = std::chrono::duration<double, std::milli>(now - request.submit_time).count();
    
    // 简单的滑动平均
    stats_.avg_end_to_end_latency_ms.store(
        (stats_.avg_end_to_end_latency_ms.load() * 0.9) + (latency * 0.1));
}

double BatchScheduler::get_gpu_memory_usage() {
    // 这里应该查询实际的GPU内存使用情况
    // 简化实现，返回模拟值
    return 0.6; // 60%
}

void BatchScheduler::update_resource_stats() {
    stats_.gpu_memory_usage.store(get_gpu_memory_usage());
    stats_.current_executing_requests.store(static_cast<int>(active_requests_.size()));
}

// === 工具函数实现 ===

namespace scheduler_utils {

std::string priority_to_string(RequestPriority priority) {
    switch (priority) {
        case RequestPriority::LOW: return "LOW";
        case RequestPriority::NORMAL: return "NORMAL";
        case RequestPriority::HIGH: return "HIGH";
        case RequestPriority::URGENT: return "URGENT";
        default: return "UNKNOWN";
    }
}

RequestPriority string_to_priority(const std::string& priority_str) {
    if (priority_str == "LOW") return RequestPriority::LOW;
    if (priority_str == "NORMAL") return RequestPriority::NORMAL;
    if (priority_str == "HIGH") return RequestPriority::HIGH;
    if (priority_str == "URGENT") return RequestPriority::URGENT;
    return RequestPriority::NORMAL;
}

std::string request_status_to_string(RequestStatus status) {
    switch (status) {
        case RequestStatus::QUEUED: return "QUEUED";
        case RequestStatus::BATCHED: return "BATCHED";
        case RequestStatus::EXECUTING: return "EXECUTING";
        case RequestStatus::STREAMING: return "STREAMING";
        case RequestStatus::COMPLETED: return "COMPLETED";
        case RequestStatus::FAILED: return "FAILED";
        case RequestStatus::CANCELLED: return "CANCELLED";
        case RequestStatus::TIMEOUT: return "TIMEOUT";
        default: return "UNKNOWN";
    }
}

double duration_to_milliseconds(const std::chrono::steady_clock::duration& duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

bool is_request_expired(const ScheduledRequest& request) {
    return std::chrono::steady_clock::now() > request.deadline;
}

double calculate_batch_efficiency(const BatchInfo& batch) {
    if (batch.requests.empty()) return 0.0;
    
    // 批次效率 = 实际处理的请求数 / 最大批次大小
    return static_cast<double>(batch.current_size) / batch.max_batch_size;
}

} // namespace scheduler_utils

} // namespace wicore 