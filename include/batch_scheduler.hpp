// batch_scheduler.hpp
#ifndef BATCH_SCHEDULER_HPP
#define BATCH_SCHEDULER_HPP

#include <memory>
#include <string>
#include <vector>
#include <queue>
#include <deque>
#include <array>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <future>
#include <condition_variable>
#include <chrono>
#include <functional>

#include "tensorrt_inference_engine.hpp"
#include "multimodal_processor.hpp"

namespace wicore {

// 前向声明
class TensorRTInferenceEngine;

// 请求优先级枚举
enum class RequestPriority {
    LOW = 0,
    NORMAL = 1,
    HIGH = 2,
    URGENT = 3
};

// 调度策略枚举
enum class SchedulingPolicy {
    FCFS,           // First Come First Serve
    PRIORITY,       // Priority-based
    SJF,            // Shortest Job First
    ADAPTIVE        // Adaptive scheduling
};

// 请求状态枚举
enum class RequestStatus {
    QUEUED,         // 在队列中等待
    BATCHED,        // 已加入批次
    EXECUTING,      // 正在执行
    STREAMING,      // 流式输出中
    COMPLETED,      // 已完成
    FAILED,         // 执行失败
    CANCELLED,      // 已取消
    TIMEOUT         // 超时
};

// 批次状态枚举
enum class BatchStatus {
    PREPARING,      // 准备中
    EXECUTING,      // 执行中
    COMPLETED,      // 已完成
    FAILED          // 失败
};

// 调度请求结构
struct ScheduledRequest {
    std::string request_id;
    ProcessedMultiModal input_data;
    RequestPriority priority = RequestPriority::NORMAL;
    int max_new_tokens = 512;
    float temperature = 1.0f;
    float top_p = 0.9f;
    int top_k = 50;
    bool stream_output = false;
    
    // 时间相关
    std::chrono::steady_clock::time_point submit_time;
    std::chrono::steady_clock::time_point deadline;
    std::chrono::milliseconds timeout_ms{30000}; // 30秒默认超时
    
    // 预估信息
    int estimated_tokens = 512;
    double estimated_duration_ms = 1000.0;
    
    // 状态管理
    RequestStatus status = RequestStatus::QUEUED;
    std::atomic<bool> is_cancelled{false};
    
    // 异步结果
    std::promise<InferenceResponse> promise;
    std::future<InferenceResponse> future;
    
    // 流式输出回调
    std::function<void(const std::string&)> stream_callback;
    
    ScheduledRequest(const std::string& id, const ProcessedMultiModal& data)
        : request_id(id), input_data(data), future(promise.get_future()) {
        submit_time = std::chrono::steady_clock::now();
        deadline = submit_time + timeout_ms;
    }
};

// 批次信息结构
struct BatchInfo {
    std::string batch_id;
    std::vector<std::shared_ptr<ScheduledRequest>> requests;
    BatchStatus status = BatchStatus::PREPARING;
    int max_batch_size = 16;
    int current_size = 0;
    
    // 时间信息
    std::chrono::steady_clock::time_point created_time;
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
    
    // 预估信息
    double estimated_total_duration_ms = 0.0;
    int total_estimated_tokens = 0;
    
    BatchInfo(const std::string& id) : batch_id(id) {
        created_time = std::chrono::steady_clock::now();
    }
};

// 调度器配置
struct SchedulerConfig {
    // 基本配置
    int max_batch_size = 16;
    int min_batch_size = 1;
    int max_queue_size = 1000;
    std::chrono::milliseconds batch_timeout_ms{10};
    std::chrono::milliseconds max_wait_time_ms{100};
    
    // 调度策略
    SchedulingPolicy scheduling_policy = SchedulingPolicy::ADAPTIVE;
    bool enable_continuous_batching = true;
    bool enable_preemption = false;
    
    // 资源限制
    int max_concurrent_batches = 4;
    int max_concurrent_requests = 64;
    double max_gpu_memory_usage = 0.9;
    double max_cpu_memory_usage = 0.8;
    
    // 自适应参数
    bool enable_adaptive_batching = true;
    double load_factor_threshold = 0.8;
    int adaptive_window_size = 100;
    
    // 超时和重试
    std::chrono::milliseconds default_timeout_ms{30000};
    int max_retry_attempts = 3;
    std::chrono::milliseconds retry_delay_ms{1000};
    
    // 性能优化
    bool enable_request_caching = true;
    bool enable_batch_prediction = true;
    int prefetch_queue_size = 32;
};

// 性能统计
struct SchedulerStats {
    // 请求统计
    std::atomic<uint64_t> total_requests{0};
    std::atomic<uint64_t> successful_requests{0};
    std::atomic<uint64_t> failed_requests{0};
    std::atomic<uint64_t> cancelled_requests{0};
    std::atomic<uint64_t> timeout_requests{0};
    
    // 批次统计
    std::atomic<uint64_t> total_batches{0};
    std::atomic<uint64_t> successful_batches{0};
    std::atomic<uint64_t> failed_batches{0};
    
    // 队列统计
    std::atomic<int> current_queue_size{0};
    std::atomic<int> current_batch_count{0};
    std::atomic<int> current_executing_requests{0};
    
    // 延迟统计
    std::atomic<double> avg_queue_wait_time_ms{0.0};
    std::atomic<double> avg_batch_execution_time_ms{0.0};
    std::atomic<double> avg_end_to_end_latency_ms{0.0};
    
    // 吞吐量统计
    std::atomic<double> requests_per_second{0.0};
    std::atomic<double> tokens_per_second{0.0};
    std::atomic<double> batch_utilization_rate{0.0};
    
    // 资源使用
    std::atomic<double> gpu_memory_usage{0.0};
    std::atomic<double> cpu_memory_usage{0.0};
    std::atomic<double> gpu_utilization{0.0};
};

// 负载预测器
class LoadPredictor {
public:
    LoadPredictor(int window_size = 100);
    
    void record_request(const ScheduledRequest& request);
    void record_batch_completion(const BatchInfo& batch);
    
    double predict_batch_duration(const std::vector<std::shared_ptr<ScheduledRequest>>& requests);
    int predict_optimal_batch_size();
    bool should_wait_for_more_requests();
    
    double get_current_load_factor() const;
    double get_average_request_rate() const;

private:
    int window_size_;
    std::deque<double> recent_durations_;
    std::deque<std::chrono::steady_clock::time_point> recent_arrivals_;
    mutable std::mutex predictor_mutex_;
    
    double calculate_moving_average(const std::deque<double>& values) const;
    double calculate_request_rate() const;
};

// 连续批处理调度器
class BatchScheduler {
public:
    explicit BatchScheduler(TensorRTInferenceEngine* inference_engine,
                           const SchedulerConfig& config = SchedulerConfig{});
    ~BatchScheduler();
    
    // 核心接口
    bool initialize();
    void shutdown();
    
    // 请求提交
    std::future<InferenceResponse> submit_request(const ProcessedMultiModal& input,
                                                 RequestPriority priority = RequestPriority::NORMAL,
                                                 const std::chrono::milliseconds& timeout = std::chrono::milliseconds{30000});
    
    // 流式请求提交
    std::future<InferenceResponse> submit_streaming_request(
        const ProcessedMultiModal& input,
        std::function<void(const std::string&)> stream_callback,
        RequestPriority priority = RequestPriority::NORMAL,
        const std::chrono::milliseconds& timeout = std::chrono::milliseconds{30000});
    
    // 请求管理
    bool cancel_request(const std::string& request_id);
    RequestStatus get_request_status(const std::string& request_id);
    std::vector<std::string> get_active_requests();
    
    // 配置管理
    void update_config(const SchedulerConfig& config);
    SchedulerConfig get_config() const;
    
    // 状态查询
    bool is_ready() const { return initialized_.load() && !shutdown_requested_.load(); }
    int get_queue_size() const;
    int get_active_batch_count() const;
    
    // 性能统计
    SchedulerStats get_stats() const;
    void reset_stats();
    
    // 手动控制
    void pause_scheduling();
    void resume_scheduling();
    bool is_paused() const { return paused_.load(); }

private:
    // 依赖组件
    TensorRTInferenceEngine* inference_engine_;
    SchedulerConfig config_;
    std::unique_ptr<LoadPredictor> load_predictor_;
    
    // 状态管理
    std::atomic<bool> initialized_{false};
    std::atomic<bool> shutdown_requested_{false};
    std::atomic<bool> paused_{false};
    
    // 请求队列（按优先级分层）
    std::array<std::queue<std::shared_ptr<ScheduledRequest>>, 4> priority_queues_;
    std::unordered_map<std::string, std::shared_ptr<ScheduledRequest>> active_requests_;
    mutable std::shared_mutex requests_mutex_;
    std::condition_variable_any request_available_cv_;
    
    // 批次管理
    std::unordered_map<std::string, std::unique_ptr<BatchInfo>> active_batches_;
    std::queue<std::unique_ptr<BatchInfo>> pending_batches_;
    mutable std::mutex batches_mutex_;
    std::condition_variable batch_ready_cv_;
    
    // 工作线程
    std::thread scheduler_thread_;
    std::thread batch_executor_thread_;
    std::thread monitoring_thread_;
    std::vector<std::thread> worker_threads_;
    int num_worker_threads_ = 2;
    
    // 统计信息
    mutable SchedulerStats stats_;
    mutable std::mutex stats_mutex_;
    
    // 内部方法 - 调度逻辑
    void scheduler_main_loop();
    void batch_executor_main_loop();
    void monitoring_main_loop();
    void worker_thread_function(int worker_id);
    
    // 请求调度
    std::shared_ptr<ScheduledRequest> get_next_request();
    std::vector<std::shared_ptr<ScheduledRequest>> create_batch();
    bool should_create_batch_now();
    int calculate_optimal_batch_size();
    
    // 批次执行
    bool execute_batch(std::unique_ptr<BatchInfo> batch);
    bool preprocess_batch(BatchInfo* batch);
    bool run_inference_batch(BatchInfo* batch);
    bool postprocess_batch(BatchInfo* batch);
    
    // 连续批处理
    bool can_add_to_existing_batch(const std::shared_ptr<ScheduledRequest>& request);
    bool try_add_to_running_batch(const std::shared_ptr<ScheduledRequest>& request);
    void update_running_batches();
    
    // 请求管理
    void cleanup_completed_requests();
    void handle_timeout_requests();
    void handle_failed_requests();
    bool retry_request(std::shared_ptr<ScheduledRequest> request);
    
    // 负载均衡
    void adjust_batch_size_adaptively();
    void update_load_metrics();
    bool is_system_overloaded();
    void apply_backpressure();
    
    // 资源监控
    double get_gpu_memory_usage();
    double get_cpu_memory_usage();
    void update_resource_stats();
    
    // 工具方法
    std::string generate_batch_id();
    std::string generate_request_id();
    InferenceResponse create_error_response(const std::string& request_id, const std::string& error_msg);
    void update_request_stats(const ScheduledRequest& request, bool success);
    void update_batch_stats(const BatchInfo& batch, bool success);
    
    std::atomic<uint64_t> next_batch_id_{1};
    std::atomic<uint64_t> next_request_id_{1};
    
    // 禁用拷贝
    BatchScheduler(const BatchScheduler&) = delete;
    BatchScheduler& operator=(const BatchScheduler&) = delete;
};

// 工具函数
namespace scheduler_utils {
    // 优先级转换
    std::string priority_to_string(RequestPriority priority);
    RequestPriority string_to_priority(const std::string& priority_str);
    
    // 状态转换
    std::string request_status_to_string(RequestStatus status);
    std::string batch_status_to_string(BatchStatus status);
    
    // 时间工具
    double duration_to_milliseconds(const std::chrono::steady_clock::duration& duration);
    bool is_request_expired(const ScheduledRequest& request);
    
    // 负载计算
    double calculate_batch_efficiency(const BatchInfo& batch);
    double calculate_resource_pressure(double gpu_usage, double cpu_usage);
    
    // 调度策略
    int compare_requests(const std::shared_ptr<ScheduledRequest>& a, 
                        const std::shared_ptr<ScheduledRequest>& b,
                        SchedulingPolicy policy);
}

} // namespace wicore

#endif // BATCH_SCHEDULER_HPP 