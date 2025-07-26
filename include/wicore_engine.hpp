// wicore_engine.hpp
#ifndef WICORE_ENGINE_HPP
#define WICORE_ENGINE_HPP

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>
#include <chrono>

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <json/json.h>

namespace wicore {

// 前向声明
class HMTMemoryManager;
class TensorRTInferenceEngine;
class MultiModalProcessor;
class BatchScheduler;
class WebServer;

// 多模态输入请求
struct MultiModalRequest {
    std::string request_id;
    std::string text_prompt;
    std::vector<cv::Mat> images;
    int max_tokens = 2048;
    float temperature = 0.7f;
    int top_k = 50;
    float top_p = 0.9f;
    std::chrono::steady_clock::time_point timestamp;
};

// 推理响应
struct InferenceResponse {
    std::string request_id;
    std::string generated_text;
    int token_count = 0;
    double latency_ms = 0.0;
    double gpu_utilization = 0.0;
    size_t memory_usage = 0;
    bool success = false;
    std::string error_message;
};

// WiCore主引擎
class WiCoreEngine {
public:
    explicit WiCoreEngine(const std::string& config_path);
    ~WiCoreEngine();
    
    // 核心生命周期接口
    bool initialize();
    bool load_model(const std::string& model_path);
    void shutdown();
    
    // 推理接口
    InferenceResponse infer(const MultiModalRequest& request);
    std::future<InferenceResponse> infer_async(const MultiModalRequest& request);
    
    // 配置管理
    bool update_config(const Json::Value& new_config);
    Json::Value get_config() const;
    
    // 性能统计
    struct EngineStats {
        std::atomic<uint64_t> total_requests{0};
        std::atomic<uint64_t> successful_requests{0};
        std::atomic<uint64_t> failed_requests{0};
        std::atomic<double> avg_latency_ms{0.0};
        std::atomic<float> gpu_utilization{0.0f};
        std::atomic<size_t> memory_usage_bytes{0};
        std::atomic<float> throughput_rps{0.0f};
        std::atomic<uint64_t> last_update_timestamp{0};
    };
    
    EngineStats get_stats() const;
    
    // 状态查询
    bool is_initialized() const { return initialized_.load(); }
    bool is_model_loaded() const { return model_loaded_.load(); }
    bool is_running() const { return running_.load(); }

private:
    // 配置结构
    struct Config {
        // 模型配置
        std::string model_path;
        std::string tokenizer_path;
        
        // 服务配置
        int server_port = 8080;
        int max_batch_size = 16;
        int batch_timeout_ms = 10;
        bool dynamic_batching = true;
        
        // 内存配置
        int max_context_length = 131072;
        size_t gpu_memory_gb = 48;
        size_t cpu_memory_gb = 128;
        std::string nvme_cache_path = "./cache/nvme";
        int memory_pool_size = 1024;
        
        // 性能配置
        int num_streams = 4;
        bool enable_cuda_graph = true;
        bool enable_hmt = true;
        bool enable_quantization = true;
        bool enable_kernel_fusion = true;
        
        // HMT配置
        float hmt_gpu_threshold = 0.85f;
        float hmt_cpu_threshold = 0.90f;
        std::string hmt_eviction_policy = "a2cr";
        float hmt_decay_factor = 0.05f;
        
        // TensorRT配置
        std::string trt_precision = "fp16";
        int trt_max_workspace_gb = 4;
        bool trt_enable_sparse = true;
        bool trt_enable_refit = true;
        
        // 多模态配置
        int image_resolution = 896;
        int max_images_per_request = 8;
        int image_preprocessing_threads = 4;
        
        // 日志和监控配置
        std::string log_level = "info";
        int stats_interval_seconds = 10;
        bool enable_performance_logging = true;
        
        // 资源限制
        int max_concurrent_requests = 128;
        int request_timeout_seconds = 30;
        int max_tokens_per_request = 4096;
    };
    
    // 核心状态
    Config config_;
    std::string config_path_;
    std::atomic<bool> initialized_{false};
    std::atomic<bool> model_loaded_{false};
    std::atomic<bool> running_{false};
    std::atomic<bool> shutdown_requested_{false};
    
    // 核心组件
    std::unique_ptr<HMTMemoryManager> memory_manager_;
    std::unique_ptr<MultiModalProcessor> mm_processor_;
    std::unique_ptr<TensorRTInferenceEngine> inference_engine_;
    std::unique_ptr<BatchScheduler> scheduler_;
    std::unique_ptr<WebServer> web_server_;
    
    // 性能统计
    mutable EngineStats stats_;
    mutable std::mutex stats_mutex_;
    std::thread stats_thread_;
    
    // 配置管理
    mutable std::shared_mutex config_mutex_;
    
    // 内部方法
    bool load_config(const std::string& config_path);
    bool validate_config(const Config& config) const;
    bool create_components();
    bool initialize_components();
    void cleanup_components();
    
    void stats_monitor_loop();
    void update_stats_from_components();
    
    InferenceResponse create_error_response(const std::string& request_id, 
                                          const std::string& error_msg) const;
    
    // 禁用拷贝和赋值
    WiCoreEngine(const WiCoreEngine&) = delete;
    WiCoreEngine& operator=(const WiCoreEngine&) = delete;
};

} // namespace wicore

#endif // WICORE_ENGINE_HPP 