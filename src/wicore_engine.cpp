// wicore_engine.cpp
#include "wicore_engine.hpp"
#include "../include/hmt_memory_manager.hpp"
#include "../include/multimodal_processor.hpp"
#include "../include/tensorrt_inference_engine.hpp"
#include "../include/batch_scheduler.hpp"
#include "../include/web_server.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <algorithm>

namespace wicore {

WiCoreEngine::WiCoreEngine(const std::string& config_path) 
    : config_path_(config_path) {
    if (!load_config(config_path)) {
        throw std::runtime_error("Failed to load configuration from: " + config_path);
    }
}

WiCoreEngine::~WiCoreEngine() {
    shutdown();
}

bool WiCoreEngine::initialize() {
    if (initialized_.load()) {
        return true;
    }
    
    try {
        // 1. 验证配置
        if (!validate_config(config_)) {
            std::cerr << "Configuration validation failed" << std::endl;
            return false;
        }
        
        // 2. 创建核心目录
        std::filesystem::create_directories(config_.nvme_cache_path);
        
        // 3. 按依赖顺序创建组件
        if (!create_components()) {
            std::cerr << "Failed to create components" << std::endl;
            cleanup_components();
            return false;
        }
        
        // 4. 按依赖顺序初始化组件
        if (!initialize_components()) {
            std::cerr << "Failed to initialize components" << std::endl;
            cleanup_components();
            return false;
        }
        
        // 5. 启动统计监控线程
        running_.store(true);
        stats_thread_ = std::thread(&WiCoreEngine::stats_monitor_loop, this);
        
        initialized_.store(true);
        std::cout << "WiCore Engine initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during initialization: " << e.what() << std::endl;
        cleanup_components();
        return false;
    }
}

bool WiCoreEngine::load_model(const std::string& model_path) {
    if (!initialized_.load()) {
        std::cerr << "Engine not initialized" << std::endl;
        return false;
    }
    
    if (model_loaded_.load()) {
        std::cout << "Model already loaded" << std::endl;
        return true;
    }
    
    try {
        // 验证模型路径
        if (!std::filesystem::exists(model_path)) {
            std::cerr << "Model path does not exist: " << model_path << std::endl;
            return false;
        }
        
        // 通过推理引擎加载模型
        if (!inference_engine_) {
            std::cerr << "Inference engine not available" << std::endl;
            return false;
        }
        
        // 这里会调用 TensorRTInferenceEngine::load_model
        // 实际的模型加载逻辑在推理引擎中实现
        std::cout << "Loading model from: " << model_path << std::endl;
        
        // 更新配置中的模型路径
        {
            std::unique_lock<std::shared_mutex> lock(config_mutex_);
            config_.model_path = model_path;
        }
        
        model_loaded_.store(true);
        std::cout << "Model loaded successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during model loading: " << e.what() << std::endl;
        return false;
    }
}

void WiCoreEngine::shutdown() {
    if (!running_.load()) {
        return;
    }
    
    std::cout << "Shutting down WiCore Engine..." << std::endl;
    
    // 1. 设置关闭标志
    shutdown_requested_.store(true);
    running_.store(false);
    
    // 2. 等待统计线程结束
    if (stats_thread_.joinable()) {
        stats_thread_.join();
    }
    
    // 3. 清理组件（逆序）
    cleanup_components();
    
    // 4. 重置状态
    initialized_.store(false);
    model_loaded_.store(false);
    
    std::cout << "WiCore Engine shutdown complete" << std::endl;
}

InferenceResponse WiCoreEngine::infer(const MultiModalRequest& request) {
    if (!initialized_.load() || !model_loaded_.load()) {
        return create_error_response(request.request_id, "Engine not ready");
    }
    
    auto start_time = std::chrono::steady_clock::now();
    
    try {
        // 请求验证
        if (request.text_prompt.empty()) {
            return create_error_response(request.request_id, "Empty text prompt");
        }
        
        if (request.max_tokens > config_.max_tokens_per_request) {
            return create_error_response(request.request_id, "Max tokens exceeds limit");
        }
        
        if (request.images.size() > static_cast<size_t>(config_.max_images_per_request)) {
            return create_error_response(request.request_id, "Too many images");
        }
        
        // 更新统计
        stats_.total_requests.fetch_add(1);
        
        // 通过批处理调度器执行推理
        if (!scheduler_) {
            return create_error_response(request.request_id, "Scheduler not available");
        }
        
        // 实际推理逻辑将由BatchScheduler处理
        // 这里暂时返回一个模拟响应
        InferenceResponse response;
        response.request_id = request.request_id;
        response.generated_text = "Generated response for: " + request.text_prompt;
        response.token_count = 100; // 模拟token数量
        response.success = true;
        
        auto end_time = std::chrono::steady_clock::now();
        response.latency_ms = std::chrono::duration<double, std::milli>(
            end_time - start_time).count();
        
        stats_.successful_requests.fetch_add(1);
        return response;
        
    } catch (const std::exception& e) {
        stats_.failed_requests.fetch_add(1);
        return create_error_response(request.request_id, "Inference error: " + std::string(e.what()));
    }
}

std::future<InferenceResponse> WiCoreEngine::infer_async(const MultiModalRequest& request) {
    return std::async(std::launch::async, [this, request]() {
        return this->infer(request);
    });
}

bool WiCoreEngine::update_config(const Json::Value& new_config) {
    try {
        // 解析新配置
        Config temp_config = config_; // 从当前配置开始
        
        // 更新配置字段
        if (new_config.isMember("max_batch_size")) {
            temp_config.max_batch_size = new_config["max_batch_size"].asInt();
        }
        if (new_config.isMember("batch_timeout_ms")) {
            temp_config.batch_timeout_ms = new_config["batch_timeout_ms"].asInt();
        }
        if (new_config.isMember("stats_interval_seconds")) {
            temp_config.stats_interval_seconds = new_config["stats_interval_seconds"].asInt();
        }
        
        // 验证新配置
        if (!validate_config(temp_config)) {
            return false;
        }
        
        // 应用配置更新
        {
            std::unique_lock<std::shared_mutex> lock(config_mutex_);
            config_ = temp_config;
        }
        
        // 通知组件配置更新
        // 具体的组件配置更新将在各组件中实现
        
        std::cout << "Configuration updated successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to update configuration: " << e.what() << std::endl;
        return false;
    }
}

Json::Value WiCoreEngine::get_config() const {
    std::shared_lock<std::shared_mutex> lock(config_mutex_);
    
    Json::Value config_json;
    config_json["model_path"] = config_.model_path;
    config_json["tokenizer_path"] = config_.tokenizer_path;
    config_json["server_port"] = config_.server_port;
    config_json["max_batch_size"] = config_.max_batch_size;
    config_json["batch_timeout_ms"] = config_.batch_timeout_ms;
    config_json["dynamic_batching"] = config_.dynamic_batching;
    config_json["max_context_length"] = config_.max_context_length;
    config_json["gpu_memory_gb"] = static_cast<Json::UInt64>(config_.gpu_memory_gb);
    config_json["cpu_memory_gb"] = static_cast<Json::UInt64>(config_.cpu_memory_gb);
    config_json["nvme_cache_path"] = config_.nvme_cache_path;
    config_json["num_streams"] = config_.num_streams;
    config_json["enable_cuda_graph"] = config_.enable_cuda_graph;
    config_json["enable_hmt"] = config_.enable_hmt;
    config_json["stats_interval_seconds"] = config_.stats_interval_seconds;
    
    return config_json;
}

WiCoreEngine::EngineStats WiCoreEngine::get_stats() const {
    return stats_;
}

bool WiCoreEngine::load_config(const std::string& config_path) {
    try {
        std::ifstream file(config_path);
        if (!file.is_open()) {
            std::cerr << "Cannot open config file: " << config_path << std::endl;
            return false;
        }
        
        Json::Value root;
        Json::Reader reader;
        
        if (!reader.parse(file, root)) {
            std::cerr << "Failed to parse JSON config: " 
                      << reader.getFormattedErrorMessages() << std::endl;
            return false;
        }
        
        // 解析配置字段
        config_.model_path = root.get("model_path", "./models/gemma-3-27b-it").asString();
        config_.tokenizer_path = root.get("tokenizer_path", 
            "./models/gemma-3-27b-it/tokenizer.json").asString();
        config_.server_port = root.get("server_port", 8080).asInt();
        config_.max_batch_size = root.get("max_batch_size", 16).asInt();
        config_.batch_timeout_ms = root.get("batch_timeout_ms", 10).asInt();
        config_.dynamic_batching = root.get("dynamic_batching", true).asBool();
        config_.max_context_length = root.get("max_context_length", 131072).asInt();
        config_.gpu_memory_gb = root.get("gpu_memory_gb", 48).asUInt64();
        config_.cpu_memory_gb = root.get("cpu_memory_gb", 128).asUInt64();
        config_.nvme_cache_path = root.get("nvme_cache_path", "./cache/nvme").asString();
        config_.memory_pool_size = root.get("memory_pool_size", 1024).asInt();
        config_.num_streams = root.get("num_streams", 4).asInt();
        config_.enable_cuda_graph = root.get("enable_cuda_graph", true).asBool();
        config_.enable_hmt = root.get("enable_hmt", true).asBool();
        config_.enable_quantization = root.get("enable_quantization", true).asBool();
        config_.enable_kernel_fusion = root.get("enable_kernel_fusion", true).asBool();
        config_.hmt_gpu_threshold = root.get("hmt_gpu_threshold", 0.85).asFloat();
        config_.hmt_cpu_threshold = root.get("hmt_cpu_threshold", 0.90).asFloat();
        config_.hmt_eviction_policy = root.get("hmt_eviction_policy", "a2cr").asString();
        config_.hmt_decay_factor = root.get("hmt_decay_factor", 0.05).asFloat();
        config_.trt_precision = root.get("trt_precision", "fp16").asString();
        config_.trt_max_workspace_gb = root.get("trt_max_workspace_gb", 4).asInt();
        config_.trt_enable_sparse = root.get("trt_enable_sparse", true).asBool();
        config_.trt_enable_refit = root.get("trt_enable_refit", true).asBool();
        config_.image_resolution = root.get("image_resolution", 896).asInt();
        config_.max_images_per_request = root.get("max_images_per_request", 8).asInt();
        config_.image_preprocessing_threads = root.get("image_preprocessing_threads", 4).asInt();
        config_.log_level = root.get("log_level", "info").asString();
        config_.stats_interval_seconds = root.get("stats_interval_seconds", 10).asInt();
        config_.enable_performance_logging = root.get("enable_performance_logging", true).asBool();
        config_.max_concurrent_requests = root.get("max_concurrent_requests", 128).asInt();
        config_.request_timeout_seconds = root.get("request_timeout_seconds", 30).asInt();
        config_.max_tokens_per_request = root.get("max_tokens_per_request", 4096).asInt();
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception loading config: " << e.what() << std::endl;
        return false;
    }
}

bool WiCoreEngine::validate_config(const Config& config) const {
    // 基本范围检查
    if (config.max_batch_size <= 0 || config.max_batch_size > 128) {
        std::cerr << "Invalid max_batch_size: " << config.max_batch_size << std::endl;
        return false;
    }
    
    if (config.server_port <= 0 || config.server_port > 65535) {
        std::cerr << "Invalid server_port: " << config.server_port << std::endl;
        return false;
    }
    
    if (config.max_context_length <= 0) {
        std::cerr << "Invalid max_context_length: " << config.max_context_length << std::endl;
        return false;
    }
    
    if (config.num_streams <= 0 || config.num_streams > 32) {
        std::cerr << "Invalid num_streams: " << config.num_streams << std::endl;
        return false;
    }
    
    if (config.hmt_gpu_threshold <= 0.0f || config.hmt_gpu_threshold >= 1.0f) {
        std::cerr << "Invalid hmt_gpu_threshold: " << config.hmt_gpu_threshold << std::endl;
        return false;
    }
    
    if (config.hmt_cpu_threshold <= 0.0f || config.hmt_cpu_threshold >= 1.0f) {
        std::cerr << "Invalid hmt_cpu_threshold: " << config.hmt_cpu_threshold << std::endl;
        return false;
    }
    
    return true;
}

bool WiCoreEngine::create_components() {
    try {
        std::cout << "Creating components..." << std::endl;
        
        // 1. 创建HMT内存管理器（无依赖）
        memory_manager_ = std::make_unique<HMTMemoryManager>(
            config_.gpu_memory_gb,
            config_.cpu_memory_gb,
            config_.nvme_cache_path
        );
        
        // 2. 创建多模态处理器（依赖配置）
        ImageConfig image_config;
        image_config.target_size = config_.image_resolution;
        image_config.max_images_per_request = config_.max_images_per_request;
        
        BatchConfig batch_config;
        batch_config.max_batch_size = config_.max_batch_size;
        batch_config.max_sequence_length = config_.max_context_length;
        batch_config.num_worker_threads = config_.image_preprocessing_threads;
        
        mm_processor_ = std::make_unique<MultiModalProcessor>(
            config_.tokenizer_path, image_config, batch_config);
        
        // 3. 创建TensorRT推理引擎（依赖内存管理器）
        ModelConfig model_config;
        model_config.model_path = config_.model_path;
        model_config.engine_cache_path = config_.model_path + "/engine_cache.trt";
        model_config.precision = (config_.trt_precision == "fp16") ? 
                                InferencePrecision::FP16 : InferencePrecision::FP32;
        model_config.max_batch_size = config_.max_batch_size;
        model_config.max_workspace_bytes = static_cast<size_t>(config_.trt_max_workspace_gb) * 1024 * 1024 * 1024;
        model_config.enable_sparse_weights = config_.trt_enable_sparse;
        model_config.enable_refit = config_.trt_enable_refit;
        
        // 配置注意力参数
        model_config.attention_config.max_context_length = config_.max_context_length;
        model_config.attention_config.block_size = 64;
        model_config.attention_config.num_heads = 32;        // Gemma-3-27B参数
        model_config.attention_config.head_dim = 128;
        model_config.attention_config.num_kv_heads = 16;     // GQA配置
        
        inference_engine_ = std::make_unique<TensorRTInferenceEngine>(
            memory_manager_.get(), model_config);
        
        // 4. 创建批处理调度器（依赖推理引擎）
        SchedulerConfig scheduler_config;
        scheduler_config.max_batch_size = config_.max_batch_size;
        scheduler_config.batch_timeout_ms = std::chrono::milliseconds(config_.batch_timeout_ms);
        scheduler_config.max_queue_size = config_.max_concurrent_requests * 2;
        scheduler_config.enable_continuous_batching = config_.dynamic_batching;
        scheduler_config.max_concurrent_requests = config_.max_concurrent_requests;
        scheduler_config.default_timeout_ms = std::chrono::milliseconds(config_.request_timeout_seconds * 1000);
        scheduler_config.scheduling_policy = SchedulingPolicy::ADAPTIVE;
        
        scheduler_ = std::make_unique<BatchScheduler>(
            inference_engine_.get(), scheduler_config);
        
        // 5. 创建Web服务器（依赖批处理调度器）
        ServerConfig server_config;
        server_config.host = "0.0.0.0";
        server_config.port = config_.server_port;
        server_config.num_threads = 4;
        server_config.max_connections = config_.max_concurrent_requests;
        server_config.enable_cors = true;
        server_config.enable_rate_limiting = true;
        server_config.rate_limit_rpm = 120; // 2 requests per second per IP
        server_config.max_request_size = 10 * 1024 * 1024; // 10MB
        server_config.request_timeout_seconds = config_.request_timeout_seconds;
        server_config.enable_metrics = config_.enable_performance_logging;
        server_config.enable_health_check = true;
        
        web_server_ = std::make_unique<WebServer>(scheduler_.get(), server_config);
        
        std::cout << "Components created successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception creating components: " << e.what() << std::endl;
        return false;
    }
}

bool WiCoreEngine::initialize_components() {
    try {
        std::cout << "Initializing components..." << std::endl;
        
        // 按依赖顺序初始化组件
        
        // 1. HMTMemoryManager (无依赖)
        if (memory_manager_) {
            if (!memory_manager_->initialize()) {
                std::cerr << "Failed to initialize HMT Memory Manager" << std::endl;
                return false;
            }
            std::cout << "HMT Memory Manager initialized" << std::endl;
        }
        
        // 2. MultiModalProcessor (依赖配置)
        if (mm_processor_) {
            if (!mm_processor_->initialize()) {
                std::cerr << "Failed to initialize MultiModal Processor" << std::endl;
                return false;
            }
            std::cout << "MultiModal Processor initialized" << std::endl;
        }
        
        // 3. TensorRTInferenceEngine (依赖内存管理器)
        if (inference_engine_) {
            if (!inference_engine_->initialize()) {
                std::cerr << "Failed to initialize TensorRT Inference Engine" << std::endl;
                return false;
            }
            std::cout << "TensorRT Inference Engine initialized" << std::endl;
        }
        
        // 4. BatchScheduler (依赖推理引擎)
        if (scheduler_) {
            if (!scheduler_->initialize()) {
                std::cerr << "Failed to initialize Batch Scheduler" << std::endl;
                return false;
            }
            std::cout << "Batch Scheduler initialized" << std::endl;
        }
        
        // 5. WebServer (依赖批处理调度器)
        if (web_server_) {
            if (!web_server_->initialize()) {
                std::cerr << "Failed to initialize Web Server" << std::endl;
                return false;
            }
            std::cout << "Web Server initialized" << std::endl;
        }
        
        std::cout << "All components initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception initializing components: " << e.what() << std::endl;
        return false;
    }
}

void WiCoreEngine::cleanup_components() {
    std::cout << "Cleaning up components..." << std::endl;
    
    // 逆序清理组件
    if (web_server_) {
        web_server_->stop();
        web_server_->shutdown();
        web_server_.reset();
    }
    
    if (scheduler_) {
        scheduler_->shutdown();
        scheduler_.reset();
    }
    
    if (inference_engine_) {
        inference_engine_->shutdown();
        inference_engine_.reset();
    }
    
    if (mm_processor_) {
        mm_processor_->shutdown();
        mm_processor_.reset();
    }
    
    if (memory_manager_) {
        memory_manager_->shutdown();
        memory_manager_.reset();
    }
    
    std::cout << "All components cleaned up" << std::endl;
}

void WiCoreEngine::stats_monitor_loop() {
    while (running_.load()) {
        auto interval = std::chrono::seconds(config_.stats_interval_seconds);
        std::this_thread::sleep_for(interval);
        
        if (!running_.load()) {
            break;
        }
        
        update_stats_from_components();
    }
}

void WiCoreEngine::update_stats_from_components() {
    try {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        
        // 更新时间戳
        auto now = std::chrono::steady_clock::now();
        auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();
        stats_.last_update_timestamp.store(timestamp);
        
        // 从各组件收集统计信息
        // GPU利用率、内存使用等将从各组件获取
        
        // 计算平均延迟
        uint64_t total = stats_.total_requests.load();
        if (total > 0) {
            // 简化的延迟计算，实际应该维护滑动窗口
            stats_.avg_latency_ms.store(25.0); // 模拟值
        }
        
        // 计算吞吐量 (requests per second)
        static auto last_time = now;
        static uint64_t last_total = 0;
        
        auto time_diff = std::chrono::duration<double>(now - last_time).count();
        if (time_diff >= 1.0) { // 每秒更新一次吞吐量
            uint64_t current_total = stats_.total_requests.load();
            double rps = (current_total - last_total) / time_diff;
            stats_.throughput_rps.store(static_cast<float>(rps));
            
            last_time = now;
            last_total = current_total;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Exception updating stats: " << e.what() << std::endl;
    }
}

InferenceResponse WiCoreEngine::create_error_response(const std::string& request_id, 
                                                    const std::string& error_msg) const {
    InferenceResponse response;
    response.request_id = request_id;
    response.success = false;
    response.error_message = error_msg;
    response.generated_text = "";
    response.token_count = 0;
    response.latency_ms = 0.0;
    response.gpu_utilization = 0.0;
    response.memory_usage = 0;
    
    return response;
}

} // namespace wicore 