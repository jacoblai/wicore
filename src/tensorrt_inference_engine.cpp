// tensorrt_inference_engine.cpp
#include "../include/tensorrt_inference_engine.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>

namespace wicore {

// === TensorRT日志器实现 ===

void TensorRTLogger::log(Severity severity, const char* msg) noexcept {
    const char* severity_str = severity_to_string(severity);
    std::cout << "[TensorRT:" << severity_str << "] " << msg << std::endl;
}

const char* TensorRTLogger::severity_to_string(Severity severity) {
    switch (severity) {
        case Severity::kINTERNAL_ERROR: return "INTERNAL_ERROR";
        case Severity::kERROR: return "ERROR";
        case Severity::kWARNING: return "WARNING";
        case Severity::kINFO: return "INFO";
        case Severity::kVERBOSE: return "VERBOSE";
        default: return "UNKNOWN";
    }
}

// === CUDA流管理器实现 ===

CudaStreamManager::CudaStreamManager(int num_streams) : num_streams_(num_streams) {
    streams_.reserve(num_streams_);
}

CudaStreamManager::~CudaStreamManager() {
    shutdown();
}

bool CudaStreamManager::initialize() {
    try {
        std::lock_guard<std::mutex> lock(streams_mutex_);
        
        // 创建CUDA流
        for (int i = 0; i < num_streams_; ++i) {
            cudaStream_t stream;
            cudaError_t result = cudaStreamCreate(&stream);
            if (result != cudaSuccess) {
                std::cerr << "Failed to create CUDA stream " << i 
                          << ": " << cudaGetErrorString(result) << std::endl;
                return false;
            }
            
            streams_.push_back(stream);
            available_streams_.push(stream);
            stream_states_[stream] = StreamState::IDLE;
        }
        
        std::cout << "CUDA Stream Manager initialized with " << num_streams_ << " streams" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception in CudaStreamManager::initialize: " << e.what() << std::endl;
        return false;
    }
}

void CudaStreamManager::shutdown() {
    std::lock_guard<std::mutex> lock(streams_mutex_);
    
    // 同步并销毁所有流
    for (auto stream : streams_) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
    
    streams_.clear();
    while (!available_streams_.empty()) {
        available_streams_.pop();
    }
    stream_states_.clear();
}

cudaStream_t CudaStreamManager::acquire_stream() {
    std::unique_lock<std::mutex> lock(streams_mutex_);
    
    stream_available_cv_.wait(lock, [this] {
        return !available_streams_.empty();
    });
    
    cudaStream_t stream = available_streams_.front();
    available_streams_.pop();
    stream_states_[stream] = StreamState::BUSY;
    
    return stream;
}

void CudaStreamManager::release_stream(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(streams_mutex_);
    
    stream_states_[stream] = StreamState::IDLE;
    available_streams_.push(stream);
    stream_available_cv_.notify_one();
}

void CudaStreamManager::set_stream_state(cudaStream_t stream, StreamState state) {
    std::lock_guard<std::mutex> lock(streams_mutex_);
    stream_states_[stream] = state;
}

StreamState CudaStreamManager::get_stream_state(cudaStream_t stream) const {
    std::lock_guard<std::mutex> lock(streams_mutex_);
    auto it = stream_states_.find(stream);
    return (it != stream_states_.end()) ? it->second : StreamState::IDLE;
}

void CudaStreamManager::synchronize_stream(cudaStream_t stream) {
    cudaStreamSynchronize(stream);
}

void CudaStreamManager::synchronize_all_streams() {
    std::lock_guard<std::mutex> lock(streams_mutex_);
    for (auto stream : streams_) {
        cudaStreamSynchronize(stream);
    }
}

int CudaStreamManager::get_active_stream_count() const {
    std::lock_guard<std::mutex> lock(streams_mutex_);
    return num_streams_ - static_cast<int>(available_streams_.size());
}

// === KV缓存管理器实现 ===

KVCacheManager::KVCacheManager(HMTMemoryManager* memory_manager, 
                               const AttentionConfig& config)
    : memory_manager_(memory_manager), config_(config) {
}

KVCacheManager::~KVCacheManager() {
    shutdown();
}

bool KVCacheManager::initialize() {
    if (!memory_manager_) {
        std::cerr << "Memory manager is null" << std::endl;
        return false;
    }
    
    // 预分配KV缓存块
    int total_blocks = (config_.max_context_length + config_.block_size - 1) / config_.block_size;
    total_blocks = std::min(total_blocks, 10000); // 限制最大块数
    
    all_blocks_.reserve(total_blocks);
    
    for (int i = 0; i < total_blocks; ++i) {
        auto block = std::make_unique<KVCacheBlock>();
        block->sequence_id = -1;
        block->is_active = false;
        free_blocks_.push(block.get());
        all_blocks_.push_back(std::move(block));
    }
    
    std::cout << "KV Cache Manager initialized with " << total_blocks << " blocks" << std::endl;
    return true;
}

void KVCacheManager::shutdown() {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    
    // 清理所有序列
    for (auto& [seq_id, blocks] : sequence_blocks_) {
        for (auto block : blocks) {
            if (block->key_block) {
                memory_manager_->deallocate_block(block->key_block);
            }
            if (block->value_block) {
                memory_manager_->deallocate_block(block->value_block);
            }
        }
    }
    
    sequence_blocks_.clear();
    all_blocks_.clear();
    while (!free_blocks_.empty()) {
        free_blocks_.pop();
    }
}

std::vector<KVCacheBlock*> KVCacheManager::allocate_kv_blocks(int sequence_id, int length) {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    
    int num_blocks = (length + config_.block_size - 1) / config_.block_size;
    std::vector<KVCacheBlock*> allocated_blocks;
    allocated_blocks.reserve(num_blocks);
    
    for (int i = 0; i < num_blocks; ++i) {
        if (free_blocks_.empty()) {
            // 如果没有空闲块，需要进行LRU置换
            break;
        }
        
        auto block = free_blocks_.front();
        free_blocks_.pop();
        
        // 分配内存
        block->key_block = memory_manager_->allocate_block(MemoryTier::GPU);
        block->value_block = memory_manager_->allocate_block(MemoryTier::GPU);
        
        if (!block->key_block || !block->value_block) {
            // 分配失败，归还块
            if (block->key_block) {
                memory_manager_->deallocate_block(block->key_block);
                block->key_block = nullptr;
            }
            free_blocks_.push(block);
            break;
        }
        
        block->sequence_id = sequence_id;
        block->start_pos = i * config_.block_size;
        block->length = std::min(config_.block_size, length - i * config_.block_size);
        block->is_active = true;
        block->last_access = std::chrono::steady_clock::now();
        
        allocated_blocks.push_back(block);
        total_allocated_blocks_.fetch_add(1);
    }
    
    if (!allocated_blocks.empty()) {
        sequence_blocks_[sequence_id] = allocated_blocks;
    }
    
    return allocated_blocks;
}

void KVCacheManager::deallocate_kv_blocks(const std::vector<KVCacheBlock*>& blocks) {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    
    for (auto block : blocks) {
        if (block->key_block) {
            memory_manager_->deallocate_block(block->key_block);
            block->key_block = nullptr;
        }
        if (block->value_block) {
            memory_manager_->deallocate_block(block->value_block);
            block->value_block = nullptr;
        }
        
        block->sequence_id = -1;
        block->is_active = false;
        free_blocks_.push(block);
        total_allocated_blocks_.fetch_sub(1);
    }
}

KVCacheBlock* KVCacheManager::get_kv_block(int sequence_id, int position) {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    
    auto it = sequence_blocks_.find(sequence_id);
    if (it == sequence_blocks_.end()) {
        cache_misses_.fetch_add(1);
        return nullptr;
    }
    
    int block_index = position / config_.block_size;
    if (block_index >= static_cast<int>(it->second.size())) {
        cache_misses_.fetch_add(1);
        return nullptr;
    }
    
    auto block = it->second[block_index];
    block->last_access = std::chrono::steady_clock::now();
    cache_hits_.fetch_add(1);
    
    return block;
}

void KVCacheManager::register_sequence(int sequence_id, int max_length) {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    sequence_blocks_[sequence_id] = std::vector<KVCacheBlock*>();
}

void KVCacheManager::unregister_sequence(int sequence_id) {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    
    auto it = sequence_blocks_.find(sequence_id);
    if (it != sequence_blocks_.end()) {
        deallocate_kv_blocks(it->second);
        sequence_blocks_.erase(it);
    }
}

int KVCacheManager::get_allocated_blocks_count() const {
    return total_allocated_blocks_.load();
}

double KVCacheManager::get_cache_hit_rate() const {
    uint64_t hits = cache_hits_.load();
    uint64_t misses = cache_misses_.load();
    uint64_t total = hits + misses;
    
    return (total > 0) ? static_cast<double>(hits) / total : 0.0;
}

// === TensorRT引擎实现 ===

TensorRTEngine::TensorRTEngine(const ModelConfig& config) : config_(config) {
}

TensorRTEngine::~TensorRTEngine() {
    shutdown();
}

bool TensorRTEngine::initialize() {
    try {
        // 创建TensorRT运行时
        static TensorRTLogger logger;
        runtime_.reset(nvinfer1::createInferRuntime(logger));
        if (!runtime_) {
            std::cerr << "Failed to create TensorRT runtime" << std::endl;
            return false;
        }
        
        std::cout << "TensorRT Engine initialized" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception in TensorRTEngine::initialize: " << e.what() << std::endl;
        return false;
    }
}

void TensorRTEngine::shutdown() {
    // 清理CUDA Graph
    if (cuda_graph_exec_) {
        cudaGraphExecDestroy(cuda_graph_exec_);
        cuda_graph_exec_ = nullptr;
    }
    if (cuda_graph_) {
        cudaGraphDestroy(cuda_graph_);
        cuda_graph_ = nullptr;
    }
    
    // 清理TensorRT对象
    context_.reset();
    engine_.reset();
    runtime_.reset();
}

bool TensorRTEngine::load_model(const std::string& model_path) {
    // 尝试从缓存加载
    std::string cache_path = config_.engine_cache_path;
    if (!cache_path.empty() && load_engine_from_cache(cache_path)) {
        std::cout << "Loaded TensorRT engine from cache: " << cache_path << std::endl;
        return true;
    }
    
    // 从ONNX构建引擎
    if (build_engine_from_onnx(model_path)) {
        if (!cache_path.empty()) {
            save_engine_to_cache(cache_path);
        }
        return true;
    }
    
    return false;
}

bool TensorRTEngine::build_engine_from_onnx(const std::string& onnx_path) {
    try {
        auto builder = create_builder();
        if (!builder) {
            return false;
        }
        
        auto network = create_network(builder.get());
        if (!network) {
            return false;
        }
        
        // 解析ONNX模型
        auto parser = std::unique_ptr<nvonnxparser::IParser>(
            nvonnxparser::createParser(*network, static_cast<nvinfer1::ILogger&>(*runtime_)));
        
        if (!parser->parseFromFile(onnx_path.c_str(), 
                                  static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
            std::cerr << "Failed to parse ONNX model: " << onnx_path << std::endl;
            return false;
        }
        
        // 优化网络
        if (!optimize_network(builder.get(), network.get())) {
            return false;
        }
        
        // 构建引擎
        auto serialized_engine = std::unique_ptr<nvinfer1::IHostMemory>(
            builder->buildSerializedNetwork(*network, *builder->createBuilderConfig()));
        
        if (!serialized_engine) {
            std::cerr << "Failed to build TensorRT engine" << std::endl;
            return false;
        }
        
        // 反序列化引擎
        engine_.reset(runtime_->deserializeCudaEngine(
            serialized_engine->data(), serialized_engine->size()));
        
        if (!engine_) {
            std::cerr << "Failed to deserialize TensorRT engine" << std::endl;
            return false;
        }
        
        // 创建执行上下文
        context_.reset(engine_->createExecutionContext());
        if (!context_) {
            std::cerr << "Failed to create execution context" << std::endl;
            return false;
        }
        
        std::cout << "TensorRT engine built successfully from ONNX: " << onnx_path << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception building TensorRT engine: " << e.what() << std::endl;
        return false;
    }
}

bool TensorRTEngine::load_engine_from_cache(const std::string& cache_path) {
    try {
        std::ifstream file(cache_path, std::ios::binary);
        if (!file.good()) {
            return false;
        }
        
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::vector<char> buffer(size);
        file.read(buffer.data(), size);
        
        engine_.reset(runtime_->deserializeCudaEngine(buffer.data(), size));
        if (!engine_) {
            return false;
        }
        
        context_.reset(engine_->createExecutionContext());
        return context_ != nullptr;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception loading engine from cache: " << e.what() << std::endl;
        return false;
    }
}

bool TensorRTEngine::save_engine_to_cache(const std::string& cache_path) {
    if (!engine_) {
        return false;
    }
    
    try {
        auto serialized_engine = std::unique_ptr<nvinfer1::IHostMemory>(
            engine_->serialize());
        
        std::ofstream file(cache_path, std::ios::binary);
        file.write(static_cast<const char*>(serialized_engine->data()), 
                  serialized_engine->size());
        
        return file.good();
        
    } catch (const std::exception& e) {
        std::cerr << "Exception saving engine to cache: " << e.what() << std::endl;
        return false;
    }
}

bool TensorRTEngine::execute_inference(cudaStream_t stream,
                                      const std::vector<void*>& input_buffers,
                                      const std::vector<void*>& output_buffers) {
    if (!context_) {
        return false;
    }
    
    // 如果有CUDA Graph且已捕获，使用Graph执行
    if (graph_captured_) {
        return execute_cuda_graph(stream);
    }
    
    // 设置绑定
    std::vector<void*> bindings;
    bindings.insert(bindings.end(), input_buffers.begin(), input_buffers.end());
    bindings.insert(bindings.end(), output_buffers.begin(), output_buffers.end());
    
    // 执行推理
    return context_->enqueueV2(bindings.data(), stream, nullptr);
}

std::unique_ptr<nvinfer1::IBuilder> TensorRTEngine::create_builder() {
    static TensorRTLogger logger;
    return std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
}

std::unique_ptr<nvinfer1::INetworkDefinition> TensorRTEngine::create_network(nvinfer1::IBuilder* builder) {
    const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    return std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(explicit_batch));
}

bool TensorRTEngine::optimize_network(nvinfer1::IBuilder* builder, nvinfer1::INetworkDefinition* network) {
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    
    // 设置最大工作空间
    config->setMaxWorkspaceSize(config_.max_workspace_bytes);
    
    // 设置精度
    if (config_.precision == InferencePrecision::FP16) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    } else if (config_.precision == InferencePrecision::INT8) {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
    }
    
    // 设置稀疏权重
    if (config_.enable_sparse_weights) {
        config->setFlag(nvinfer1::BuilderFlag::kSPARSE_WEIGHTS);
    }
    
    return true;
}

// === TensorRT推理引擎主类实现 ===

TensorRTInferenceEngine::TensorRTInferenceEngine(HMTMemoryManager* memory_manager,
                                               const ModelConfig& config)
    : memory_manager_(memory_manager), config_(config) {
    
    std::cout << "TensorRT Inference Engine created" << std::endl;
    std::cout << "Model path: " << config_.model_path << std::endl;
    std::cout << "Precision: " << (config_.precision == InferencePrecision::FP16 ? "FP16" : "FP32") << std::endl;
    std::cout << "Max batch size: " << config_.max_batch_size << std::endl;
}

TensorRTInferenceEngine::~TensorRTInferenceEngine() {
    shutdown();
}

bool TensorRTInferenceEngine::initialize() {
    if (initialized_.load()) {
        return true;
    }
    
    try {
        // 1. 初始化CUDA上下文
        if (!initialize_cuda_context()) {
            std::cerr << "Failed to initialize CUDA context" << std::endl;
            return false;
        }
        
        // 2. 初始化组件
        if (!initialize_components()) {
            std::cerr << "Failed to initialize components" << std::endl;
            return false;
        }
        
        // 3. 启动工作线程
        shutdown_requested_.store(false);
        for (int i = 0; i < num_worker_threads_; ++i) {
            worker_threads_.emplace_back(&TensorRTInferenceEngine::worker_thread_function, this);
        }
        
        initialized_.store(true);
        std::cout << "TensorRT Inference Engine initialized successfully" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during TensorRT Inference Engine initialization: " << e.what() << std::endl;
        return false;
    }
}

void TensorRTInferenceEngine::shutdown() {
    if (!initialized_.load() || shutdown_requested_.load()) {
        return;
    }
    
    std::cout << "Shutting down TensorRT Inference Engine..." << std::endl;
    
    // 停止工作线程
    shutdown_requested_.store(true);
    request_available_cv_.notify_all();
    
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    // 清理组件
    cleanup_components();
    
    initialized_.store(false);
    model_loaded_.store(false);
    std::cout << "TensorRT Inference Engine shutdown complete" << std::endl;
}

bool TensorRTInferenceEngine::load_model(const std::string& model_path) {
    if (!initialized_.load()) {
        std::cerr << "Engine not initialized" << std::endl;
        return false;
    }
    
    if (!trt_engine_->load_model(model_path)) {
        std::cerr << "Failed to load TensorRT model: " << model_path << std::endl;
        return false;
    }
    
    model_loaded_.store(true);
    std::cout << "Model loaded successfully: " << model_path << std::endl;
    return true;
}

InferenceResponse TensorRTInferenceEngine::infer(const ProcessedMultiModal& input) {
    if (!is_ready()) {
        return create_error_response(input.request_id, "Engine not ready");
    }
    
    // 创建推理请求
    auto request = std::make_unique<InferenceRequest>(input.request_id, input);
    request->submit_time = std::chrono::steady_clock::now();
    
    // 同步执行
    bool success = process_request(request.get());
    
    // 返回结果
    if (success) {
        try {
            return request->future.get();
        } catch (const std::exception& e) {
            return create_error_response(input.request_id, "Future exception: " + std::string(e.what()));
        }
    } else {
        return create_error_response(input.request_id, "Request processing failed");
    }
}

std::future<InferenceResponse> TensorRTInferenceEngine::infer_async(const ProcessedMultiModal& input) {
    return std::async(std::launch::async, [this, input]() {
        return this->infer(input);
    });
}

bool TensorRTInferenceEngine::initialize_cuda_context() {
    // 初始化cuBLAS
    cublasStatus_t status = cublasCreate(&cublas_handle_);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Failed to create cuBLAS handle" << std::endl;
        return false;
    }
    
    return true;
}

bool TensorRTInferenceEngine::initialize_components() {
    // 1. 初始化TensorRT引擎
    trt_engine_ = std::make_unique<TensorRTEngine>(config_);
    if (!trt_engine_->initialize()) {
        return false;
    }
    
    // 2. 初始化CUDA流管理器
    stream_manager_ = std::make_unique<CudaStreamManager>(config_.max_batch_size);
    if (!stream_manager_->initialize()) {
        return false;
    }
    
    // 3. 初始化KV缓存管理器
    kv_cache_manager_ = std::make_unique<KVCacheManager>(memory_manager_, config_.attention_config);
    if (!kv_cache_manager_->initialize()) {
        return false;
    }
    
    return true;
}

void TensorRTInferenceEngine::cleanup_components() {
    kv_cache_manager_.reset();
    stream_manager_.reset();
    trt_engine_.reset();
    
    if (cublas_handle_) {
        cublasDestroy(cublas_handle_);
        cublas_handle_ = nullptr;
    }
}

void TensorRTInferenceEngine::worker_thread_function() {
    while (!shutdown_requested_.load()) {
        std::unique_lock<std::mutex> lock(requests_mutex_);
        
        request_available_cv_.wait(lock, [this] {
            return !pending_requests_.empty() || shutdown_requested_.load();
        });
        
        if (shutdown_requested_.load()) {
            break;
        }
        
        if (!pending_requests_.empty()) {
            auto request = std::move(pending_requests_.front());
            pending_requests_.pop();
            
            active_requests_[request->request_id] = std::move(request);
            auto* req_ptr = active_requests_[request->request_id].get();
            
            lock.unlock();
            
            // 处理请求
            process_request(req_ptr);
            
            // 移除活跃请求
            lock.lock();
            active_requests_.erase(req_ptr->request_id);
        }
    }
}

bool TensorRTInferenceEngine::process_request(InferenceRequest* request) {
    if (!request) return false;
    
    try {
        request->start_time = std::chrono::steady_clock::now();
        request->state = InferenceState::PREPROCESSING;
        
        // 1. 预处理
        if (!preprocess_request(request)) {
            request->state = InferenceState::FAILED;
            return false;
        }
        
        // 2. 执行推理
        request->state = InferenceState::EXECUTING;
        if (!execute_inference_internal(request)) {
            request->state = InferenceState::FAILED;
            return false;
        }
        
        // 3. 后处理
        request->state = InferenceState::POSTPROCESSING;
        if (!postprocess_request(request)) {
            request->state = InferenceState::FAILED;
            return false;
        }
        
        request->state = InferenceState::COMPLETED;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception in process_request: " << e.what() << std::endl;
        request->state = InferenceState::FAILED;
        return false;
    }
}

bool TensorRTInferenceEngine::preprocess_request(InferenceRequest* request) {
    // 分配序列ID
    request->sequence_id = generate_sequence_id();
    
    // 分配CUDA流
    request->assigned_stream = stream_manager_->acquire_stream();
    
    // 分配KV缓存
    int context_length = request->input_data.text.processed_length;
    request->kv_blocks = kv_cache_manager_->allocate_kv_blocks(request->sequence_id, context_length);
    
    return request->assigned_stream != nullptr && !request->kv_blocks.empty();
}

bool TensorRTInferenceEngine::execute_inference_internal(InferenceRequest* request) {
    // 这里是推理执行的占位符实现
    // 实际实现需要准备输入缓冲区，调用TensorRT引擎，处理输出
    
    // 模拟推理延迟
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    return true;
}

bool TensorRTInferenceEngine::postprocess_request(InferenceRequest* request) {
    // 创建响应
    InferenceResponse response;
    response.request_id = request->request_id;
    response.success = true;
    response.generated_text = "Generated response for: " + request->input_data.request_id;
    response.tokens_generated = 100;
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - request->start_time);
    response.inference_time_ms = static_cast<double>(duration.count());
    
    // 释放资源
    if (request->assigned_stream) {
        stream_manager_->release_stream(request->assigned_stream);
    }
    kv_cache_manager_->deallocate_kv_blocks(request->kv_blocks);
    
    // 设置Promise结果
    request->promise.set_value(response);
    
    // 更新统计
    update_stats(true, response.inference_time_ms, response.tokens_generated);
    
    return true;
}

InferenceResponse TensorRTInferenceEngine::create_error_response(const std::string& request_id,
                                                               const std::string& error_msg) {
    InferenceResponse response;
    response.request_id = request_id;
    response.success = false;
    response.error_message = error_msg;
    response.inference_time_ms = 0.0;
    response.tokens_generated = 0;
    
    return response;
}

void TensorRTInferenceEngine::update_stats(bool success, double latency_ms, int tokens_generated) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    stats_.total_requests.fetch_add(1);
    if (success) {
        stats_.successful_requests.fetch_add(1);
        stats_.total_tokens_generated.fetch_add(tokens_generated);
        
        // 更新平均延迟（简化的滑动平均）
        stats_.avg_inference_latency_ms.store(
            (stats_.avg_inference_latency_ms.load() * 0.9) + (latency_ms * 0.1));
        
        // 更新吞吐量
        if (latency_ms > 0) {
            double tokens_per_sec = (tokens_generated * 1000.0) / latency_ms;
            stats_.avg_throughput_tokens_per_sec.store(
                (stats_.avg_throughput_tokens_per_sec.load() * 0.9) + (tokens_per_sec * 0.1));
        }
    } else {
        stats_.failed_requests.fetch_add(1);
    }
}

int TensorRTInferenceEngine::generate_sequence_id() {
    return next_sequence_id_.fetch_add(1);
}

TensorRTInferenceEngine::EngineStats TensorRTInferenceEngine::get_stats() const {
    return stats_;
}

} // namespace wicore 