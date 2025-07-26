// tensorrt_inference_engine.hpp
#ifndef TENSORRT_INFERENCE_ENGINE_HPP
#define TENSORRT_INFERENCE_ENGINE_HPP

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <future>
#include <queue>
#include <condition_variable>
#include <chrono>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>

#include "hmt_memory_manager.hpp"
#include "multimodal_processor.hpp"

namespace wicore {

// 前向声明
class HMTMemoryManager;

// TensorRT推理精度
enum class InferencePrecision {
    FP32,
    FP16,
    INT8,
    MIXED  // FP16 + INT8混合精度
};

// CUDA流状态
enum class StreamState {
    IDLE,
    BUSY,
    GRAPH_CAPTURING,
    GRAPH_EXECUTING
};

// 推理请求状态
enum class InferenceState {
    PENDING,
    PREPROCESSING,
    EXECUTING,
    POSTPROCESSING,
    COMPLETED,
    FAILED
};

// KV缓存块描述
struct KVCacheBlock {
    MemoryBlock* key_block = nullptr;     // 键缓存块
    MemoryBlock* value_block = nullptr;   // 值缓存块
    int sequence_id = -1;                 // 序列ID
    int start_pos = 0;                    // 在序列中的起始位置
    int length = 0;                       // 当前长度
    bool is_active = false;               // 是否活跃
    std::chrono::steady_clock::time_point last_access;
};

// Attention配置
struct AttentionConfig {
    int num_heads = 32;                   // 注意力头数
    int head_dim = 128;                   // 每个头的维度  
    int num_kv_heads = 8;                 // KV头数(GQA)
    int max_context_length = 131072;      // 128K上下文
    int block_size = 64;                  // KV缓存块大小
    bool use_sliding_window = false;      // 是否使用滑动窗口
    int sliding_window_size = 4096;       // 滑动窗口大小
};

// 模型配置
struct ModelConfig {
    std::string model_path;
    std::string engine_cache_path;
    InferencePrecision precision = InferencePrecision::FP16;
    int max_batch_size = 16;
    size_t max_workspace_bytes = 8ULL * 1024 * 1024 * 1024; // 8GB
    bool enable_sparse_weights = true;
    bool enable_refit = true;
    bool enable_timing_cache = true;
    AttentionConfig attention_config;
};

// CUDA流管理器
class CudaStreamManager {
public:
    explicit CudaStreamManager(int num_streams = 4);
    ~CudaStreamManager();
    
    bool initialize();
    void shutdown();
    
    // 流分配与释放
    cudaStream_t acquire_stream();
    void release_stream(cudaStream_t stream);
    
    // 流状态管理
    void set_stream_state(cudaStream_t stream, StreamState state);
    StreamState get_stream_state(cudaStream_t stream) const;
    
    // 同步
    void synchronize_stream(cudaStream_t stream);
    void synchronize_all_streams();
    
    // 统计
    int get_active_stream_count() const;
    int get_total_stream_count() const { return num_streams_; }

private:
    int num_streams_;
    std::vector<cudaStream_t> streams_;
    std::queue<cudaStream_t> available_streams_;
    std::unordered_map<cudaStream_t, StreamState> stream_states_;
    mutable std::mutex streams_mutex_;
    std::condition_variable stream_available_cv_;
};

// KV缓存管理器
class KVCacheManager {
public:
    explicit KVCacheManager(HMTMemoryManager* memory_manager, 
                           const AttentionConfig& config);
    ~KVCacheManager();
    
    bool initialize();
    void shutdown();
    
    // 缓存分配
    std::vector<KVCacheBlock*> allocate_kv_blocks(int sequence_id, int length);
    void deallocate_kv_blocks(const std::vector<KVCacheBlock*>& blocks);
    
    // 缓存访问
    KVCacheBlock* get_kv_block(int sequence_id, int position);
    void update_kv_cache(int sequence_id, int position, 
                        const void* key_data, const void* value_data);
    
    // 序列管理
    void register_sequence(int sequence_id, int max_length);
    void unregister_sequence(int sequence_id);
    void extend_sequence(int sequence_id, int new_length);
    
    // 统计
    int get_allocated_blocks_count() const;
    double get_cache_hit_rate() const;

private:
    HMTMemoryManager* memory_manager_;
    AttentionConfig config_;
    
    // KV块管理
    std::unordered_map<int, std::vector<KVCacheBlock*>> sequence_blocks_;
    std::vector<std::unique_ptr<KVCacheBlock>> all_blocks_;
    std::queue<KVCacheBlock*> free_blocks_;
    
    mutable std::shared_mutex cache_mutex_;
    
    // 统计
    std::atomic<int> total_allocated_blocks_{0};
    std::atomic<uint64_t> cache_hits_{0};
    std::atomic<uint64_t> cache_misses_{0};
};

// TensorRT引擎包装器
class TensorRTEngine {
public:
    explicit TensorRTEngine(const ModelConfig& config);
    ~TensorRTEngine();
    
    bool initialize();
    void shutdown();
    
    // 模型加载
    bool load_model(const std::string& model_path);
    bool build_engine_from_onnx(const std::string& onnx_path);
    bool load_engine_from_cache(const std::string& cache_path);
    bool save_engine_to_cache(const std::string& cache_path);
    
    // 推理执行
    bool execute_inference(cudaStream_t stream,
                          const std::vector<void*>& input_buffers,
                          const std::vector<void*>& output_buffers);
    
    // CUDA Graph支持
    bool capture_cuda_graph(cudaStream_t stream,
                           const std::vector<void*>& input_buffers,
                           const std::vector<void*>& output_buffers);
    bool execute_cuda_graph(cudaStream_t stream);
    
    // 信息查询
    int get_input_count() const;
    int get_output_count() const;
    std::vector<nvinfer1::Dims> get_input_shapes() const;
    std::vector<nvinfer1::Dims> get_output_shapes() const;
    size_t get_binding_size(int binding_index) const;

private:
    ModelConfig config_;
    
    // TensorRT对象
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    
    // CUDA Graph
    cudaGraph_t cuda_graph_ = nullptr;
    cudaGraphExec_t cuda_graph_exec_ = nullptr;
    bool graph_captured_ = false;
    
    // 工具方法
    std::unique_ptr<nvinfer1::IBuilder> create_builder();
    std::unique_ptr<nvinfer1::INetworkDefinition> create_network(nvinfer1::IBuilder* builder);
    bool optimize_network(nvinfer1::IBuilder* builder, nvinfer1::INetworkDefinition* network);
};

// 推理请求结构
struct InferenceRequest {
    std::string request_id;
    ProcessedMultiModal input_data;
    int sequence_id = -1;
    int max_new_tokens = 512;
    float temperature = 1.0f;
    float top_p = 0.9f;
    int top_k = 50;
    bool stream_output = false;
    
    // 内部状态
    InferenceState state = InferenceState::PENDING;
    cudaStream_t assigned_stream = nullptr;
    std::vector<KVCacheBlock*> kv_blocks;
    std::chrono::steady_clock::time_point submit_time;
    std::chrono::steady_clock::time_point start_time;
    
    // 异步执行
    std::promise<InferenceResponse> promise;
    std::future<InferenceResponse> future;
    
    InferenceRequest(const std::string& id, const ProcessedMultiModal& data)
        : request_id(id), input_data(data), future(promise.get_future()) {}
};

// TensorRT推理引擎
class TensorRTInferenceEngine {
public:
    explicit TensorRTInferenceEngine(HMTMemoryManager* memory_manager,
                                   const ModelConfig& config);
    ~TensorRTInferenceEngine();
    
    // 核心接口
    bool initialize();
    void shutdown();
    bool load_model(const std::string& model_path);
    
    // 推理执行
    InferenceResponse infer(const ProcessedMultiModal& input);
    std::future<InferenceResponse> infer_async(const ProcessedMultiModal& input);
    
    // 批处理推理
    std::vector<InferenceResponse> infer_batch(const std::vector<ProcessedMultiModal>& inputs);
    std::future<std::vector<InferenceResponse>> infer_batch_async(const std::vector<ProcessedMultiModal>& inputs);
    
    // 配置管理
    void update_model_config(const ModelConfig& config);
    ModelConfig get_model_config() const;
    
    // 状态查询
    bool is_ready() const { return initialized_.load() && model_loaded_.load(); }
    int get_active_requests_count() const;
    
    // 性能统计
    struct EngineStats {
        std::atomic<uint64_t> total_requests{0};
        std::atomic<uint64_t> successful_requests{0};
        std::atomic<uint64_t> failed_requests{0};
        std::atomic<double> avg_inference_latency_ms{0.0};
        std::atomic<double> avg_throughput_tokens_per_sec{0.0};
        std::atomic<uint64_t> total_tokens_generated{0};
        std::atomic<int> current_concurrent_requests{0};
        std::atomic<double> gpu_utilization_percent{0.0};
    };
    
    EngineStats get_stats() const;

private:
    // 依赖组件
    HMTMemoryManager* memory_manager_;
    ModelConfig config_;
    
    // 核心组件
    std::unique_ptr<TensorRTEngine> trt_engine_;
    std::unique_ptr<CudaStreamManager> stream_manager_;
    std::unique_ptr<KVCacheManager> kv_cache_manager_;
    
    // 状态
    std::atomic<bool> initialized_{false};
    std::atomic<bool> model_loaded_{false};
    std::atomic<bool> shutdown_requested_{false};
    
    // 请求队列
    std::queue<std::unique_ptr<InferenceRequest>> pending_requests_;
    std::unordered_map<std::string, std::unique_ptr<InferenceRequest>> active_requests_;
    mutable std::mutex requests_mutex_;
    std::condition_variable request_available_cv_;
    
    // 工作线程
    std::vector<std::thread> worker_threads_;
    int num_worker_threads_ = 4;
    
    // 统计
    mutable EngineStats stats_;
    mutable std::mutex stats_mutex_;
    
    // cuBLAS句柄
    cublasHandle_t cublas_handle_ = nullptr;
    
    // 内部方法
    bool initialize_cuda_context();
    bool initialize_components();
    void cleanup_components();
    
    // 请求处理
    void worker_thread_function();
    bool process_request(InferenceRequest* request);
    bool preprocess_request(InferenceRequest* request);
    bool execute_inference_internal(InferenceRequest* request);
    bool postprocess_request(InferenceRequest* request);
    
    // 内存管理
    bool allocate_inference_buffers(InferenceRequest* request);
    void deallocate_inference_buffers(InferenceRequest* request);
    
    // 工具方法
    InferenceResponse create_error_response(const std::string& request_id,
                                          const std::string& error_msg);
    void update_stats(bool success, double latency_ms, int tokens_generated);
    int generate_sequence_id();
    std::atomic<int> next_sequence_id_{1};
    
    // 禁用拷贝
    TensorRTInferenceEngine(const TensorRTInferenceEngine&) = delete;
    TensorRTInferenceEngine& operator=(const TensorRTInferenceEngine&) = delete;
};

// TensorRT日志器
class TensorRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
    
private:
    static const char* severity_to_string(Severity severity);
};

} // namespace wicore

#endif // TENSORRT_INFERENCE_ENGINE_HPP 