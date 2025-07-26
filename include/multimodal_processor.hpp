// multimodal_processor.hpp
#ifndef MULTIMODAL_PROCESSOR_HPP
#define MULTIMODAL_PROCESSOR_HPP

#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <future>
#include <queue>
#include <condition_variable>
#include <functional>

#include <opencv2/opencv.hpp>
#include <sentencepiece_processor.h>

namespace wicore {

// 前向声明
struct MultiModalRequest;

// Gemma-3特殊token定义
struct GemmaTokens {
    static constexpr int32_t BOS_TOKEN_ID = 1;      // 开始token
    static constexpr int32_t EOS_TOKEN_ID = 2;      // 结束token
    static constexpr int32_t UNK_TOKEN_ID = 3;      // 未知token
    static constexpr int32_t PAD_TOKEN_ID = 0;      // 填充token
    static constexpr int32_t IMAGE_TOKEN_ID = 32000; // 图像token
    static constexpr int32_t NEWLINE_TOKEN_ID = 108; // 换行token
};

// 图像预处理参数
struct ImageConfig {
    int target_size = 896;                          // Gemma-3标准尺寸
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};  // ImageNet均值
    std::vector<float> std = {0.229f, 0.224f, 0.225f};   // ImageNet标准差
    bool normalize = true;
    bool to_rgb = true;
    int max_images_per_request = 8;
};

// 处理后的文本数据
struct ProcessedText {
    std::vector<int32_t> input_ids;
    std::vector<int32_t> attention_mask;
    std::vector<int32_t> position_ids;
    int original_length = 0;
    int processed_length = 0;
    bool has_images = false;
    int image_token_count = 0;
};

// 处理后的图像数据
struct ProcessedImage {
    std::vector<float> pixel_values;    // CHW格式，float32
    int batch_size = 0;
    int channels = 3;
    int height = 896;
    int width = 896;
    std::vector<cv::Size> original_sizes;
};

// 多模态处理结果
struct ProcessedMultiModal {
    ProcessedText text;
    ProcessedImage images;
    
    // 元数据
    std::string request_id;
    int total_sequence_length = 0;
    double processing_time_ms = 0.0;
    bool success = false;
    std::string error_message;
};

// 批处理配置
struct BatchConfig {
    int max_batch_size = 16;
    int max_sequence_length = 131072;  // 128K上下文
    bool dynamic_padding = true;
    bool parallel_processing = true;
    int num_worker_threads = 4;
};

// 多模态预处理器
class MultiModalProcessor {
public:
    explicit MultiModalProcessor(const std::string& tokenizer_path,
                               const ImageConfig& image_config = ImageConfig{},
                               const BatchConfig& batch_config = BatchConfig{});
    ~MultiModalProcessor();
    
    // 核心接口
    bool initialize();
    void shutdown();
    
    // 单个请求处理
    ProcessedMultiModal process_request(const MultiModalRequest& request);
    std::future<ProcessedMultiModal> process_request_async(const MultiModalRequest& request);
    
    // 批处理
    std::vector<ProcessedMultiModal> process_batch(const std::vector<MultiModalRequest>& requests);
    std::future<std::vector<ProcessedMultiModal>> process_batch_async(const std::vector<MultiModalRequest>& requests);
    
    // 单独处理接口
    ProcessedText process_text(const std::string& text, 
                              int num_images = 0,
                              int max_length = -1);
    ProcessedImage process_images(const std::vector<cv::Mat>& images);
    
    // 配置管理
    void update_image_config(const ImageConfig& config);
    void update_batch_config(const BatchConfig& config);
    ImageConfig get_image_config() const;
    BatchConfig get_batch_config() const;
    
    // 状态查询
    bool is_initialized() const { return initialized_.load(); }
    size_t get_vocab_size() const;
    
    // 性能统计
    struct ProcessorStats {
        std::atomic<uint64_t> total_requests{0};
        std::atomic<uint64_t> successful_requests{0};
        std::atomic<uint64_t> failed_requests{0};
        std::atomic<double> avg_text_processing_ms{0.0};
        std::atomic<double> avg_image_processing_ms{0.0};
        std::atomic<double> avg_total_processing_ms{0.0};
        std::atomic<uint64_t> total_tokens_processed{0};
        std::atomic<uint64_t> total_images_processed{0};
    };
    
    ProcessorStats get_stats() const;

private:
    // 配置
    std::string tokenizer_path_;
    ImageConfig image_config_;
    BatchConfig batch_config_;
    
    // 核心组件
    std::unique_ptr<sentencepiece::SentencePieceProcessor> tokenizer_;
    
    // 状态
    std::atomic<bool> initialized_{false};
    std::atomic<bool> shutdown_requested_{false};
    
    // 线程池
    std::vector<std::thread> worker_threads_;
    std::queue<std::function<void()>> task_queue_;
    std::mutex task_mutex_;
    std::condition_variable task_cv_;
    
    // 统计
    mutable ProcessorStats stats_;
    mutable std::mutex stats_mutex_;
    
    // 内部方法 - 文本处理
    bool initialize_tokenizer();
    std::vector<int32_t> tokenize_text(const std::string& text);
    std::vector<int32_t> add_special_tokens(const std::vector<int32_t>& tokens, 
                                           int num_images = 0);
    std::vector<int32_t> generate_attention_mask(const std::vector<int32_t>& input_ids);
    std::vector<int32_t> generate_position_ids(const std::vector<int32_t>& input_ids);
    ProcessedText create_processed_text(const std::vector<int32_t>& input_ids,
                                       int original_length,
                                       int num_images);
    
    // 内部方法 - 图像处理
    cv::Mat preprocess_single_image(const cv::Mat& image);
    cv::Mat resize_image(const cv::Mat& image, int target_size);
    cv::Mat normalize_image(const cv::Mat& image);
    std::vector<float> convert_to_chw(const cv::Mat& image);
    ProcessedImage create_processed_images(const std::vector<cv::Mat>& images);
    
    // 批处理内部方法
    std::vector<ProcessedText> process_text_batch(const std::vector<std::string>& texts,
                                                 const std::vector<int>& num_images_list);
    std::vector<ProcessedImage> process_image_batch(const std::vector<std::vector<cv::Mat>>& image_batches);
    
    // 工具方法
    ProcessedMultiModal create_error_response(const std::string& request_id,
                                             const std::string& error_msg);
    void update_stats(double text_time_ms, double image_time_ms, 
                     int tokens_count, int images_count, bool success);
    bool validate_request(const MultiModalRequest& request, std::string& error_msg);
    
    // 线程池管理
    void worker_thread_function();
    void enqueue_task(std::function<void()> task);
    
    // 禁用拷贝
    MultiModalProcessor(const MultiModalProcessor&) = delete;
    MultiModalProcessor& operator=(const MultiModalProcessor&) = delete;
};

// 工具函数
namespace multimodal_utils {
    // 图像验证
    bool is_valid_image(const cv::Mat& image);
    cv::Size get_optimal_resize_size(cv::Size original, int target_size);
    
    // 文本验证
    bool is_valid_text(const std::string& text);
    std::string sanitize_text(const std::string& text);
    
    // 批处理工具
    std::vector<std::vector<cv::Mat>> group_images_for_batch(
        const std::vector<std::vector<cv::Mat>>& all_images, 
        int max_batch_size);
    
    // 性能工具
    class Timer {
    public:
        Timer() : start_(std::chrono::high_resolution_clock::now()) {}
        double elapsed_ms() const {
            auto end = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(end - start_).count();
        }
    private:
        std::chrono::high_resolution_clock::time_point start_;
    };
}

} // namespace wicore

#endif // MULTIMODAL_PROCESSOR_HPP 