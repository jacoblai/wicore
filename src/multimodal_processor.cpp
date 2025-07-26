// multimodal_processor.cpp
#include "../include/multimodal_processor.hpp"
#include "../include/wicore_engine.hpp"
#include <iostream>
#include <algorithm>
#include <sstream>
#include <regex>

namespace wicore {

MultiModalProcessor::MultiModalProcessor(const std::string& tokenizer_path,
                                       const ImageConfig& image_config,
                                       const BatchConfig& batch_config)
    : tokenizer_path_(tokenizer_path)
    , image_config_(image_config)
    , batch_config_(batch_config)
    , tokenizer_(std::make_unique<sentencepiece::SentencePieceProcessor>()) {
    
    std::cout << "MultiModal Processor created" << std::endl;
    std::cout << "Tokenizer path: " << tokenizer_path_ << std::endl;
    std::cout << "Image target size: " << image_config_.target_size << "x" << image_config_.target_size << std::endl;
    std::cout << "Max batch size: " << batch_config_.max_batch_size << std::endl;
}

MultiModalProcessor::~MultiModalProcessor() {
    shutdown();
}

bool MultiModalProcessor::initialize() {
    if (initialized_.load()) {
        return true;
    }
    
    try {
        // 1. 初始化tokenizer
        if (!initialize_tokenizer()) {
            std::cerr << "Failed to initialize tokenizer" << std::endl;
            return false;
        }
        
        // 2. 启动线程池
        shutdown_requested_.store(false);
        for (int i = 0; i < batch_config_.num_worker_threads; ++i) {
            worker_threads_.emplace_back(&MultiModalProcessor::worker_thread_function, this);
        }
        
        initialized_.store(true);
        std::cout << "MultiModal Processor initialized successfully" << std::endl;
        std::cout << "Vocabulary size: " << get_vocab_size() << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during MultiModal Processor initialization: " << e.what() << std::endl;
        return false;
    }
}

void MultiModalProcessor::shutdown() {
    if (!initialized_.load() || shutdown_requested_.load()) {
        return;
    }
    
    std::cout << "Shutting down MultiModal Processor..." << std::endl;
    
    // 停止线程池
    shutdown_requested_.store(true);
    task_cv_.notify_all();
    
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    initialized_.store(false);
    std::cout << "MultiModal Processor shutdown complete" << std::endl;
}

ProcessedMultiModal MultiModalProcessor::process_request(const MultiModalRequest& request) {
    if (!initialized_.load()) {
        return create_error_response(request.request_id, "Processor not initialized");
    }
    
    multimodal_utils::Timer timer;
    stats_.total_requests.fetch_add(1);
    
    // 验证请求
    std::string error_msg;
    if (!validate_request(request, error_msg)) {
        stats_.failed_requests.fetch_add(1);
        return create_error_response(request.request_id, error_msg);
    }
    
    try {
        multimodal_utils::Timer text_timer;
        
        // 1. 处理文本
        auto processed_text = process_text(request.text_prompt, 
                                         static_cast<int>(request.images.size()));
        double text_time_ms = text_timer.elapsed_ms();
        
        multimodal_utils::Timer image_timer;
        
        // 2. 处理图像
        ProcessedImage processed_images;
        if (!request.images.empty()) {
            processed_images = process_images(request.images);
        }
        double image_time_ms = image_timer.elapsed_ms();
        
        // 3. 创建结果
        ProcessedMultiModal result;
        result.request_id = request.request_id;
        result.text = processed_text;
        result.images = processed_images;
        result.total_sequence_length = processed_text.processed_length;
        result.processing_time_ms = timer.elapsed_ms();
        result.success = true;
        
        // 4. 更新统计
        update_stats(text_time_ms, image_time_ms, 
                    processed_text.processed_length, 
                    static_cast<int>(request.images.size()), true);
        
        stats_.successful_requests.fetch_add(1);
        return result;
        
    } catch (const std::exception& e) {
        stats_.failed_requests.fetch_add(1);
        return create_error_response(request.request_id, 
                                   "Processing error: " + std::string(e.what()));
    }
}

std::future<ProcessedMultiModal> MultiModalProcessor::process_request_async(const MultiModalRequest& request) {
    return std::async(std::launch::async, [this, request]() {
        return this->process_request(request);
    });
}

std::vector<ProcessedMultiModal> MultiModalProcessor::process_batch(const std::vector<MultiModalRequest>& requests) {
    std::vector<ProcessedMultiModal> results;
    results.reserve(requests.size());
    
    if (!batch_config_.parallel_processing || requests.size() == 1) {
        // 串行处理
        for (const auto& request : requests) {
            results.push_back(process_request(request));
        }
    } else {
        // 并行处理
        std::vector<std::future<ProcessedMultiModal>> futures;
        futures.reserve(requests.size());
        
        for (const auto& request : requests) {
            futures.push_back(process_request_async(request));
        }
        
        for (auto& future : futures) {
            results.push_back(future.get());
        }
    }
    
    return results;
}

std::future<std::vector<ProcessedMultiModal>> MultiModalProcessor::process_batch_async(const std::vector<MultiModalRequest>& requests) {
    return std::async(std::launch::async, [this, requests]() {
        return this->process_batch(requests);
    });
}

ProcessedText MultiModalProcessor::process_text(const std::string& text, int num_images, int max_length) {
    if (!initialized_.load()) {
        return ProcessedText{};
    }
    
    try {
        // 1. 文本清理和验证
        std::string clean_text = multimodal_utils::sanitize_text(text);
        int original_length = static_cast<int>(clean_text.length());
        
        // 2. Tokenization
        auto tokens = tokenize_text(clean_text);
        
        // 3. 添加特殊token
        auto tokens_with_special = add_special_tokens(tokens, num_images);
        
        // 4. 长度截断（如果需要）
        if (max_length > 0 && static_cast<int>(tokens_with_special.size()) > max_length) {
            tokens_with_special.resize(max_length - 1);
            tokens_with_special.push_back(GemmaTokens::EOS_TOKEN_ID);
        }
        
        // 5. 创建结果
        return create_processed_text(tokens_with_special, original_length, num_images);
        
    } catch (const std::exception& e) {
        std::cerr << "Error processing text: " << e.what() << std::endl;
        return ProcessedText{};
    }
}

ProcessedImage MultiModalProcessor::process_images(const std::vector<cv::Mat>& images) {
    if (!initialized_.load() || images.empty()) {
        return ProcessedImage{};
    }
    
    try {
        ProcessedImage result;
        result.batch_size = static_cast<int>(images.size());
        result.channels = 3;
        result.height = image_config_.target_size;
        result.width = image_config_.target_size;
        
        // 预分配内存
        size_t total_pixels = result.batch_size * result.channels * result.height * result.width;
        result.pixel_values.reserve(total_pixels);
        result.original_sizes.reserve(images.size());
        
        // 并行处理图像
        std::vector<std::future<std::vector<float>>> futures;
        for (const auto& image : images) {
            if (!multimodal_utils::is_valid_image(image)) {
                throw std::runtime_error("Invalid image data");
            }
            
            result.original_sizes.push_back(image.size());
            
            // 异步处理单张图像
            futures.push_back(std::async(std::launch::async, [this, image]() {
                auto processed = preprocess_single_image(image);
                return convert_to_chw(processed);
            }));
        }
        
        // 收集结果
        for (auto& future : futures) {
            auto image_data = future.get();
            result.pixel_values.insert(result.pixel_values.end(), 
                                     image_data.begin(), image_data.end());
        }
        
        return result;
        
    } catch (const std::exception& e) {
        std::cerr << "Error processing images: " << e.what() << std::endl;
        return ProcessedImage{};
    }
}

void MultiModalProcessor::update_image_config(const ImageConfig& config) {
    image_config_ = config;
}

void MultiModalProcessor::update_batch_config(const BatchConfig& config) {
    batch_config_ = config;
}

ImageConfig MultiModalProcessor::get_image_config() const {
    return image_config_;
}

BatchConfig MultiModalProcessor::get_batch_config() const {
    return batch_config_;
}

size_t MultiModalProcessor::get_vocab_size() const {
    if (!tokenizer_) {
        return 0;
    }
    return tokenizer_->GetPieceSize();
}

MultiModalProcessor::ProcessorStats MultiModalProcessor::get_stats() const {
    return stats_;
}

// === 私有方法实现 ===

bool MultiModalProcessor::initialize_tokenizer() {
    try {
        auto status = tokenizer_->Load(tokenizer_path_);
        if (!status.ok()) {
            std::cerr << "Failed to load tokenizer from " << tokenizer_path_ 
                      << ": " << status.ToString() << std::endl;
            return false;
        }
        
        std::cout << "Tokenizer loaded successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception loading tokenizer: " << e.what() << std::endl;
        return false;
    }
}

std::vector<int32_t> MultiModalProcessor::tokenize_text(const std::string& text) {
    std::vector<int> pieces;
    tokenizer_->Encode(text, &pieces);
    
    std::vector<int32_t> tokens;
    tokens.reserve(pieces.size());
    for (int piece : pieces) {
        tokens.push_back(static_cast<int32_t>(piece));
    }
    
    return tokens;
}

std::vector<int32_t> MultiModalProcessor::add_special_tokens(const std::vector<int32_t>& tokens, int num_images) {
    std::vector<int32_t> result;
    result.reserve(tokens.size() + 10); // 预留空间给特殊token
    
    // 添加BOS token
    result.push_back(GemmaTokens::BOS_TOKEN_ID);
    
    // 如果有图像，在开头添加图像token
    for (int i = 0; i < num_images; ++i) {
        result.push_back(GemmaTokens::IMAGE_TOKEN_ID);
    }
    
    // 添加原始token
    result.insert(result.end(), tokens.begin(), tokens.end());
    
    // 注意：不在这里添加EOS token，让模型自己决定何时结束
    
    return result;
}

std::vector<int32_t> MultiModalProcessor::generate_attention_mask(const std::vector<int32_t>& input_ids) {
    // 对于Gemma-3，所有token都参与attention
    return std::vector<int32_t>(input_ids.size(), 1);
}

std::vector<int32_t> MultiModalProcessor::generate_position_ids(const std::vector<int32_t>& input_ids) {
    std::vector<int32_t> position_ids;
    position_ids.reserve(input_ids.size());
    
    for (size_t i = 0; i < input_ids.size(); ++i) {
        position_ids.push_back(static_cast<int32_t>(i));
    }
    
    return position_ids;
}

ProcessedText MultiModalProcessor::create_processed_text(const std::vector<int32_t>& input_ids,
                                                        int original_length,
                                                        int num_images) {
    ProcessedText result;
    result.input_ids = input_ids;
    result.attention_mask = generate_attention_mask(input_ids);
    result.position_ids = generate_position_ids(input_ids);
    result.original_length = original_length;
    result.processed_length = static_cast<int>(input_ids.size());
    result.has_images = (num_images > 0);
    result.image_token_count = num_images;
    
    return result;
}

cv::Mat MultiModalProcessor::preprocess_single_image(const cv::Mat& image) {
    // 1. 调整尺寸
    auto resized = resize_image(image, image_config_.target_size);
    
    // 2. 转换颜色空间
    cv::Mat rgb_image;
    if (image_config_.to_rgb && resized.channels() == 3) {
        cv::cvtColor(resized, rgb_image, cv::COLOR_BGR2RGB);
    } else {
        rgb_image = resized;
    }
    
    // 3. 归一化
    cv::Mat normalized;
    if (image_config_.normalize) {
        normalized = normalize_image(rgb_image);
    } else {
        rgb_image.convertTo(normalized, CV_32F, 1.0/255.0);
    }
    
    return normalized;
}

cv::Mat MultiModalProcessor::resize_image(const cv::Mat& image, int target_size) {
    cv::Size optimal_size = multimodal_utils::get_optimal_resize_size(image.size(), target_size);
    
    cv::Mat resized;
    cv::resize(image, resized, optimal_size, 0, 0, cv::INTER_LINEAR);
    
    // 如果需要，进行中心裁剪到目标尺寸
    if (resized.rows != target_size || resized.cols != target_size) {
        int x = (resized.cols - target_size) / 2;
        int y = (resized.rows - target_size) / 2;
        x = std::max(0, x);
        y = std::max(0, y);
        
        cv::Rect crop_rect(x, y, target_size, target_size);
        crop_rect &= cv::Rect(0, 0, resized.cols, resized.rows);
        
        cv::Mat cropped = resized(crop_rect);
        
        // 如果裁剪后尺寸不够，进行padding
        if (cropped.rows != target_size || cropped.cols != target_size) {
            cv::Mat padded;
            cv::copyMakeBorder(cropped, padded, 
                             (target_size - cropped.rows) / 2,
                             (target_size - cropped.rows + 1) / 2,
                             (target_size - cropped.cols) / 2,
                             (target_size - cropped.cols + 1) / 2,
                             cv::BORDER_REFLECT);
            return padded;
        }
        
        return cropped;
    }
    
    return resized;
}

cv::Mat MultiModalProcessor::normalize_image(const cv::Mat& image) {
    cv::Mat float_image;
    image.convertTo(float_image, CV_32F, 1.0/255.0);
    
    cv::Mat normalized;
    cv::Mat channels[3];
    cv::split(float_image, channels);
    
    // 应用ImageNet标准化
    for (int i = 0; i < 3; ++i) {
        channels[i] = (channels[i] - image_config_.mean[i]) / image_config_.std[i];
    }
    
    cv::merge(channels, 3, normalized);
    return normalized;
}

std::vector<float> MultiModalProcessor::convert_to_chw(const cv::Mat& image) {
    std::vector<float> result;
    int total_pixels = image.rows * image.cols * image.channels();
    result.reserve(total_pixels);
    
    // 转换为CHW格式
    cv::Mat channels[3];
    cv::split(image, channels);
    
    for (int c = 0; c < 3; ++c) {
        cv::Mat channel = channels[c];
        for (int h = 0; h < channel.rows; ++h) {
            for (int w = 0; w < channel.cols; ++w) {
                result.push_back(channel.at<float>(h, w));
            }
        }
    }
    
    return result;
}

ProcessedImage MultiModalProcessor::create_processed_images(const std::vector<cv::Mat>& images) {
    return process_images(images);
}

ProcessedMultiModal MultiModalProcessor::create_error_response(const std::string& request_id,
                                                              const std::string& error_msg) {
    ProcessedMultiModal response;
    response.request_id = request_id;
    response.success = false;
    response.error_message = error_msg;
    response.processing_time_ms = 0.0;
    response.total_sequence_length = 0;
    
    return response;
}

void MultiModalProcessor::update_stats(double text_time_ms, double image_time_ms, 
                                      int tokens_count, int images_count, bool success) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    // 更新计数
    stats_.total_tokens_processed.fetch_add(tokens_count);
    stats_.total_images_processed.fetch_add(images_count);
    
    // 更新平均时间（简化的滑动平均）
    double total_time = text_time_ms + image_time_ms;
    stats_.avg_text_processing_ms.store(
        (stats_.avg_text_processing_ms.load() * 0.9) + (text_time_ms * 0.1));
    stats_.avg_image_processing_ms.store(
        (stats_.avg_image_processing_ms.load() * 0.9) + (image_time_ms * 0.1));
    stats_.avg_total_processing_ms.store(
        (stats_.avg_total_processing_ms.load() * 0.9) + (total_time * 0.1));
}

bool MultiModalProcessor::validate_request(const MultiModalRequest& request, std::string& error_msg) {
    // 文本验证
    if (!multimodal_utils::is_valid_text(request.text_prompt)) {
        error_msg = "Invalid text input";
        return false;
    }
    
    // 图像数量验证
    if (static_cast<int>(request.images.size()) > image_config_.max_images_per_request) {
        error_msg = "Too many images (max: " + std::to_string(image_config_.max_images_per_request) + ")";
        return false;
    }
    
    // 图像验证
    for (const auto& image : request.images) {
        if (!multimodal_utils::is_valid_image(image)) {
            error_msg = "Invalid image data";
            return false;
        }
    }
    
    // Token数量验证
    if (request.max_tokens > batch_config_.max_sequence_length) {
        error_msg = "Max tokens exceeds limit";
        return false;
    }
    
    return true;
}

void MultiModalProcessor::worker_thread_function() {
    while (!shutdown_requested_.load()) {
        std::function<void()> task;
        
        {
            std::unique_lock<std::mutex> lock(task_mutex_);
            task_cv_.wait(lock, [this] {
                return !task_queue_.empty() || shutdown_requested_.load();
            });
            
            if (shutdown_requested_.load()) {
                break;
            }
            
            if (!task_queue_.empty()) {
                task = std::move(task_queue_.front());
                task_queue_.pop();
            }
        }
        
        if (task) {
            try {
                task();
            } catch (const std::exception& e) {
                std::cerr << "Error in worker thread: " << e.what() << std::endl;
            }
        }
    }
}

void MultiModalProcessor::enqueue_task(std::function<void()> task) {
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        task_queue_.push(std::move(task));
    }
    task_cv_.notify_one();
}

// === 工具函数实现 ===

namespace multimodal_utils {

bool is_valid_image(const cv::Mat& image) {
    return !image.empty() && image.channels() >= 1 && image.channels() <= 4 &&
           image.rows > 0 && image.cols > 0;
}

cv::Size get_optimal_resize_size(cv::Size original, int target_size) {
    double scale = static_cast<double>(target_size) / std::max(original.width, original.height);
    int new_width = static_cast<int>(original.width * scale);
    int new_height = static_cast<int>(original.height * scale);
    
    return cv::Size(new_width, new_height);
}

bool is_valid_text(const std::string& text) {
    return !text.empty() && text.length() <= 1000000; // 1M字符限制
}

std::string sanitize_text(const std::string& text) {
    // 移除控制字符，保留必要的空白字符
    std::string result;
    result.reserve(text.length());
    
    for (char c : text) {
        if (c >= 32 || c == '\n' || c == '\t') {
            result.push_back(c);
        }
    }
    
    return result;
}

std::vector<std::vector<cv::Mat>> group_images_for_batch(
    const std::vector<std::vector<cv::Mat>>& all_images, 
    int max_batch_size) {
    
    std::vector<std::vector<cv::Mat>> batches;
    
    // 简化实现：每个请求的图像作为一个单独的批次
    for (const auto& images : all_images) {
        batches.push_back(images);
    }
    
    return batches;
}

} // namespace multimodal_utils

} // namespace wicore 