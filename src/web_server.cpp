// web_server.cpp
#include "../include/web_server.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <event2/buffer.h>
#include <event2/bufferevent.h>

namespace wicore {

// === ChatCompletionRequest实现 ===

bool ChatCompletionRequest::parse_from_json(const Json::Value& json, std::string& error) {
    try {
        if (json.isMember("model") && json["model"].isString()) {
            model = json["model"].asString();
        }
        
        if (!json.isMember("messages") || !json["messages"].isArray()) {
            error = "Missing required field: messages";
            return false;
        }
        
        messages = json["messages"];
        
        if (json.isMember("temperature")) {
            temperature = json["temperature"].asFloat();
        }
        
        if (json.isMember("top_p")) {
            top_p = json["top_p"].asFloat();
        }
        
        if (json.isMember("top_k")) {
            top_k = json["top_k"].asInt();
        }
        
        if (json.isMember("max_tokens")) {
            max_tokens = json["max_tokens"].asInt();
        }
        
        if (json.isMember("stream")) {
            stream = json["stream"].asBool();
        }
        
        if (json.isMember("stop") && json["stop"].isArray()) {
            for (const auto& stop_token : json["stop"]) {
                if (stop_token.isString()) {
                    stop.push_back(stop_token.asString());
                }
            }
        }
        
        if (json.isMember("user")) {
            user = json["user"].asString();
        }
        
        // WiCore扩展参数
        if (json.isMember("priority")) {
            std::string priority_str = json["priority"].asString();
            priority = scheduler_utils::string_to_priority(priority_str);
        }
        
        if (json.isMember("timeout_seconds")) {
            timeout_seconds = json["timeout_seconds"].asInt();
        }
        
        return true;
        
    } catch (const std::exception& e) {
        error = "JSON parsing error: " + std::string(e.what());
        return false;
    }
}

// === ChatCompletionResponse实现 ===

Json::Value ChatCompletionResponse::to_json() const {
    Json::Value json;
    json["id"] = id;
    json["object"] = object;
    json["created"] = created;
    json["model"] = model;
    
    Json::Value choices_array(Json::arrayValue);
    for (const auto& choice : choices) {
        Json::Value choice_json;
        choice_json["index"] = choice.index;
        choice_json["message"] = choice.message;
        choice_json["finish_reason"] = choice.finish_reason;
        choices_array.append(choice_json);
    }
    json["choices"] = choices_array;
    
    Json::Value usage_json;
    usage_json["prompt_tokens"] = usage.prompt_tokens;
    usage_json["completion_tokens"] = usage.completion_tokens;
    usage_json["total_tokens"] = usage.total_tokens;
    json["usage"] = usage_json;
    
    return json;
}

// === StreamChunk实现 ===

Json::Value StreamChunk::to_json() const {
    Json::Value json;
    json["id"] = id;
    json["object"] = object;
    json["created"] = created;
    json["model"] = model;
    
    Json::Value choices_array(Json::arrayValue);
    for (const auto& choice : choices) {
        Json::Value choice_json;
        choice_json["index"] = choice.index;
        
        Json::Value delta_json;
        if (!choice.delta.role.empty()) {
            delta_json["role"] = choice.delta.role;
        }
        if (!choice.delta.content.empty()) {
            delta_json["content"] = choice.delta.content;
        }
        choice_json["delta"] = delta_json;
        
        if (!choice.finish_reason.empty()) {
            choice_json["finish_reason"] = choice.finish_reason;
        }
        
        choices_array.append(choice_json);
    }
    json["choices"] = choices_array;
    
    return json;
}

// === RateLimiter实现 ===

RateLimiter::RateLimiter(int requests_per_minute, int burst_size)
    : requests_per_minute_(requests_per_minute)
    , burst_size_(burst_size)
    , refill_rate_(static_cast<double>(requests_per_minute) / 60.0) {
}

bool RateLimiter::allow_request(const std::string& client_ip) {
    std::lock_guard<std::mutex> lock(buckets_mutex_);
    
    auto now = std::chrono::steady_clock::now();
    auto& bucket = client_buckets_[client_ip];
    
    if (bucket.tokens == 0) {
        bucket.tokens = burst_size_;
        bucket.last_refill = now;
    }
    
    refill_bucket(bucket, now);
    
    if (bucket.tokens > 0) {
        bucket.tokens--;
        return true;
    }
    
    return false;
}

void RateLimiter::refill_bucket(ClientBucket& bucket, const std::chrono::steady_clock::time_point& now) {
    auto elapsed = std::chrono::duration<double>(now - bucket.last_refill).count();
    int tokens_to_add = static_cast<int>(elapsed * refill_rate_);
    
    if (tokens_to_add > 0) {
        bucket.tokens = std::min(burst_size_, bucket.tokens + tokens_to_add);
        bucket.last_refill = now;
    }
}

int RateLimiter::get_remaining_requests(const std::string& client_ip) {
    std::lock_guard<std::mutex> lock(buckets_mutex_);
    auto it = client_buckets_.find(client_ip);
    return (it != client_buckets_.end()) ? it->second.tokens : burst_size_;
}

// === WebSocketManager实现 ===

WebSocketManager::WebSocketManager() {
    cleanup_thread_ = std::thread(&WebSocketManager::cleanup_dead_connections, this);
}

WebSocketManager::~WebSocketManager() {
    cleanup_running_.store(false);
    if (cleanup_thread_.joinable()) {
        cleanup_thread_.join();
    }
}

bool WebSocketManager::add_connection(evhtp_request_t* req, const std::string& connection_id) {
    std::unique_lock<std::shared_mutex> lock(connections_mutex_);
    
    auto connection = std::make_unique<Connection>();
    connection->request = req;
    connection->created_time = std::chrono::steady_clock::now();
    
    connections_[connection_id] = std::move(connection);
    return true;
}

void WebSocketManager::remove_connection(const std::string& connection_id) {
    std::unique_lock<std::shared_mutex> lock(connections_mutex_);
    connections_.erase(connection_id);
}

bool WebSocketManager::send_message(const std::string& connection_id, const std::string& message) {
    std::shared_lock<std::shared_mutex> lock(connections_mutex_);
    
    auto it = connections_.find(connection_id);
    if (it == connections_.end() || !it->second->is_active.load()) {
        return false;
    }
    
    // 发送WebSocket消息的实现
    // 这里需要具体的WebSocket协议实现
    return true;
}

void WebSocketManager::cleanup_dead_connections() {
    while (cleanup_running_.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(30));
        
        std::unique_lock<std::shared_mutex> lock(connections_mutex_);
        auto now = std::chrono::steady_clock::now();
        
        auto it = connections_.begin();
        while (it != connections_.end()) {
            auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(
                now - it->second->created_time).count();
            
            if (elapsed > 60 || !it->second->is_active.load()) { // 60分钟超时
                it = connections_.erase(it);
            } else {
                ++it;
            }
        }
    }
}

// === WebServer实现 ===

WebServer::WebServer(BatchScheduler* scheduler, const ServerConfig& config)
    : scheduler_(scheduler), config_(config) {
    
    if (config_.enable_rate_limiting) {
        rate_limiter_ = std::make_unique<RateLimiter>(config_.rate_limit_rpm, config_.rate_limit_burst);
    }
    
    websocket_manager_ = std::make_unique<WebSocketManager>();
    
    std::cout << "Web Server created" << std::endl;
    std::cout << "Listen address: " << config_.host << ":" << config_.port << std::endl;
    std::cout << "API prefix: " << config_.api_prefix << std::endl;
}

WebServer::~WebServer() {
    shutdown();
}

bool WebServer::initialize() {
    if (initialized_.load()) {
        return true;
    }
    
    try {
        // 1. 设置evhtp
        if (!setup_evhtp()) {
            std::cerr << "Failed to setup evhtp" << std::endl;
            return false;
        }
        
        // 2. 设置SSL（如果启用）
        if (config_.enable_ssl && !setup_ssl()) {
            std::cerr << "Failed to setup SSL" << std::endl;
            return false;
        }
        
        // 3. 设置默认路由
        setup_default_routes();
        
        // 4. 启动监控
        if (config_.enable_metrics) {
            start_metrics_collection();
        }
        
        initialized_.store(true);
        std::cout << "Web Server initialized successfully" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during Web Server initialization: " << e.what() << std::endl;
        return false;
    }
}

void WebServer::shutdown() {
    if (!initialized_.load()) {
        return;
    }
    
    std::cout << "Shutting down Web Server..." << std::endl;
    
    shutdown_requested_.store(true);
    stop();
    
    // 等待工作线程结束
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    if (metrics_thread_.joinable()) {
        metrics_thread_.join();
    }
    
    // 清理evhtp资源
    if (evhtp_) {
        evhtp_free(evhtp_);
        evhtp_ = nullptr;
    }
    
    if (evbase_) {
        event_base_free(evbase_);
        evbase_ = nullptr;
    }
    
    initialized_.store(false);
    std::cout << "Web Server shutdown complete" << std::endl;
}

bool WebServer::start() {
    if (!initialized_.load() || running_.load()) {
        return false;
    }
    
    try {
        // 绑定并监听
        int result = evhtp_bind_socket(evhtp_, config_.host.c_str(), config_.port, 1024);
        if (result != 0) {
            std::cerr << "Failed to bind to " << config_.host << ":" << config_.port << std::endl;
            return false;
        }
        
        running_.store(true);
        
        // 启动事件循环
        std::cout << "Web Server started on " << config_.host << ":" << config_.port << std::endl;
        event_base_dispatch(evbase_);
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception starting Web Server: " << e.what() << std::endl;
        return false;
    }
}

void WebServer::stop() {
    if (running_.load()) {
        running_.store(false);
        event_base_loopbreak(evbase_);
    }
}

bool WebServer::setup_evhtp() {
    evbase_ = event_base_new();
    if (!evbase_) {
        return false;
    }
    
    evhtp_ = evhtp_new(evbase_, nullptr);
    if (!evhtp_) {
        return false;
    }
    
    // 设置通用回调
    evhtp_set_gencb(evhtp_, WebServer::on_request, this);
    
    return true;
}

void WebServer::setup_default_routes() {
    std::unique_lock<std::shared_mutex> lock(routes_mutex_);
    
    // 聊天完成接口
    std::string chat_path = config_.api_prefix + "/chat/completions";
    route_handlers_[chat_path] = [this](const ApiRequest& req) {
        return handle_chat_completions(req);
    };
    
    // 模型列表接口
    std::string models_path = config_.api_prefix + "/models";
    route_handlers_[models_path] = [this](const ApiRequest& req) {
        return handle_models(req);
    };
    
    // 状态接口
    std::string status_path = config_.api_prefix + "/status";
    route_handlers_[status_path] = [this](const ApiRequest& req) {
        return handle_status(req);
    };
    
    // 健康检查
    if (config_.enable_health_check) {
        route_handlers_[config_.health_endpoint] = [this](const ApiRequest& req) {
            return handle_health(req);
        };
    }
    
    // 性能指标
    if (config_.enable_metrics) {
        route_handlers_[config_.metrics_endpoint] = [this](const ApiRequest& req) {
            return handle_metrics(req);
        };
    }
}

void WebServer::on_request(evhtp_request_t* req, void* arg) {
    auto* server = static_cast<WebServer*>(arg);
    server->handle_request(req);
}

void WebServer::handle_request(evhtp_request_t* req) {
    auto start_time = std::chrono::high_resolution_clock::now();
    metrics_.total_requests.fetch_add(1);
    
    try {
        // 解析请求
        ApiRequest api_req = parse_request(req);
        
        // 验证请求
        ApiResponse error_response;
        if (!validate_request(api_req, error_response)) {
            send_response(req, error_response);
            return;
        }
        
        // 查找路由处理器
        ApiResponse response;
        {
            std::shared_lock<std::shared_mutex> lock(routes_mutex_);
            auto it = route_handlers_.find(api_req.path);
            if (it != route_handlers_.end()) {
                response = it->second(api_req);
            } else {
                // 尝试静态文件服务
                if (config_.enable_static_files) {
                    response = serve_static_file(api_req.path);
                } else {
                    response = create_error_response(HttpStatus::NOT_FOUND, "Not Found");
                }
            }
        }
        
        // 添加CORS头
        if (config_.enable_cors) {
            web_utils::add_cors_headers(response, config_.allowed_origins);
        }
        
        // 发送响应
        send_response(req, response);
        
        // 更新统计
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        update_metrics(api_req, response, duration);
        
    } catch (const std::exception& e) {
        std::cerr << "Exception handling request: " << e.what() << std::endl;
        auto error_response = create_error_response(HttpStatus::INTERNAL_ERROR, "Internal Server Error");
        send_response(req, error_response);
        metrics_.failed_requests.fetch_add(1);
    }
}

ApiRequest WebServer::parse_request(evhtp_request_t* req) {
    ApiRequest api_req;
    
    // 基本信息
    api_req.method = web_utils::http_method_to_string(
        static_cast<HttpMethod>(evhtp_request_get_method(req)));
    
    const char* uri = req->uri->path->full;
    api_req.path = uri ? uri : "";
    
    const char* query = req->uri->query_raw;
    api_req.query_string = query ? query : "";
    api_req.query_params = web_utils::parse_query_string(api_req.query_string);
    
    // 客户端IP
    api_req.client_ip = get_client_ip(req);
    
    // 请求头
    evhtp_headers_t* headers = req->headers_in;
    if (headers) {
        evhtp_kv_t* kv;
        TAILQ_FOREACH(kv, headers, next) {
            if (kv->key && kv->val) {
                api_req.headers[kv->key] = kv->val;
            }
        }
    }
    
    // 请求体
    if (req->buffer_in) {
        size_t len = evbuffer_get_length(req->buffer_in);
        if (len > 0 && len <= config_.max_request_size) {
            char* data = new char[len + 1];
            evbuffer_copyout(req->buffer_in, data, len);
            data[len] = '\0';
            api_req.body = std::string(data, len);
            delete[] data;
            
            // 解析JSON
            if (api_req.headers["content-type"].find("application/json") != std::string::npos) {
                std::string error;
                web_utils::string_to_json(api_req.body, api_req.json_body, error);
            }
        }
        
        metrics_.total_bytes_received.fetch_add(len);
    }
    
    return api_req;
}

void WebServer::send_response(evhtp_request_t* req, const ApiResponse& response) {
    // 设置状态码
    evhtp_send_reply_start(req, static_cast<evhtp_res>(response.status));
    
    // 设置响应头
    for (const auto& [key, value] : response.headers) {
        evhtp_headers_add_header(req->headers_out, 
                                evhtp_header_new(key.c_str(), value.c_str(), 0, 0));
    }
    
    // 发送响应体
    std::string body;
    if (response.is_json) {
        body = web_utils::json_to_string(response.json_body);
    } else {
        body = response.body;
    }
    
    if (!body.empty()) {
        evbuffer_add(req->buffer_out, body.c_str(), body.length());
        metrics_.total_bytes_sent.fetch_add(body.length());
    }
    
    evhtp_send_reply_end(req);
}

ApiResponse WebServer::handle_chat_completions(const ApiRequest& request) {
    if (request.method != "POST") {
        return create_error_response(HttpStatus::METHOD_NOT_ALLOWED, "Method not allowed");
    }
    
    try {
        // 解析聊天请求
        ChatCompletionRequest chat_req;
        std::string error;
        if (!chat_req.parse_from_json(request.json_body, error)) {
            return create_error_response(HttpStatus::BAD_REQUEST, error);
        }
        
        // 构建多模态输入
        ProcessedMultiModal mm_input;
        mm_input.request_id = generate_request_id();
        
        // 简化：将消息转换为单个文本提示
        std::ostringstream prompt;
        for (const auto& message : chat_req.messages) {
            if (message.isMember("role") && message.isMember("content")) {
                prompt << message["role"].asString() << ": " 
                       << message["content"].asString() << "\n";
            }
        }
        
        // 创建文本输入（简化实现）
        mm_input.text.input_ids = {1, 2, 3}; // 占位符
        mm_input.text.processed_length = static_cast<int>(prompt.str().length());
        mm_input.success = true;
        
        // 提交到调度器
        auto future = scheduler_->submit_request(mm_input, chat_req.priority, 
                                               std::chrono::milliseconds(chat_req.timeout_seconds * 1000));
        
        // 等待结果
        auto inference_response = future.get();
        
        // 构建聊天响应
        ChatCompletionResponse chat_response;
        chat_response.id = mm_input.request_id;
        chat_response.created = web_utils::get_unix_timestamp();
        chat_response.model = chat_req.model;
        
        ChatCompletionResponse::Choice choice;
        choice.index = 0;
        choice.message["role"] = "assistant";
        choice.message["content"] = inference_response.generated_text;
        choice.finish_reason = inference_response.success ? "stop" : "error";
        
        chat_response.choices.push_back(choice);
        
        chat_response.usage.prompt_tokens = mm_input.text.processed_length;
        chat_response.usage.completion_tokens = inference_response.tokens_generated;
        chat_response.usage.total_tokens = chat_response.usage.prompt_tokens + 
                                          chat_response.usage.completion_tokens;
        
        ApiResponse response;
        response.set_json(chat_response.to_json());
        return response;
        
    } catch (const std::exception& e) {
        return create_error_response(HttpStatus::INTERNAL_ERROR, 
                                   "Processing error: " + std::string(e.what()));
    }
}

ApiResponse WebServer::handle_models(const ApiRequest& request) {
    Json::Value models_response;
    models_response["object"] = "list";
    
    Json::Value models_array(Json::arrayValue);
    Json::Value model;
    model["id"] = "gemma-3-27b-it";
    model["object"] = "model";
    model["created"] = web_utils::get_unix_timestamp();
    model["owned_by"] = "wicore";
    models_array.append(model);
    
    models_response["data"] = models_array;
    
    ApiResponse response;
    response.set_json(models_response);
    return response;
}

ApiResponse WebServer::handle_status(const ApiRequest& request) {
    Json::Value status;
    status["status"] = "healthy";
    status["version"] = "1.0.0";
    status["uptime_seconds"] = 3600; // 占位符
    
    // 调度器状态
    if (scheduler_) {
        status["scheduler"]["queue_size"] = scheduler_->get_queue_size();
        status["scheduler"]["active_batches"] = scheduler_->get_active_batch_count();
        
        auto scheduler_stats = scheduler_->get_stats();
        status["scheduler"]["total_requests"] = static_cast<Json::UInt64>(scheduler_stats.total_requests.load());
        status["scheduler"]["successful_requests"] = static_cast<Json::UInt64>(scheduler_stats.successful_requests.load());
        status["scheduler"]["requests_per_second"] = scheduler_stats.requests_per_second.load();
    }
    
    // 服务器统计
    status["server"]["total_requests"] = static_cast<Json::UInt64>(metrics_.total_requests.load());
    status["server"]["active_connections"] = metrics_.active_connections.load();
    status["server"]["avg_latency_ms"] = metrics_.avg_request_latency_ms.load();
    
    ApiResponse response;
    response.set_json(status);
    return response;
}

ApiResponse WebServer::handle_health(const ApiRequest& request) {
    bool healthy = is_healthy();
    
    Json::Value health;
    health["status"] = healthy ? "healthy" : "unhealthy";
    health["timestamp"] = web_utils::get_iso8601_timestamp();
    
    ApiResponse response;
    response.status = healthy ? HttpStatus::OK : HttpStatus::SERVICE_UNAVAILABLE;
    response.set_json(health);
    return response;
}

bool WebServer::is_healthy() const {
    // 检查调度器状态
    if (!scheduler_ || !scheduler_->is_ready()) {
        return false;
    }
    
    // 检查错误率
    uint64_t total = metrics_.total_requests.load();
    uint64_t failed = metrics_.failed_requests.load();
    if (total > 100 && (static_cast<double>(failed) / total) > 0.1) { // 10%错误率阈值
        return false;
    }
    
    return true;
}

bool WebServer::validate_request(const ApiRequest& request, ApiResponse& error_response) {
    // 检查认证
    if (config_.enable_auth && !authenticate_request(request)) {
        error_response = create_error_response(HttpStatus::UNAUTHORIZED, "Unauthorized");
        return false;
    }
    
    // 检查速率限制
    if (config_.enable_rate_limiting && !check_rate_limit(request.client_ip)) {
        error_response = create_error_response(HttpStatus::TOO_MANY_REQUESTS, "Rate limit exceeded");
        return false;
    }
    
    // 检查请求大小
    if (request.body.size() > config_.max_request_size) {
        error_response = create_error_response(HttpStatus::BAD_REQUEST, "Request too large");
        return false;
    }
    
    return true;
}

bool WebServer::authenticate_request(const ApiRequest& request) {
    if (config_.api_key.empty()) {
        return true; // 无需认证
    }
    
    std::string token = web_utils::extract_bearer_token(request);
    return web_utils::validate_api_key(token, config_.api_key);
}

bool WebServer::check_rate_limit(const std::string& client_ip) {
    if (!rate_limiter_) {
        return true;
    }
    
    bool allowed = rate_limiter_->allow_request(client_ip);
    if (!allowed) {
        metrics_.rate_limited_requests.fetch_add(1);
    }
    
    return allowed;
}

std::string WebServer::get_client_ip(evhtp_request_t* req) {
    // 尝试从代理头获取真实IP
    const char* forwarded = evhtp_header_find(req->headers_in, "x-forwarded-for");
    if (forwarded) {
        return std::string(forwarded);
    }
    
    const char* real_ip = evhtp_header_find(req->headers_in, "x-real-ip");
    if (real_ip) {
        return std::string(real_ip);
    }
    
    // 获取连接的远程地址
    struct sockaddr* sa = req->conn->saddr;
    if (sa->sa_family == AF_INET) {
        struct sockaddr_in* sin = (struct sockaddr_in*)sa;
        char ip_str[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &sin->sin_addr, ip_str, INET_ADDRSTRLEN);
        return std::string(ip_str);
    }
    
    return "unknown";
}

std::string WebServer::generate_request_id() {
    return "req_" + std::to_string(next_request_id_.fetch_add(1));
}

ApiResponse WebServer::create_error_response(HttpStatus status, const std::string& message, const std::string& code) {
    Json::Value error;
    error["error"]["message"] = message;
    error["error"]["type"] = "error";
    if (!code.empty()) {
        error["error"]["code"] = code;
    }
    
    ApiResponse response;
    response.status = status;
    response.set_json(error);
    
    // 更新错误统计
    int status_code = static_cast<int>(status);
    if (status_code >= 400 && status_code < 500) {
        metrics_.http_4xx_errors.fetch_add(1);
    } else if (status_code >= 500) {
        metrics_.http_5xx_errors.fetch_add(1);
    }
    
    return response;
}

void WebServer::update_metrics(const ApiRequest& request, const ApiResponse& response, double processing_time_ms) {
    // 更新延迟统计（简化的滑动平均）
    metrics_.avg_request_latency_ms.store(
        (metrics_.avg_request_latency_ms.load() * 0.9) + (processing_time_ms * 0.1));
    
    // 更新成功/失败计数
    if (response.status == HttpStatus::OK) {
        metrics_.successful_requests.fetch_add(1);
    } else {
        metrics_.failed_requests.fetch_add(1);
    }
}

void WebServer::start_metrics_collection() {
    metrics_thread_ = std::thread(&WebServer::collect_system_metrics, this);
}

void WebServer::collect_system_metrics() {
    while (!shutdown_requested_.load()) {
        try {
            // 收集系统指标（简化实现）
            // 实际实现中可以使用系统API获取真实指标
            
            std::this_thread::sleep_for(std::chrono::seconds(10));
        } catch (const std::exception& e) {
            std::cerr << "Error collecting metrics: " << e.what() << std::endl;
        }
    }
}

ApiResponse WebServer::serve_static_file(const std::string& file_path) {
    std::string full_path = config_.static_root + file_path;
    
    if (!web_utils::file_exists(full_path)) {
        return create_error_response(HttpStatus::NOT_FOUND, "File not found");
    }
    
    std::string content = web_utils::read_file(full_path);
    if (content.empty()) {
        return create_error_response(HttpStatus::INTERNAL_ERROR, "Failed to read file");
    }
    
    ApiResponse response;
    response.set_text(content);
    response.headers["Content-Type"] = get_mime_type(full_path);
    
    return response;
}

std::string WebServer::get_mime_type(const std::string& file_path) {
    size_t dot_pos = file_path.find_last_of('.');
    if (dot_pos == std::string::npos) {
        return "application/octet-stream";
    }
    
    std::string ext = file_path.substr(dot_pos + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    static const std::unordered_map<std::string, std::string> mime_types = {
        {"html", "text/html"},
        {"css", "text/css"},
        {"js", "application/javascript"},
        {"json", "application/json"},
        {"png", "image/png"},
        {"jpg", "image/jpeg"},
        {"jpeg", "image/jpeg"},
        {"gif", "image/gif"},
        {"ico", "image/x-icon"}
    };
    
    auto it = mime_types.find(ext);
    return (it != mime_types.end()) ? it->second : "application/octet-stream";
}

// === 工具函数实现 ===

namespace web_utils {

std::string http_status_to_string(HttpStatus status) {
    switch (status) {
        case HttpStatus::OK: return "OK";
        case HttpStatus::BAD_REQUEST: return "Bad Request";
        case HttpStatus::UNAUTHORIZED: return "Unauthorized";
        case HttpStatus::FORBIDDEN: return "Forbidden";
        case HttpStatus::NOT_FOUND: return "Not Found";
        case HttpStatus::TOO_MANY_REQUESTS: return "Too Many Requests";
        case HttpStatus::INTERNAL_ERROR: return "Internal Server Error";
        default: return "Unknown";
    }
}

std::string http_method_to_string(HttpMethod method) {
    switch (method) {
        case HttpMethod::GET: return "GET";
        case HttpMethod::POST: return "POST";
        case HttpMethod::PUT: return "PUT";
        case HttpMethod::DELETE: return "DELETE";
        case HttpMethod::OPTIONS: return "OPTIONS";
        default: return "UNKNOWN";
    }
}

std::string json_to_string(const Json::Value& json, bool pretty) {
    if (pretty) {
        Json::StreamWriterBuilder builder;
        builder["indentation"] = "  ";
        return Json::writeString(builder, json);
    } else {
        Json::StreamWriterBuilder builder;
        builder["indentation"] = "";
        return Json::writeString(builder, json);
    }
}

bool string_to_json(const std::string& str, Json::Value& json, std::string& error) {
    Json::CharReaderBuilder builder;
    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    
    return reader->parse(str.c_str(), str.c_str() + str.length(), &json, &error);
}

int64_t get_unix_timestamp() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string get_iso8601_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::ostringstream oss;
    oss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

void add_cors_headers(ApiResponse& response, const std::vector<std::string>& allowed_origins) {
    response.headers["Access-Control-Allow-Origin"] = "*"; // 简化实现
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS";
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization";
}

std::string extract_bearer_token(const ApiRequest& request) {
    auto it = request.headers.find("authorization");
    if (it != request.headers.end()) {
        const std::string& auth_header = it->second;
        if (auth_header.substr(0, 7) == "Bearer ") {
            return auth_header.substr(7);
        }
    }
    return "";
}

bool validate_api_key(const std::string& api_key, const std::string& expected) {
    return api_key == expected;
}

bool file_exists(const std::string& path) {
    return std::filesystem::exists(path);
}

std::string read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return "";
    }
    
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::string content(size, '\0');
    file.read(&content[0], size);
    
    return content;
}

std::unordered_map<std::string, std::string> parse_query_string(const std::string& query) {
    std::unordered_map<std::string, std::string> params;
    
    if (query.empty()) {
        return params;
    }
    
    std::istringstream iss(query);
    std::string pair;
    
    while (std::getline(iss, pair, '&')) {
        size_t eq_pos = pair.find('=');
        if (eq_pos != std::string::npos) {
            std::string key = pair.substr(0, eq_pos);
            std::string value = pair.substr(eq_pos + 1);
            params[key] = value;
        }
    }
    
    return params;
}

} // namespace web_utils

} // namespace wicore 