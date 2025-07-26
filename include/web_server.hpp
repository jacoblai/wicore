// web_server.hpp
#ifndef WEB_SERVER_HPP
#define WEB_SERVER_HPP

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <chrono>
#include <functional>
#include <queue>
#include <iomanip>
#include <sstream>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <evhtp.h>
#include <json/json.h>

#include "batch_scheduler.hpp"

namespace wicore {

// 前向声明
class BatchScheduler;

// HTTP方法枚举
enum class HttpMethod {
    GET,
    POST,
    PUT,
    DELETE,
    OPTIONS
};

// 响应状态码
enum class HttpStatus {
    OK = 200,
    CREATED = 201,
    ACCEPTED = 202,
    BAD_REQUEST = 400,
    UNAUTHORIZED = 401,
    FORBIDDEN = 403,
    NOT_FOUND = 404,
    METHOD_NOT_ALLOWED = 405,
    TOO_MANY_REQUESTS = 429,
    INTERNAL_ERROR = 500,
    BAD_GATEWAY = 502,
    SERVICE_UNAVAILABLE = 503
};

// API请求结构
struct ApiRequest {
    std::string method;
    std::string path;
    std::string query_string;
    std::unordered_map<std::string, std::string> headers;
    std::string body;
    std::string client_ip;
    std::chrono::steady_clock::time_point received_time;
    
    // 解析后的数据
    Json::Value json_body;
    std::unordered_map<std::string, std::string> query_params;
    
    ApiRequest() : received_time(std::chrono::steady_clock::now()) {}
};

// API响应结构
struct ApiResponse {
    HttpStatus status = HttpStatus::OK;
    std::unordered_map<std::string, std::string> headers;
    std::string body;
    Json::Value json_body;
    bool is_json = true;
    
    void set_json(const Json::Value& json) {
        json_body = json;
        is_json = true;
        headers["Content-Type"] = "application/json";
    }
    
    void set_text(const std::string& text) {
        body = text;
        is_json = false;
        headers["Content-Type"] = "text/plain";
    }
};

// 聊天完成请求（OpenAI兼容）
struct ChatCompletionRequest {
    std::string model = "gemma-3-27b-it";
    std::vector<Json::Value> messages;
    float temperature = 1.0f;
    float top_p = 0.9f;
    int top_k = 50;
    int max_tokens = 512;
    bool stream = false;
    std::vector<std::string> stop;
    std::string user;
    
    // WiCore扩展参数
    RequestPriority priority = RequestPriority::NORMAL;
    int timeout_seconds = 30;
    
    bool parse_from_json(const Json::Value& json, std::string& error);
};

// 聊天完成响应（OpenAI兼容）
struct ChatCompletionResponse {
    std::string id;
    std::string object = "chat.completion";
    int64_t created;
    std::string model;
    
    struct Choice {
        int index = 0;
        Json::Value message;
        std::string finish_reason = "stop";
    };
    std::vector<Choice> choices;
    
    struct Usage {
        int prompt_tokens = 0;
        int completion_tokens = 0;
        int total_tokens = 0;
    };
    Usage usage;
    
    Json::Value to_json() const;
};

// 流式响应片段
struct StreamChunk {
    std::string id;
    std::string object = "chat.completion.chunk";
    int64_t created;
    std::string model;
    
    struct Delta {
        std::string role;
        std::string content;
    };
    
    struct Choice {
        int index = 0;
        Delta delta;
        std::string finish_reason;
    };
    std::vector<Choice> choices;
    
    Json::Value to_json() const;
};

// 速率限制器
class RateLimiter {
public:
    explicit RateLimiter(int requests_per_minute = 60, int burst_size = 10);
    
    bool allow_request(const std::string& client_ip);
    void reset_client(const std::string& client_ip);
    int get_remaining_requests(const std::string& client_ip);
    
private:
    struct ClientBucket {
        int tokens = 0;
        std::chrono::steady_clock::time_point last_refill;
    };
    
    int requests_per_minute_;
    int burst_size_;
    double refill_rate_; // tokens per second
    
    std::unordered_map<std::string, ClientBucket> client_buckets_;
    mutable std::mutex buckets_mutex_;
    
    void refill_bucket(ClientBucket& bucket, 
                      const std::chrono::steady_clock::time_point& now);
};

// WebSocket连接管理
class WebSocketManager {
public:
    WebSocketManager();
    ~WebSocketManager();
    
    bool add_connection(evhtp_request_t* req, const std::string& connection_id);
    void remove_connection(const std::string& connection_id);
    bool send_message(const std::string& connection_id, const std::string& message);
    bool broadcast_message(const std::string& message);
    
    size_t get_connection_count() const;
    std::vector<std::string> get_active_connections() const;

private:
    struct Connection {
        evhtp_request_t* request;
        std::chrono::steady_clock::time_point created_time;
        std::atomic<bool> is_active{true};
    };
    
    std::unordered_map<std::string, std::unique_ptr<Connection>> connections_;
    mutable std::shared_mutex connections_mutex_;
    
    void cleanup_dead_connections();
    std::thread cleanup_thread_;
    std::atomic<bool> cleanup_running_{true};
};

// 服务器配置
struct ServerConfig {
    // 基本配置
    std::string host = "0.0.0.0";
    int port = 8080;
    int num_threads = 4;
    int max_connections = 1000;
    
    // SSL配置
    bool enable_ssl = false;
    std::string ssl_cert_file;
    std::string ssl_key_file;
    
    // API配置
    std::string api_prefix = "/v1";
    bool enable_cors = true;
    std::vector<std::string> allowed_origins = {"*"};
    
    // 速率限制
    bool enable_rate_limiting = true;
    int rate_limit_rpm = 60;    // requests per minute
    int rate_limit_burst = 10;
    
    // 请求限制
    size_t max_request_size = 10 * 1024 * 1024; // 10MB
    int request_timeout_seconds = 30;
    
    // 认证
    bool enable_auth = false;
    std::string api_key;
    
    // 监控
    bool enable_metrics = true;
    std::string metrics_endpoint = "/metrics";
    bool enable_health_check = true;
    std::string health_endpoint = "/health";
    
    // 静态文件服务
    bool enable_static_files = true;
    std::string static_root = "./static";
    std::string index_file = "index.html";
};

// 性能指标
struct ServerMetrics {
    // 请求统计
    std::atomic<uint64_t> total_requests{0};
    std::atomic<uint64_t> successful_requests{0};
    std::atomic<uint64_t> failed_requests{0};
    std::atomic<uint64_t> rate_limited_requests{0};
    
    // 延迟统计
    std::atomic<double> avg_request_latency_ms{0.0};
    std::atomic<double> avg_processing_time_ms{0.0};
    std::atomic<double> p95_latency_ms{0.0};
    
    // 连接统计
    std::atomic<int> active_connections{0};
    std::atomic<int> websocket_connections{0};
    std::atomic<uint64_t> total_bytes_sent{0};
    std::atomic<uint64_t> total_bytes_received{0};
    
    // 错误统计
    std::atomic<uint64_t> http_4xx_errors{0};
    std::atomic<uint64_t> http_5xx_errors{0};
    std::atomic<uint64_t> timeouts{0};
    
    // 系统资源
    std::atomic<double> cpu_usage{0.0};
    std::atomic<double> memory_usage_mb{0.0};
    std::atomic<int> open_file_descriptors{0};
};

// Web服务器主类
class WebServer {
public:
    explicit WebServer(BatchScheduler* scheduler, const ServerConfig& config = ServerConfig{});
    ~WebServer();
    
    // 核心接口
    bool initialize();
    void shutdown();
    bool start();
    void stop();
    
    // 配置管理
    void update_config(const ServerConfig& config);
    ServerConfig get_config() const;
    
    // 状态查询
    bool is_running() const { return running_.load(); }
    bool is_healthy() const;
    
    // 性能统计
    ServerMetrics get_metrics() const;
    void reset_metrics();
    
    // 路由注册
    void register_route(HttpMethod method, const std::string& path, 
                       std::function<ApiResponse(const ApiRequest&)> handler);
    void register_websocket_handler(const std::string& path,
                                   std::function<void(const std::string&, const std::string&)> handler);

private:
    // 依赖组件
    BatchScheduler* scheduler_;
    ServerConfig config_;
    
    // 核心组件
    std::unique_ptr<RateLimiter> rate_limiter_;
    std::unique_ptr<WebSocketManager> websocket_manager_;
    
    // evhtp相关
    evbase_t* evbase_ = nullptr;
    evhtp_t* evhtp_ = nullptr;
    std::vector<std::thread> worker_threads_;
    
    // 状态管理
    std::atomic<bool> initialized_{false};
    std::atomic<bool> running_{false};
    std::atomic<bool> shutdown_requested_{false};
    
    // 统计信息
    mutable ServerMetrics metrics_;
    mutable std::mutex metrics_mutex_;
    
    // 路由管理
    std::unordered_map<std::string, std::function<ApiResponse(const ApiRequest&)>> route_handlers_;
    std::unordered_map<std::string, std::function<void(const std::string&, const std::string&)>> websocket_handlers_;
    mutable std::shared_mutex routes_mutex_;
    
    // 内部方法 - 初始化
    bool setup_evhtp();
    bool setup_ssl();
    void setup_default_routes();
    void setup_cors_headers();
    
    // 请求处理
    static void on_request(evhtp_request_t* req, void* arg);
    void handle_request(evhtp_request_t* req);
    ApiRequest parse_request(evhtp_request_t* req);
    void send_response(evhtp_request_t* req, const ApiResponse& response);
    
    // API端点实现
    ApiResponse handle_chat_completions(const ApiRequest& request);
    ApiResponse handle_models(const ApiRequest& request);
    ApiResponse handle_status(const ApiRequest& request);
    ApiResponse handle_health(const ApiRequest& request);
    ApiResponse handle_metrics(const ApiRequest& request);
    
    // 流式处理
    void handle_streaming_request(const ChatCompletionRequest& chat_req,
                                 evhtp_request_t* req);
    void send_stream_chunk(evhtp_request_t* req, const StreamChunk& chunk);
    void send_stream_end(evhtp_request_t* req);
    
    // WebSocket处理
    static void on_websocket_message(evhtp_request_t* req, void* arg);
    void handle_websocket_connection(evhtp_request_t* req);
    
    // 请求验证
    bool validate_request(const ApiRequest& request, ApiResponse& error_response);
    bool authenticate_request(const ApiRequest& request);
    bool check_rate_limit(const std::string& client_ip);
    
    // 工具方法
    std::string get_client_ip(evhtp_request_t* req);
    std::string generate_request_id();
    void update_metrics(const ApiRequest& request, const ApiResponse& response, 
                       double processing_time_ms);
    
    // 错误处理
    ApiResponse create_error_response(HttpStatus status, const std::string& message,
                                    const std::string& code = "");
    void log_request(const ApiRequest& request, const ApiResponse& response,
                    double processing_time_ms);
    
    // 静态文件服务
    ApiResponse serve_static_file(const std::string& file_path);
    std::string get_mime_type(const std::string& file_path);
    
    // 监控
    void start_metrics_collection();
    void collect_system_metrics();
    std::thread metrics_thread_;
    
    std::atomic<uint64_t> next_request_id_{1};
    
    // 禁用拷贝
    WebServer(const WebServer&) = delete;
    WebServer& operator=(const WebServer&) = delete;
};

// 工具函数
namespace web_utils {
    // HTTP工具
    std::string http_status_to_string(HttpStatus status);
    std::string http_method_to_string(HttpMethod method);
    HttpMethod string_to_http_method(const std::string& method);
    
    // JSON工具
    std::string json_to_string(const Json::Value& json, bool pretty = false);
    bool string_to_json(const std::string& str, Json::Value& json, std::string& error);
    
    // URL工具
    std::string url_encode(const std::string& str);
    std::string url_decode(const std::string& str);
    std::unordered_map<std::string, std::string> parse_query_string(const std::string& query);
    
    // 时间工具
    int64_t get_unix_timestamp();
    std::string get_iso8601_timestamp();
    
    // CORS工具
    void add_cors_headers(ApiResponse& response, const std::vector<std::string>& allowed_origins);
    bool is_preflight_request(const ApiRequest& request);
    
    // 认证工具
    std::string extract_bearer_token(const ApiRequest& request);
    bool validate_api_key(const std::string& api_key, const std::string& expected);
    
    // 文件工具
    bool file_exists(const std::string& path);
    std::string read_file(const std::string& path);
    size_t get_file_size(const std::string& path);
}

} // namespace wicore

#endif // WEB_SERVER_HPP 