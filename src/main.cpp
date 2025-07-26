// main.cpp - WiCore推理引擎主程序
#include "../include/wicore_engine.hpp"
#include <iostream>
#include <signal.h>
#include <fstream>

using namespace wicore;

std::unique_ptr<WiCoreEngine> g_engine;

void signal_handler(int signal) {
    std::cout << "\n收到退出信号，正在关闭WiCore引擎..." << std::endl;
    if (g_engine) {
        g_engine->shutdown();
    }
    exit(0);
}

bool load_config(const std::string& config_path, Json::Value& config) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        std::cerr << "无法打开配置文件: " << config_path << std::endl;
        return false;
    }
    
    Json::Reader reader;
    if (!reader.parse(file, config)) {
        std::cerr << "配置文件解析错误: " << reader.getFormattedErrorMessages() << std::endl;
        return false;
    }
    
    return true;
}

void print_banner() {
    std::cout << R"(
████╗    ██╗██╗ ██████╗ ██████╗ ██████╗ ███████╗
██╔═██╗  ██║██║██╔════╝██╔═══██╗██╔══██╗██╔════╝
██║██╔██╗██║██║██║     ██║   ██║██████╔╝█████╗  
██║╚═╝ ██╗██║██║██║     ██║   ██║██╔══██╗██╔══╝  
██║    ╚═╝██║██║╚██████╗╚██████╔╝██║  ██║███████╗
╚═╝       ╚═╝╚═╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝

WiCore C++推理引擎 v1.0
面向Gemma-3-27B-IT的极致性能实现
)" << std::endl;
}

void print_system_info() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    std::cout << "=== 系统信息 ===" << std::endl;
    std::cout << "GPU设备数量: " << device_count << std::endl;
    
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        std::cout << "GPU " << i << ": " << prop.name << std::endl;
        std::cout << "  显存: " << prop.totalGlobalMem / (1024*1024) << "MB" << std::endl;
        std::cout << "  SM数量: " << prop.multiProcessorCount << std::endl;
        std::cout << "  计算能力: " << prop.major << "." << prop.minor << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    // 打印启动信息
    print_banner();
    print_system_info();
    
    // 检查命令行参数
    if (argc != 2) {
        std::cerr << "用法: " << argv[0] << " <config.json>" << std::endl;
        return 1;
    }
    
    // 注册信号处理器
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    try {
        // 加载配置
        Json::Value config;
        if (!load_config(argv[1], config)) {
            return 1;
        }
        
        std::cout << "正在加载配置文件: " << argv[1] << std::endl;
        
        // 创建WiCore引擎
        g_engine = std::make_unique<WiCoreEngine>(argv[1]);
        
        // 初始化引擎
        std::cout << "正在初始化WiCore引擎..." << std::endl;
        if (!g_engine->initialize()) {
            std::cerr << "引擎初始化失败!" << std::endl;
            return 1;
        }
        
        // 加载模型
        std::string model_path = config["model_path"].asString();
        std::cout << "正在加载模型: " << model_path << std::endl;
        
        if (!g_engine->load_model(model_path)) {
            std::cerr << "模型加载失败!" << std::endl;
            return 1;
        }
        
        std::cout << "✅ WiCore引擎启动成功!" << std::endl;
        std::cout << "服务端口: " << config.get("server_port", 8080).asInt() << std::endl;
        std::cout << "最大批处理大小: " << config.get("max_batch_size", 16).asInt() << std::endl;
        std::cout << "最大上下文长度: " << config.get("max_context_length", 131072).asInt() << std::endl;
        std::cout << "\n按 Ctrl+C 退出服务..." << std::endl;
        
        // 主循环 - 定期打印统计信息
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(10));
            
            auto stats = g_engine->get_stats();
            std::cout << "\n=== 运行统计 ===" << std::endl;
            std::cout << "总请求数: " << stats.total_requests << std::endl;
            std::cout << "成功请求: " << stats.successful_requests << std::endl;
            std::cout << "平均延迟: " << stats.avg_latency << "ms" << std::endl;
            std::cout << "GPU利用率: " << stats.gpu_utilization << "%" << std::endl;
            std::cout << "内存使用: " << stats.memory_usage / (1024*1024) << "MB" << std::endl;
            std::cout << "吞吐量: " << stats.throughput << " req/s" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 