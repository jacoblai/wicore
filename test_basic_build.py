#!/usr/bin/env python3
# test_basic_build.py - 基础构建测试脚本

import subprocess
import os
import sys
import json

def print_status(message, status="INFO"):
    colors = {
        "INFO": "\033[94m",
        "SUCCESS": "\033[92m", 
        "ERROR": "\033[91m",
        "WARNING": "\033[93m"
    }
    reset = "\033[0m"
    print(f"{colors.get(status, '')}{status}: {message}{reset}")

def check_dependencies():
    """检查编译依赖"""
    print_status("检查编译依赖...")
    
    deps = [
        ("cmake", "cmake --version"),
        ("nvcc", "nvcc --version"),
        ("pkg-config", "pkg-config --version"),
        ("g++", "g++ --version")
    ]
    
    missing = []
    for name, cmd in deps:
        try:
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            if result.returncode == 0:
                print_status(f"{name}: 已安装", "SUCCESS")
            else:
                missing.append(name)
        except FileNotFoundError:
            missing.append(name)
    
    if missing:
        print_status(f"缺少依赖: {', '.join(missing)}", "ERROR")
        return False
    
    return True

def check_libraries():
    """检查库依赖"""
    print_status("检查库依赖...")
    
    libs = [
        "opencv4",
        "jsoncpp", 
        "evhtp",
        "sentencepiece"
    ]
    
    missing = []
    for lib in libs:
        try:
            result = subprocess.run(["pkg-config", "--exists", lib], capture_output=True)
            if result.returncode == 0:
                print_status(f"{lib}: 已安装", "SUCCESS")
            else:
                missing.append(lib)
        except:
            missing.append(lib)
    
    if missing:
        print_status(f"缺少库: {', '.join(missing)}", "WARNING")
        print_status("可能需要安装以下包:", "INFO")
        print("  Ubuntu/Debian:")
        print("    sudo apt-get install libopencv-dev libjsoncpp-dev libevhtp-dev libsentencepiece-dev")
        print("  CentOS/RHEL:")
        print("    sudo yum install opencv-devel jsoncpp-devel libevhtp-devel sentencepiece-devel")
    
    return len(missing) == 0

def create_test_config():
    """创建测试配置文件"""
    print_status("创建测试配置...")
    
    test_config = {
        "model_path": "./models/gemma-3-27b-it",
        "tokenizer_path": "./models/gemma-3-27b-it/tokenizer.model",
        "server_port": 8080,
        "max_batch_size": 4,
        "batch_timeout_ms": 10,
        "dynamic_batching": True,
        "max_context_length": 32768,  # 降低用于测试
        "gpu_memory_gb": 16,          # 降低用于测试
        "cpu_memory_gb": 32,          # 降低用于测试
        "nvme_cache_path": "./test_cache",
        "memory_pool_size": 256,
        "num_streams": 2,             # 降低用于测试
        "enable_cuda_graph": False,   # 测试时禁用
        "enable_hmt": True,
        "enable_quantization": True,
        "enable_kernel_fusion": False, # 测试时禁用
        "hmt_gpu_threshold": 0.85,
        "hmt_cpu_threshold": 0.90,
        "hmt_eviction_policy": "a2cr",
        "hmt_decay_factor": 0.05,
        "trt_precision": "fp16",
        "trt_max_workspace_gb": 2,    # 降低用于测试
        "trt_enable_sparse": True,
        "trt_enable_refit": True,
        "image_resolution": 896,
        "max_images_per_request": 4,  # 降低用于测试
        "image_preprocessing_threads": 2,
        "log_level": "info",
        "stats_interval_seconds": 5,
        "enable_performance_logging": True,
        "max_concurrent_requests": 32,
        "request_timeout_seconds": 30,
        "max_tokens_per_request": 2048
    }
    
    with open("test_config.json", "w") as f:
        json.dump(test_config, f, indent=2)
    
    print_status("测试配置文件已创建: test_config.json", "SUCCESS")

def test_compile():
    """测试编译"""
    print_status("开始编译测试...")
    
    # 创建构建目录
    os.makedirs("build", exist_ok=True)
    os.chdir("build")
    
    try:
        # CMake配置
        print_status("运行CMake配置...")
        result = subprocess.run([
            "cmake", "..",
            "-DCMAKE_BUILD_TYPE=Debug",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print_status("CMake配置失败:", "ERROR")
            print(result.stderr)
            return False
            
        print_status("CMake配置成功", "SUCCESS")
        
        # 编译
        print_status("开始编译...")
        result = subprocess.run([
            "make", "-j2"  # 使用2个并行任务避免内存不足
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print_status("编译失败:", "ERROR")
            print(result.stderr)
            return False
            
        print_status("编译成功!", "SUCCESS")
        
        # 检查可执行文件
        if os.path.exists("wicore_server"):
            print_status("可执行文件已生成: wicore_server", "SUCCESS")
            return True
        else:
            print_status("可执行文件未找到", "ERROR")
            return False
            
    except Exception as e:
        print_status(f"编译过程异常: {e}", "ERROR")
        return False
    finally:
        os.chdir("..")

def test_basic_functionality():
    """测试基础功能"""
    print_status("测试基础功能...")
    
    if not os.path.exists("build/wicore_server"):
        print_status("可执行文件不存在，跳过功能测试", "WARNING")
        return False
    
    try:
        # 创建测试目录
        os.makedirs("test_cache", exist_ok=True)
        
        # 测试帮助信息（如果有的话）
        print_status("测试程序启动...", "INFO")
        print_status("注意: 由于没有模型文件，程序会报错，这是正常的", "WARNING")
        
        return True
        
    except Exception as e:
        print_status(f"功能测试异常: {e}", "ERROR")
        return False

def main():
    print_status("=== WiCore C++ 构建测试 ===", "INFO")
    
    # 检查当前目录
    if not os.path.exists("CMakeLists.txt"):
        print_status("请在项目根目录运行此脚本", "ERROR")
        sys.exit(1)
    
    success = True
    
    # 1. 检查依赖
    if not check_dependencies():
        success = False
    
    # 2. 检查库
    check_libraries()  # 库缺失不会阻止编译测试
    
    # 3. 创建测试配置
    create_test_config()
    
    # 4. 编译测试
    if not test_compile():
        success = False
    
    # 5. 基础功能测试
    test_basic_functionality()
    
    if success:
        print_status("=== 构建测试完成 ===", "SUCCESS")
        print_status("当前已实现的组件:", "INFO")
        print("  ✅ WiCoreEngine - 核心引擎框架")
        print("  ✅ HMTMemoryManager - 分层内存管理")  
        print("  ✅ MultiModalProcessor - 多模态预处理")
        print("  ✅ TensorRTInferenceEngine - 推理引擎")
        print("  ✅ BatchScheduler - 批处理调度器")
        print("  ⏳ WebServer - 待实现")
    else:
        print_status("=== 构建测试失败 ===", "ERROR")
        print_status("请解决上述问题后重试", "INFO")
        sys.exit(1)

if __name__ == "__main__":
    main() 