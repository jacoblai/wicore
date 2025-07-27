#!/bin/bash
# WiCore C++ 推理引擎 - 一键部署脚本
# 适用于 Ubuntu 22.04 + NVIDIA GPU

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查root权限
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "此脚本需要root权限运行。请使用: sudo $0"
        exit 1
    fi
}

# 检查系统版本
check_system() {
    log_info "检查系统版本..."
    
    if ! grep -q "Ubuntu 22.04" /etc/os-release; then
        log_warning "建议使用 Ubuntu 22.04 LTS"
    fi
    
    log_success "系统检查完成"
}

# 检查GPU
check_gpu() {
    log_info "检查NVIDIA GPU..."
    
    if ! lspci | grep -i nvidia >/dev/null; then
        log_error "未检测到NVIDIA GPU"
        exit 1
    fi
    
    gpu_info=$(lspci | grep -i nvidia | head -1)
    log_success "检测到GPU: $gpu_info"
}

# 更新系统
update_system() {
    log_info "更新系统包..."
    apt update && apt upgrade -y
    apt install -y wget curl git build-essential cmake ninja-build \
        pkg-config software-properties-common gpg-agent
    log_success "系统更新完成"
}

# 安装NVIDIA驱动
install_nvidia_driver() {
    log_info "安装NVIDIA驱动..."
    
    # 检查是否已安装
    if nvidia-smi >/dev/null 2>&1; then
        log_success "NVIDIA驱动已安装"
        return
    fi
    
    # 添加NVIDIA仓库
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
    dpkg -i cuda-keyring_1.0-1_all.deb
    apt update
    
    # 安装驱动
    apt install -y nvidia-driver-535 nvidia-dkms-535
    
    log_warning "NVIDIA驱动安装完成，请重启系统后继续"
    log_warning "重启后运行: nvidia-smi 验证安装"
    
    read -p "是否现在重启? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        reboot
    fi
}

# 安装CUDA
install_cuda() {
    log_info "安装CUDA 12.2..."
    
    # 检查是否已安装
    if nvcc --version >/dev/null 2>&1; then
        log_success "CUDA已安装"
        return
    fi
    
    # 下载并安装CUDA
    CUDA_INSTALLER="cuda_12.2.0_535.54.03_linux.run"
    if [[ ! -f $CUDA_INSTALLER ]]; then
        wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/$CUDA_INSTALLER
    fi
    
    sh $CUDA_INSTALLER --silent --toolkit
    
    # 添加环境变量
    cat >> /etc/environment << EOF
PATH="/usr/local/cuda-12.2/bin:$PATH"
LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH"
EOF
    
    # 添加到当前session
    export PATH=/usr/local/cuda-12.2/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
    
    log_success "CUDA安装完成"
}

# 安装TensorRT
install_tensorrt() {
    log_info "安装TensorRT..."
    
    # 通过APT安装
    apt install -y tensorrt libnvinfer-plugin8 libnvonnxparsers8
    
    # 安装Python bindings (可选)
    apt install -y python3-libnvinfer python3-libnvinfer-dev
    
    log_success "TensorRT安装完成"
}

# 安装依赖库
install_dependencies() {
    log_info "安装依赖库..."
    
    apt install -y \
        libopencv-dev \
        libjsoncpp-dev \
        libevhtp-dev \
        libevent-dev \
        libsentencepiece-dev \
        protobuf-compiler \
        libprotobuf-dev \
        python3-pip \
        apache2-utils
    
    # 安装Python包
    pip3 install huggingface-hub transformers torch
    
    log_success "依赖库安装完成"
}

# 检查依赖版本
check_dependencies() {
    log_info "检查依赖版本..."
    
    # 检查版本
    opencv_version=$(pkg-config --modversion opencv4 2>/dev/null || echo "未找到")
    jsoncpp_version=$(pkg-config --modversion jsoncpp 2>/dev/null || echo "未找到")
    
    log_info "OpenCV版本: $opencv_version"
    log_info "JsonCpp版本: $jsoncpp_version"
    
    # 验证CUDA和TensorRT
    if nvcc --version >/dev/null 2>&1; then
        cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        log_info "CUDA版本: $cuda_version"
    fi
    
    if python3 -c "import tensorrt" >/dev/null 2>&1; then
        trt_version=$(python3 -c "import tensorrt as trt; print(trt.__version__)")
        log_info "TensorRT版本: $trt_version"
    fi
    
    log_success "依赖检查完成"
}

# 创建工作目录
setup_workspace() {
    log_info "创建工作目录..."
    
    WORK_DIR="/opt/wicore_deployment"
    mkdir -p $WORK_DIR/{models,logs,static}
    chmod 755 $WORK_DIR
    
    # 创建缓存目录
    mkdir -p /tmp/wicore_cache
    chmod 777 /tmp/wicore_cache
    
    log_success "工作目录创建完成: $WORK_DIR"
    echo "WORK_DIR=$WORK_DIR" > /etc/environment
}

# 下载模型
download_model() {
    log_info "下载Gemma-3-27B-IT模型..."
    
    WORK_DIR="/opt/wicore_deployment"
    MODEL_DIR="$WORK_DIR/models/gemma-3-27b-it"
    
    if [[ -d $MODEL_DIR && -f $MODEL_DIR/config.json ]]; then
        log_success "模型已存在"
        return
    fi
    
    # 检查磁盘空间 (需要约80GB)
    available_space=$(df $WORK_DIR | tail -1 | awk '{print $4}')
    required_space=$((80 * 1024 * 1024))  # 80GB in KB
    
    if [[ $available_space -lt $required_space ]]; then
        log_error "磁盘空间不足，需要至少80GB可用空间"
        exit 1
    fi
    
    log_info "开始下载模型 (约50-60GB)，请耐心等待..."
    
    # 使用huggingface-cli下载
    su - $SUDO_USER -c "
        cd $WORK_DIR
        huggingface-cli download google/gemma-3-27b-it \
            --local-dir ./models/gemma-3-27b-it \
            --local-dir-use-symlinks False
    " || {
        log_warning "Hugging Face下载失败，尝试git clone..."
        su - $SUDO_USER -c "
            cd $WORK_DIR
            git lfs install
            git clone https://huggingface.co/google/gemma-3-27b-it ./models/gemma-3-27b-it
        "
    }
    
    # 创建引擎缓存目录
    mkdir -p $MODEL_DIR/engine_cache
    chmod 755 $MODEL_DIR/engine_cache
    
    log_success "模型下载完成"
}

# 生成配置文件
generate_config() {
    log_info "生成配置文件..."
    
    WORK_DIR="/opt/wicore_deployment"
    CONFIG_FILE="$WORK_DIR/production_config.json"
    
    # 自动检测GPU内存
    gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    gpu_memory_gb=$((gpu_memory / 1024 - 4))  # 预留4GB
    
    # 自动检测CPU内存
    cpu_memory=$(free -g | awk 'NR==2{print $2}')
    cpu_memory_gb=$((cpu_memory - 8))  # 预留8GB
    
    cat > $CONFIG_FILE << EOF
{
  "model_path": "$WORK_DIR/models/gemma-3-27b-it",
  "tokenizer_path": "$WORK_DIR/models/gemma-3-27b-it/tokenizer.model",
  "server_port": 8080,
  "max_batch_size": 16,
  "max_context_length": 131072,
  "max_concurrent_requests": 32,
  
  "gpu_memory_gb": $gpu_memory_gb,
  "cpu_memory_gb": $cpu_memory_gb,
  "nvme_cache_path": "/tmp/wicore_cache",
  
  "trt_precision": "fp16",
  "trt_max_workspace_gb": 8,
  "trt_enable_sparse": true,
  "trt_enable_refit": true,
  
  "image_resolution": 896,
  "image_preprocessing_threads": 4,
  "max_images_per_request": 10,
  
  "dynamic_batching": true,
  "batch_timeout_ms": 10,
  "request_timeout_seconds": 300,
  
  "enable_performance_logging": true,
  "performance_log_interval_seconds": 60,
  "log_level": "info"
}
EOF
    
    log_success "配置文件生成完成: $CONFIG_FILE"
    log_info "GPU内存配置: ${gpu_memory_gb}GB"
    log_info "CPU内存配置: ${cpu_memory_gb}GB"
}

# 创建systemd服务
create_service() {
    log_info "创建systemd服务..."
    
    WORK_DIR="/opt/wicore_deployment"
    SERVICE_USER=${SUDO_USER:-"wicore"}
    
    cat > /etc/systemd/system/wicore.service << EOF
[Unit]
Description=WiCore Inference Engine
After=network.target

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_USER
WorkingDirectory=$WORK_DIR
ExecStart=$WORK_DIR/wicore_src/build/wicore_server $WORK_DIR/production_config.json
Restart=always
RestartSec=10
Environment=CUDA_VISIBLE_DEVICES=0
Environment=OMP_NUM_THREADS=8
Environment=LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib

[Install]
WantedBy=multi-user.target
EOF
    
    systemctl daemon-reload
    systemctl enable wicore
    
    log_success "systemd服务创建完成"
}

# 创建测试脚本
create_test_scripts() {
    log_info "创建测试脚本..."
    
    WORK_DIR="/opt/wicore_deployment"
    
    # API测试脚本
    cat > $WORK_DIR/test_api.sh << 'EOF'
#!/bin/bash
echo "=== WiCore API测试 ==="

echo "1. 健康检查"
curl -s http://localhost:8080/health | jq .

echo -e "\n2. 模型列表"
curl -s http://localhost:8080/v1/models | jq .

echo -e "\n3. 简单对话测试"
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-27b-it",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }' | jq .

echo -e "\n测试完成!"
EOF
    
    # 性能测试脚本
    cat > $WORK_DIR/benchmark.py << 'EOF'
#!/usr/bin/env python3
import time
import requests
import statistics
import concurrent.futures

def single_request():
    start_time = time.time()
    try:
        response = requests.post('http://localhost:8080/v1/chat/completions', 
            json={
                "model": "gemma-3-27b-it",
                "messages": [{"role": "user", "content": "Count from 1 to 10"}],
                "max_tokens": 100
            }, timeout=30)
        
        latency = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            tokens = data.get('usage', {}).get('completion_tokens', 0)
            return True, latency, tokens
        else:
            return False, latency, 0
    except Exception as e:
        return False, time.time() - start_time, 0

def run_benchmark(num_requests=20, concurrency=1):
    print(f"运行基准测试: {num_requests} 请求, 并发度: {concurrency}")
    
    results = []
    
    if concurrency == 1:
        # 串行测试
        for i in range(num_requests):
            success, latency, tokens = single_request()
            results.append((success, latency, tokens))
            print(f"请求 {i+1}: {'成功' if success else '失败'} {latency:.2f}s {tokens} tokens")
    else:
        # 并发测试
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(single_request) for _ in range(num_requests)]
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                success, latency, tokens = future.result()
                results.append((success, latency, tokens))
                print(f"请求 {i+1}: {'成功' if success else '失败'} {latency:.2f}s {tokens} tokens")
    
    # 统计结果
    successful = [r for r in results if r[0]]
    if successful:
        latencies = [r[1] for r in successful]
        token_counts = [r[2] for r in successful]
        
        avg_latency = statistics.mean(latencies)
        avg_tokens = statistics.mean(token_counts)
        total_tokens = sum(token_counts)
        total_time = sum(latencies)
        
        print(f"\n=== 基准测试结果 ===")
        print(f"成功请求: {len(successful)}/{num_requests}")
        print(f"平均延迟: {avg_latency:.2f}s")
        print(f"平均Token数: {avg_tokens:.1f}")
        print(f"总体吞吐量: {total_tokens/total_time:.1f} tokens/s")
        print(f"单请求吞吐量: {avg_tokens/avg_latency:.1f} tokens/s")

if __name__ == "__main__":
    print("开始性能测试...")
    run_benchmark(num_requests=10, concurrency=1)
EOF
    
    chmod +x $WORK_DIR/test_api.sh
    chmod +x $WORK_DIR/benchmark.py
    
    log_success "测试脚本创建完成"
}

# 显示部署摘要
show_summary() {
    WORK_DIR="/opt/wicore_deployment"
    
    echo
    log_success "=== WiCore部署完成! ==="
    echo
    echo "📁 工作目录: $WORK_DIR"
    echo "⚙️  配置文件: $WORK_DIR/production_config.json"
    echo "📝 日志目录: $WORK_DIR/logs"
    echo "🧪 测试脚本: $WORK_DIR/test_api.sh"
    echo "📊 性能测试: $WORK_DIR/benchmark.py"
    echo
    echo "🚀 下一步操作:"
    echo "1. 编译项目:"
    echo "   cd $WORK_DIR/wicore_src && mkdir -p build && cd build"
    echo "   cmake .. && make -j\$(nproc)"
    echo
    echo "2. 启动服务:"
    echo "   sudo systemctl start wicore"
    echo "   sudo systemctl status wicore"
    echo
    echo "3. 测试API:"
    echo "   cd $WORK_DIR && ./test_api.sh"
    echo
    echo "4. 性能测试:"
    echo "   cd $WORK_DIR && python3 benchmark.py"
    echo
    echo "5. 查看日志:"
    echo "   sudo journalctl -u wicore -f"
    echo
    log_warning "注意: 请确保将WiCore源代码上传到 $WORK_DIR/wicore_src/"
}

# 主函数
main() {
    echo "========================================"
    echo "WiCore C++ 推理引擎 - 一键部署脚本"
    echo "适用于 Ubuntu 22.04 + NVIDIA GPU"
    echo "========================================"
    echo
    
    # 检查权限
    check_root
    
    # 保存原用户
    if [[ -z $SUDO_USER ]]; then
        log_error "请使用sudo运行此脚本"
        exit 1
    fi
    
    # 系统检查
    check_system
    check_gpu
    
    # 询问用户要执行的步骤
    echo "请选择要执行的步骤 (可多选，用空格分隔):"
    echo "1) 更新系统"
    echo "2) 安装NVIDIA驱动"
    echo "3) 安装CUDA"
    echo "4) 安装TensorRT"
    echo "5) 安装依赖库"
    echo "6) 设置工作目录"
    echo "7) 下载模型"
    echo "8) 生成配置"
    echo "9) 创建服务"
    echo "a) 全部执行"
    echo
    read -p "请输入选择 (默认: a): " choices
    choices=${choices:-a}
    
    # 执行选择的步骤
    if [[ $choices == *"a"* || $choices == *"1"* ]]; then
        update_system
    fi
    
    if [[ $choices == *"a"* || $choices == *"2"* ]]; then
        install_nvidia_driver
    fi
    
    if [[ $choices == *"a"* || $choices == *"3"* ]]; then
        install_cuda
    fi
    
    if [[ $choices == *"a"* || $choices == *"4"* ]]; then
        install_tensorrt
    fi
    
    if [[ $choices == *"a"* || $choices == *"5"* ]]; then
        install_dependencies
        check_dependencies
    fi
    
    if [[ $choices == *"a"* || $choices == *"6"* ]]; then
        setup_workspace
    fi
    
    if [[ $choices == *"a"* || $choices == *"7"* ]]; then
        download_model
    fi
    
    if [[ $choices == *"a"* || $choices == *"8"* ]]; then
        generate_config
    fi
    
    if [[ $choices == *"a"* || $choices == *"9"* ]]; then
        create_service
        create_test_scripts
    fi
    
    # 显示摘要
    show_summary
}

# 错误处理
trap 'log_error "脚本执行失败，请查看错误信息"; exit 1' ERR

# 执行主函数
main "$@" 