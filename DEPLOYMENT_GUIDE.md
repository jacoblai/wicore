# WiCore C++ 推理引擎 - 生产部署指南

## 🎯 概述

本指南将指导您在 Ubuntu 22.04 + GPU 服务器上完整部署 WiCore C++ 推理引擎，从模型获取到服务测试的全流程。

**目标环境**: Ubuntu 22.04 LTS + NVIDIA GPU  
**测试模型**: Google Gemma-3-27B-IT  
**预期性能**: 150+ tokens/s @ RTX 4090, 128K上下文支持

---

## 📋 系统要求

### 硬件要求
- **GPU**: NVIDIA RTX 3090/4090 或更高 (≥24GB VRAM)
- **CPU**: Intel/AMD 8核心以上
- **内存**: 64GB RAM 推荐 (32GB 最低)
- **存储**: 500GB 可用空间 (NVMe SSD 推荐)

### 软件环境
- Ubuntu 22.04 LTS
- NVIDIA Driver ≥ 525.x
- CUDA 12.0+
- TensorRT 8.6+
- Docker (可选)

---

## 🔧 第一步：环境准备

### 1.1 更新系统
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y wget curl git build-essential cmake ninja-build
```

### 1.2 安装 NVIDIA 驱动
```bash
# 检查显卡
lspci | grep -i nvidia

# 安装驱动 (建议使用官方驱动)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y nvidia-driver-535 nvidia-dkms-535

# 重启系统
sudo reboot
```

### 1.3 验证驱动安装
```bash
nvidia-smi
# 应该看到 GPU 信息和驱动版本
```

### 1.4 安装 CUDA 12.2
```bash
# 下载 CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
sudo sh cuda_12.2.0_535.54.03_linux.run

# 添加环境变量
echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证安装
nvcc --version
```

### 1.5 安装 TensorRT 8.6
```bash
# 下载 TensorRT (需要NVIDIA开发者账号)
# 方法1: 通过 APT (推荐)
sudo apt install -y tensorrt

# 方法2: 手动下载安装
# 从 https://developer.nvidia.com/tensorrt 下载 TensorRT-8.6.x.x.Ubuntu-22.04.x86_64-gnu.cuda-12.0.tar.gz
# tar -xzf TensorRT-8.6.x.x.Ubuntu-22.04.x86_64-gnu.cuda-12.0.tar.gz
# sudo cp -r TensorRT-8.6.x.x/* /usr/local/
# echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc

# 验证 TensorRT
python3 -c "import tensorrt as trt; print(f'TensorRT version: {trt.__version__}')"
```

---

## 📦 第二步：依赖库安装

### 2.1 安装基础依赖
```bash
sudo apt install -y \
    libopencv-dev \
    libjsoncpp-dev \
    libevhtp-dev \
    libevent-dev \
    pkg-config \
    libsentencepiece-dev \
    protobuf-compiler \
    libprotobuf-dev
```

### 2.2 检查依赖版本
```bash
pkg-config --modversion opencv4  # >= 4.5
pkg-config --modversion jsoncpp  # >= 1.9
pkg-config --modversion sentencepiece  # >= 0.1.96
```

### 2.3 安装 Python 工具 (可选)
```bash
sudo apt install -y python3-pip
pip3 install torch torchvision transformers accelerate
```

---

## 🤖 第三步：模型获取和准备

### 3.1 创建工作目录
```bash
mkdir -p ~/wicore_deployment/models
cd ~/wicore_deployment
```

### 3.2 方法一：使用 Hugging Face Hub (推荐)
```bash
# 安装 huggingface-hub
pip3 install huggingface-hub

# 下载 Gemma-3-27B-IT 模型
huggingface-cli download google/gemma-3-27b-it \
    --local-dir ./models/gemma-3-27b-it \
    --local-dir-use-symlinks False

# 模型文件结构应该是:
# models/gemma-3-27b-it/
# ├── config.json
# ├── model.safetensors.index.json
# ├── model-00001-of-00055.safetensors
# ├── ...
# ├── tokenizer.model
# └── tokenizer_config.json
```

### 3.3 方法二：手动下载 (如果Hub访问困难)
```bash
# 使用 git-lfs 克隆模型
git lfs install
git clone https://huggingface.co/google/gemma-3-27b-it ./models/gemma-3-27b-it

# 或者使用镜像站点
git clone https://hf-mirror.com/google/gemma-3-27b-it ./models/gemma-3-27b-it
```

### 3.4 验证模型文件
```bash
ls -la ./models/gemma-3-27b-it/
du -sh ./models/gemma-3-27b-it/  # 应该约 50-60GB

# 检查 tokenizer
python3 -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('./models/gemma-3-27b-it')
print(f'Tokenizer vocab size: {tokenizer.vocab_size}')
print('Tokenizer loaded successfully!')
"
```

### 3.5 准备 TensorRT 引擎缓存目录
```bash
mkdir -p ./models/gemma-3-27b-it/engine_cache
chmod 755 ./models/gemma-3-27b-it/engine_cache
```

---

## 🛠️ 第四步：代码编译

### 4.1 克隆项目代码
```bash
cd ~/wicore_deployment
git clone <YOUR_REPO_URL> wicore_src
# 或者上传你的代码包
# scp -r /path/to/wicore user@server:~/wicore_deployment/wicore_src
```

### 4.2 编译项目
```bash
cd wicore_src

# 创建构建目录
mkdir -p build && cd build

# 配置 CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.2 \
    -DTensorRT_ROOT=/usr/local/tensorrt \
    -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89"

# 编译 (使用所有 CPU 核心)
make -j$(nproc)

# 验证编译结果
ls -la wicore_server
ldd wicore_server  # 检查动态库依赖
```

### 4.3 解决常见编译问题
```bash
# 如果 TensorRT 路径不对
export TensorRT_ROOT=/usr/lib/x86_64-linux-gnu  # APT 安装路径

# 如果 CUDA 架构不匹配
# RTX 3090: 86, RTX 4090: 89, A100: 80
cmake .. -DCMAKE_CUDA_ARCHITECTURES="89"

# 如果链接错误
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib:$LD_LIBRARY_PATH
```

---

## ⚙️ 第五步：配置文件准备

### 5.1 创建生产配置
```bash
cd ~/wicore_deployment
cp wicore_src/config_template.json production_config.json
```

### 5.2 编辑配置文件
```bash
nano production_config.json
```

**关键配置项**:
```json
{
  "model_path": "/home/username/wicore_deployment/models/gemma-3-27b-it",
  "tokenizer_path": "/home/username/wicore_deployment/models/gemma-3-27b-it/tokenizer.model",
  "server_port": 8080,
  "max_batch_size": 16,
  "max_context_length": 131072,
  "max_concurrent_requests": 32,
  
  "gpu_memory_gb": 20,
  "cpu_memory_gb": 32,
  "nvme_cache_path": "/tmp/wicore_cache",
  
  "trt_precision": "fp16",
  "trt_max_workspace_gb": 8,
  "trt_enable_sparse": true,
  
  "image_resolution": 896,
  "image_preprocessing_threads": 4,
  
  "enable_performance_logging": true,
  "log_level": "info"
}
```

### 5.3 创建缓存目录
```bash
sudo mkdir -p /tmp/wicore_cache
sudo chmod 777 /tmp/wicore_cache
```

### 5.4 创建静态文件目录 (可选)
```bash
mkdir -p static
echo '<h1>WiCore Inference Engine</h1><p>Server is running!</p>' > static/index.html
```

---

## 🚀 第六步：服务启动

### 6.1 首次启动测试
```bash
cd ~/wicore_deployment

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# 启动服务 (前台模式，方便调试)
./wicore_src/build/wicore_server production_config.json
```

**预期输出**:
```
WiCore Engine starting...
Loading configuration from production_config.json
HMT Memory Manager initialized
MultiModal Processor initialized  
TensorRT Inference Engine initialized
Batch Scheduler initialized
Web Server initialized
Server started on 0.0.0.0:8080
Ready to accept requests!
```

### 6.2 后台服务启动
```bash
# 创建日志目录
mkdir -p logs

# 后台启动
nohup ./wicore_src/build/wicore_server production_config.json > logs/wicore.log 2>&1 &

# 查看进程
ps aux | grep wicore_server

# 查看日志
tail -f logs/wicore.log
```

### 6.3 创建系统服务 (可选)
```bash
sudo tee /etc/systemd/system/wicore.service > /dev/null <<EOF
[Unit]
Description=WiCore Inference Engine
After=network.target

[Service]
Type=simple
User=wicore
WorkingDirectory=/home/wicore/wicore_deployment
ExecStart=/home/wicore/wicore_deployment/wicore_src/build/wicore_server production_config.json
Restart=always
RestartSec=10
Environment=CUDA_VISIBLE_DEVICES=0
Environment=OMP_NUM_THREADS=8

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable wicore
sudo systemctl start wicore
sudo systemctl status wicore
```

---

## 🧪 第七步：功能测试

### 7.1 基础连通性测试
```bash
# 健康检查
curl http://localhost:8080/health

# 模型列表
curl http://localhost:8080/v1/models

# 系统状态
curl http://localhost:8080/v1/status
```

### 7.2 聊天完成测试
```bash
# 简单文本对话
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-27b-it",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

### 7.3 流式输出测试
```bash
# 流式对话
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-27b-it",
    "messages": [
      {"role": "user", "content": "Write a short story about AI"}
    ],
    "max_tokens": 300,
    "stream": true
  }'
```

### 7.4 多模态测试 (图像+文本)
```bash
# 准备测试图像
wget https://example.com/test_image.jpg -O test_image.jpg

# 多模态对话 (Base64编码)
base64_image=$(base64 -w 0 test_image.jpg)

curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-27b-it",
    "messages": [
      {
        "role": "user", 
        "content": [
          {"type": "text", "text": "What is in this image?"},
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'$base64_image'"}}
        ]
      }
    ],
    "max_tokens": 200
  }'
```

### 7.5 压力测试
```bash
# 安装测试工具
sudo apt install -y apache2-utils

# 并发测试
ab -n 100 -c 10 -H "Content-Type: application/json" \
   -p test_payload.json \
   http://localhost:8080/v1/chat/completions

# 创建测试载荷
echo '{
  "model": "gemma-3-27b-it",
  "messages": [{"role": "user", "content": "Test message"}],
  "max_tokens": 50
}' > test_payload.json
```

---

## 📊 第八步：性能监控

### 8.1 系统监控
```bash
# GPU 使用率
watch -n 1 nvidia-smi

# 内存使用
watch -n 1 'free -h && df -h'

# 网络连接
ss -tulpn | grep :8080
```

### 8.2 应用监控
```bash
# 性能指标
curl http://localhost:8080/metrics

# 实时日志
tail -f logs/wicore.log

# 进程资源使用
top -p $(pgrep wicore_server)
```

### 8.3 基准测试
```bash
# 创建基准测试脚本
cat > benchmark.py << 'EOF'
#!/usr/bin/env python3
import time
import requests
import threading
import statistics

def benchmark_request():
    start_time = time.time()
    response = requests.post('http://localhost:8080/v1/chat/completions', 
        json={
            "model": "gemma-3-27b-it",
            "messages": [{"role": "user", "content": "Count from 1 to 10"}],
            "max_tokens": 100
        })
    latency = time.time() - start_time
    
    if response.status_code == 200:
        data = response.json()
        tokens = data.get('usage', {}).get('completion_tokens', 0)
        return latency, tokens
    return latency, 0

# 运行基准测试
latencies = []
token_counts = []

print("Running benchmark (20 requests)...")
for i in range(20):
    latency, tokens = benchmark_request()
    latencies.append(latency)
    token_counts.append(tokens)
    print(f"Request {i+1}: {latency:.2f}s, {tokens} tokens")

# 统计结果
avg_latency = statistics.mean(latencies)
avg_tokens = statistics.mean(token_counts)
tokens_per_second = avg_tokens / avg_latency if avg_latency > 0 else 0

print(f"\nBenchmark Results:")
print(f"Average Latency: {avg_latency:.2f}s")
print(f"Average Tokens: {avg_tokens:.1f}")
print(f"Tokens/Second: {tokens_per_second:.1f}")
EOF

python3 benchmark.py
```

---

## 🔧 第九步：故障排除

### 9.1 常见问题

**问题**: CUDA out of memory
```bash
# 解决方案
# 1. 降低 max_batch_size
# 2. 降低 gpu_memory_gb
# 3. 使用 TensorRT 量化
```

**问题**: TensorRT 引擎构建失败
```bash
# 检查 CUDA 架构匹配
nvidia-smi --query-gpu=name,compute_cap --format=csv

# 重新构建引擎
rm -rf models/gemma-3-27b-it/engine_cache/*
```

**问题**: 模型加载慢
```bash
# 使用 NVMe 存储
# 预先构建 TensorRT 引擎
# 启用模型并行加载
```

### 9.2 日志分析
```bash
# 错误日志
grep -i error logs/wicore.log

# 性能日志
grep -i "tokens/s\|latency\|throughput" logs/wicore.log

# 内存日志
grep -i "memory\|oom\|allocation" logs/wicore.log
```

### 9.3 调试模式
```bash
# 启用详细日志
./wicore_src/build/wicore_server production_config.json --log-level debug

# 使用 gdb 调试
gdb ./wicore_src/build/wicore_server
(gdb) run production_config.json
```

---

## 📈 第十步：性能优化

### 10.1 TensorRT 优化
```bash
# 修改配置启用更多优化
"trt_enable_fp16": true,
"trt_enable_int8": false,  # 需要校准数据集
"trt_enable_sparse": true,
"trt_enable_refit": true,
"trt_optimization_level": 5
```

### 10.2 系统调优
```bash
# CPU 调优
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 内存调优
echo 1 | sudo tee /proc/sys/vm/swappiness
echo 3 | sudo tee /proc/sys/vm/drop_caches

# 网络调优
echo 'net.core.rmem_max = 16777216' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max = 16777216' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

### 10.3 GPU 调优
```bash
# 设置 GPU 性能模式
sudo nvidia-smi -pm 1
sudo nvidia-smi -ac 1215,210  # 根据GPU型号调整
```

---

## 🎯 预期性能指标

### RTX 4090 + Gemma-3-27B-IT
- **吞吐量**: 150-200 tokens/s
- **延迟**: 50-100ms (首token)
- **并发**: 32 并发请求
- **内存使用**: GPU 20GB, CPU 16GB
- **上下文长度**: 128K tokens

### RTX 3090 + Gemma-3-27B-IT  
- **吞吐量**: 100-150 tokens/s
- **延迟**: 80-150ms (首token)
- **并发**: 16 并发请求
- **内存使用**: GPU 22GB, CPU 16GB
- **上下文长度**: 64K tokens

---

## ✅ 完成检查清单

- [ ] Ubuntu 22.04 环境准备
- [ ] NVIDIA 驱动安装 (≥525.x)
- [ ] CUDA 12.2 安装和配置
- [ ] TensorRT 8.6 安装和验证
- [ ] 依赖库安装完成
- [ ] Gemma-3-27B-IT 模型下载
- [ ] WiCore 源码编译成功
- [ ] 生产配置文件准备
- [ ] 服务启动和运行
- [ ] API 功能测试通过
- [ ] 性能基准测试完成
- [ ] 监控系统配置
- [ ] 故障排除文档熟悉

---

## 📞 技术支持

如果在部署过程中遇到问题，请提供以下信息：

1. **系统信息**: `uname -a`, `nvidia-smi`, `nvcc --version`
2. **错误日志**: 具体的错误信息和日志
3. **配置文件**: 使用的配置参数
4. **硬件规格**: GPU型号、内存大小等

**部署完成后，您的 WiCore 推理引擎即可为生产环境提供高性能的多模态AI服务！** 🚀 