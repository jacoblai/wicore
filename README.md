# WiCore HMT推理引擎

<div align="center">

🧠 **支持千亿模型单卡部署的分层内存管理推理引擎**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-orange.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

## 🌟 核心特性

### 🚀 **HMT (Hierarchical Memory Tiering) 分层内存管理**
集成2024-2025最新内存优化技术，支持千亿模型单卡部署：

- **🔄 MiniKV**: 2位量化KV缓存 (ArXiv 2411.18077)
- **🏗️ LaCache**: 3层阶梯形缓存结构 (ArXiv 2507.14204)  
- **🎯 HeadInfer**: 头级别KV缓存offloading (ArXiv 2502.12574)
- **🎵 SYMPHONY**: 多轮交互优化 (ArXiv 2412.16434)
- **📦 vTensor**: GPU虚拟内存管理 (ArXiv 2407.15309)
- **🧩 Jenga**: 异构嵌入内存分配 (ArXiv 2503.18292)

### 💻 **生产特性**
- ✅ 单GPU 16GB内存支持7B-70B模型
- ✅ FastAPI异步API服务
- ✅ 流式推理支持
- ✅ 多模型动态加载
- ✅ 详细性能监控和日志

## 📋 环境要求

### 硬件要求
- **GPU**: NVIDIA GPU with 16GB+ VRAM (推荐RTX 4090/A100)
- **内存**: 32GB+ 系统内存
- **存储**: 100GB+ 可用空间

### 软件环境
- **操作系统**: Linux (推荐Ubuntu 20.04+)
- **Python**: 3.8+
- **CUDA**: 12.1+ 
- **驱动**: NVIDIA Driver 530+

## 🛠️ 快速开始

### 1️⃣ 环境安装

```bash
# 克隆项目
git clone <repository-url>
cd wicore

# 安装依赖 (使用阿里云镜像)
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 可选：安装量化支持
pip install bitsandbytes -i https://mirrors.aliyun.com/pypi/simple/
```

### 2️⃣ 模型下载

使用内置脚本下载Qwen2.5-7B模型：

```bash
# 从ModelScope下载 (推荐国内用户)
python3 download_qwen_simple.py

# 下载完成后模型将位于: models/Qwen2.5-7B-Instruct/
```

### 3️⃣ 启动服务

```bash
# 使用生产配置启动
python3 -m wicore --config configs/production.yaml

# 服务启动后访问: http://localhost:8000
```

### 4️⃣ API使用

```bash
# 测试API
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "你好，请介绍一下人工智能"}],
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

## ⚙️ 配置文件

### 生产配置 (`configs/production.yaml`)
完整的HMT生产环境配置，启用所有内存优化技术：

```yaml
# 模型配置
model:
  model_path: "models/Qwen2.5-7B-Instruct"
  model_type: "qwen"
  device_map: "cuda:0"
  torch_dtype: "float16"

# HMT内存管理
hmt:
  enable_hmt: true
  enable_minikv: true      # 2位量化缓存
  enable_lacache: true     # 阶梯形缓存
  enable_head_offload: true # 头级别offloading
  enable_symphony: true    # 多轮优化
  enable_vtensor: true     # 虚拟内存
  enable_jenga: true       # 异构分配
```

### 示例配置 (`configs/qwen25_7b.yaml`)
针对Qwen2.5-7B优化的示例配置。

## 🔬 HMT技术验证

运行完整的HMT验证测试：

```bash
python3 test_hmt_validation.py
```

验证报告将显示所有HMT技术的运行状态：

```
🔬 HMT核心技术验证:
   分层内存管理: ✅ 验证通过
   MiniKV量化缓存: ✅ 验证通过  
   LaCache阶梯缓存: ✅ 验证通过
   HeadInfer offloading: ✅ 验证通过
   SYMPHONY多轮优化: ✅ 验证通过
   vTensor虚拟内存: ✅ 验证通过
   Jenga异构分配: ✅ 验证通过
```

## 📊 性能监控

### 内存监控
```bash
# GPU内存使用情况
nvidia-smi

# 系统内存监控
htop
```

### API统计
访问 `http://localhost:8000/stats` 查看详细性能统计。

## 🚧 故障排除

### 常见问题

**Q: 模型加载失败 "CUDA out of memory"**
```bash
# 启用INT4量化
pip install bitsandbytes
# 在配置文件中设置: enable_quantization: true
```

**Q: 依赖安装失败**
```bash
# 使用阿里云镜像
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

**Q: NCCL错误**
```bash
# 重新安装PyTorch
pip uninstall torch torchvision torchaudio
pip install torch==2.4.0 torchvision torchaudio -i https://mirrors.aliyun.com/pypi/simple/
```

### 日志查看
```bash
# 查看运行日志
tail -f logs/wicore.log

# 查看HMT验证报告
cat logs/hmt_validation_report.json
```

## 📚 技术文档

- [HMT技术设计](hmt.md) - 分层内存管理技术详解
- [核心架构设计](WICORE_MOJO_DESIGN.md) - 系统架构文档

## 🤝 支持的模型

当前支持的模型系列：
- **Qwen2.5** (7B/14B/32B/72B)
- **Llama3.1/3.2** (8B/70B/405B)  
- **Gemma2/3** (2B/9B/27B)
- **其他Transformer架构模型**

## 🎯 设计目标

WiCore致力于实现：
- 🚀 千亿模型单卡部署
- 💾 128K上下文支持
- ⚡ 毫秒级推理延迟
- 🔋 最优内存利用率

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

<div align="center">

**🌟 如果此项目对您有帮助，请给我们一个Star！ 🌟**

</div>