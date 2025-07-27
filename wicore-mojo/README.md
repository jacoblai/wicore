# WiCore Mojo 推理引擎

🚀 **基于 Mojo 语言的自主可控高性能 AI 推理引擎**

WiCore 是专为中国算力受限环境设计的异构硬件统一调度推理平台，集成了 HMT 分层内存管理和 MoR 动态路由技术，支持千亿参数模型的高效推理。

## 🎯 核心特性

### 技术优势
- **🔒 自主可控**: 摆脱 NVIDIA TensorRT 依赖，避免技术封锁风险
- **🌐 硬件无关**: 支持所有 GPU 品牌（NVIDIA、AMD、Intel、国产等）
- **⚡ 极致性能**: Mojo 68,000x Python 性能，原生硬件编译优化
- **🧠 智能调度**: HMT 三级存储 + A²CR 缓存算法 + MoR 动态路由
- **🔌 生态兼容**: 100% Python 兼容，OpenAI API 标准接口

### 目标性能 (T10 双卡)
- **吞吐量**: 100-150 tokens/s
- **延迟**: 50-100ms (首 token)
- **并发**: 16-32 并发请求
- **内存利用率**: >85%

## 🏗️ 系统架构

```
┌─────────────────────────────────────────┐
│           WiCore Mojo Engine            │
├─────────────────────────────────────────┤
│  Web API Layer (FastAPI/Mojo)          │
├─────────────────────────────────────────┤
│  Request Orchestrator (Mojo)           │
├─────────────────────────────────────────┤
│  Compute Scheduler (MAX Engine)        │
├─────────────────────────────────────────┤
│ ┌─────────┐ ┌─────────┐ ┌─────────┐     │
│ │Compute  │ │Compute  │ │Compute  │     │
│ │Node 1   │ │Node 2   │ │Node N   │     │
│ │(GPU/CPU)│ │(GPU/CPU)│ │(...)    │     │
│ └─────────┘ └─────────┘ └─────────┘     │
├─────────────────────────────────────────┤
│  HMT Memory Manager (Mojo + MAX)       │
├─────────────────────────────────────────┤
│  Hardware Abstraction Layer (MAX)      │
├─────────────────────────────────────────┤
│ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐         │
│ │T10#1│ │T10#2│ │ CPU │ │ ... │         │
│ └─────┘ └─────┘ └─────┘ └─────┘         │
└─────────────────────────────────────────┘
```

### 核心组件

1. **设备管理器** (`device_manager.mojo`)
   - 异构硬件设备发现和抽象
   - 设备拓扑构建和带宽优化
   - NUMA 感知的设备绑定

2. **HMT 分层内存管理器** (`hmt_memory_manager.mojo`)
   - GPU 显存 → CPU 内存 → NVMe 存储三级缓存
   - A²CR (Attention-Aware Cache Replacement) 智能置换算法
   - 零拷贝内存池和异步数据迁移

3. **模型执行器** (`model_executor.mojo`)
   - Gemma-3-27B 模型加载和推理
   - MoR (Mixture of Routers) 动态路由
   - 多 GPU 协调和批处理优化

4. **请求调度器** (`request_scheduler.mojo`)
   - 优先级队列和智能批处理
   - 异步执行和负载均衡
   - 超时处理和错误恢复

5. **Web 服务器** (`web_server.mojo`)
   - OpenAI 兼容 API 接口
   - 流式输出和健康监控
   - 高并发请求处理

## 🛠️ 安装指南

### 系统要求

**开发环境** (macOS/Linux):
- Python 3.8+
- 8GB+ RAM
- 支持模拟模式开发

**生产环境** (Linux):
- NVIDIA T10 双卡 (或其他 GPU)
- 64GB+ RAM
- NVMe SSD
- Modular SDK + MAX Engine

### 快速安装

```bash
# 1. 克隆项目
git clone <repository_url>
cd wicore-mojo

# 2. 运行环境搭建脚本
chmod +x scripts/setup.sh
./scripts/setup.sh

# 3. 激活 Python 环境
source venv/bin/activate

# 4. 运行测试验证
python scripts/test_engine.py
```

### 生产环境部署

```bash
# 1. 下载 Gemma-3-27B 模型（详见下方模型下载指南）
mkdir -p models
# 按照模型下载指南下载模型文件

# 2. 配置生产环境
cp configs/development.json configs/production.json
# 编辑 production.json 设置 GPU 配置

# 3. 启动推理引擎
./scripts/start_engine.sh --config configs/production.json
```

## 📦 模型下载指南

### 支持的模型

WiCore 目前主要支持以下模型：

| 模型名称 | 参数量 | 存储空间 | 推荐硬件 | 状态 |
|---------|-------|----------|----------|------|
| Gemma-3-27B-IT | 27B | ~54GB | T10 双卡 | ✅ 主要支持 |
| Gemma-3-9B-IT | 9B | ~18GB | T10 单卡 | 🔄 开发中 |
| Llama-3.1-8B | 8B | ~16GB | T10 单卡 | 🔄 开发中 |

### 方法一：使用 Hugging Face Hub（推荐）

```bash
# 安装 huggingface-hub
pixi add huggingface-hub

# 下载 Gemma-3-27B-IT 模型
cd models
pixi run python -c "
from huggingface_hub import snapshot_download
import os

# 创建模型目录
os.makedirs('gemma-3-27b-it', exist_ok=True)

# 下载模型文件
snapshot_download(
    repo_id='google/gemma-2-27b-it',
    cache_dir='./cache',
    local_dir='./gemma-3-27b-it',
    local_dir_use_symlinks=False,
    resume_download=True
)

print('✅ 模型下载完成')
"
```

### 方法二：使用 Git LFS

```bash
# 安装 Git LFS
sudo apt install git-lfs  # Ubuntu/Debian
# 或 brew install git-lfs  # macOS

# 初始化 Git LFS
git lfs install

# 克隆模型仓库
cd models
git clone https://huggingface.co/google/gemma-2-27b-it gemma-3-27b-it

# 验证下载完整性
cd gemma-3-27b-it
git lfs ls-files  # 查看 LFS 文件列表
```

### 方法三：手动下载（适用于离线环境）

如果无法直接访问 Hugging Face，可以通过以下方式获取模型：

1. **通过镜像站下载**：
```bash
# 使用国内镜像（如 ModelScope）
cd models
git clone https://modelscope.cn/google/gemma-2-27b-it.git gemma-3-27b-it
```

2. **分块下载**：
```bash
# 使用 wget 分块下载（适用于网络不稳定的情况）
cd models/gemma-3-27b-it
wget -c https://huggingface.co/google/gemma-2-27b-it/resolve/main/model-00001-of-00109.safetensors
wget -c https://huggingface.co/google/gemma-2-27b-it/resolve/main/model-00002-of-00109.safetensors
# ... 继续下载所有分片文件
```

### 模型文件结构验证

下载完成后，验证模型文件结构：

```bash
cd models/gemma-3-27b-it
ls -la

# 期望的文件结构：
# config.json                    # 模型配置
# generation_config.json         # 生成配置  
# model-00001-of-00109.safetensors  # 模型权重（分片1）
# model-00002-of-00109.safetensors  # 模型权重（分片2）
# ...
# model-00109-of-00109.safetensors  # 模型权重（分片109）
# model.safetensors.index.json    # 权重索引
# special_tokens_map.json         # 特殊token映射
# tokenizer.json                  # 分词器
# tokenizer_config.json          # 分词器配置
```

### 验证模型完整性

运行以下脚本验证模型文件完整性：

```bash
# 创建验证脚本
cat > verify_model.py << 'EOF'
import os
import json
from pathlib import Path

def verify_gemma_model(model_path):
    """验证 Gemma 模型文件完整性"""
    model_path = Path(model_path)
    
    # 必需文件列表
    required_files = [
        'config.json',
        'generation_config.json', 
        'model.safetensors.index.json',
        'special_tokens_map.json',
        'tokenizer.json',
        'tokenizer_config.json'
    ]
    
    print(f"🔍 验证模型目录: {model_path}")
    
    # 检查必需文件
    missing_files = []
    for file in required_files:
        if not (model_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 缺少文件: {missing_files}")
        return False
    
    # 检查权重文件
    try:
        with open(model_path / 'model.safetensors.index.json', 'r') as f:
            index = json.load(f)
        
        weight_files = set(index['weight_map'].values())
        
        missing_weights = []
        for weight_file in weight_files:
            if not (model_path / weight_file).exists():
                missing_weights.append(weight_file)
        
        if missing_weights:
            print(f"❌ 缺少权重文件: {missing_weights[:5]}...")  # 只显示前5个
            return False
        
        print(f"✅ 找到 {len(weight_files)} 个权重文件")
        
    except Exception as e:
        print(f"❌ 验证权重文件时出错: {e}")
        return False
    
    # 计算总大小
    total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
    total_size_gb = total_size / (1024**3)
    
    print(f"📊 模型总大小: {total_size_gb:.1f} GB")
    
    if total_size_gb < 50:  # Gemma-3-27B 应该大约 54GB
        print("⚠️  模型大小异常，可能下载不完整")
        return False
    
    print("✅ 模型验证通过")
    return True

if __name__ == "__main__":
    model_path = "gemma-3-27b-it"
    verify_gemma_model(model_path)
EOF

# 运行验证
pixi run python verify_model.py
```

### 存储空间要求

**磁盘空间建议**：
- **Gemma-3-27B**: 至少 60GB 可用空间（模型 54GB + 缓存 6GB）
- **系统总计**: 建议 100GB+ 自由空间用于运行时缓存

**存储性能建议**：
- 生产环境：NVMe SSD（读取速度 >3GB/s）
- 开发环境：普通 SSD 即可
- 避免使用机械硬盘（HDD）

### 环境变量配置

下载完成后，设置模型路径：

```bash
# 在 ~/.bashrc 或 ~/.zshrc 中添加
export WICORE_MODEL_PATH="/path/to/models/gemma-3-27b-it"

# 或在项目配置文件中设置
echo '{
  "model_path": "models/gemma-3-27b-it",
  "model_name": "gemma-3-27b-it"
}' > configs/model_config.json
```

### 常见问题

**Q: 下载速度太慢怎么办？**
A: 
1. 使用国内镜像站（ModelScope）
2. 使用断点续传工具（wget -c）
3. 考虑在网络好的环境下载后传输

**Q: 磁盘空间不足怎么办？**
A:
1. 使用符号链接将模型放在大容量磁盘上
2. 考虑使用较小的模型（Gemma-3-9B）
3. 清理不必要的缓存文件

**Q: 模型验证失败怎么办？**
A:
1. 重新下载缺失的文件
2. 检查网络连接和存储设备
3. 对比 SHA256 校验和

## 🚀 使用指南

### API 接口

WiCore 提供 OpenAI 兼容的 REST API：

```bash
# 聊天完成
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-27b-it",
    "messages": [
      {"role": "user", "content": "解释量子计算的基本原理"}
    ],
    "max_tokens": 512,
    "temperature": 0.7
  }'

# 健康检查
curl http://localhost:8000/health

# 系统状态
curl http://localhost:8000/status
```

### Python 客户端

```python
import requests

def chat_with_wicore(message: str) -> str:
    response = requests.post("http://localhost:8000/v1/chat/completions", 
        json={
            "model": "gemma-3-27b-it",
            "messages": [{"role": "user", "content": message}],
            "max_tokens": 512
        })
    
    return response.json()["choices"][0]["message"]["content"]

# 使用示例
result = chat_with_wicore("你好，请介绍一下你自己")
print(result)
```

## 📊 性能调优

### HMT 内存管理配置

```json
{
  "hmt_config": {
    "enable_a2cr": true,
    "nvme_cache_path": "/nvme/wicore_cache",
    "time_decay_factor": 0.05,
    "attention_weight": 0.4,
    "frequency_weight": 0.3,
    "recency_weight": 0.3
  }
}
```

### MoR 动态路由配置

```json
{
  "mor_config": {
    "enable_mor_routing": true,
    "routing_threshold": 0.5,
    "cpu_depth": 8,
    "gpu_depth": 32
  }
}
```

## 🧪 测试和验证

### 开发环境测试

```bash
# 运行完整测试套件
python scripts/test_engine.py

# 单独测试组件
python scripts/test_device_manager.py
python scripts/test_memory_manager.py
python scripts/test_model_executor.py
```

### 生产环境基准测试

```bash
# 性能基准测试
python scripts/benchmark.py --config configs/production.json

# 压力测试
python scripts/stress_test.py --concurrent 32 --duration 300

# 内存泄漏检测
python scripts/memory_leak_test.py --iterations 1000
```

## 📁 项目结构

```
wicore-mojo/
├── src/                           # Mojo 源代码
│   ├── wicore_engine.mojo        # 主引擎
│   ├── device_manager.mojo       # 设备管理
│   ├── hmt_memory_manager.mojo   # HMT 内存管理
│   ├── model_executor.mojo       # 模型执行
│   ├── request_scheduler.mojo    # 请求调度
│   └── web_server.mojo          # Web 服务
├── simulation/                   # 模拟环境
│   └── max_simulation.py        # MAX Engine 模拟
├── scripts/                     # 脚本工具
│   ├── setup.sh                # 环境搭建
│   ├── test_engine.py          # 测试脚本
│   └── benchmark.py            # 性能测试
├── configs/                    # 配置文件
│   ├── development.json       # 开发配置
│   └── production.json        # 生产配置
├── models/                    # 模型文件
│   └── gemma-3-27b-it/       # Gemma-3 模型
├── docs/                     # 文档
└── requirements.txt          # Python 依赖
```

## 🛡️ 技术方案

### 关键技术决策

1. **Mojo 语言选择**
   - 68,000x Python 性能提升
   - 原生硬件编译优化
   - Python 生态完全兼容

2. **HMT 分层内存管理**
   - GPU 显存：热数据 FP16 存储
   - CPU 内存：温数据 Q8_K 量化
   - NVMe 存储：冷数据 Q4_K 压缩

3. **A²CR 缓存算法**
   - 注意力感知的智能置换
   - 时间衰减 + 频率统计
   - 动态阈值自适应调整

4. **MoR 动态路由**
   - 轻量级路由决策 (<3μs)
   - 重要 token → GPU 深度计算
   - 普通 token → CPU 浅层处理

### 风险控制

- **技术风险**: 保持 Python 兼容性，渐进式 Mojo 采用
- **性能风险**: 早期验证，多后端支持
- **硬件风险**: 分层支持，CPU fallback 机制

## 🤝 贡献指南

1. Fork 项目仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🎯 路线图

### Phase 1: 基础验证 ✅
- [x] Mojo 环境搭建
- [x] 模拟环境开发
- [x] 核心组件实现
- [x] 集成测试验证

### Phase 2: 生产就绪 🔄
- [ ] GPU 服务器部署
- [ ] Gemma-3-27B 模型加载
- [ ] T10 双卡性能优化
- [ ] 稳定性测试

### Phase 3: 扩展支持 📋
- [ ] 多模型支持
- [ ] 国产硬件适配
- [ ] 云原生部署
- [ ] 监控告警系统

---

**🎉 WiCore Mojo 推理引擎 - 自主可控的 AI 推理未来** 

如有问题或建议，请创建 Issue 或联系开发团队。 