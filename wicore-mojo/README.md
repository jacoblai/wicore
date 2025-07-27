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
# 1. 下载 Gemma-3-27B 模型
mkdir -p models
cd models
# 下载模型文件到此目录

# 2. 配置生产环境
cp configs/development.json configs/production.json
# 编辑 production.json 设置 GPU 配置

# 3. 启动推理引擎
./scripts/start_engine.sh --config configs/production.json
```

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