#!/bin/bash

# WiCore Mojo 推理引擎环境搭建脚本
# 使用 Pixi 管理 Modular 平台（官方推荐方式）

set -e

echo "🚀 开始搭建 WiCore Mojo 推理引擎环境..."

# 检测操作系统
OS_TYPE=$(uname -s)
echo "🖥️  检测到操作系统: $OS_TYPE"

# 初始化配置变量
ENABLE_MULTI_GPU=false
TARGET_DEVICES='"cpu:0"'

# 检查 NVIDIA GPU 环境
echo "🔍 检查 GPU 环境..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU 驱动已安装"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    
    # 检查 GPU 数量
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "📊 检测到 $GPU_COUNT 个 GPU"
    
    if [ "$GPU_COUNT" -gt 1 ]; then
        ENABLE_MULTI_GPU=true
        TARGET_DEVICES='"gpu:0", "gpu:1"'
        echo "🎯 多GPU配置: $GPU_COUNT 卡"
    else
        ENABLE_MULTI_GPU=false
        TARGET_DEVICES='"gpu:0"'
        echo "🎯 单GPU配置"
    fi
elif [ "$OS_TYPE" = "Darwin" ]; then
    echo "🍎 macOS 环境 - 将使用 CPU 模拟模式"
    TARGET_DEVICES='"cpu:0"'
else
    echo "⚠️  未检测到 NVIDIA GPU，将使用 CPU 模拟模式"
    TARGET_DEVICES='"cpu:0"'
fi

# 安装 Pixi（Modular 官方推荐的包管理器）
echo "📦 安装 Pixi 包管理器..."
if ! command -v pixi &> /dev/null; then
    echo "正在下载 Pixi..."
    curl -fsSL https://pixi.sh/install.sh | bash
    
    # 添加到当前会话的 PATH
    export PATH="$HOME/.pixi/bin:$PATH"
    
    echo "✅ Pixi 安装完成"
else
    echo "✅ Pixi 已安装"
fi

# 配置 Pixi 默认渠道（包含 Modular 渠道）
echo "⚙️  配置 Pixi 默认渠道..."
mkdir -p "$HOME/.pixi"
echo 'default-channels = ["https://conda.modular.com/max-nightly", "conda-forge"]' > "$HOME/.pixi/config.toml"
echo "✅ Pixi 渠道配置完成"

# 初始化 WiCore 项目
echo "🏗️  初始化 WiCore 项目..."
if [ ! -f "pixi.toml" ]; then
    pixi init . --name wicore-mojo
    echo "✅ 项目初始化完成"
else
    echo "✅ 项目已存在"
fi

# 安装 Modular 平台
echo "🔧 安装 Modular 平台..."
pixi add "modular=*" "python==3.11" || {
    echo "⚠️  Modular 安装失败，尝试使用稳定版本..."
    pixi add "modular~=25.4" "python==3.11"
}

# 安装 Python 依赖
echo "📦 安装 Python 依赖..."
pixi add torch torchvision torchaudio transformers accelerate
pixi add fastapi uvicorn pydantic
pixi add numpy pandas psutil requests
pixi add matplotlib seaborn

echo "✅ 依赖安装完成"

# 验证 Modular 安装
echo "🧪 验证 Modular 安装..."
pixi run python -c "
import sys
print(f'Python 版本: {sys.version}')

try:
    # 检查 Modular 是否可用
    import subprocess
    result = subprocess.run(['modular', '--version'], capture_output=True, text=True)
    if result.returncode == 0:
        print(f'✅ Modular 版本: {result.stdout.strip()}')
    else:
        print('⚠️  Modular 命令行工具未正确安装')
except Exception as e:
    print(f'⚠️  Modular 验证异常: {e}')

# 检查基础 Python 包
try:
    import torch
    print(f'✅ PyTorch 版本: {torch.__version__}')
    print(f'✅ CUDA 可用: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✅ GPU 数量: {torch.cuda.device_count()}')
except ImportError:
    print('❌ PyTorch 未正确安装')

try:
    import transformers
    print(f'✅ Transformers 版本: {transformers.__version__}')
except ImportError:
    print('❌ Transformers 未正确安装')
"

# 创建环境配置
echo "⚙️  创建环境配置..."
mkdir -p configs

cat > configs/production.json << EOF
{
    "model_path": "models/gemma-3-27b-it",
    "server_port": 8000,
    "max_batch_size": 16,
    "max_context_length": 131072,
    "gpu_memory_limit_gb": 15.0,
    "enable_multi_gpu": $ENABLE_MULTI_GPU,
    "target_devices": [$TARGET_DEVICES],
    "hmt_config": {
        "enable_a2cr": true,
        "nvme_cache_path": "/tmp/wicore_cache",
        "time_decay_factor": 0.05,
        "attention_weight": 0.4,
        "frequency_weight": 0.3,
        "recency_weight": 0.3,
        "max_cache_size_gb": 100
    },
    "performance_config": {
        "enable_batching": true,
        "batch_timeout_ms": 50,
        "max_concurrent_requests": 32,
        "request_timeout_seconds": 30,
        "enable_streaming": true
    },
    "logging": {
        "level": "INFO",
        "enable_request_logging": true,
        "enable_performance_logging": true,
        "log_file": "logs/wicore.log"
    }
}
EOF

# 创建开发配置
cat > configs/development.json << EOF
{
    "model_path": "models/gemma-3-27b-it",
    "server_port": 8000,
    "max_batch_size": 8,
    "max_context_length": 32768,
    "gpu_memory_limit_gb": 8.0,
    "enable_multi_gpu": false,
    "target_devices": ["cpu:0"],
    "simulation_mode": true,
    "hmt_config": {
        "enable_a2cr": true,
        "nvme_cache_path": "./cache",
        "time_decay_factor": 0.05,
        "attention_weight": 0.4,
        "frequency_weight": 0.3,
        "recency_weight": 0.3
    },
    "logging": {
        "level": "DEBUG",
        "enable_request_logging": true,
        "enable_performance_logging": true
    }
}
EOF

echo "✅ 配置文件已创建"

# 创建必要目录
echo "📁 创建项目目录结构..."
mkdir -p {models,cache,logs,build}

# 创建模拟环境（用于开发测试）
echo "🎭 创建 MAX Engine 模拟环境..."
mkdir -p simulation

cat > simulation/max_simulation.py << 'EOF'
"""
MAX Engine 模拟环境
用于在没有真实 MAX Engine 的情况下进行开发测试
"""

import numpy as np
import time
import threading
from typing import List, Dict, Optional, Any
import psutil


class Device:
    """模拟计算设备"""
    def __init__(self, device_type: str, device_id: str, memory_total: int = 16):
        self.type = device_type
        self.id = device_id
        self.memory_total = memory_total * 1024 * 1024 * 1024  # GB -> Bytes
        self.memory_allocated = 0
        self.memory_available = self.memory_total
        self.compute_capability = 7.5 if device_type == "gpu" else 1.0
        self.is_available = True
        
    def allocate_memory(self, size: int) -> bool:
        """分配内存"""
        if self.memory_available >= size:
            self.memory_allocated += size
            self.memory_available -= size
            return True
        return False
    
    def free_memory(self, size: int):
        """释放内存"""
        self.memory_allocated = max(0, self.memory_allocated - size)
        self.memory_available = min(self.memory_total, self.memory_available + size)
    
    def get_memory_info(self) -> Dict[str, int]:
        """获取内存信息"""
        return {
            "total": self.memory_total,
            "allocated": self.memory_allocated,
            "available": self.memory_available
        }


class Tensor:
    """模拟张量"""
    def __init__(self, shape: tuple, dtype: str = "float16", device: str = "cpu:0"):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.data = np.random.randn(*shape).astype(self._numpy_dtype())
        self.size = self.data.nbytes
        
    def _numpy_dtype(self):
        """转换数据类型"""
        dtype_map = {
            "float16": np.float16,
            "float32": np.float32,
            "int32": np.int32,
            "int8": np.int8
        }
        return dtype_map.get(self.dtype, np.float16)
    
    def to(self, device: str):
        """移动到指定设备"""
        self.device = device
        return self
    
    def __str__(self):
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, device={self.device})"


class ModelGraph:
    """模拟计算图"""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.layers = []
        self.parameters = {}
        self.memory_requirement = 0
        
    def add_layer(self, layer_name: str, params: Dict):
        """添加层"""
        self.layers.append({"name": layer_name, "params": params})
        
    def execute(self, inputs: List[Tensor], device_ids: List[str]) -> List[Tensor]:
        """执行计算图"""
        # 模拟推理延迟
        batch_size = inputs[0].shape[0] if inputs else 1
        base_latency = 0.01  # 10ms基础延迟
        compute_latency = batch_size * 0.005  # 每个sample 5ms
        
        time.sleep(base_latency + compute_latency)
        
        # 模拟输出
        if inputs:
            input_tensor = inputs[0]
            # 模拟语言模型输出（vocab_size=32000）
            output_shape = (input_tensor.shape[0], input_tensor.shape[1], 32000)
            output = Tensor(output_shape, "float16", input_tensor.device)
            return [output]
        else:
            return [Tensor((1, 1, 32000), "float16", "cpu:0")]


class Engine:
    """模拟 MAX Engine"""
    
    def __init__(self):
        self.devices = []
        self.loaded_models = {}
        
    def discover_devices(self) -> List[Device]:
        """发现设备"""
        if not self.devices:
            # 模拟 CPU 设备
            cpu_count = psutil.cpu_count(logical=False)
            for i in range(min(cpu_count, 2)):  # 最多模拟2个CPU核心
                device = Device("cpu", f"cpu:{i}", memory_total=8)
                self.devices.append(device)
            
            # 在生产环境下会发现 GPU
            # 这里可以通过环境变量控制是否模拟 GPU
            import os
            if os.getenv("WICORE_SIMULATE_GPU", "false").lower() == "true":
                for i in range(2):
                    device = Device("gpu", f"gpu:{i}", memory_total=16)
                    self.devices.append(device)
        
        return self.devices.copy()
    
    def allocate_device_memory(self, device_id: str, size: int) -> Optional[int]:
        """在指定设备上分配内存"""
        device = self._get_device(device_id)
        if device and device.allocate_memory(size):
            # 返回模拟的内存地址
            return id(device) + size
        return None
    
    def free_device_memory(self, device_id: str, ptr: int, size: int):
        """释放设备内存"""
        device = self._get_device(device_id)
        if device:
            device.free_memory(size)
    
    def load_model(self, model_path: str, device_ids: List[str]) -> ModelGraph:
        """加载模型"""
        print(f"🤖 [模拟] 加载模型: {model_path}")
        
        # 模拟模型加载时间
        time.sleep(2.0)  # 2秒加载时间
        
        # 创建模拟的模型图
        graph = ModelGraph(model_path)
        
        # 模拟 Gemma-3-27B 的结构
        for i in range(32):  # 32层 Transformer
            graph.add_layer(f"transformer_layer_{i}", {
                "hidden_size": 4096,
                "num_heads": 32,
                "intermediate_size": 11008
            })
        
        # 估算内存需求（简化）
        graph.memory_requirement = 27 * 1024 * 1024 * 1024  # 27GB
        
        # 检查设备内存
        total_available = sum(
            self._get_device(device_id).memory_available 
            for device_id in device_ids 
            if self._get_device(device_id)
        )
        
        if total_available < graph.memory_requirement:
            print(f"⚠️  [模拟] 内存不足：需要 {graph.memory_requirement/1e9:.1f}GB，可用 {total_available/1e9:.1f}GB")
        
        self.loaded_models[model_path] = graph
        print(f"✅ [模拟] 模型加载完成: {model_path}")
        return graph
    
    def execute_graph(self, graph: ModelGraph, inputs: List[Tensor], device_ids: List[str]) -> List[Tensor]:
        """执行计算图"""
        return graph.execute(inputs, device_ids)
    
    def _get_device(self, device_id: str) -> Optional[Device]:
        """获取设备"""
        for device in self.devices:
            if device.id == device_id:
                return device
        return None


class Graph:
    """模拟计算图模块"""
    
    @staticmethod
    def from_torch_model(torch_model) -> ModelGraph:
        """从 PyTorch 模型创建计算图"""
        print("🔄 [模拟] 转换 PyTorch 模型为 MAX 计算图...")
        time.sleep(1.0)  # 模拟转换时间
        
        # 创建模拟图
        graph = ModelGraph("converted_model")
        
        # 简化的转换逻辑
        if hasattr(torch_model, 'config'):
            config = torch_model.config
            num_layers = getattr(config, 'num_hidden_layers', 32)
            
            for i in range(num_layers):
                graph.add_layer(f"layer_{i}", {
                    "type": "transformer_block",
                    "hidden_size": getattr(config, 'hidden_size', 4096)
                })
        
        print("✅ [模拟] 模型转换完成")
        return graph


# 全局实例
engine = Engine()
graph = Graph()

# 模拟 max 模块的结构
class MaxModule:
    def __init__(self):
        self.engine = engine
        self.graph = graph

# 创建模块实例以支持 from max import engine, graph
max_module = MaxModule()
EOF

echo "✅ 模拟环境创建完成"

# 创建测试脚本
echo "🧪 创建测试脚本..."
cat > scripts/test_engine.py << 'EOF'
#!/usr/bin/env python3
"""
WiCore Mojo 推理引擎测试脚本
验证所有核心组件的功能和集成
"""

import sys
import os
import time
import json
import subprocess
from typing import Dict, Any

# 添加模拟环境到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulation'))

def test_environment():
    """测试基础环境"""
    print("🔧 测试环境配置...")
    
    # 测试 Python 版本
    print(f"Python 版本: {sys.version}")
    
    # 测试基础包
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU 数量: {torch.cuda.device_count()}")
    except ImportError:
        print("❌ PyTorch 未安装")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers 未安装")
        return False
    
    return True

def test_modular_integration():
    """测试 Modular 集成"""
    print("\n🔧 测试 Modular 集成...")
    
    try:
        # 尝试导入模拟的 max 模块
        import max_simulation as max_sim
        engine = max_sim.engine
        
        print("✅ MAX Engine 模拟环境导入成功")
        
        # 测试设备发现
        devices = engine.discover_devices()
        print(f"✅ 发现 {len(devices)} 个设备")
        for device in devices:
            print(f"   - {device.type}: {device.id}")
        
        return True
    except Exception as e:
        print(f"❌ Modular 集成测试失败: {e}")
        return False

def test_configuration():
    """测试配置文件"""
    print("\n📋 测试配置文件...")
    
    try:
        with open('configs/production.json', 'r') as f:
            prod_config = json.load(f)
        print("✅ 生产配置文件加载成功")
        
        with open('configs/development.json', 'r') as f:
            dev_config = json.load(f)
        print("✅ 开发配置文件加载成功")
        
        return True
    except Exception as e:
        print(f"❌ 配置文件测试失败: {e}")
        return False

def test_pixi_environment():
    """测试 Pixi 环境"""
    print("\n📦 测试 Pixi 环境...")
    
    try:
        # 检查 pixi.toml 文件
        if os.path.exists('pixi.toml'):
            print("✅ pixi.toml 文件存在")
        else:
            print("⚠️  pixi.toml 文件不存在")
        
        # 测试 pixi 命令
        result = subprocess.run(['pixi', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Pixi 版本: {result.stdout.strip()}")
            return True
        else:
            print("❌ Pixi 命令不可用")
            return False
    except Exception as e:
        print(f"❌ Pixi 环境测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始 WiCore Mojo 推理引擎测试...")
    print("=" * 50)
    
    tests = [
        ("环境配置", test_environment),
        ("Modular 集成", test_modular_integration),
        ("配置文件", test_configuration),
        ("Pixi 环境", test_pixi_environment),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 测试通过")
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"🎯 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！WiCore 环境配置成功")
        return 0
    else:
        print("⚠️  部分测试失败，请检查环境配置")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

chmod +x scripts/test_engine.py

echo "✅ 目录结构创建完成"

# 显示总结信息
echo ""
echo "🎉 WiCore Mojo 推理引擎环境搭建完成！"
echo "=" * 50
echo "🎯 目标设备: $TARGET_DEVICES"
echo "📦 包管理器: Pixi (官方推荐)"
echo "📄 配置文件: configs/production.json, configs/development.json"
echo ""
echo "🚀 下一步操作:"
echo "   1. 测试环境: pixi run python scripts/test_engine.py"
echo "   2. 下载模型: 下载 Gemma-3-27B 到 models/ 目录"
echo "   3. 构建项目: pixi run ./scripts/build.sh"
echo "   4. 启动引擎: pixi run python src/wicore_engine.py"
echo ""
echo "💡 Pixi 常用命令:"
echo "   - pixi run <command>    # 在环境中运行命令"
echo "   - pixi shell           # 进入环境 shell"
echo "   - pixi add <package>   # 添加包"
echo "   - pixi update          # 更新包"
echo ""
echo "📚 文档位置:"
echo "   - API 文档: docs/API.md"
echo "   - 性能优化: docs/PERFORMANCE.md"
echo "   - 部署指南: docs/DEPLOYMENT.md" 