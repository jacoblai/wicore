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
