"""
WiCore设备管理器  
基于已验证的硬件检测逻辑，支持异构硬件统一抽象

核心功能:
- 实时硬件发现和状态监控
- 异构设备统一抽象（GPU、CPU、NPU等）
- 设备拓扑分析和优化建议
- 内存和计算资源管理
- 设备健康状态监控
"""

import torch
import logging
import subprocess
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json
import psutil
import numpy as np

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """设备类型"""
    GPU = "gpu"
    CPU = "cpu" 
    NPU = "npu"
    TPU = "tpu"
    FPGA = "fpga"


@dataclass  
class DeviceInfo:
    """设备信息"""
    device_id: str
    device_type: DeviceType
    name: str
    memory_total: int           # 总内存（字节）
    memory_available: int       # 可用内存（字节）
    compute_capability: float   # 计算能力评分
    numa_node: int             # NUMA节点
    pcie_bandwidth: float      # PCIe带宽（GB/s）
    power_usage: float         # 功耗（瓦特）
    temperature: float         # 温度（摄氏度）
    utilization: float         # 利用率（0-1）
    is_available: bool         # 是否可用
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result['device_type'] = self.device_type.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeviceInfo':
        """从字典创建"""
        data['device_type'] = DeviceType(data['device_type'])
        return cls(**data)


@dataclass
class DeviceTopology:
    """设备拓扑信息"""
    device_connections: Dict[str, List[str]]         # 设备连接关系
    bandwidth_matrix: Dict[str, Dict[str, float]]   # 带宽矩阵
    numa_topology: Dict[int, List[str]]             # NUMA拓扑
    optimal_placement: Dict[str, str]               # 最优放置建议


class DeviceManager:
    """设备管理器"""
    
    def __init__(self):
        self.devices: Dict[str, DeviceInfo] = {}
        self.topology: Optional[DeviceTopology] = None
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.monitor_interval = 5.0  # 5秒监控间隔
        
        # 统计信息
        self.discovery_time = 0
        self.total_compute_power = 0.0
        self.total_memory = 0
        self.available_memory = 0
        
        # 执行初始硬件发现
        self._discover_devices()
        self._analyze_topology()
        
        logger.info(f"设备管理器初始化完成: 发现{len(self.devices)}个设备")
    
    def _discover_devices(self):
        """发现硬件设备"""
        start_time = time.time()
        logger.info("🔍 开始设备发现...")
        
        try:
            # 发现GPU设备
            self._discover_gpu_devices()
            
            # 发现CPU设备  
            self._discover_cpu_devices()
            
            # 发现其他设备（NPU、TPU等）
            self._discover_other_devices()
            
            self.discovery_time = time.time() - start_time
            
            # 计算总体统计
            self._calculate_system_stats()
            
            logger.info(f"✅ 设备发现完成: {len(self.devices)}个设备, 用时{self.discovery_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ 设备发现失败: {e}")
    
    def _discover_gpu_devices(self):
        """发现GPU设备"""
        logger.info("📊 检测GPU设备...")
        
        # 检测NVIDIA GPU
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,compute_cap,temperature.gpu,power.draw,utilization.gpu", 
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True
            )
            
            gpu_output = result.stdout.strip()
            if gpu_output:
                gpu_lines = gpu_output.split('\n')
                logger.info(f"✅ 发现 {len(gpu_lines)} 个NVIDIA GPU:")
                
                for line in gpu_lines:
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 8:
                            gpu_id = f"gpu_{parts[0]}"
                            name = parts[1]
                            memory_total = int(parts[2]) * 1024 * 1024  # MB to bytes
                            memory_free = int(parts[3]) * 1024 * 1024   # MB to bytes
                            compute_cap = float(parts[4])
                            temperature = float(parts[5]) if parts[5] != 'N/A' else 0.0
                            power_usage = float(parts[6]) if parts[6] != 'N/A' else 0.0
                            utilization = float(parts[7]) / 100.0 if parts[7] != 'N/A' else 0.0
                            
                            device_info = DeviceInfo(
                                device_id=gpu_id,
                                device_type=DeviceType.GPU,
                                name=name,
                                memory_total=memory_total,
                                memory_available=memory_free,
                                compute_capability=compute_cap,
                                numa_node=0,  # 需要进一步检测
                                pcie_bandwidth=16.0,  # 默认PCIe 4.0 x16
                                power_usage=power_usage,
                                temperature=temperature,
                                utilization=utilization,
                                is_available=True
                            )
                            
                            self.devices[gpu_id] = device_info
                            
                            logger.info(f"    GPU {parts[0]}: {name}")
                            logger.info(f"      内存: {memory_free//1024//1024}MB / {memory_total//1024//1024}MB")
                            logger.info(f"      计算能力: {compute_cap}, 温度: {temperature}°C")
                            logger.info(f"      功耗: {power_usage}W, 利用率: {utilization*100:.1f}%")
        
        except subprocess.CalledProcessError:
            logger.info("⚠️  nvidia-smi命令执行失败，可能没有NVIDIA GPU")
        except Exception as e:
            logger.warning(f"⚠️  GPU检测异常: {e}")
        
        # 检测AMD GPU（如果需要）
        # 检测Intel GPU（如果需要）
    
    def _discover_cpu_devices(self):
        """发现CPU设备"""
        logger.info("🖥️ 检测CPU设备...")
        
        try:
            # 获取CPU信息
            cpu_info = {}
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        if key == 'model name' and 'model_name' not in cpu_info:
                            cpu_info['model_name'] = value
                        elif key == 'cpu cores' and 'cores' not in cpu_info:
                            cpu_info['cores'] = int(value)
                        elif key == 'cpu MHz' and 'frequency' not in cpu_info:
                            cpu_info['frequency'] = float(value)
            
            # 获取内存信息
            memory_info = psutil.virtual_memory()
            
            # 创建CPU设备信息
            device_info = DeviceInfo(
                device_id="cpu_0",
                device_type=DeviceType.CPU,
                name=cpu_info.get('model_name', 'Unknown CPU'),
                memory_total=memory_info.total,
                memory_available=memory_info.available,
                compute_capability=cpu_info.get('cores', 1) * cpu_info.get('frequency', 2000) / 1000,
                numa_node=0,
                pcie_bandwidth=0.0,
                power_usage=0.0,  # CPU功耗需要特殊检测
                temperature=0.0,  # CPU温度需要特殊检测
                utilization=psutil.cpu_percent() / 100.0,
                is_available=True
            )
            
            self.devices["cpu_0"] = device_info
            
            logger.info(f"✅ CPU: {device_info.name}")
            logger.info(f"    内存: {memory_info.available//1024//1024//1024}GB / {memory_info.total//1024//1024//1024}GB")
            logger.info(f"    核心数: {cpu_info.get('cores', 'Unknown')}")
            logger.info(f"    频率: {cpu_info.get('frequency', 'Unknown')}MHz")
            
        except Exception as e:
            logger.warning(f"⚠️  CPU信息检测失败: {e}")
    
    def _discover_other_devices(self):
        """发现其他设备（NPU、TPU等）"""
        # 这里可以添加对其他设备的检测逻辑
        # 例如：Intel Neural Compute Stick、Google TPU、华为昇腾NPU等
        pass
    
    def _analyze_topology(self):
        """分析设备拓扑"""
        logger.info("🔗 分析设备拓扑...")
        
        try:
            # 获取GPU拓扑信息
            gpu_topology = self._get_gpu_topology()
            
            # 构建拓扑结构
            device_connections = {}
            bandwidth_matrix = {}
            numa_topology = {0: list(self.devices.keys())}  # 简化的NUMA拓扑
            optimal_placement = {}
            
            # 分析GPU连接
            if gpu_topology:
                for connection in gpu_topology:
                    gpu_a = f"gpu_{connection['gpu_a']}"
                    gpu_b = f"gpu_{connection['gpu_b']}"
                    link_type = connection['link_type']
                    
                    # 添加连接关系
                    if gpu_a not in device_connections:
                        device_connections[gpu_a] = []
                    if gpu_b not in device_connections:
                        device_connections[gpu_b] = []
                    
                    device_connections[gpu_a].append(gpu_b)
                    device_connections[gpu_b].append(gpu_a)
                    
                    # 估算带宽
                    bandwidth = self._estimate_bandwidth(link_type)
                    
                    if gpu_a not in bandwidth_matrix:
                        bandwidth_matrix[gpu_a] = {}
                    if gpu_b not in bandwidth_matrix:
                        bandwidth_matrix[gpu_b] = {}
                    
                    bandwidth_matrix[gpu_a][gpu_b] = bandwidth
                    bandwidth_matrix[gpu_b][gpu_a] = bandwidth
                    
                    logger.info(f"    {gpu_a} <-> {gpu_b}: {link_type} ({bandwidth:.1f} GB/s)")
            
            self.topology = DeviceTopology(
                device_connections=device_connections,
                bandwidth_matrix=bandwidth_matrix,
                numa_topology=numa_topology,
                optimal_placement=optimal_placement
            )
            
        except Exception as e:
            logger.warning(f"⚠️  拓扑分析失败: {e}")
    
    def _get_gpu_topology(self) -> List[Dict[str, Any]]:
        """获取GPU拓扑信息"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "topo", "-m"],
                capture_output=True, text=True, check=True
            )
            
            # 解析nvidia-smi topo输出
            lines = result.stdout.strip().split('\n')
            topology = []
            
            for line in lines:
                if 'GPU' in line and 'PHB' in line:
                    # 简化的解析，实际应该更复杂
                    parts = line.split()
                    if len(parts) >= 3:
                        gpu_a = 0  # 提取GPU索引
                        gpu_b = 1  # 提取GPU索引
                        link_type = "PHB"  # PCIe + Host Bridge
                        
                        topology.append({
                            'gpu_a': gpu_a,
                            'gpu_b': gpu_b,
                            'link_type': link_type
                        })
            
            return topology
            
        except subprocess.CalledProcessError:
            logger.info("⚠️  无法获取GPU拓扑信息")
            return []
    
    def _estimate_bandwidth(self, link_type: str) -> float:
        """估算带宽"""
        bandwidth_map = {
            'NV1': 25.0,  # NVLink 1.0
            'NV2': 50.0,  # NVLink 2.0  
            'NV3': 112.0, # NVLink 3.0
            'NV4': 112.0, # NVLink 4.0
            'PHB': 16.0,  # PCIe Host Bridge
            'PIX': 16.0,  # PCIe
            'SYS': 10.0,  # System connection
        }
        return bandwidth_map.get(link_type, 10.0)
    
    def _calculate_system_stats(self):
        """计算系统统计信息"""
        self.total_compute_power = sum(
            device.compute_capability for device in self.devices.values()
        )
        self.total_memory = sum(
            device.memory_total for device in self.devices.values()
        )
        self.available_memory = sum(
            device.memory_available for device in self.devices.values()
        )
    
    def start_monitoring(self):
        """开始设备监控"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_devices, daemon=True)
        self.monitor_thread.start()
        
        logger.info("🔄 设备监控已启动")
    
    def stop_monitoring(self):
        """停止设备监控"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        logger.info("⏹️ 设备监控已停止")
    
    def _monitor_devices(self):
        """监控设备状态"""
        while self.monitoring_active:
            try:
                # 更新GPU状态
                self._update_gpu_status()
                
                # 更新CPU状态
                self._update_cpu_status()
                
                # 重新计算统计信息
                self._calculate_system_stats()
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.warning(f"设备监控异常: {e}")
                time.sleep(self.monitor_interval)
    
    def _update_gpu_status(self):
        """更新GPU状态"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.free,temperature.gpu,power.draw,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True
            )
            
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 5:
                        gpu_id = f"gpu_{parts[0]}"
                        if gpu_id in self.devices:
                            device = self.devices[gpu_id]
                            device.memory_available = int(parts[1]) * 1024 * 1024
                            device.temperature = float(parts[2]) if parts[2] != 'N/A' else 0.0
                            device.power_usage = float(parts[3]) if parts[3] != 'N/A' else 0.0
                            device.utilization = float(parts[4]) / 100.0 if parts[4] != 'N/A' else 0.0
                            
        except subprocess.CalledProcessError:
            pass
    
    def _update_cpu_status(self):
        """更新CPU状态"""
        try:
            if "cpu_0" in self.devices:
                memory_info = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent()
                
                device = self.devices["cpu_0"]
                device.memory_available = memory_info.available
                device.utilization = cpu_percent / 100.0
                
        except Exception:
            pass
    
    def get_device(self, device_id: str) -> Optional[DeviceInfo]:
        """获取设备信息"""
        return self.devices.get(device_id)
    
    def get_devices_by_type(self, device_type: DeviceType) -> List[DeviceInfo]:
        """按类型获取设备"""
        return [device for device in self.devices.values() if device.device_type == device_type]
    
    def get_available_devices(self) -> List[DeviceInfo]:
        """获取可用设备"""
        return [device for device in self.devices.values() if device.is_available]
    
    def get_optimal_device(self, memory_required: int = 0) -> Optional[DeviceInfo]:
        """获取最优设备"""
        available_devices = self.get_available_devices()
        
        # 按计算能力和可用内存排序
        scored_devices = []
        for device in available_devices:
            if device.memory_available >= memory_required:
                score = device.compute_capability * (1 - device.utilization)
                scored_devices.append((score, device))
        
        if scored_devices:
            scored_devices.sort(reverse=True)
            return scored_devices[0][1]
        
        return None
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        gpu_devices = self.get_devices_by_type(DeviceType.GPU)
        cpu_devices = self.get_devices_by_type(DeviceType.CPU)
        
        return {
            "total_devices": len(self.devices),
            "gpu_count": len(gpu_devices),
            "cpu_count": len(cpu_devices),
            "total_compute_power": self.total_compute_power,
            "total_memory_gb": self.total_memory / 1024 / 1024 / 1024,
            "available_memory_gb": self.available_memory / 1024 / 1024 / 1024,
            "memory_utilization": 1 - (self.available_memory / max(self.total_memory, 1)),
            "discovery_time": self.discovery_time,
            "monitoring_active": self.monitoring_active,
            "devices": [device.to_dict() for device in self.devices.values()]
        }
    
    def export_config(self, filepath: str):
        """导出设备配置"""
        config = {
            "devices": {device_id: device.to_dict() for device_id, device in self.devices.items()},
            "topology": {
                "device_connections": self.topology.device_connections if self.topology else {},
                "bandwidth_matrix": self.topology.bandwidth_matrix if self.topology else {},
                "numa_topology": self.topology.numa_topology if self.topology else {},
                "optimal_placement": self.topology.optimal_placement if self.topology else {}
            } if self.topology else {},
            "system_stats": self.get_system_stats()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"设备配置已导出: {filepath}")
    
    def __del__(self):
        """清理资源"""
        self.stop_monitoring() 