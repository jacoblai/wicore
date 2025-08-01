"""
WiCore通用设备管理器
设备无关的硬件能力发现和优化框架

核心设计原则:
- 硬件无关：自动适配任意GPU配置
- 运行时发现：动态检测设备能力和拓扑
- 可扩展架构：插件式优化策略
- 自适应优化：根据实际硬件选择最优策略
"""

import torch
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import time
import threading
from collections import defaultdict
import psutil

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """设备类型"""
    GPU = "gpu"
    CPU = "cpu" 
    DISK = "disk"
    NETWORK = "network"


class InterconnectType(Enum):
    """互联类型"""
    PCIE = "pcie"
    NVLINK = "nvlink" 
    P2P = "p2p"
    SYSTEM_MEMORY = "system_memory"
    NETWORK = "network"


@dataclass
class DeviceCapability:
    """设备能力描述"""
    device_id: str
    device_type: DeviceType
    compute_capability: float
    memory_total: int
    memory_bandwidth: float  # GB/s
    compute_units: int
    frequency: float  # MHz
    supports_fp16: bool = True
    supports_bf16: bool = False
    supports_int8: bool = True
    power_limit: float = 0.0  # Watts
    thermal_limit: float = 85.0  # Celsius


@dataclass
class InterconnectInfo:
    """设备互联信息"""
    src_device: str
    dst_device: str
    interconnect_type: InterconnectType
    bandwidth: float  # GB/s
    latency: float = 0.0  # microseconds
    bidirectional: bool = True


@dataclass
class DeviceTopology:
    """设备拓扑结构"""
    devices: Dict[str, DeviceCapability]
    interconnects: List[InterconnectInfo]
    numa_nodes: Dict[str, int]
    
    def get_bandwidth(self, src: str, dst: str) -> float:
        """获取两设备间带宽"""
        for interconnect in self.interconnects:
            if ((interconnect.src_device == src and interconnect.dst_device == dst) or
                (interconnect.bidirectional and 
                 interconnect.src_device == dst and interconnect.dst_device == src)):
                return interconnect.bandwidth
        return 0.0
    
    def get_optimal_path(self, src: str, dst: str) -> List[str]:
        """获取最优传输路径"""
        # 简化实现：直接连接或通过系统内存
        if self.get_bandwidth(src, dst) > 0:
            return [src, dst]
        else:
            # 通过系统内存中转
            return [src, "system_memory", dst]


class UniversalDeviceManager:
    """通用设备管理器
    
    自动发现和管理任意硬件配置的通用框架：
    - 运行时硬件能力检测
    - 动态拓扑发现
    - 自适应优化策略选择
    - 设备无关的负载均衡
    """
    
    def __init__(self):
        """初始化通用设备管理器"""
        self.device_topology: Optional[DeviceTopology] = None
        self.optimization_strategies: Dict[str, Any] = {}
        self.performance_history: Dict[str, List[float]] = defaultdict(list)
        
        # 动态监控
        self.monitoring_enabled = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self._monitor_lock = threading.Lock()
        
        # 自动发现硬件
        self._discover_hardware()
        
        logger.info("通用设备管理器初始化完成")
    
    def _discover_hardware(self):
        """自动发现硬件配置"""
        logger.info("🔍 自动发现硬件配置...")
        
        devices = {}
        interconnects = []
        
        # 发现GPU设备
        gpu_devices = self._discover_gpu_devices()
        devices.update(gpu_devices)
        
        # 发现CPU设备
        cpu_devices = self._discover_cpu_devices()
        devices.update(cpu_devices)
        
        # 发现设备互联
        interconnects.extend(self._discover_gpu_interconnects(gpu_devices))
        interconnects.extend(self._discover_system_interconnects(devices))
        
        # NUMA拓扑
        numa_nodes = self._discover_numa_topology(devices)
        
        self.device_topology = DeviceTopology(
            devices=devices,
            interconnects=interconnects,
            numa_nodes=numa_nodes
        )
        
        self._log_topology_summary()
    
    def _discover_gpu_devices(self) -> Dict[str, DeviceCapability]:
        """发现GPU设备"""
        gpu_devices = {}
        
        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                
                # 检测精度支持
                supports_bf16 = hasattr(torch, 'bfloat16') and torch.cuda.is_bf16_supported()
                
                # 兼容性处理：获取时钟频率
                try:
                    frequency = props.max_clock_rate / 1000.0  # Convert to MHz
                except AttributeError:
                    # 在某些PyTorch版本中，使用不同的属性名
                    frequency = getattr(props, 'clock_rate', 1500.0) / 1000.0
                
                device_id = f"cuda:{i}"
                capability = DeviceCapability(
                    device_id=device_id,
                    device_type=DeviceType.GPU,
                    compute_capability=props.major + props.minor * 0.1,
                    memory_total=props.total_memory,
                    memory_bandwidth=self._estimate_memory_bandwidth(props),
                    compute_units=props.multi_processor_count,
                    frequency=frequency,
                    supports_fp16=True,
                    supports_bf16=supports_bf16,
                    supports_int8=True,
                    power_limit=self._get_gpu_power_limit(i),
                    thermal_limit=self._get_gpu_thermal_limit(i)
                )
                
                gpu_devices[device_id] = capability
                logger.info(f"发现GPU: {device_id} - {props.name}")
                
            except Exception as e:
                logger.warning(f"发现GPU {i} 失败: {e}")
        
        return gpu_devices
    
    def _discover_cpu_devices(self) -> Dict[str, DeviceCapability]:
        """发现CPU设备"""
        try:
            cpu_count = psutil.cpu_count(logical=False)
            cpu_freq = psutil.cpu_freq()
            memory_info = psutil.virtual_memory()
            
            device_id = "cpu"
            capability = DeviceCapability(
                device_id=device_id,
                device_type=DeviceType.CPU,
                compute_capability=1.0,  # 标准化为1.0
                memory_total=memory_info.total,
                memory_bandwidth=50.0,  # 估算CPU内存带宽
                compute_units=cpu_count,
                frequency=cpu_freq.current if cpu_freq else 2000.0,
                supports_fp16=False,  # CPU通常使用FP32
                supports_bf16=False,
                supports_int8=True
            )
            
            logger.info(f"发现CPU: {cpu_count}核心, {memory_info.total//1024**3}GB内存")
            return {device_id: capability}
            
        except Exception as e:
            logger.warning(f"发现CPU失败: {e}")
            return {}
    
    def _discover_gpu_interconnects(self, gpu_devices: Dict[str, DeviceCapability]) -> List[InterconnectInfo]:
        """发现GPU间互联"""
        interconnects = []
        gpu_ids = [int(dev.split(':')[1]) for dev in gpu_devices.keys() if dev.startswith('cuda:')]
        
        for i, gpu_a in enumerate(gpu_ids):
            for gpu_b in gpu_ids[i+1:]:
                try:
                    # 检测P2P支持
                    if torch.cuda.can_device_access_peer(gpu_a, gpu_b):
                        # 测量P2P带宽
                        bandwidth = self._measure_p2p_bandwidth(gpu_a, gpu_b)
                        interconnect_type = InterconnectType.P2P
                        
                        if bandwidth > 100.0:  # 高带宽，可能是NVLink
                            interconnect_type = InterconnectType.NVLINK
                        
                        interconnects.append(InterconnectInfo(
                            src_device=f"cuda:{gpu_a}",
                            dst_device=f"cuda:{gpu_b}",
                            interconnect_type=interconnect_type,
                            bandwidth=bandwidth,
                            bidirectional=True
                        ))
                        
                        logger.info(f"发现GPU互联: cuda:{gpu_a} <-> cuda:{gpu_b} ({bandwidth:.1f} GB/s)")
                    
                except Exception as e:
                    logger.warning(f"检测GPU {gpu_a}-{gpu_b} 互联失败: {e}")
        
        return interconnects
    
    def _discover_system_interconnects(self, devices: Dict[str, DeviceCapability]) -> List[InterconnectInfo]:
        """发现系统级互联"""
        interconnects = []
        
        # 所有设备都通过系统总线连接到CPU/系统内存
        cpu_bandwidth = 20.0  # 估算PCIe带宽
        
        for device_id in devices.keys():
            if device_id != "cpu":
                interconnects.append(InterconnectInfo(
                    src_device=device_id,
                    dst_device="cpu",
                    interconnect_type=InterconnectType.PCIE,
                    bandwidth=cpu_bandwidth,
                    bidirectional=True
                ))
        
        return interconnects
    
    def _discover_numa_topology(self, devices: Dict[str, DeviceCapability]) -> Dict[str, int]:
        """发现NUMA拓扑"""
        numa_nodes = {}
        
        # 简化实现：所有设备分配到NUMA节点0
        for device_id in devices.keys():
            numa_nodes[device_id] = 0
        
        return numa_nodes
    
    def _estimate_memory_bandwidth(self, props) -> float:
        """估算GPU内存带宽"""
        # 基于GPU型号和规格估算
        if "A100" in props.name:
            return 1555.0  # A100的HBM2e带宽
        elif "V100" in props.name:
            return 900.0   # V100的HBM2带宽
        elif "RTX" in props.name:
            return 800.0   # RTX系列的GDDR6X带宽
        elif "GTX" in props.name:
            return 400.0   # GTX系列的GDDR5带宽
        else:
            # 基于内存大小粗略估算
            memory_gb = props.total_memory / (1024**3)
            return min(memory_gb * 50, 1000.0)  # 保守估算
    
    def _get_gpu_power_limit(self, gpu_id: int) -> float:
        """获取GPU功耗限制"""
        try:
            # 这里应该使用nvidia-ml-py，简化返回估算值
            props = torch.cuda.get_device_properties(gpu_id)
            if "A100" in props.name:
                return 400.0
            elif "V100" in props.name:
                return 300.0
            else:
                return 250.0
        except:
            return 250.0
    
    def _get_gpu_thermal_limit(self, gpu_id: int) -> float:
        """获取GPU温度限制"""
        try:
            # 大多数GPU的热节流温度
            return 83.0
        except:
            return 83.0
    
    def _measure_p2p_bandwidth(self, gpu_a: int, gpu_b: int) -> float:
        """测量P2P带宽"""
        try:
            # 创建测试数据
            test_size = 128 * 1024 * 1024  # 128MB
            test_data = torch.randn(test_size // 4, dtype=torch.float32, device=f'cuda:{gpu_a}')
            
            # 预热
            for _ in range(3):
                _ = test_data.to(f'cuda:{gpu_b}')
                torch.cuda.synchronize()
            
            # 测量传输时间
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(10):
                transferred = test_data.to(f'cuda:{gpu_b}')
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            # 计算带宽
            total_bytes = test_size * 10
            total_time = end_time - start_time
            bandwidth_gbps = (total_bytes / total_time) / (1024**3)
            
            return bandwidth_gbps
            
        except Exception as e:
            logger.warning(f"测量P2P带宽失败: {e}")
            return 16.0  # 默认PCIe 3.0 x16带宽
    
    def _log_topology_summary(self):
        """记录拓扑摘要"""
        if not self.device_topology:
            return
        
        logger.info("=== 设备拓扑摘要 ===")
        
        # 设备摘要
        device_counts = defaultdict(int)
        for device in self.device_topology.devices.values():
            device_counts[device.device_type.value] += 1
        
        for device_type, count in device_counts.items():
            logger.info(f"{device_type.upper()}: {count}个设备")
        
        # 互联摘要
        interconnect_counts = defaultdict(int)
        for interconnect in self.device_topology.interconnects:
            interconnect_counts[interconnect.interconnect_type.value] += 1
        
        for interconnect_type, count in interconnect_counts.items():
            logger.info(f"{interconnect_type.upper()}互联: {count}个")
    
    def get_optimal_device_allocation(self, workload_size: int, 
                                    memory_requirement: int) -> Dict[str, float]:
        """获取最优设备分配"""
        if not self.device_topology:
            return {}
        
        allocation = {}
        
        # 计算每个设备的适合度分数
        device_scores = {}
        for device_id, capability in self.device_topology.devices.items():
            if capability.device_type != DeviceType.GPU:
                continue
            
            # 综合评分：计算能力 + 内存容量 + 可用性
            compute_score = capability.compute_capability
            memory_score = min(capability.memory_total / memory_requirement, 1.0)
            utilization_score = 1.0 - self._get_current_utilization(device_id)
            
            total_score = compute_score * 0.4 + memory_score * 0.4 + utilization_score * 0.2
            device_scores[device_id] = total_score
        
        # 按分数排序并分配
        sorted_devices = sorted(device_scores.items(), key=lambda x: x[1], reverse=True)
        
        remaining_workload = 1.0
        for device_id, score in sorted_devices:
            if remaining_workload <= 0:
                break
            
            # 根据设备能力分配工作负载比例
            allocation_ratio = min(score, remaining_workload)
            allocation[device_id] = allocation_ratio
            remaining_workload -= allocation_ratio
        
        return allocation
    
    def _get_current_utilization(self, device_id: str) -> float:
        """获取当前设备利用率"""
        if device_id.startswith('cuda:'):
            try:
                gpu_id = int(device_id.split(':')[1])
                return torch.cuda.utilization(gpu_id) / 100.0
            except:
                return 0.0
        return 0.0
    
    def get_transfer_strategy(self, src_device: str, dst_device: str, 
                            data_size: int) -> Dict[str, Any]:
        """获取数据传输策略"""
        if not self.device_topology:
            return {"strategy": "direct"}
        
        bandwidth = self.device_topology.get_bandwidth(src_device, dst_device)
        
        if bandwidth > 50.0:  # 高带宽连接
            return {
                "strategy": "direct_p2p",
                "chunk_size": min(data_size, 256 * 1024 * 1024),  # 256MB chunks
                "async": True
            }
        elif bandwidth > 0:  # 中等带宽
            return {
                "strategy": "direct",
                "chunk_size": min(data_size, 64 * 1024 * 1024),   # 64MB chunks
                "async": False
            }
        else:  # 无直连，通过系统内存
            return {
                "strategy": "via_system_memory",
                "chunk_size": min(data_size, 32 * 1024 * 1024),   # 32MB chunks
                "async": False
            }
    
    def get_device_topology(self) -> Optional[DeviceTopology]:
        """获取设备拓扑"""
        return self.device_topology
    
    def get_device_recommendations(self) -> Dict[str, Any]:
        """获取设备使用建议"""
        if not self.device_topology:
            return {}
        
        recommendations = {
            "optimal_strategies": {},
            "bottlenecks": [],
            "optimizations": []
        }
        
        gpu_devices = [d for d in self.device_topology.devices.values() 
                      if d.device_type == DeviceType.GPU]
        
        if len(gpu_devices) == 1:
            recommendations["optimal_strategies"]["inference"] = "single_gpu_optimized"
            recommendations["optimizations"].append("启用混合精度推理")
            recommendations["optimizations"].append("激活内存池化")
        
        elif len(gpu_devices) == 2:
            recommendations["optimal_strategies"]["inference"] = "dual_gpu_pipeline"
            recommendations["optimizations"].append("启用GPU间流水线")
            recommendations["optimizations"].append("使用P2P传输优化")
        
        elif len(gpu_devices) > 2:
            recommendations["optimal_strategies"]["inference"] = "multi_gpu_distributed"
            recommendations["optimizations"].append("启用模型并行")
            recommendations["optimizations"].append("使用集合通信优化")
        
        return recommendations
    
    def start_monitoring(self):
        """启动性能监控"""
        if self.monitoring_enabled:
            return
        
        self.monitoring_enabled = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        logger.info("设备性能监控已启动")
    
    def stop_monitoring(self):
        """停止性能监控"""
        self.monitoring_enabled = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        logger.info("设备性能监控已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_enabled:
            try:
                with self._monitor_lock:
                    self._update_performance_metrics()
                time.sleep(5.0)  # 每5秒更新一次
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                time.sleep(10.0)
    
    def _update_performance_metrics(self):
        """更新性能指标"""
        if not self.device_topology:
            return
        
        for device_id, capability in self.device_topology.devices.items():
            if capability.device_type == DeviceType.GPU:
                utilization = self._get_current_utilization(device_id)
                self.performance_history[device_id].append(utilization)
                
                # 保持历史长度
                if len(self.performance_history[device_id]) > 100:
                    self.performance_history[device_id] = self.performance_history[device_id][-50:] 