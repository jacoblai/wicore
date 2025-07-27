"""
设备管理器 - WiCore Mojo 推理引擎
负责硬件设备的发现、管理和优化调度
支持异构硬件统一抽象：GPU、CPU、NPU等
"""

from max import engine
from collections import Dict, List
from python import Python
import time

struct DeviceInfo:
    var device_id: String
    var device_type: String  # "gpu", "cpu", "npu"
    var memory_total: Int
    var memory_available: Int
    var compute_capability: Float64
    var numa_node: Int
    var pcie_bandwidth: Float64  # GB/s
    var is_available: Bool

struct DeviceTopology:
    var device_connections: Dict[String, List[String]]
    var bandwidth_matrix: Dict[String, Dict[String, Float64]]
    
    fn __init__(inout self):
        self.device_connections = Dict[String, List[String]]()
        self.bandwidth_matrix = Dict[String, Dict[String, Float64]]()
    
    fn add_connection(inout self, device_a: String, device_b: String, bandwidth: Float64):
        """添加设备间连接"""
        if device_a not in self.device_connections:
            self.device_connections[device_a] = List[String]()
        if device_b not in self.device_connections:
            self.device_connections[device_b] = List[String]()
        
        self.device_connections[device_a].append(device_b)
        self.device_connections[device_b].append(device_a)
        
        if device_a not in self.bandwidth_matrix:
            self.bandwidth_matrix[device_a] = Dict[String, Float64]()
        if device_b not in self.bandwidth_matrix:
            self.bandwidth_matrix[device_b] = Dict[String, Float64]()
        
        self.bandwidth_matrix[device_a][device_b] = bandwidth
        self.bandwidth_matrix[device_b][device_a] = bandwidth
    
    fn get_optimal_path(self, source: String, target: String) -> List[String]:
        """获取设备间最优数据传输路径"""
        # 简化的路径查找算法
        if source == target:
            return [source]
        
        if source in self.device_connections and target in self.device_connections[source]:
            return [source, target]
        
        # 返回直接路径（实际应用中需要实现 Dijkstra 或类似算法）
        return [source, target]

struct DeviceManager:
    var devices: List[DeviceInfo]
    var device_topology: DeviceTopology
    var target_devices: List[String]
    
    fn __init__(inout self, target_devices: List[String]):
        """初始化设备管理器"""
        self.devices = List[DeviceInfo]()
        self.device_topology = DeviceTopology()
        self.target_devices = target_devices
    
    fn initialize(self) -> Bool:
        """发现和初始化所有设备"""
        print("🔍 开始设备发现...")
        
        try:
            # 使用 MAX Engine 发现硬件设备
            discovered_devices = engine.discover_devices()
            
            print(f"📊 发现 {len(discovered_devices)} 个计算设备")
            
            # 转换为 DeviceInfo 格式
            for device in discovered_devices:
                device_info = DeviceInfo(
                    device_id=device.id,
                    device_type=device.type,
                    memory_total=device.memory_total,
                    memory_available=device.memory_available,
                    compute_capability=device.compute_capability,
                    numa_node=self._get_numa_node(device.id),
                    pcie_bandwidth=self._get_pcie_bandwidth(device.id),
                    is_available=True
                )
                self.devices.append(device_info)
                
                print(f"✅ 设备: {device.type}:{device.id}")
                print(f"   内存: {device.memory_available/1e9:.1f}GB / {device.memory_total/1e9:.1f}GB")
                print(f"   计算能力: {device.compute_capability}")
            
            # 构建设备拓扑图
            self._build_topology()
            
            # 验证目标设备可用性
            available_devices = [d.device_id for d in self.devices if d.is_available]
            for target in self.target_devices:
                if target not in available_devices:
                    print(f"⚠️  目标设备 {target} 不可用")
                    return False
            
            print("✅ 设备管理器初始化完成")
            return True
            
        except Exception as e:
            print(f"❌ 设备发现失败: {e}")
            return False
    
    fn get_optimal_devices(self, memory_requirement: Int, 
                          compute_requirement: Float64) -> List[String]:
        """根据需求选择最优设备组合"""
        suitable_devices = List[String]()
        
        for device in self.devices:
            if (device.is_available and 
                device.memory_available >= memory_requirement and
                device.compute_capability >= compute_requirement):
                suitable_devices.append(device.device_id)
        
        # 优先选择 GPU 设备
        gpu_devices = [d for d in suitable_devices if "gpu" in d]
        if len(gpu_devices) > 0:
            return gpu_devices[:2]  # 最多选择2个GPU
        
        # 回退到 CPU 设备
        cpu_devices = [d for d in suitable_devices if "cpu" in d]
        return cpu_devices[:1]  # 选择1个CPU
    
    fn get_device_info(self, device_id: String) -> Optional[DeviceInfo]:
        """获取设备信息"""
        for device in self.devices:
            if device.device_id == device_id:
                return device
        return None
    
    fn _build_topology(self):
        """构建设备拓扑图"""
        print("🔗 构建设备拓扑图...")
        
        # 检测 GPU 间的 NVLink/PCIe 连接
        gpu_devices = [d for d in self.devices if d.device_type == "gpu"]
        
        for i in range(len(gpu_devices)):
            for j in range(i + 1, len(gpu_devices)):
                device_a = gpu_devices[i].device_id
                device_b = gpu_devices[j].device_id
                
                # 检测连接类型和带宽
                bandwidth = self._detect_inter_gpu_bandwidth(device_a, device_b)
                
                if bandwidth > 0:
                    self.device_topology.add_connection(device_a, device_b, bandwidth)
                    print(f"🔗 检测到连接: {device_a} ↔ {device_b} ({bandwidth:.1f} GB/s)")
        
        print("✅ 设备拓扑图构建完成")
    
    fn _get_numa_node(self, device_id: String) -> Int:
        """获取设备的 NUMA 节点"""
        # 简化实现，实际应读取 /sys/bus/pci/devices/*/numa_node
        if "gpu:0" in device_id:
            return 0
        elif "gpu:1" in device_id:
            return 1
        else:
            return 0
    
    fn _get_pcie_bandwidth(self, device_id: String) -> Float64:
        """获取 PCIe 带宽"""
        # 简化实现，T10 通常为 PCIe 4.0 x16
        if "gpu" in device_id:
            return 32.0  # PCIe 4.0 x16 = ~32 GB/s
        else:
            return 8.0   # CPU 连接带宽
    
    fn _detect_inter_gpu_bandwidth(self, device_a: String, device_b: String) -> Float64:
        """检测 GPU 间连接带宽"""
        # 实际实现应使用 nvidia-ml-py 或 nvml
        # 这里简化为固定值
        return 50.0  # NVLink 带宽，T10 支持 NVLink
    
    fn cleanup(self):
        """清理设备管理器"""
        print("🧹 清理设备管理器...")
        
        # 释放设备资源
        for device in self.devices:
            if device.is_available:
                # 这里可以添加设备特定的清理逻辑
                pass
        
        self.devices.clear()
        print("✅ 设备管理器清理完成") 