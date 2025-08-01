"""
WiCore通用负载均衡器
设备无关的智能负载分配和动态均衡算法

核心算法:
- 自适应负载分配：基于设备能力的智能分配
- 动态重平衡：实时监控和调整负载分布
- 多目标优化：兼顾性能、效率和公平性
- 容错处理：自动处理设备故障和性能波动
"""

import torch
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import time
import threading
import math
from collections import defaultdict, deque

from .universal_device_manager import UniversalDeviceManager, DeviceTopology, DeviceType, DeviceCapability

logger = logging.getLogger(__name__)


class LoadBalanceStrategy(Enum):
    """负载均衡策略"""
    ROUND_ROBIN = "round_robin"           # 轮询分配
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"  # 加权轮询
    LEAST_CONNECTIONS = "least_connections"        # 最少连接
    CAPABILITY_BASED = "capability_based"          # 基于能力
    ADAPTIVE = "adaptive"                 # 自适应
    PERFORMANCE_AWARE = "performance_aware"        # 性能感知


class LoadMetric(Enum):
    """负载指标"""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_USAGE = "memory_usage"
    COMPUTE_UTILIZATION = "compute_utilization"
    BANDWIDTH_USAGE = "bandwidth_usage"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"


@dataclass
class DeviceLoad:
    """设备负载状态"""
    device_id: str
    utilization: float          # 0-1, 设备利用率
    memory_usage: float         # 0-1, 内存使用率
    queue_length: int          # 排队任务数
    avg_response_time: float   # 平均响应时间(ms)
    temperature: float         # 设备温度
    power_usage: float         # 功耗使用率
    last_update: float         # 最后更新时间
    
    def get_composite_load(self) -> float:
        """获取综合负载分数"""
        return (self.utilization * 0.4 + 
                self.memory_usage * 0.3 + 
                min(self.queue_length / 10.0, 1.0) * 0.2 + 
                min(self.avg_response_time / 100.0, 1.0) * 0.1)


@dataclass
class LoadBalanceDecision:
    """负载均衡决策"""
    target_device: str
    confidence: float           # 决策置信度
    expected_completion_time: float  # 预期完成时间
    load_after_assignment: float     # 分配后负载
    reasoning: str             # 决策理由


@dataclass
class WorkloadRequest:
    """工作负载请求"""
    request_id: str
    compute_requirement: float  # 计算需求(GFLOPS)
    memory_requirement: int     # 内存需求(bytes)
    expected_duration: float    # 预期执行时间(ms)
    priority: int = 1          # 优先级(1-10)
    affinity: Set[str] = None  # 设备亲和性


class UniversalLoadBalancer:
    """通用负载均衡器
    
    智能的设备无关负载均衡系统：
    - 多策略支持：适应不同场景需求
    - 实时监控：持续跟踪设备状态
    - 预测性调度：基于历史数据预测最优分配
    - 自动故障处理：设备异常时的负载转移
    """
    
    def __init__(self, device_manager: UniversalDeviceManager, 
                 strategy: LoadBalanceStrategy = LoadBalanceStrategy.ADAPTIVE):
        """
        初始化通用负载均衡器
        
        Args:
            device_manager: 通用设备管理器
            strategy: 负载均衡策略
        """
        self.device_manager = device_manager
        self.strategy = strategy
        self.topology = device_manager.get_device_topology()
        
        # 负载状态跟踪
        self.device_loads: Dict[str, DeviceLoad] = {}
        self.load_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.workload_queue: List[WorkloadRequest] = []
        
        # 性能统计
        self.assignment_history: List[Tuple[str, str, float]] = []  # (request_id, device, duration)
        self.balance_operations: int = 0
        self.successful_assignments: int = 0
        self.failed_assignments: int = 0
        
        # 监控和控制
        self.monitoring_enabled = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.balance_lock = threading.RLock()
        
        # 配置参数
        self.config = {
            "balance_interval": 5.0,      # 负载均衡间隔(秒)
            "load_threshold": 0.8,        # 负载阈值
            "migration_threshold": 0.3,   # 迁移阈值
            "history_window": 60,         # 历史窗口(秒)
            "prediction_window": 30,      # 预测窗口(秒)
        }
        
        self._initialize_device_loads()
        logger.info(f"通用负载均衡器初始化完成，策略: {strategy.value}")
    
    def _initialize_device_loads(self):
        """初始化设备负载状态"""
        if not self.topology:
            return
        
        for device_id, capability in self.topology.devices.items():
            if capability.device_type in [DeviceType.GPU, DeviceType.CPU]:
                self.device_loads[device_id] = DeviceLoad(
                    device_id=device_id,
                    utilization=0.0,
                    memory_usage=0.0,
                    queue_length=0,
                    avg_response_time=0.0,
                    temperature=25.0,  # 室温
                    power_usage=0.0,
                    last_update=time.time()
                )
    
    def assign_workload(self, request: WorkloadRequest) -> Optional[LoadBalanceDecision]:
        """分配工作负载到最优设备"""
        with self.balance_lock:
            try:
                # 更新设备状态
                self._update_device_loads()
                
                # 筛选候选设备
                candidates = self._filter_candidates(request)
                if not candidates:
                    logger.warning(f"没有找到合适的设备处理请求: {request.request_id}")
                    self.failed_assignments += 1
                    return None
                
                # 根据策略选择设备
                decision = self._select_device(request, candidates)
                
                if decision:
                    # 更新设备负载
                    self._update_load_after_assignment(decision.target_device, request)
                    self.successful_assignments += 1
                    
                    logger.debug(f"分配请求 {request.request_id} 到 {decision.target_device}")
                
                return decision
                
            except Exception as e:
                logger.error(f"工作负载分配失败: {e}")
                self.failed_assignments += 1
                return None
    
    def _filter_candidates(self, request: WorkloadRequest) -> List[str]:
        """筛选候选设备"""
        candidates = []
        
        for device_id, load in self.device_loads.items():
            # 检查设备类型
            device_capability = self.topology.devices.get(device_id)
            if not device_capability:
                continue
            
            # 检查亲和性
            if request.affinity and device_id not in request.affinity:
                continue
            
            # 检查内存容量
            if device_capability.memory_total < request.memory_requirement:
                continue
            
            # 检查负载状态
            if load.utilization > self.config["load_threshold"]:
                continue
            
            # 检查设备健康状态
            if self._is_device_healthy(device_id):
                candidates.append(device_id)
        
        return candidates
    
    def _select_device(self, request: WorkloadRequest, 
                      candidates: List[str]) -> Optional[LoadBalanceDecision]:
        """根据策略选择设备"""
        if not candidates:
            return None
        
        if self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
            return self._round_robin_selection(request, candidates)
        elif self.strategy == LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(request, candidates)
        elif self.strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(request, candidates)
        elif self.strategy == LoadBalanceStrategy.CAPABILITY_BASED:
            return self._capability_based_selection(request, candidates)
        elif self.strategy == LoadBalanceStrategy.PERFORMANCE_AWARE:
            return self._performance_aware_selection(request, candidates)
        else:  # ADAPTIVE
            return self._adaptive_selection(request, candidates)
    
    def _round_robin_selection(self, request: WorkloadRequest, 
                              candidates: List[str]) -> LoadBalanceDecision:
        """轮询选择"""
        # 简单轮询：基于请求计数
        device_index = self.successful_assignments % len(candidates)
        target_device = candidates[device_index]
        
        return LoadBalanceDecision(
            target_device=target_device,
            confidence=0.7,
            expected_completion_time=request.expected_duration,
            load_after_assignment=self.device_loads[target_device].get_composite_load() + 0.1,
            reasoning="轮询分配"
        )
    
    def _weighted_round_robin_selection(self, request: WorkloadRequest, 
                                       candidates: List[str]) -> LoadBalanceDecision:
        """加权轮询选择"""
        # 基于设备能力计算权重
        weights = []
        for device_id in candidates:
            capability = self.topology.devices[device_id]
            load = self.device_loads[device_id]
            
            # 权重 = 计算能力 × (1 - 当前负载)
            weight = capability.compute_capability * (1.0 - load.get_composite_load())
            weights.append(weight)
        
        # 加权随机选择
        total_weight = sum(weights)
        if total_weight == 0:
            return self._round_robin_selection(request, candidates)
        
        import random
        r = random.uniform(0, total_weight)
        cumulative = 0
        
        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                target_device = candidates[i]
                break
        else:
            target_device = candidates[0]
        
        return LoadBalanceDecision(
            target_device=target_device,
            confidence=0.8,
            expected_completion_time=request.expected_duration * (1.0 + self.device_loads[target_device].get_composite_load()),
            load_after_assignment=self.device_loads[target_device].get_composite_load() + 0.15,
            reasoning="加权轮询分配"
        )
    
    def _least_connections_selection(self, request: WorkloadRequest, 
                                    candidates: List[str]) -> LoadBalanceDecision:
        """最少连接选择"""
        # 选择队列长度最短的设备
        target_device = min(candidates, 
                           key=lambda x: self.device_loads[x].queue_length)
        
        return LoadBalanceDecision(
            target_device=target_device,
            confidence=0.75,
            expected_completion_time=request.expected_duration * (1.0 + self.device_loads[target_device].queue_length * 0.1),
            load_after_assignment=self.device_loads[target_device].get_composite_load() + 0.1,
            reasoning="最少队列长度"
        )
    
    def _capability_based_selection(self, request: WorkloadRequest, 
                                   candidates: List[str]) -> LoadBalanceDecision:
        """基于能力选择"""
        # 计算每个设备的能力匹配度
        scores = {}
        for device_id in candidates:
            capability = self.topology.devices[device_id]
            load = self.device_loads[device_id]
            
            # 计算能力分数
            compute_score = min(capability.compute_capability / (request.compute_requirement / 1000.0), 2.0)
            memory_score = min(capability.memory_total / request.memory_requirement, 2.0)
            load_score = 1.0 - load.get_composite_load()
            
            total_score = compute_score * 0.4 + memory_score * 0.3 + load_score * 0.3
            scores[device_id] = total_score
        
        # 选择得分最高的设备
        target_device = max(scores.keys(), key=lambda x: scores[x])
        
        return LoadBalanceDecision(
            target_device=target_device,
            confidence=0.85,
            expected_completion_time=request.expected_duration / scores[target_device],
            load_after_assignment=self.device_loads[target_device].get_composite_load() + 0.12,
            reasoning=f"能力匹配(分数: {scores[target_device]:.2f})"
        )
    
    def _performance_aware_selection(self, request: WorkloadRequest, 
                                    candidates: List[str]) -> LoadBalanceDecision:
        """性能感知选择"""
        # 基于历史性能选择
        scores = {}
        
        for device_id in candidates:
            # 获取历史性能数据
            history = self.load_history[device_id]
            if len(history) < 5:
                # 历史数据不足，使用能力分数
                capability = self.topology.devices[device_id]
                scores[device_id] = capability.compute_capability
            else:
                # 计算平均性能
                recent_loads = list(history)[-10:]  # 最近10次
                avg_load = sum(recent_loads) / len(recent_loads)
                performance_score = 1.0 - avg_load
                scores[device_id] = performance_score
        
        target_device = max(scores.keys(), key=lambda x: scores[x])
        
        return LoadBalanceDecision(
            target_device=target_device,
            confidence=0.9,
            expected_completion_time=request.expected_duration * (1.0 + (1.0 - scores[target_device])),
            load_after_assignment=self.device_loads[target_device].get_composite_load() + 0.1,
            reasoning=f"历史性能(分数: {scores[target_device]:.2f})"
        )
    
    def _adaptive_selection(self, request: WorkloadRequest, 
                           candidates: List[str]) -> LoadBalanceDecision:
        """自适应选择"""
        # 综合多种因素的自适应算法
        scores = {}
        
        for device_id in candidates:
            capability = self.topology.devices[device_id]
            load = self.device_loads[device_id]
            
            # 多维度评分
            compute_factor = min(capability.compute_capability / 8.0, 1.0)
            memory_factor = min(capability.memory_total / request.memory_requirement, 2.0) / 2.0
            load_factor = 1.0 - load.get_composite_load()
            
            # 历史性能因子
            history = self.load_history[device_id]
            if len(history) >= 5:
                recent_performance = 1.0 - (sum(list(history)[-5:]) / 5)
                performance_factor = recent_performance
            else:
                performance_factor = 0.5  # 默认值
            
            # 温度因子（防止过热）
            temp_factor = max(0.5, 1.0 - (load.temperature - 60.0) / 30.0)
            
            # 综合评分
            total_score = (compute_factor * 0.25 + 
                          memory_factor * 0.2 + 
                          load_factor * 0.25 + 
                          performance_factor * 0.2 + 
                          temp_factor * 0.1)
            
            scores[device_id] = total_score
        
        target_device = max(scores.keys(), key=lambda x: scores[x])
        
        return LoadBalanceDecision(
            target_device=target_device,
            confidence=0.95,
            expected_completion_time=request.expected_duration / max(scores[target_device], 0.1),
            load_after_assignment=self.device_loads[target_device].get_composite_load() + 0.08,
            reasoning=f"自适应算法(综合分数: {scores[target_device]:.2f})"
        )
    
    def _is_device_healthy(self, device_id: str) -> bool:
        """检查设备健康状态"""
        load = self.device_loads.get(device_id)
        if not load:
            return False
        
        # 检查温度
        if load.temperature > 85.0:
            return False
        
        # 检查响应时间
        if load.avg_response_time > 1000.0:  # 超过1秒
            return False
        
        # 检查最后更新时间
        if time.time() - load.last_update > 30.0:  # 超过30秒未更新
            return False
        
        return True
    
    def _update_device_loads(self):
        """更新设备负载状态"""
        current_time = time.time()
        
        for device_id, load in self.device_loads.items():
            try:
                # 获取实时负载数据
                utilization = self._get_device_utilization(device_id)
                memory_usage = self._get_memory_usage(device_id)
                temperature = self._get_device_temperature(device_id)
                
                # 更新负载状态
                load.utilization = utilization
                load.memory_usage = memory_usage
                load.temperature = temperature
                load.last_update = current_time
                
                # 记录历史
                composite_load = load.get_composite_load()
                self.load_history[device_id].append(composite_load)
                
            except Exception as e:
                logger.warning(f"更新设备 {device_id} 负载失败: {e}")
    
    def _get_device_utilization(self, device_id: str) -> float:
        """获取设备利用率"""
        if device_id.startswith('cuda:'):
            try:
                gpu_id = int(device_id.split(':')[1])
                return torch.cuda.utilization(gpu_id) / 100.0
            except:
                return 0.0
        elif device_id == 'cpu':
            try:
                import psutil
                return psutil.cpu_percent(interval=0.1) / 100.0
            except:
                return 0.0
        return 0.0
    
    def _get_memory_usage(self, device_id: str) -> float:
        """获取内存使用率"""
        if device_id.startswith('cuda:'):
            try:
                gpu_id = int(device_id.split(':')[1])
                memory_used = torch.cuda.memory_allocated(gpu_id)
                memory_total = torch.cuda.get_device_properties(gpu_id).total_memory
                return memory_used / memory_total
            except:
                return 0.0
        elif device_id == 'cpu':
            try:
                import psutil
                memory = psutil.virtual_memory()
                return memory.percent / 100.0
            except:
                return 0.0
        return 0.0
    
    def _get_device_temperature(self, device_id: str) -> float:
        """获取设备温度"""
        # 简化实现，实际应该使用nvidia-ml-py
        if device_id.startswith('cuda:'):
            try:
                # 基于利用率估算温度
                utilization = self._get_device_utilization(device_id)
                return 35.0 + utilization * 45.0  # 35-80°C范围
            except:
                return 50.0
        return 25.0  # CPU温度
    
    def _update_load_after_assignment(self, device_id: str, request: WorkloadRequest):
        """分配后更新设备负载"""
        load = self.device_loads.get(device_id)
        if not load:
            return
        
        # 增加队列长度
        load.queue_length += 1
        
        # 预估负载增加
        capability = self.topology.devices[device_id]
        load_increase = request.compute_requirement / (capability.compute_capability * 1000.0)
        load.utilization = min(load.utilization + load_increase, 1.0)
    
    def complete_workload(self, device_id: str, request_id: str, 
                         actual_duration: float):
        """完成工作负载"""
        with self.balance_lock:
            load = self.device_loads.get(device_id)
            if load:
                # 减少队列长度
                load.queue_length = max(load.queue_length - 1, 0)
                
                # 更新平均响应时间
                if load.avg_response_time == 0:
                    load.avg_response_time = actual_duration
                else:
                    load.avg_response_time = (load.avg_response_time * 0.8 + 
                                            actual_duration * 0.2)
            
            # 记录完成历史
            self.assignment_history.append((request_id, device_id, actual_duration))
            
            logger.debug(f"工作负载完成: {request_id} on {device_id} ({actual_duration:.2f}ms)")
    
    def rebalance_loads(self) -> Dict[str, Any]:
        """重新平衡负载"""
        with self.balance_lock:
            self.balance_operations += 1
            
            # 分析负载分布
            load_distribution = {}
            for device_id, load in self.device_loads.items():
                load_distribution[device_id] = load.get_composite_load()
            
            # 识别过载和空闲设备
            avg_load = sum(load_distribution.values()) / len(load_distribution)
            overloaded = {k: v for k, v in load_distribution.items() 
                         if v > avg_load + self.config["migration_threshold"]}
            underloaded = {k: v for k, v in load_distribution.items() 
                          if v < avg_load - self.config["migration_threshold"]}
            
            rebalance_actions = []
            
            # 生成重平衡建议
            for overloaded_device, load in overloaded.items():
                if underloaded:
                    # 找到最空闲的设备
                    target_device = min(underloaded.keys(), 
                                       key=lambda x: underloaded[x])
                    
                    action = {
                        "action": "migrate_workload",
                        "from": overloaded_device,
                        "to": target_device,
                        "load_difference": load - underloaded[target_device]
                    }
                    rebalance_actions.append(action)
            
            return {
                "timestamp": time.time(),
                "avg_load": avg_load,
                "load_distribution": load_distribution,
                "overloaded_devices": list(overloaded.keys()),
                "underloaded_devices": list(underloaded.keys()),
                "rebalance_actions": rebalance_actions
            }
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """获取负载均衡器统计"""
        total_assignments = self.successful_assignments + self.failed_assignments
        success_rate = self.successful_assignments / max(total_assignments, 1)
        
        return {
            "strategy": self.strategy.value,
            "total_assignments": total_assignments,
            "successful_assignments": self.successful_assignments,
            "failed_assignments": self.failed_assignments,
            "success_rate": success_rate,
            "balance_operations": self.balance_operations,
            "device_loads": {
                device_id: {
                    "utilization": load.utilization,
                    "memory_usage": load.memory_usage,
                    "queue_length": load.queue_length,
                    "composite_load": load.get_composite_load(),
                    "temperature": load.temperature
                }
                for device_id, load in self.device_loads.items()
            }
        } 