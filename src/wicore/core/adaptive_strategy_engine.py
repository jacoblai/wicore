"""
WiCore自适应策略引擎
根据硬件配置自动选择最优推理策略的智能引擎

核心功能:
- 策略自动选择：基于硬件能力智能匹配策略
- 性能预测：评估不同策略的预期性能
- 动态调整：运行时根据实际性能调整策略
- 可扩展框架：支持插件式策略注册
"""

import torch
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import time
import math

from .universal_device_manager import UniversalDeviceManager, DeviceTopology, DeviceType, DeviceCapability

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """策略类型"""
    INFERENCE = "inference"
    MEMORY_MANAGEMENT = "memory_management"
    LOAD_BALANCING = "load_balancing"
    DATA_TRANSFER = "data_transfer"
    PRECISION = "precision"


class PerformanceProfile(Enum):
    """性能配置文件"""
    LATENCY_OPTIMIZED = "latency_optimized"      # 延迟优化
    THROUGHPUT_OPTIMIZED = "throughput_optimized" # 吞吐量优化
    MEMORY_OPTIMIZED = "memory_optimized"        # 内存优化
    POWER_OPTIMIZED = "power_optimized"          # 功耗优化
    BALANCED = "balanced"                        # 平衡模式


@dataclass
class StrategyConfig:
    """策略配置"""
    strategy_id: str
    strategy_type: StrategyType
    description: str
    min_devices: int = 1
    max_devices: int = 1
    memory_requirement: int = 0  # bytes
    compute_requirement: float = 0.0
    supported_precisions: List[str] = None
    performance_profile: PerformanceProfile = PerformanceProfile.BALANCED


@dataclass
class StrategyEvaluation:
    """策略评估结果"""
    strategy_id: str
    compatibility_score: float  # 0-1, 硬件兼容性
    performance_score: float    # 0-1, 预期性能
    efficiency_score: float     # 0-1, 资源效率
    total_score: float         # 综合评分
    estimated_latency: float   # 预估延迟(ms)
    estimated_throughput: float # 预估吞吐量(tokens/s)
    memory_usage: int          # 预估内存使用(bytes)


class OptimizationStrategy(ABC):
    """优化策略基类"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
    
    @abstractmethod
    def is_compatible(self, topology: DeviceTopology) -> bool:
        """检查策略是否兼容当前硬件"""
        pass
    
    @abstractmethod
    def evaluate_performance(self, topology: DeviceTopology, 
                           workload: Dict[str, Any]) -> StrategyEvaluation:
        """评估策略性能"""
        pass
    
    @abstractmethod
    def apply_strategy(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """应用策略"""
        pass


class SingleGPUStrategy(OptimizationStrategy):
    """单GPU优化策略"""
    
    def __init__(self):
        config = StrategyConfig(
            strategy_id="single_gpu_optimized",
            strategy_type=StrategyType.INFERENCE,
            description="单GPU内存和计算优化",
            min_devices=1,
            max_devices=1,
            supported_precisions=["fp16", "fp32", "int8"]
        )
        super().__init__(config)
    
    def is_compatible(self, topology: DeviceTopology) -> bool:
        gpu_count = sum(1 for d in topology.devices.values() 
                       if d.device_type == DeviceType.GPU)
        return gpu_count >= 1
    
    def evaluate_performance(self, topology: DeviceTopology, 
                           workload: Dict[str, Any]) -> StrategyEvaluation:
        gpu_devices = [d for d in topology.devices.values() 
                      if d.device_type == DeviceType.GPU]
        
        if not gpu_devices:
            return StrategyEvaluation(
                strategy_id=self.config.strategy_id,
                compatibility_score=0.0,
                performance_score=0.0,
                efficiency_score=0.0,
                total_score=0.0,
                estimated_latency=1000.0,
                estimated_throughput=0.0,
                memory_usage=0
            )
        
        # 选择最强的GPU
        best_gpu = max(gpu_devices, key=lambda x: x.compute_capability)
        
        # 评估分数
        compatibility_score = 1.0  # 总是兼容
        performance_score = min(best_gpu.compute_capability / 8.0, 1.0)  # 基于计算能力
        efficiency_score = 0.9  # 单GPU效率较高
        
        total_score = (compatibility_score * 0.3 + 
                      performance_score * 0.5 + 
                      efficiency_score * 0.2)
        
        # 估算性能指标
        estimated_latency = 50.0 / performance_score  # ms
        estimated_throughput = performance_score * 100.0  # tokens/s
        memory_usage = workload.get("model_size", 0)
        
        return StrategyEvaluation(
            strategy_id=self.config.strategy_id,
            compatibility_score=compatibility_score,
            performance_score=performance_score,
            efficiency_score=efficiency_score,
            total_score=total_score,
            estimated_latency=estimated_latency,
            estimated_throughput=estimated_throughput,
            memory_usage=memory_usage
        )
    
    def apply_strategy(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "forward_strategy": "sequential",
            "memory_strategy": "conservative",
            "precision": "fp16",
            "batch_size": 1,
            "enable_cache": True
        }


class DualGPUPipelineStrategy(OptimizationStrategy):
    """双GPU流水线策略"""
    
    def __init__(self):
        config = StrategyConfig(
            strategy_id="dual_gpu_pipeline",
            strategy_type=StrategyType.INFERENCE,
            description="双GPU流水线并行优化",
            min_devices=2,
            max_devices=2,
            supported_precisions=["fp16", "fp32"]
        )
        super().__init__(config)
    
    def is_compatible(self, topology: DeviceTopology) -> bool:
        gpu_count = sum(1 for d in topology.devices.values() 
                       if d.device_type == DeviceType.GPU)
        return gpu_count >= 2
    
    def evaluate_performance(self, topology: DeviceTopology, 
                           workload: Dict[str, Any]) -> StrategyEvaluation:
        gpu_devices = [d for d in topology.devices.values() 
                      if d.device_type == DeviceType.GPU]
        
        if len(gpu_devices) < 2:
            return StrategyEvaluation(
                strategy_id=self.config.strategy_id,
                compatibility_score=0.0,
                performance_score=0.0,
                efficiency_score=0.0,
                total_score=0.0,
                estimated_latency=1000.0,
                estimated_throughput=0.0,
                memory_usage=0
            )
        
        # 评估双GPU性能
        avg_compute = sum(d.compute_capability for d in gpu_devices[:2]) / 2
        
        # 检查GPU间互联
        gpu_ids = [d.device_id for d in gpu_devices[:2]]
        interconnect_bandwidth = topology.get_bandwidth(gpu_ids[0], gpu_ids[1])
        
        compatibility_score = 1.0 if len(gpu_devices) >= 2 else 0.0
        performance_score = min(avg_compute / 8.0 * 1.5, 1.0)  # 流水线加速
        
        # 互联带宽影响效率
        if interconnect_bandwidth > 50.0:
            efficiency_score = 0.85  # 高速互联
        elif interconnect_bandwidth > 20.0:
            efficiency_score = 0.75  # 中速互联
        else:
            efficiency_score = 0.65  # 低速互联
        
        total_score = (compatibility_score * 0.3 + 
                      performance_score * 0.5 + 
                      efficiency_score * 0.2)
        
        # 流水线性能估算
        estimated_latency = 40.0 / performance_score
        estimated_throughput = performance_score * 150.0  # 流水线提升吞吐量
        memory_usage = workload.get("model_size", 0)
        
        return StrategyEvaluation(
            strategy_id=self.config.strategy_id,
            compatibility_score=compatibility_score,
            performance_score=performance_score,
            efficiency_score=efficiency_score,
            total_score=total_score,
            estimated_latency=estimated_latency,
            estimated_throughput=estimated_throughput,
            memory_usage=memory_usage
        )
    
    def apply_strategy(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "forward_strategy": "pipeline",
            "memory_strategy": "distributed",
            "precision": "fp16",
            "batch_size": 2,
            "enable_p2p": True,
            "pipeline_depth": 2
        }


class MultiGPUDistributedStrategy(OptimizationStrategy):
    """多GPU分布式策略"""
    
    def __init__(self):
        config = StrategyConfig(
            strategy_id="multi_gpu_distributed",
            strategy_type=StrategyType.INFERENCE,
            description="多GPU分布式并行优化",
            min_devices=3,
            max_devices=16,
            supported_precisions=["fp16", "bf16", "fp32"]
        )
        super().__init__(config)
    
    def is_compatible(self, topology: DeviceTopology) -> bool:
        gpu_count = sum(1 for d in topology.devices.values() 
                       if d.device_type == DeviceType.GPU)
        return gpu_count >= 3
    
    def evaluate_performance(self, topology: DeviceTopology, 
                           workload: Dict[str, Any]) -> StrategyEvaluation:
        gpu_devices = [d for d in topology.devices.values() 
                      if d.device_type == DeviceType.GPU]
        
        gpu_count = len(gpu_devices)
        if gpu_count < 3:
            return StrategyEvaluation(
                strategy_id=self.config.strategy_id,
                compatibility_score=0.0,
                performance_score=0.0,
                efficiency_score=0.0,
                total_score=0.0,
                estimated_latency=1000.0,
                estimated_throughput=0.0,
                memory_usage=0
            )
        
        # 评估多GPU性能
        avg_compute = sum(d.compute_capability for d in gpu_devices) / gpu_count
        scaling_factor = min(math.sqrt(gpu_count), 4.0)  # 非线性扩展
        
        compatibility_score = 1.0
        performance_score = min(avg_compute / 8.0 * scaling_factor, 1.0)
        
        # 多GPU通信开销
        efficiency_score = max(0.9 - (gpu_count - 3) * 0.05, 0.6)
        
        total_score = (compatibility_score * 0.3 + 
                      performance_score * 0.5 + 
                      efficiency_score * 0.2)
        
        estimated_latency = 35.0 / scaling_factor
        estimated_throughput = performance_score * 200.0  # 高吞吐量
        memory_usage = workload.get("model_size", 0) // gpu_count
        
        return StrategyEvaluation(
            strategy_id=self.config.strategy_id,
            compatibility_score=compatibility_score,
            performance_score=performance_score,
            efficiency_score=efficiency_score,
            total_score=total_score,
            estimated_latency=estimated_latency,
            estimated_throughput=estimated_throughput,
            memory_usage=memory_usage
        )
    
    def apply_strategy(self, context: Dict[str, Any]) -> Dict[str, Any]:
        gpu_count = context.get("gpu_count", 4)
        return {
            "forward_strategy": "distributed",
            "memory_strategy": "sharded",
            "precision": "fp16",
            "batch_size": gpu_count,
            "enable_collective": True,
            "sharding_strategy": "layer_wise"
        }


class AdaptiveStrategyEngine:
    """自适应策略引擎
    
    智能选择最优推理策略的核心引擎：
    - 自动策略匹配
    - 性能预测和评估
    - 动态策略调整
    - 可扩展策略框架
    """
    
    def __init__(self, device_manager: UniversalDeviceManager):
        """
        初始化自适应策略引擎
        
        Args:
            device_manager: 通用设备管理器
        """
        self.device_manager = device_manager
        self.strategies: Dict[str, OptimizationStrategy] = {}
        self.active_strategies: Dict[StrategyType, str] = {}
        self.performance_history: Dict[str, List[float]] = {}
        
        # 注册内置策略
        self._register_builtin_strategies()
        
        logger.info("自适应策略引擎初始化完成")
    
    def _register_builtin_strategies(self):
        """注册内置策略"""
        builtin_strategies = [
            SingleGPUStrategy(),
            DualGPUPipelineStrategy(),
            MultiGPUDistributedStrategy()
        ]
        
        for strategy in builtin_strategies:
            self.register_strategy(strategy)
        
        logger.info(f"注册了 {len(builtin_strategies)} 个内置策略")
    
    def register_strategy(self, strategy: OptimizationStrategy):
        """注册优化策略"""
        self.strategies[strategy.config.strategy_id] = strategy
        logger.debug(f"注册策略: {strategy.config.strategy_id}")
    
    def select_optimal_strategy(self, workload: Dict[str, Any], 
                               performance_profile: PerformanceProfile = PerformanceProfile.BALANCED) -> Optional[str]:
        """选择最优策略"""
        topology = self.device_manager.get_device_topology()
        if not topology:
            logger.error("设备拓扑未初始化")
            return None
        
        strategy_evaluations = []
        
        # 评估所有兼容策略
        for strategy_id, strategy in self.strategies.items():
            if not strategy.is_compatible(topology):
                continue
            
            evaluation = strategy.evaluate_performance(topology, workload)
            
            # 根据性能配置文件调整评分
            adjusted_score = self._adjust_score_for_profile(evaluation, performance_profile)
            evaluation.total_score = adjusted_score
            
            strategy_evaluations.append(evaluation)
        
        if not strategy_evaluations:
            logger.warning("没有找到兼容的策略")
            return None
        
        # 选择评分最高的策略
        best_strategy = max(strategy_evaluations, key=lambda x: x.total_score)
        
        logger.info(f"选择最优策略: {best_strategy.strategy_id} (评分: {best_strategy.total_score:.3f})")
        
        return best_strategy.strategy_id
    
    def _adjust_score_for_profile(self, evaluation: StrategyEvaluation, 
                                 profile: PerformanceProfile) -> float:
        """根据性能配置文件调整评分"""
        base_score = evaluation.total_score
        
        if profile == PerformanceProfile.LATENCY_OPTIMIZED:
            # 优先考虑低延迟
            latency_factor = max(0.5, 1.0 - evaluation.estimated_latency / 100.0)
            return base_score * 0.7 + latency_factor * 0.3
        
        elif profile == PerformanceProfile.THROUGHPUT_OPTIMIZED:
            # 优先考虑高吞吐量
            throughput_factor = min(1.0, evaluation.estimated_throughput / 200.0)
            return base_score * 0.7 + throughput_factor * 0.3
        
        elif profile == PerformanceProfile.MEMORY_OPTIMIZED:
            # 优先考虑内存效率
            memory_factor = evaluation.efficiency_score
            return base_score * 0.6 + memory_factor * 0.4
        
        elif profile == PerformanceProfile.POWER_OPTIMIZED:
            # 优先考虑功耗效率
            power_factor = evaluation.efficiency_score
            return base_score * 0.8 + power_factor * 0.2
        
        else:  # BALANCED
            return base_score
    
    def apply_strategy(self, strategy_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """应用指定策略"""
        if strategy_id not in self.strategies:
            logger.error(f"策略不存在: {strategy_id}")
            return {}
        
        strategy = self.strategies[strategy_id]
        config = strategy.apply_strategy(context)
        
        logger.info(f"应用策略: {strategy_id}")
        logger.debug(f"策略配置: {config}")
        
        return config
    
    def get_strategy_recommendations(self, workload: Dict[str, Any]) -> Dict[str, Any]:
        """获取策略推荐"""
        topology = self.device_manager.get_device_topology()
        if not topology:
            return {}
        
        recommendations = {
            "optimal_strategy": {},
            "alternative_strategies": [],
            "performance_comparison": {},
            "hardware_analysis": {}
        }
        
        # 评估所有策略
        evaluations = []
        for strategy_id, strategy in self.strategies.items():
            if strategy.is_compatible(topology):
                evaluation = strategy.evaluate_performance(topology, workload)
                evaluations.append(evaluation)
        
        # 按评分排序
        evaluations.sort(key=lambda x: x.total_score, reverse=True)
        
        if evaluations:
            # 最优策略
            best = evaluations[0]
            recommendations["optimal_strategy"] = {
                "strategy_id": best.strategy_id,
                "score": best.total_score,
                "estimated_latency": best.estimated_latency,
                "estimated_throughput": best.estimated_throughput
            }
            
            # 备选策略
            for eval in evaluations[1:3]:  # 取前3个备选
                recommendations["alternative_strategies"].append({
                    "strategy_id": eval.strategy_id,
                    "score": eval.total_score,
                    "estimated_latency": eval.estimated_latency,
                    "estimated_throughput": eval.estimated_throughput
                })
            
            # 性能对比
            recommendations["performance_comparison"] = {
                eval.strategy_id: {
                    "latency": eval.estimated_latency,
                    "throughput": eval.estimated_throughput,
                    "efficiency": eval.efficiency_score
                }
                for eval in evaluations
            }
        
        # 硬件分析
        gpu_count = sum(1 for d in topology.devices.values() 
                       if d.device_type == DeviceType.GPU)
        
        recommendations["hardware_analysis"] = {
            "gpu_count": gpu_count,
            "total_memory": sum(d.memory_total for d in topology.devices.values() 
                              if d.device_type == DeviceType.GPU),
            "avg_compute_capability": sum(d.compute_capability for d in topology.devices.values() 
                                        if d.device_type == DeviceType.GPU) / max(gpu_count, 1),
            "high_speed_interconnects": len([i for i in topology.interconnects 
                                           if i.bandwidth > 50.0])
        }
        
        return recommendations
    
    def update_strategy_performance(self, strategy_id: str, 
                                  actual_latency: float, 
                                  actual_throughput: float):
        """更新策略实际性能"""
        if strategy_id not in self.performance_history:
            self.performance_history[strategy_id] = []
        
        # 记录性能指标
        performance_score = 100.0 / max(actual_latency, 1.0) + actual_throughput / 100.0
        self.performance_history[strategy_id].append(performance_score)
        
        # 保持历史长度
        if len(self.performance_history[strategy_id]) > 50:
            self.performance_history[strategy_id] = self.performance_history[strategy_id][-25:]
        
        logger.debug(f"更新策略 {strategy_id} 性能: 延迟={actual_latency:.2f}ms, 吞吐量={actual_throughput:.1f}")
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """获取策略统计"""
        return {
            "registered_strategies": list(self.strategies.keys()),
            "active_strategies": self.active_strategies,
            "performance_history": {
                strategy_id: {
                    "avg_performance": sum(history) / len(history) if history else 0.0,
                    "sample_count": len(history)
                }
                for strategy_id, history in self.performance_history.items()
            }
        } 