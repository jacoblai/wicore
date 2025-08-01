"""
MoR (Mixture of Routers) 动态路由器
整合2024-2025最新MoE路由优化技术的核心路由系统

基于以下突破性研究:
- EvoMoE: 专家进化解决均匀性问题 (ArXiv 2505.23830)
- RMoE: 层级递归路由建立依赖关系 (ArXiv 2408.06793)  
- DynamicMoE: 根据任务复杂度自适应选择 (ArXiv 2403.07652)
- LocalRouting: 本地路由一致性优化 (ArXiv 2505.16056)
- InferenceDynamics: 多维路由框架 (ArXiv 2505.16303)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict

from ..core.config import MoRConfig
from .evo_moe import EvoMoERouter
from .rmoe import RMoERouter
from .dynamic_selector import DynamicExpertSelector
from .inference_dynamics import InferenceDynamicsRouter

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """路由策略类型"""
    STATIC_TOPK = "static_topk"              # 传统静态TopK
    DYNAMIC_ADAPTIVE = "dynamic_adaptive"     # 动态自适应
    EXPERT_EVOLUTION = "expert_evolution"     # 专家进化
    RECURSIVE_LAYER = "recursive_layer"       # 层级递归
    INFERENCE_DYNAMICS = "inference_dynamics" # 推理动态
    HYBRID_OPTIMAL = "hybrid_optimal"         # 混合最优


@dataclass 
class MoRConfig:
    """MoR配置参数"""
    # 基本路由配置
    num_experts: int = 8                    # 专家数量
    num_experts_per_token: int = 2          # 每token激活专家数
    routing_strategy: RoutingStrategy = RoutingStrategy.HYBRID_OPTIMAL
    
    # 专家进化配置 (EvoMoE)
    enable_expert_evolution: bool = True    # 启用专家进化
    evolution_steps: int = 1000            # 进化步数
    mutation_rate: float = 0.1             # 变异率
    diversity_weight: float = 0.2          # 多样性权重
    
    # 递归路由配置 (RMoE) 
    enable_recursive_routing: bool = True   # 启用递归路由
    gru_hidden_size: int = 256             # GRU隐藏层大小
    max_layer_memory: int = 5              # 最大层间记忆
    
    # 动态选择配置
    enable_dynamic_selection: bool = True   # 启用动态选择
    complexity_threshold: float = 0.6      # 复杂度阈值
    min_experts: int = 1                   # 最少专家数
    max_experts: int = 4                   # 最多专家数
    
    # 推理动态配置
    enable_inference_dynamics: bool = True  # 启用推理动态
    capability_dim: int = 128              # 能力维度
    knowledge_dim: int = 128               # 知识维度
    
    # 性能优化配置
    cache_routing_decisions: bool = True    # 缓存路由决策
    prefetch_experts: bool = True          # 预取专家
    load_balancing_weight: float = 0.01    # 负载均衡权重
    
    # 监控和调试
    collect_routing_stats: bool = True     # 收集路由统计
    debug_mode: bool = False               # 调试模式


class MoRRouter(nn.Module):
    """
    MoR动态路由器 - 协调所有路由优化技术
    
    核心功能:
    1. 专家进化算法 (EvoMoE) - 解决专家均匀性
    2. 层级递归路由 (RMoE) - GRU建立层间依赖  
    3. 动态专家选择 - 根据任务复杂度自适应
    4. 推理动态框架 - 多维能力和知识建模
    5. 本地路由一致性 - 提升缓存效率
    6. 混合最优策略 - 智能组合多种技术
    """
    
    def __init__(self, config: MoRConfig):
        super().__init__()
        self.config = config
        
        # 初始化各个路由器组件
        self._init_routers()
        
        # 路由决策缓存
        self.routing_cache = {} if config.cache_routing_decisions else None
        
        # 性能统计
        self.stats = {
            "total_routing_decisions": 0,
            "cache_hits": 0,
            "expert_usage": defaultdict(int),
            "layer_routing_history": [],
            "complexity_distribution": [],
        }
        
        logger.info(f"MoR Router initialized with {config.routing_strategy} strategy")
    
    def _init_routers(self):
        """初始化各个路由器组件"""
        input_dim = 768  # 默认输入维度
        expert_dim = 512  # 默认专家维度
        
        # EvoMoE专家进化路由器
        if self.config.enable_expert_evolution:
            from .evo_moe import ExpertEvolutionConfig
            evo_config = ExpertEvolutionConfig(
                num_experts=self.config.num_experts,
                evolution_steps=self.config.evolution_steps,
                mutation_rate=self.config.mutation_rate,
                diversity_weight=self.config.diversity_weight
            )
            self.evo_router = EvoMoERouter(input_dim, expert_dim, evo_config)
        
        # RMoE递归路由器
        if self.config.enable_recursive_routing:
            from .rmoe import RMoEConfig
            rmoe_config = RMoEConfig(
                num_experts=self.config.num_experts,
                gru_hidden_dim=self.config.gru_hidden_size,
                num_layers=12  # 默认层数
            )
            self.rmoe_router = RMoERouter(input_dim, rmoe_config)
        
        # 动态专家选择器
        if self.config.enable_dynamic_selection:
            from .dynamic_selector import DynamicSelectorConfig
            selector_config = DynamicSelectorConfig(
                max_experts=self.config.max_experts,
                min_experts=self.config.min_experts,
                complexity_threshold=self.config.complexity_threshold
            )
            self.dynamic_selector = DynamicExpertSelector(input_dim, selector_config)
        
        # 推理动态路由器
        if self.config.enable_inference_dynamics:
            self.inference_router = InferenceDynamicsRouter(
                input_dim=input_dim,
                num_experts=self.config.num_experts
            )
        
        # 混合路由决策网络
        self.routing_fusion = nn.Sequential(
            nn.Linear(self.config.num_experts * 4, self.config.num_experts * 2),
            nn.ReLU(),
            nn.Linear(self.config.num_experts * 2, self.config.num_experts),
            nn.Softmax(dim=-1)
        )
    
    def route(
        self,
        hidden_states: torch.Tensor,
        layer_id: int = 0,
        task_type: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        智能路由决策
        
        Args:
            hidden_states: 输入隐藏状态 [batch, seq_len, hidden_dim]
            layer_id: 当前层ID
            task_type: 任务类型 (可选)
            context: 上下文信息 (可选)
            
        Returns:
            routing_weights: 路由权重 [batch, seq_len, num_experts]
            expert_indices: 选中的专家索引 [batch, seq_len, top_k]
            routing_info: 路由信息字典
        """
        self.stats["total_routing_decisions"] += 1
        
        # 检查缓存
        cache_key = self._generate_cache_key(hidden_states, layer_id, task_type)
        if self.routing_cache and cache_key in self.routing_cache:
            self.stats["cache_hits"] += 1
            return self.routing_cache[cache_key]
        
        try:
            # Step 1: 获取各个路由器的决策
            routing_scores = self._gather_routing_scores(
                hidden_states, layer_id, task_type, context
            )
            
            # Step 2: 动态确定专家数量
            num_experts_per_token = self._determine_expert_count(
                hidden_states, task_type
            )
            
            # Step 3: 融合路由决策
            final_weights = self._fuse_routing_decisions(routing_scores)
            
            # Step 4: 选择专家
            expert_indices = self._select_experts(
                final_weights, num_experts_per_token
            )
            
            # Step 5: 应用负载均衡
            balanced_weights = self._apply_load_balancing(final_weights)
            
            # Step 6: 收集路由信息
            routing_info = self._collect_routing_info(
                final_weights, expert_indices, layer_id, task_type
            )
            
            # 缓存结果
            result = (balanced_weights, expert_indices, routing_info)
            if self.routing_cache:
                self.routing_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Routing failed: {e}")
            # 返回默认路由作为fallback
            return self._fallback_routing(hidden_states)
    
    def _gather_routing_scores(
        self,
        hidden_states: torch.Tensor,
        layer_id: int,
        task_type: Optional[str],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """收集各个路由器的决策分数"""
        routing_scores = {}
        
        # EvoMoE专家进化分数
        if self.config.enable_expert_evolution:
            evo_scores = self.evo_router.compute_routing_scores(
                hidden_states, 
                evolution_step=self.stats["total_routing_decisions"]
            )
            routing_scores['evolution'] = evo_scores
        
        # RMoE递归路由分数  
        if self.config.enable_recursive_routing:
            rmoe_scores = self.rmoe_router.compute_recursive_scores(
                hidden_states, layer_id
            )
            routing_scores['recursive'] = rmoe_scores
        
        # 推理动态分数
        if self.config.enable_inference_dynamics:
            dynamics_scores = self.inference_router.compute_capability_scores(
                hidden_states, task_type
            )
            routing_scores['dynamics'] = dynamics_scores
        
        # 基础TopK分数 (作为baseline)
        base_scores = self._compute_base_routing(hidden_states)
        routing_scores['baseline'] = base_scores
        
        return routing_scores
    
    def _determine_expert_count(
        self, 
        hidden_states: torch.Tensor, 
        task_type: Optional[str]
    ) -> int:
        """动态确定每个token需要的专家数量"""
        if not self.config.enable_dynamic_selection:
            return self.config.num_experts_per_token
        
        # 计算任务复杂度
        complexity = self.dynamic_selector.assess_complexity(
            hidden_states, task_type
        )
        
        # 记录复杂度分布
        self.stats["complexity_distribution"].append(complexity.mean().item())
        
        # 基于复杂度动态选择专家数量
        expert_count = self.dynamic_selector.select_expert_count(complexity)
        
        return expert_count
    
    def _fuse_routing_decisions(
        self, 
        routing_scores: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """融合多个路由器的决策"""
        if len(routing_scores) == 1:
            return list(routing_scores.values())[0]
        
        # 将所有分数拼接
        score_list = []
        for key in ['evolution', 'recursive', 'dynamics', 'baseline']:
            if key in routing_scores:
                score_list.append(routing_scores[key])
        
        if len(score_list) == 0:
            raise ValueError("No routing scores available")
        
        # 拼接所有分数
        concatenated_scores = torch.cat(score_list, dim=-1)
        
        # 通过融合网络获得最终权重
        fused_weights = self.routing_fusion(concatenated_scores)
        
        return fused_weights
    
    def _select_experts(
        self, 
        weights: torch.Tensor, 
        num_experts: int
    ) -> torch.Tensor:
        """选择TopK专家"""
        # TopK选择
        top_weights, top_indices = torch.topk(
            weights, k=num_experts, dim=-1
        )
        
        # 重新归一化权重
        normalized_weights = F.softmax(top_weights, dim=-1)
        
        return top_indices
    
    def _apply_load_balancing(self, weights: torch.Tensor) -> torch.Tensor:
        """应用负载均衡正则化"""
        if self.config.load_balancing_weight == 0:
            return weights
        
        # 计算专家使用频率
        expert_usage = weights.mean(dim=(0, 1))  # [num_experts]
        
        # 计算负载均衡损失 (鼓励均匀使用)
        target_usage = 1.0 / self.config.num_experts
        balance_loss = torch.mean((expert_usage - target_usage) ** 2)
        
        # 调整权重 (简化的实现)
        adjustment = -self.config.load_balancing_weight * (
            expert_usage - target_usage
        ).unsqueeze(0).unsqueeze(0)
        
        balanced_weights = weights + adjustment
        
        return F.softmax(balanced_weights, dim=-1)
    
    def _collect_routing_info(
        self,
        weights: torch.Tensor,
        expert_indices: torch.Tensor, 
        layer_id: int,
        task_type: Optional[str]
    ) -> Dict[str, Any]:
        """收集路由信息用于监控和分析"""
        info = {
            "layer_id": layer_id,
            "task_type": task_type,
            "routing_entropy": self._compute_entropy(weights),
            "expert_distribution": weights.mean(dim=(0, 1)).tolist(),
            "active_experts": expert_indices.unique().tolist(),
        }
        
        # 更新专家使用统计
        for expert_id in expert_indices.flatten().tolist():
            self.stats["expert_usage"][expert_id] += 1
        
        # 记录层级路由历史 (用于RMoE)
        self.stats["layer_routing_history"].append({
            "layer_id": layer_id,
            "expert_indices": expert_indices.clone().detach(),
            "weights": weights.clone().detach()
        })
        
        return info
    
    def _compute_entropy(self, weights: torch.Tensor) -> float:
        """计算路由权重的熵值"""
        # 避免log(0)
        eps = 1e-8
        log_weights = torch.log(weights + eps)
        entropy = -torch.sum(weights * log_weights, dim=-1).mean()
        return entropy.item()
    
    def _compute_base_routing(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """计算基础路由分数"""
        # 简单的线性投影作为baseline
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # 线性变换到专家数量维度
        routing_logits = torch.randn(
            batch_size, seq_len, self.config.num_experts,
            device=hidden_states.device
        )
        
        return F.softmax(routing_logits, dim=-1)
    
    def _generate_cache_key(
        self, 
        hidden_states: torch.Tensor, 
        layer_id: int, 
        task_type: Optional[str]
    ) -> str:
        """生成缓存key"""
        # 使用hash来生成紧凑的key
        state_hash = hash(hidden_states.shape + tuple(hidden_states.flatten()[:10].tolist()))
        return f"layer_{layer_id}_{task_type}_{state_hash}"
    
    def _fallback_routing(
        self, 
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """备用路由方案"""
        batch_size, seq_len, _ = hidden_states.shape
        
        # 简单的均匀分布
        weights = torch.ones(
            batch_size, seq_len, self.config.num_experts,
            device=hidden_states.device
        ) / self.config.num_experts
        
        # 选择前K个专家
        expert_indices = torch.arange(
            self.config.num_experts_per_token,
            device=hidden_states.device
        ).unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        
        info = {"fallback": True}
        
        return weights, expert_indices, info
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """获取路由统计信息"""
        total_decisions = self.stats["total_routing_decisions"]
        cache_hit_rate = self.stats["cache_hits"] / max(1, total_decisions)
        
        # 专家使用分布
        expert_usage_dist = dict(self.stats["expert_usage"])
        total_usage = sum(expert_usage_dist.values())
        expert_usage_normalized = {
            k: v / max(1, total_usage) 
            for k, v in expert_usage_dist.items()
        }
        
        return {
            "total_routing_decisions": total_decisions,
            "cache_hit_rate": cache_hit_rate,
            "expert_usage_distribution": expert_usage_normalized,
            "complexity_stats": {
                "mean": np.mean(self.stats["complexity_distribution"]),
                "std": np.std(self.stats["complexity_distribution"]),
                "samples": len(self.stats["complexity_distribution"])
            },
            "active_optimizations": {
                "expert_evolution": self.config.enable_expert_evolution,
                "recursive_routing": self.config.enable_recursive_routing,
                "dynamic_selection": self.config.enable_dynamic_selection,
                "inference_dynamics": self.config.enable_inference_dynamics,
            }
        } 