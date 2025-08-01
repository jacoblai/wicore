"""
RMoE (Recursive Mixture of Experts) 层级递归路由器
基于ArXiv 2408.06793研究的递归路由机制

核心创新:
- 层级递归路由：GRU建立连续层间路由决策依赖关系
- 历史感知路由：利用前层路由信息指导当前层决策
- 渐进式专家激活：根据序列复杂度递进激活专家
- 路由一致性优化：减少层间路由震荡提升稳定性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class RMoEConfig:
    """RMoE递归路由配置"""
    num_experts: int = 8                    # 专家数量
    num_layers: int = 12                    # 网络层数
    hidden_dim: int = 768                   # 隐藏维度
    gru_hidden_dim: int = 256               # GRU隐藏维度
    max_top_k: int = 4                      # 最大激活专家数
    min_top_k: int = 1                      # 最小激活专家数
    consistency_weight: float = 0.1         # 一致性损失权重
    history_length: int = 5                 # 历史路由长度
    temperature: float = 1.0                # 路由温度
    load_balance_weight: float = 0.01       # 负载均衡权重


class LayerRoutingState(NamedTuple):
    """层路由状态"""
    routing_weights: torch.Tensor           # [batch, seq, num_experts]
    expert_indices: torch.Tensor            # [batch, seq, top_k]
    expert_weights: torch.Tensor            # [batch, seq, top_k]
    gru_hidden: torch.Tensor                # [batch, seq, gru_hidden_dim]
    complexity_score: torch.Tensor          # [batch, seq, 1]


class RecursiveRoutingGRU(nn.Module):
    """递归路由GRU单元"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        
        # GRU单元
        self.gru = nn.GRU(
            input_size=input_dim + num_experts,  # 输入特征 + 上层路由权重
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # 复杂度评估网络
        self.complexity_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 路由决策网络
        self.routing_head = nn.Linear(hidden_dim, num_experts)
        
        # 专家数量决策网络
        self.top_k_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        prev_routing: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入特征 [batch, seq, input_dim]
            prev_routing: 上层路由权重 [batch, seq, num_experts]
            hidden: GRU隐藏状态 [batch, seq, hidden_dim]
        
        Returns:
            routing_logits: 路由logits [batch, seq, num_experts]
            complexity_score: 复杂度评分 [batch, seq, 1]
            top_k_score: 专家数量评分 [batch, seq, 1]
            new_hidden: 新的隐藏状态 [batch, seq, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # 准备GRU输入
        if prev_routing is not None:
            gru_input = torch.cat([x, prev_routing], dim=-1)
        else:
            # 第一层，使用零填充
            zero_routing = torch.zeros(batch_size, seq_len, self.num_experts, device=x.device)
            gru_input = torch.cat([x, zero_routing], dim=-1)
        
        # GRU前向传播
        gru_output, new_hidden = self.gru(gru_input, hidden)
        
        # 复杂度评估
        complexity_score = self.complexity_estimator(gru_output)
        
        # 路由决策
        routing_logits = self.routing_head(gru_output)
        
        # 专家数量决策
        top_k_score = self.top_k_head(gru_output)
        
        return routing_logits, complexity_score, top_k_score, new_hidden


class RMoERouter(nn.Module):
    """RMoE层级递归路由器"""
    
    def __init__(
        self,
        input_dim: int,
        config: Optional[RMoEConfig] = None
    ):
        super().__init__()
        self.config = config or RMoEConfig()
        self.input_dim = input_dim
        
        # 为每一层创建递归路由单元
        self.layer_routers = nn.ModuleList([
            RecursiveRoutingGRU(
                input_dim=input_dim,
                hidden_dim=self.config.gru_hidden_dim,
                num_experts=self.config.num_experts
            )
            for _ in range(self.config.num_layers)
        ])
        
        # 路由历史缓存
        self.routing_history = deque(maxlen=self.config.history_length)
        
        # 层间一致性评估
        self.consistency_tracker = nn.ModuleList([
            nn.Linear(self.config.num_experts * 2, 1)
            for _ in range(self.config.num_layers - 1)
        ])
        
        logger.info(f"初始化RMoE路由器: {self.config.num_layers}层 x {self.config.num_experts}专家")
    
    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int,
        prev_routing_state: Optional[LayerRoutingState] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, LayerRoutingState, Dict[str, Any]]:
        """单层递归路由前向计算"""
        batch_size, seq_len, hidden_dim = x.shape
        
        if layer_idx >= self.config.num_layers:
            raise ValueError(f"层索引超出范围: {layer_idx} >= {self.config.num_layers}")
        
        router = self.layer_routers[layer_idx]
        
        # 获取上层路由信息
        prev_routing = None
        prev_hidden = None
        if prev_routing_state is not None:
            prev_routing = prev_routing_state.routing_weights
            prev_hidden = prev_routing_state.gru_hidden
        
        # 递归路由计算
        routing_logits, complexity_score, top_k_score, new_hidden = router(
            x, prev_routing, prev_hidden
        )
        
        # 温度缩放
        routing_logits = routing_logits / self.config.temperature
        
        # 动态Top-K选择
        dynamic_top_k = self._compute_dynamic_top_k(
            complexity_score, top_k_score
        )
        
        # 专家选择和权重计算
        routing_weights = F.softmax(routing_logits, dim=-1)
        expert_weights, expert_indices = self._select_experts(
            routing_weights, dynamic_top_k
        )
        
        # 创建当前层路由状态
        current_state = LayerRoutingState(
            routing_weights=routing_weights,
            expert_indices=expert_indices,
            expert_weights=expert_weights,
            gru_hidden=new_hidden,
            complexity_score=complexity_score
        )
        
        # 计算损失指标
        losses = self._compute_routing_losses(
            current_state, prev_routing_state, layer_idx
        )
        
        # 路由统计信息
        routing_info = {
            "layer_idx": layer_idx,
            "avg_top_k": dynamic_top_k.float().mean().item(),
            "avg_complexity": complexity_score.mean().item(),
            "expert_utilization": self._compute_expert_utilization(expert_indices),
            "routing_entropy": self._compute_routing_entropy(routing_weights),
            **losses
        }
        
        return expert_weights, expert_indices, current_state, routing_info
    
    def _compute_dynamic_top_k(
        self,
        complexity_score: torch.Tensor,
        top_k_score: torch.Tensor
    ) -> torch.Tensor:
        """计算动态Top-K专家数量"""
        # 基于复杂度和专家数量评分计算Top-K
        combined_score = 0.7 * complexity_score + 0.3 * top_k_score
        
        # 映射到[min_top_k, max_top_k]范围
        top_k_range = self.config.max_top_k - self.config.min_top_k
        dynamic_top_k = self.config.min_top_k + (combined_score * top_k_range)
        
        # 四舍五入到整数
        dynamic_top_k = torch.round(dynamic_top_k).long()
        
        return dynamic_top_k
    
    def _select_experts(
        self,
        routing_weights: torch.Tensor,
        dynamic_top_k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """选择Top-K专家"""
        batch_size, seq_len, num_experts = routing_weights.shape
        
        # 获取最大Top-K用于统一处理
        max_k = self.config.max_top_k
        top_weights, top_indices = torch.topk(routing_weights, max_k, dim=-1)
        
        # 根据dynamic_top_k创建掩码
        k_mask = torch.arange(max_k, device=routing_weights.device)[None, None, :] < dynamic_top_k.unsqueeze(-1)
        
        # 应用掩码并重新归一化
        masked_weights = top_weights * k_mask.float()
        weight_sums = masked_weights.sum(dim=-1, keepdim=True)
        weight_sums = torch.where(weight_sums > 0, weight_sums, torch.ones_like(weight_sums))
        expert_weights = masked_weights / weight_sums
        
        return expert_weights, top_indices
    
    def _compute_routing_losses(
        self,
        current_state: LayerRoutingState,
        prev_state: Optional[LayerRoutingState],
        layer_idx: int
    ) -> Dict[str, torch.Tensor]:
        """计算路由相关损失"""
        losses = {}
        
        # 负载均衡损失
        expert_usage = current_state.routing_weights.mean(dim=[0, 1])
        ideal_usage = 1.0 / self.config.num_experts
        load_balance_loss = F.mse_loss(
            expert_usage, 
            torch.full_like(expert_usage, ideal_usage)
        )
        losses["load_balance_loss"] = load_balance_loss * self.config.load_balance_weight
        
        # 层间一致性损失
        if prev_state is not None and layer_idx > 0:
            consistency_loss = self._compute_consistency_loss(
                current_state.routing_weights,
                prev_state.routing_weights,
                layer_idx - 1
            )
            losses["consistency_loss"] = consistency_loss * self.config.consistency_weight
        else:
            losses["consistency_loss"] = torch.tensor(0.0, device=current_state.routing_weights.device)
        
        # 复杂度正则化损失
        complexity_reg = torch.mean(current_state.complexity_score ** 2)
        losses["complexity_reg_loss"] = complexity_reg * 0.001
        
        return losses
    
    def _compute_consistency_loss(
        self,
        current_routing: torch.Tensor,
        prev_routing: torch.Tensor,
        consistency_idx: int
    ) -> torch.Tensor:
        """计算层间路由一致性损失"""
        batch_size, seq_len, num_experts = current_routing.shape
        
        # 拼接当前层和前一层的路由权重
        combined_routing = torch.cat([current_routing, prev_routing], dim=-1)
        combined_routing = combined_routing.view(-1, num_experts * 2)
        
        # 使用一致性评估网络
        consistency_scorer = self.consistency_tracker[consistency_idx]
        consistency_scores = consistency_scorer(combined_routing)
        
        # 一致性损失：希望相邻层路由相似但不完全相同
        target_consistency = 0.7  # 目标一致性水平
        consistency_loss = F.mse_loss(
            torch.sigmoid(consistency_scores),
            torch.full_like(consistency_scores, target_consistency)
        )
        
        return consistency_loss
    
    def _compute_expert_utilization(self, expert_indices: torch.Tensor) -> torch.Tensor:
        """计算专家利用率"""
        batch_size, seq_len, max_k = expert_indices.shape
        utilization = torch.zeros(self.config.num_experts, device=expert_indices.device)
        
        for i in range(self.config.num_experts):
            expert_count = (expert_indices == i).sum().float()
            total_selections = batch_size * seq_len * max_k
            utilization[i] = expert_count / total_selections
        
        return utilization
    
    def _compute_routing_entropy(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """计算路由熵（衡量路由多样性）"""
        # 避免log(0)
        routing_weights = torch.clamp(routing_weights, min=1e-10)
        entropy = -torch.sum(routing_weights * torch.log(routing_weights), dim=-1)
        return entropy.mean()
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """获取路由统计信息"""
        stats = {
            "num_layers": self.config.num_layers,
            "num_experts": self.config.num_experts,
            "history_length": len(self.routing_history),
            "config": self.config
        }
        
        # 历史路由统计
        if self.routing_history:
            recent_routing = list(self.routing_history)
            avg_entropy = np.mean([
                state.get("routing_entropy", 0.0) for state in recent_routing
            ])
            avg_top_k = np.mean([
                state.get("avg_top_k", 0.0) for state in recent_routing
            ])
            stats.update({
                "avg_routing_entropy": avg_entropy,
                "avg_top_k": avg_top_k
            })
        
        return stats
    
    def reset_routing_history(self):
        """重置路由历史"""
        self.routing_history.clear()
        logger.info("路由历史已重置")
    
    def save_routing_checkpoint(self, path: str):
        """保存路由检查点"""
        checkpoint = {
            "config": self.config,
            "layer_routers_state": [router.state_dict() for router in self.layer_routers],
            "consistency_tracker_state": [tracker.state_dict() for tracker in self.consistency_tracker],
            "routing_history": list(self.routing_history)
        }
        torch.save(checkpoint, path)
        logger.info(f"RMoE路由检查点已保存: {path}")
    
    def load_routing_checkpoint(self, path: str):
        """加载路由检查点"""
        checkpoint = torch.load(path)
        
        # 恢复路由器状态
        for router, state in zip(self.layer_routers, checkpoint["layer_routers_state"]):
            router.load_state_dict(state)
        
        # 恢复一致性跟踪器状态
        for tracker, state in zip(self.consistency_tracker, checkpoint["consistency_tracker_state"]):
            tracker.load_state_dict(state)
        
        # 恢复路由历史
        self.routing_history = deque(checkpoint["routing_history"], maxlen=self.config.history_length)
        
        logger.info(f"RMoE路由检查点已加载: {path}") 