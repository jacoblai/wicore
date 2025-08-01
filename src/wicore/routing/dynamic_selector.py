"""
动态专家选择器
根据任务复杂度和推理动态自适应调整激活专家数量

核心功能:
- 任务复杂度实时评估
- 自适应Top-K专家选择
- 推理效率优化
- 负载均衡智能调度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DynamicSelectorConfig:
    """动态选择器配置"""
    max_experts: int = 8                    # 最大专家数
    min_experts: int = 1                    # 最小专家数
    complexity_threshold: float = 0.5       # 复杂度阈值
    efficiency_weight: float = 0.3          # 效率权重
    quality_weight: float = 0.7             # 质量权重
    adaptation_rate: float = 0.1            # 适应速率


class DynamicExpertSelector(nn.Module):
    """动态专家选择器"""
    
    def __init__(
        self,
        input_dim: int,
        config: Optional[DynamicSelectorConfig] = None
    ):
        super().__init__()
        self.config = config or DynamicSelectorConfig()
        self.input_dim = input_dim
        
        # 复杂度评估网络
        self.complexity_estimator = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(), 
            nn.Linear(input_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # 专家数量预测网络
        self.expert_count_predictor = nn.Sequential(
            nn.Linear(input_dim + 1, input_dim // 2),  # +1 for complexity
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )
        
        logger.info(f"初始化动态专家选择器: {self.config.max_experts}最大专家")
    
    def forward(
        self,
        x: torch.Tensor,
        routing_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        动态选择专家数量
        
        Args:
            x: 输入特征 [batch, seq, input_dim] 
            routing_weights: 路由权重 [batch, seq, num_experts]
            
        Returns:
            selected_weights: 选中的专家权重
            selected_indices: 选中的专家索引
            selection_info: 选择信息
        """
        batch_size, seq_len, _ = x.shape
        
        # 评估复杂度
        complexity_scores = self.complexity_estimator(x)  # [batch, seq, 1]
        
        # 预测所需专家数量
        combined_input = torch.cat([x, complexity_scores], dim=-1)
        expert_ratio = self.expert_count_predictor(combined_input)  # [batch, seq, 1]
        
        # 转换为专家数量
        expert_counts = self._ratio_to_expert_count(expert_ratio)
        
        # 动态选择专家
        selected_weights, selected_indices = self._select_dynamic_experts(
            routing_weights, expert_counts
        )
        
        # 计算选择统计
        selection_info = {
            "avg_complexity": complexity_scores.mean().item(),
            "avg_expert_count": expert_counts.float().mean().item(),
            "complexity_std": complexity_scores.std().item(),
            "expert_count_distribution": self._get_count_distribution(expert_counts)
        }
        
        return selected_weights, selected_indices, selection_info
    
    def _ratio_to_expert_count(self, ratio: torch.Tensor) -> torch.Tensor:
        """将比例转换为专家数量"""
        expert_range = self.config.max_experts - self.config.min_experts
        expert_count = self.config.min_experts + ratio * expert_range
        return torch.round(expert_count).long()
    
    def _select_dynamic_experts(
        self,
        routing_weights: torch.Tensor,
        expert_counts: torch.Tensor  
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """根据动态数量选择专家"""
        batch_size, seq_len, num_experts = routing_weights.shape
        max_k = self.config.max_experts
        
        # 获取Top-K权重和索引
        top_weights, top_indices = torch.topk(routing_weights, max_k, dim=-1)
        
        # 创建选择掩码
        k_range = torch.arange(max_k, device=routing_weights.device)
        k_mask = k_range[None, None, :] < expert_counts.unsqueeze(-1)
        
        # 应用掩码
        masked_weights = top_weights * k_mask.float()
        
        # 重新归一化
        weight_sums = masked_weights.sum(dim=-1, keepdim=True)
        weight_sums = torch.where(weight_sums > 0, weight_sums, torch.ones_like(weight_sums))
        normalized_weights = masked_weights / weight_sums
        
        return normalized_weights, top_indices
    
    def _get_count_distribution(self, expert_counts: torch.Tensor) -> Dict[int, float]:
        """获取专家数量分布"""
        distribution = {}
        total_count = expert_counts.numel()
        
        for k in range(self.config.min_experts, self.config.max_experts + 1):
            count = (expert_counts == k).sum().item()
            distribution[k] = count / total_count
            
        return distribution 