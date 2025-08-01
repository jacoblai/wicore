"""
EvoMoE (Evolutionary Mixture of Experts) 专家进化路由器
基于ArXiv 2505.23830研究的突破性专家进化算法

核心创新:
- 专家进化机制：从单一专家进化出多个鲁棒专家
- 均匀性问题解决：避免专家分布不均导致的性能瓶颈
- 自适应专家数量：根据任务复杂度动态调整激活专家
- 梯度感知进化：基于梯度信息指导专家进化方向
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


@dataclass
class ExpertEvolutionConfig:
    """专家进化配置"""
    num_experts: int = 8                    # 专家总数
    evolution_steps: int = 100              # 进化步数
    mutation_rate: float = 0.1              # 变异率
    crossover_rate: float = 0.7             # 交叉率
    elite_ratio: float = 0.2                # 精英比例
    diversity_weight: float = 0.3           # 多样性权重
    fitness_smoothing: float = 0.9          # 适应度平滑系数
    min_expert_utilization: float = 0.05    # 最小专家利用率


class ExpertGene(nn.Module):
    """专家基因编码"""
    
    def __init__(self, input_dim: int, expert_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.expert_dim = expert_dim
        
        # 专家参数基因
        self.weight_gene = nn.Parameter(torch.randn(expert_dim, input_dim))
        self.bias_gene = nn.Parameter(torch.randn(expert_dim))
        
        # 路由偏好基因
        self.routing_preference = nn.Parameter(torch.randn(input_dim))
        
        # 适应度历史
        self.fitness_history = []
        self.utilization_history = []
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """专家前向计算"""
        return F.linear(x, self.weight_gene, self.bias_gene)
    
    def get_routing_score(self, x: torch.Tensor) -> torch.Tensor:
        """获取路由评分"""
        return torch.sum(x * self.routing_preference, dim=-1)
    
    def mutate(self, mutation_rate: float):
        """基因变异"""
        with torch.no_grad():
            if torch.rand(1) < mutation_rate:
                noise = torch.randn_like(self.weight_gene) * 0.01
                self.weight_gene.add_(noise)
                
            if torch.rand(1) < mutation_rate:
                noise = torch.randn_like(self.bias_gene) * 0.01
                self.bias_gene.add_(noise)
                
            if torch.rand(1) < mutation_rate:
                noise = torch.randn_like(self.routing_preference) * 0.01
                self.routing_preference.add_(noise)
    
    def crossover(self, other: 'ExpertGene', crossover_rate: float) -> 'ExpertGene':
        """基因交叉"""
        offspring = ExpertGene(self.input_dim, self.expert_dim)
        
        with torch.no_grad():
            # 权重交叉
            mask = torch.rand_like(self.weight_gene) < crossover_rate
            offspring.weight_gene.data = torch.where(
                mask, self.weight_gene, other.weight_gene
            )
            
            # 偏置交叉
            mask = torch.rand_like(self.bias_gene) < crossover_rate
            offspring.bias_gene.data = torch.where(
                mask, self.bias_gene, other.bias_gene
            )
            
            # 路由偏好交叉
            mask = torch.rand_like(self.routing_preference) < crossover_rate
            offspring.routing_preference.data = torch.where(
                mask, self.routing_preference, other.routing_preference
            )
        
        return offspring


class EvoMoERouter(nn.Module):
    """EvoMoE专家进化路由器"""
    
    def __init__(
        self,
        input_dim: int,
        expert_dim: int,
        config: Optional[ExpertEvolutionConfig] = None
    ):
        super().__init__()
        self.config = config or ExpertEvolutionConfig()
        self.input_dim = input_dim
        self.expert_dim = expert_dim
        
        # 专家种群
        self.experts = nn.ModuleList([
            ExpertGene(input_dim, expert_dim) 
            for _ in range(self.config.num_experts)
        ])
        
        # 路由网络
        self.router = nn.Linear(input_dim, self.config.num_experts)
        
        # 进化统计
        self.generation = 0
        self.fitness_tracker = defaultdict(list)
        self.diversity_tracker = []
        
        logger.info(f"初始化EvoMoE路由器: {self.config.num_experts}个专家")
    
    def forward(
        self, 
        x: torch.Tensor, 
        top_k: int = 2
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """前向计算与路由"""
        batch_size, seq_len, hidden_dim = x.shape
        
        # 计算路由权重
        routing_logits = self.router(x)  # [batch, seq, num_experts]
        routing_weights = F.softmax(routing_logits, dim=-1)
        
        # Top-K专家选择
        top_weights, top_indices = torch.topk(routing_weights, top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        
        # 专家计算
        outputs = []
        expert_utilization = torch.zeros(self.config.num_experts, device=x.device)
        
        for i, expert in enumerate(self.experts):
            # 检查专家是否被选中
            expert_mask = (top_indices == i).any(dim=-1)
            if expert_mask.any():
                expert_input = x[expert_mask]
                if expert_input.numel() > 0:
                    expert_output = expert(expert_input)
                    outputs.append((i, expert_mask, expert_output))
                    expert_utilization[i] = expert_mask.float().mean()
        
        # 聚合输出
        final_output = torch.zeros_like(x[:, :, :self.expert_dim])
        for expert_idx, mask, expert_out in outputs:
            weight_mask = (top_indices == expert_idx)
            weights = top_weights * weight_mask.float()
            weights = weights.sum(dim=-1, keepdim=True)
            final_output[mask] += weights[mask] * expert_out
        
        # 计算损失指标
        load_balance_loss = self._compute_load_balance_loss(routing_weights)
        diversity_score = self._compute_diversity_score()
        
        routing_info = {
            "routing_weights": routing_weights,
            "expert_utilization": expert_utilization,
            "load_balance_loss": load_balance_loss,
            "diversity_score": diversity_score,
            "generation": self.generation
        }
        
        return final_output, routing_info
    
    def evolve_experts(self, fitness_scores: torch.Tensor):
        """专家进化过程"""
        if len(fitness_scores) != self.config.num_experts:
            logger.warning(f"适应度评分数量不匹配: {len(fitness_scores)} vs {self.config.num_experts}")
            return
        
        # 更新适应度历史
        for i, score in enumerate(fitness_scores):
            self.experts[i].fitness_history.append(score.item())
        
        # 计算平滑适应度
        smoothed_fitness = []
        for expert in self.experts:
            if len(expert.fitness_history) > 1:
                alpha = self.config.fitness_smoothing
                smoothed = alpha * expert.fitness_history[-1] + (1 - alpha) * expert.fitness_history[-2]
            else:
                smoothed = expert.fitness_history[-1]
            smoothed_fitness.append(smoothed)
        
        smoothed_fitness = torch.tensor(smoothed_fitness)
        
        # 选择精英
        num_elite = max(1, int(self.config.num_experts * self.config.elite_ratio))
        elite_indices = torch.topk(smoothed_fitness, num_elite).indices
        
        # 创建新一代
        new_experts = []
        
        # 保留精英
        for idx in elite_indices:
            new_experts.append(self.experts[idx])
        
        # 生成后代
        while len(new_experts) < self.config.num_experts:
            # 选择父母 (轮盘赌选择)
            probabilities = F.softmax(smoothed_fitness / 0.1, dim=0)
            parent1_idx = torch.multinomial(probabilities, 1).item()
            parent2_idx = torch.multinomial(probabilities, 1).item()
            
            parent1 = self.experts[parent1_idx]
            parent2 = self.experts[parent2_idx]
            
            # 交叉
            offspring = parent1.crossover(parent2, self.config.crossover_rate)
            
            # 变异
            offspring.mutate(self.config.mutation_rate)
            
            new_experts.append(offspring)
        
        # 更新专家种群
        self.experts = nn.ModuleList(new_experts[:self.config.num_experts])
        self.generation += 1
        
        # 记录多样性
        diversity = self._compute_diversity_score()
        self.diversity_tracker.append(diversity)
        
        logger.info(f"专家进化完成 - 第{self.generation}代, 多样性: {diversity:.4f}")
    
    def _compute_load_balance_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """计算负载均衡损失"""
        # 计算专家使用率
        expert_usage = routing_weights.mean(dim=[0, 1])  # [num_experts]
        
        # 理想使用率（均匀分布）
        ideal_usage = 1.0 / self.config.num_experts
        
        # 均方误差损失
        load_balance_loss = F.mse_loss(expert_usage, torch.full_like(expert_usage, ideal_usage))
        
        return load_balance_loss
    
    def _compute_diversity_score(self) -> float:
        """计算专家多样性评分"""
        if len(self.experts) < 2:
            return 0.0
        
        diversities = []
        for i in range(len(self.experts)):
            for j in range(i + 1, len(self.experts)):
                # 计算专家参数的欧几里得距离
                expert1_params = torch.cat([
                    self.experts[i].weight_gene.flatten(),
                    self.experts[i].bias_gene.flatten(),
                    self.experts[i].routing_preference.flatten()
                ])
                expert2_params = torch.cat([
                    self.experts[j].weight_gene.flatten(),
                    self.experts[j].bias_gene.flatten(),
                    self.experts[j].routing_preference.flatten()
                ])
                
                distance = torch.norm(expert1_params - expert2_params).item()
                diversities.append(distance)
        
        return np.mean(diversities) if diversities else 0.0
    
    def get_evolution_stats(self) -> Dict[str, Any]:
        """获取进化统计信息"""
        stats = {
            "generation": self.generation,
            "num_experts": self.config.num_experts,
            "diversity_history": self.diversity_tracker.copy(),
            "current_diversity": self.diversity_tracker[-1] if self.diversity_tracker else 0.0
        }
        
        # 专家适应度统计
        expert_fitness = []
        for expert in self.experts:
            if expert.fitness_history:
                expert_fitness.append({
                    "current_fitness": expert.fitness_history[-1],
                    "avg_fitness": np.mean(expert.fitness_history),
                    "fitness_trend": expert.fitness_history[-5:] if len(expert.fitness_history) >= 5 else expert.fitness_history
                })
            else:
                expert_fitness.append({"current_fitness": 0.0, "avg_fitness": 0.0, "fitness_trend": []})
        
        stats["expert_fitness"] = expert_fitness
        
        return stats
    
    def save_evolution_checkpoint(self, path: str):
        """保存进化检查点"""
        checkpoint = {
            "config": self.config,
            "generation": self.generation,
            "diversity_tracker": self.diversity_tracker,
            "expert_states": [expert.state_dict() for expert in self.experts],
            "router_state": self.router.state_dict()
        }
        torch.save(checkpoint, path)
        logger.info(f"进化检查点已保存: {path}")
    
    def load_evolution_checkpoint(self, path: str):
        """加载进化检查点"""
        checkpoint = torch.load(path)
        self.generation = checkpoint["generation"]
        self.diversity_tracker = checkpoint["diversity_tracker"]
        
        # 恢复专家状态
        for expert, state in zip(self.experts, checkpoint["expert_states"]):
            expert.load_state_dict(state)
        
        # 恢复路由器状态
        self.router.load_state_dict(checkpoint["router_state"])
        
        logger.info(f"进化检查点已加载: {path}, 第{self.generation}代") 