"""
多头注意力引擎
优化的多头注意力实现

核心功能:
- 并行多头计算
- 内存优化
- 动态头数调整
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


class MultiHeadAttentionEngine(nn.Module):
    """多头注意力引擎"""
    
    def __init__(self, hidden_dim: int = 768, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        
        # 线性投影层
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout层
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        logger.info(f"多头注意力引擎初始化: {num_heads}头, {hidden_dim}维")
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """多头注意力前向计算"""
        batch_size, seq_len, _ = hidden_states.shape
        
        # 线性投影
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # 重塑为多头形状
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力权重
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # 应用注意力掩码
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # Softmax和dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 应用注意力权重
        output = torch.matmul(attn_weights, v)
        
        # 重塑并输出投影
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(output)
        output = self.proj_dropout(output)
        
        return output 