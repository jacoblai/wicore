"""
WiCore注意力引擎模块
基于2024-2025最新注意力优化技术

核心技术:
- FlashAttention-3：硬件感知的注意力优化
- FlashInfer：可定制模板的JIT编译注意力引擎  
- Multi-Head Attention优化：并行计算和内存优化
- Sparse Attention：稀疏注意力模式支持
- KV Cache集成：与内存管理系统无缝集成
"""

from .flash_engine import FlashAttentionEngine
from .flashinfer import FlashInferEngine  
from .multi_head import MultiHeadAttentionEngine
from .sparse_attention import SparseAttentionEngine

__all__ = [
    "FlashAttentionEngine",
    "FlashInferEngine", 
    "MultiHeadAttentionEngine",
    "SparseAttentionEngine",
] 