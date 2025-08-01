"""
WiCore: 世界级高性能LLM推理引擎
基于2024-2025最新研究的HMT (Hierarchical Memory Tiering) 架构

核心技术栈:
- Jenga异构内存分配 + HeadInfer offloading + vTensor虚拟内存
- EvoMoE专家进化 + RMoE递归路由 + 动态专家选择  
- MiniKV 2位量化 + LaCache阶梯形缓存 + SYMPHONY多轮优化
- FlashInfer可定制注意力引擎

Copyright (c) 2024 WiCore Project
"""

__version__ = "1.0.0"
__author__ = "WiCore Team"

from .core import WiCoreEngine
from .memory import HMTManager
from .routing import MoRRouter
from .attention import AttentionEngine

__all__ = [
    "WiCoreEngine",
    "HMTManager", 
    "MoRRouter",
    "AttentionEngine",
] 