"""
HMT (Hierarchical Memory Tiering) 内存管理系统
基于2024-2025最新研究的分层内存优化技术

核心技术:
- Jenga异构内存分配框架 (79.6%内存利用率提升)
- HeadInfer头级别offloading (92%内存占用减少)  
- vTensor GPU虚拟内存管理 (1.86x性能提升)
- MiniKV 2位量化缓存 (86%压缩率保持98.5%精度)
- LaCache阶梯形KV缓存 (超长上下文支持)
- SYMPHONY多轮交互优化 (8x请求处理能力)
"""

from .hmt_manager import HMTManager
from .vtensor import VTensorManager
from .jenga_allocator import JengaAllocator
from .kv_cache import KVCacheManager, MiniKVCache, LaCacheManager
from .head_infer import HeadInferOffloader
from .symphony import SymphonyManager

__all__ = [
    "HMTManager",
    "VTensorManager", 
    "JengaAllocator",
    "KVCacheManager",
    "MiniKVCache",
    "LaCacheManager",
    "HeadInferOffloader",
    "SymphonyManager",
] 