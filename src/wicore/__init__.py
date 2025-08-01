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

# 核心引擎
from .core.engine import WiCoreEngine
from .core.config import WiCoreConfig, ConfigManager
from .core.device_manager import DeviceManager
from .core.performance_monitor import PerformanceMonitor
from .core.simple_model_loader import SimpleModelLoader
from .core.inference_engine import InferenceEngine, InferenceRequest, InferenceResponse

# HMT内存管理系统
from .memory.hmt_manager import HMTManager
from .memory.vtensor import VTensorManager  
from .memory.jenga_allocator import JengaAllocator
from .memory.kv_cache import KVCacheManager, MiniKVCache, LaCacheManager
from .memory.head_infer import HeadInferOffloader
from .memory.symphony import SymphonyManager

# MoR动态路由系统  
from .routing.mor_router import MoRRouter
from .routing.evo_moe import EvoMoERouter
from .routing.rmoe import RMoERouter
from .routing.dynamic_selector import DynamicExpertSelector
from .routing.inference_dynamics import InferenceDynamicsRouter

# 注意力引擎
from .attention.flash_engine import FlashAttentionEngine
from .attention.flashinfer import FlashInferEngine
from .attention.multi_head import MultiHeadAttentionEngine
from .attention.sparse_attention import SparseAttentionEngine

# API服务器
from .server.api_server import create_app, start_server

__all__ = [
    # 核心引擎
    "WiCoreEngine",
    "WiCoreConfig", 
    "ConfigManager",
    "DeviceManager",
    "PerformanceMonitor",
    "SimpleModelLoader",
    "InferenceEngine",
    "InferenceRequest",
    "InferenceResponse",
    
    # HMT内存管理
    "HMTManager",
    "VTensorManager",
    "JengaAllocator", 
    "KVCacheManager",
    "MiniKVCache",
    "LaCacheManager",
    "HeadInferOffloader",
    "SymphonyManager",
    
    # MoR动态路由
    "MoRRouter",
    "EvoMoERouter",
    "RMoERouter",
    "DynamicExpertSelector",
    "InferenceDynamicsRouter",
    
    # 注意力引擎
    "FlashAttentionEngine",
    "FlashInferEngine",
    "MultiHeadAttentionEngine",
    "SparseAttentionEngine",
    
    # API服务器
    "create_app",
    "start_server",
] 