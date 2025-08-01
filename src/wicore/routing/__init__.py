"""
MoR (Mixture of Routers) 动态路由系统
基于2024-2025最新MoE研究的智能路由技术

核心技术:
- EvoMoE专家进化算法 (解决专家均匀性问题)
- RMoE层级递归路由 (GRU建立层间依赖关系) 
- 动态专家选择 (根据任务复杂度自适应调整)
- 本地路由一致性优化 (提升缓存效率)
- 多维路由框架 (能力和知识建模)
"""

from .mor_router import MoRRouter
from .evo_moe import EvoMoERouter
from .rmoe import RMoERouter  
from .dynamic_selector import DynamicExpertSelector
from .inference_dynamics import InferenceDynamicsRouter

__all__ = [
    "MoRRouter",
    "EvoMoERouter", 
    "RMoERouter",
    "DynamicExpertSelector",
    "InferenceDynamicsRouter",
] 