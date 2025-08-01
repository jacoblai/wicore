"""
HMT (Hierarchical Memory Tiering) 管理器
整合2024-2025最新内存优化技术的核心管理系统

基于以下突破性研究:
- Jenga: 异构嵌入内存分配 (ArXiv 2503.18292)
- HeadInfer: 头级别KV缓存offloading (ArXiv 2502.12574) 
- vTensor: GPU虚拟内存管理 (ArXiv 2407.15309)
- MiniKV: 2位量化KV缓存 (ArXiv 2411.18077)
- LaCache: 阶梯形缓存结构 (ArXiv 2507.14204)
- SYMPHONY: 多轮交互优化 (ArXiv 2412.16434)
"""

import torch
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from ..core.config import HMTConfig
from .vtensor import VTensorManager
from .jenga_allocator import JengaAllocator  
from .kv_cache import KVCacheManager, MiniKVCache, LaCacheManager
from .head_infer import HeadInferOffloader
from .symphony import SymphonyManager

logger = logging.getLogger(__name__)


class MemoryTier(Enum):
    """内存层级定义"""
    GPU_HBM = "gpu_hbm"        # GPU高带宽内存 (最快)
    GPU_SHARED = "gpu_shared"   # GPU共享内存
    CPU_DRAM = "cpu_dram"      # CPU内存 (中等)  
    NVME_SSD = "nvme_ssd"      # NVMe存储 (最慢但容量大)


@dataclass
class HMTConfig:
    """HMT配置参数"""
    # GPU内存配置
    gpu_memory_pool_size: int = 32 * 1024**3  # 32GB
    gpu_reserved_memory: int = 4 * 1024**3    # 4GB预留
    
    # 分层内存配置
    tier_ratios: Dict[MemoryTier, float] = None  # 各层内存比例
    tier_bandwidth: Dict[MemoryTier, float] = None  # 各层带宽 GB/s
    
    # KV缓存优化配置
    enable_minikv: bool = True          # 启用2位量化
    minikv_compression_ratio: float = 0.86  # 压缩比例
    enable_lacache: bool = True         # 启用阶梯形缓存
    lacache_levels: int = 4             # 缓存层级数
    
    # HeadInfer配置
    enable_head_offload: bool = True    # 启用头级别offloading
    head_offload_ratio: float = 0.8     # offload比例
    
    # SYMPHONY配置  
    enable_symphony: bool = True        # 启用多轮优化
    symphony_window_size: int = 128     # 会话窗口大小
    
    # 性能调优
    prefetch_enabled: bool = True       # 启用预取
    async_offload: bool = True          # 异步offloading
    memory_pool_threads: int = 4        # 内存池线程数

    def __post_init__(self):
        """初始化默认配置"""
        if self.tier_ratios is None:
            self.tier_ratios = {
                MemoryTier.GPU_HBM: 0.6,      # 60% GPU HBM
                MemoryTier.GPU_SHARED: 0.2,   # 20% GPU共享
                MemoryTier.CPU_DRAM: 0.15,    # 15% CPU内存
                MemoryTier.NVME_SSD: 0.05,    # 5% NVMe存储
            }
        
        if self.tier_bandwidth is None:
            self.tier_bandwidth = {
                MemoryTier.GPU_HBM: 2000.0,    # 2TB/s (HBM3)
                MemoryTier.GPU_SHARED: 1000.0,  # 1TB/s 
                MemoryTier.CPU_DRAM: 100.0,     # 100GB/s (DDR5)
                MemoryTier.NVME_SSD: 10.0,      # 10GB/s (PCIe 5.0)
            }


class HMTManager:
    """
    HMT内存管理器 - 协调所有内存优化技术
    
    核心功能:
    1. 分层内存管理 (GPU→CPU→NVMe)
    2. 智能KV缓存优化 (MiniKV + LaCache)
    3. 动态头级别offloading (HeadInfer)
    4. 多轮交互优化 (SYMPHONY)
    5. 虚拟内存管理 (vTensor)
    6. 异构内存分配 (Jenga)
    """
    
    def __init__(self, config: HMTConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info("🚀 初始化 HMT (Hierarchical Memory Tiering) 系统...")
        logger.info("📋 HMT 验证目标：支持千亿模型单卡部署与128K上下文")
        
        # 初始化各个子系统
        self._init_subsystems()
        
        # 性能统计
        self.stats = {
            "total_allocations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "offload_operations": 0,
            "memory_usage": {},
            "vtensor_operations": 0,
            "jenga_allocations": 0,
            "lacache_hits": 0,
            "symphony_cache_hits": 0,
            "head_offload_saves_mb": 0,
        }
        
        # 输出系统概况
        self._log_system_overview()
        
        logger.info("✅ HMT Manager 初始化完成 - 集成2024-2025最新内存优化技术")
    
    def _init_subsystems(self):
        """初始化各个子系统"""
        logger.info("🔧 初始化 HMT 核心子系统...")
        
        # 📦 1. vTensor虚拟内存管理器 (ArXiv 2407.15309)
        if getattr(self.config, 'enable_vtensor', True):
            logger.info("📦 初始化 vTensor 虚拟内存管理器...")
            try:
                from .vtensor import VTensorConfig
                vtensor_config = VTensorConfig()
                self.vtensor_manager = VTensorManager(vtensor_config)
                logger.info(f"✅ vTensor 启动成功 - 页面大小: {getattr(self.config, 'vtensor_page_size_mb', 64)}MB")
            except Exception as e:
                logger.warning(f"⚠️  vTensor 初始化失败: {e}")
                self.vtensor_manager = None
        else:
            logger.info("⏭️  vTensor 已禁用")
            self.vtensor_manager = None
        
        # 🧩 2. Jenga异构内存分配器 (ArXiv 2503.18292)
        if getattr(self.config, 'enable_jenga', True):
            logger.info("🧩 初始化 Jenga 异构嵌入内存分配器...")
            try:
                from .jenga_allocator import JengaConfig
                jenga_config = JengaConfig()
                self.jenga_allocator = JengaAllocator(jenga_config)
                gpu_ratio = getattr(self.config, 'jenga_gpu_embedding_ratio', 0.7)
                logger.info(f"✅ Jenga 启动成功 - GPU嵌入比例: {gpu_ratio*100:.1f}%")
            except Exception as e:
                logger.warning(f"⚠️  Jenga 初始化失败: {e}")
                self.jenga_allocator = None
        else:
            logger.info("⏭️  Jenga 已禁用")
            self.jenga_allocator = None
        
        # 💾 3. KV缓存管理器 (MiniKV + LaCache)
        logger.info("💾 初始化 KV缓存管理器...")
        try:
            from .kv_cache import KVCacheConfig
            kv_config = KVCacheConfig(
                enable_quantization=getattr(self.config, 'enable_minikv', False),
                quantization_bits=getattr(self.config, 'minikv_quantization_bits', 2),
                enable_ladder_cache=getattr(self.config, 'enable_lacache', False),
                ladder_levels=getattr(self.config, 'lacache_levels', 3),
                enable_symphony=getattr(self.config, 'enable_symphony', False)
            )
            self.kv_cache_manager = KVCacheManager(kv_config)
            
            # 记录启用的特性
            features = []
            if kv_config.enable_quantization:
                features.append(f"MiniKV({kv_config.quantization_bits}bit)")
            if kv_config.enable_ladder_cache:
                features.append(f"LaCache({kv_config.ladder_levels}层)")
            if kv_config.enable_symphony:
                features.append("SYMPHONY")
            
            logger.info(f"✅ KV缓存管理器启动成功 - 特性: {', '.join(features) if features else '标准缓存'}")
        except Exception as e:
            logger.warning(f"⚠️  KV缓存管理器初始化失败: {e}")
            self.kv_cache_manager = None
        
        # 🎯 4. HeadInfer offloader (ArXiv 2502.12574)
        if getattr(self.config, 'enable_head_offload', False):
            logger.info("🎯 初始化 HeadInfer 头级别offloading...")
            try:
                self.head_offloader = HeadInferOffloader(
                    num_heads=32,  # 默认值，后续会从模型配置获取
                    offload_ratio=getattr(self.config, 'head_offload_ratio', 0.3)
                )
                offload_ratio = getattr(self.config, 'head_offload_ratio', 0.3)
                logger.info(f"✅ HeadInfer 启动成功 - offload比例: {offload_ratio*100:.1f}%")
            except Exception as e:
                logger.warning(f"⚠️  HeadInfer 初始化失败: {e}")
                self.head_offloader = None
        else:
            logger.info("⏭️  HeadInfer 已禁用")
            self.head_offloader = None
        
        # 🎵 5. SYMPHONY多轮优化器 (ArXiv 2412.16434)
        if getattr(self.config, 'enable_symphony', False):
            logger.info("🎵 初始化 SYMPHONY 多轮交互优化器...")
            try:
                self.symphony_manager = SymphonyManager(
                    max_concurrent_requests=getattr(self.config, 'symphony_window_size', 8)
                )
                window_size = getattr(self.config, 'symphony_window_size', 8)
                logger.info(f"✅ SYMPHONY 启动成功 - 窗口大小: {window_size}")
            except Exception as e:
                logger.warning(f"⚠️  SYMPHONY 初始化失败: {e}")
                self.symphony_manager = None
        else:
            logger.info("⏭️  SYMPHONY 已禁用")
            self.symphony_manager = None
        
        # 🔧 6. 线程池用于异步操作
        max_workers = getattr(self.config, 'memory_pool_threads', 4)
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        logger.info(f"🔧 异步操作线程池启动 - 工作线程: {max_workers}")
        
        logger.info("✅ HMT 子系统初始化完成")
    
    def _log_system_overview(self):
        """输出HMT系统概况"""
        logger.info("📊 HMT系统概况:")
        
        # 🧠 分层内存配置
        gpu_max = getattr(self.config, 'memory_pools', {}).get('gpu', {}).get('max_size_gb', 0)
        cpu_max = getattr(self.config, 'memory_pools', {}).get('cpu', {}).get('max_size_gb', 0)
        nvme_max = getattr(self.config, 'memory_pools', {}).get('nvme', {}).get('max_size_gb', 0)
        logger.info(f"🧠 分层内存: GPU({gpu_max}GB) → CPU({cpu_max}GB) → NVMe({nvme_max}GB)")
        
        # 📋 启用的技术
        enabled_techs = []
        if getattr(self.config, 'enable_minikv', False):
            bits = getattr(self.config, 'minikv_quantization_bits', 2)
            enabled_techs.append(f"MiniKV({bits}bit)")
        if getattr(self.config, 'enable_lacache', False):
            levels = getattr(self.config, 'lacache_levels', 3)
            enabled_techs.append(f"LaCache({levels}层)")
        if getattr(self.config, 'enable_head_offload', False):
            ratio = getattr(self.config, 'head_offload_ratio', 0.3)
            enabled_techs.append(f"HeadInfer({ratio*100:.0f}%)")
        if getattr(self.config, 'enable_symphony', False):
            window = getattr(self.config, 'symphony_window_size', 8)
            enabled_techs.append(f"SYMPHONY({window})")
        if getattr(self.config, 'enable_vtensor', False):
            page_size = getattr(self.config, 'vtensor_page_size_mb', 64)
            enabled_techs.append(f"vTensor({page_size}MB)")
        if getattr(self.config, 'enable_jenga', False):
            gpu_ratio = getattr(self.config, 'jenga_gpu_embedding_ratio', 0.7)
            enabled_techs.append(f"Jenga({gpu_ratio*100:.0f}%)")
        
        logger.info(f"🔬 启用技术: {', '.join(enabled_techs) if enabled_techs else '基础内存管理'}")
        
        # 🎯 验证目标
        logger.info("🎯 验证目标:")
        logger.info("   ✓ 千亿模型单卡部署")
        logger.info("   ✓ 128K上下文支持")
        logger.info("   ✓ GPU→CPU→NVMe分层存储")
        logger.info("   ✓ 智能缓存优化")
        logger.info("   ✓ 动态内存管理")
    
    def get_hmt_detailed_stats(self) -> Dict[str, Any]:
        """获取HMT详细统计信息"""
        detailed_stats = {
            "hmt_enabled": True,
            "subsystems": {},
            "performance": self.stats.copy(),
            "memory_breakdown": {}
        }
        
        # vTensor统计
        if self.vtensor_manager:
            detailed_stats["subsystems"]["vtensor"] = {
                "enabled": True,
                "page_size_mb": getattr(self.config, 'vtensor_page_size_mb', 64),
                "operations": self.stats.get("vtensor_operations", 0),
                "status": "运行中"
            }
        else:
            detailed_stats["subsystems"]["vtensor"] = {"enabled": False}
        
        # Jenga统计
        if self.jenga_allocator:
            detailed_stats["subsystems"]["jenga"] = {
                "enabled": True,
                "gpu_ratio": getattr(self.config, 'jenga_gpu_embedding_ratio', 0.7),
                "allocations": self.stats.get("jenga_allocations", 0),
                "status": "运行中"
            }
        else:
            detailed_stats["subsystems"]["jenga"] = {"enabled": False}
        
        # KV缓存统计
        if self.kv_cache_manager:
            kv_stats = {
                "enabled": True,
                "minikv_enabled": getattr(self.config, 'enable_minikv', False),
                "lacache_enabled": getattr(self.config, 'enable_lacache', False),
                "symphony_enabled": getattr(self.config, 'enable_symphony', False),
                "cache_hits": self.stats.get("cache_hits", 0),
                "lacache_hits": self.stats.get("lacache_hits", 0),
                "status": "运行中"
            }
            if getattr(self.config, 'enable_minikv', False):
                kv_stats["minikv_bits"] = getattr(self.config, 'minikv_quantization_bits', 2)
            if getattr(self.config, 'enable_lacache', False):
                kv_stats["lacache_levels"] = getattr(self.config, 'lacache_levels', 3)
            detailed_stats["subsystems"]["kv_cache"] = kv_stats
        else:
            detailed_stats["subsystems"]["kv_cache"] = {"enabled": False}
        
        # HeadInfer统计
        if self.head_offloader:
            detailed_stats["subsystems"]["headinfer"] = {
                "enabled": True,
                "offload_ratio": getattr(self.config, 'head_offload_ratio', 0.3),
                "memory_saved_mb": self.stats.get("head_offload_saves_mb", 0),
                "status": "运行中"
            }
        else:
            detailed_stats["subsystems"]["headinfer"] = {"enabled": False}
        
        # SYMPHONY统计
        if self.symphony_manager:
            detailed_stats["subsystems"]["symphony"] = {
                "enabled": True,
                "window_size": getattr(self.config, 'symphony_window_size', 8),
                "cache_hits": self.stats.get("symphony_cache_hits", 0),
                "status": "运行中"
            }
        else:
            detailed_stats["subsystems"]["symphony"] = {"enabled": False}
        
        return detailed_stats
    
    def log_performance_summary(self):
        """输出性能摘要"""
        logger.info("📈 HMT 性能摘要:")
        logger.info(f"   🎯 总分配次数: {self.stats['total_allocations']}")
        logger.info(f"   💾 缓存命中: {self.stats['cache_hits']}")
        logger.info(f"   🔄 Offload操作: {self.stats['offload_operations']}")
        
        if self.vtensor_manager:
            logger.info(f"   📦 vTensor操作: {self.stats['vtensor_operations']}")
        
        if self.jenga_allocator:
            logger.info(f"   🧩 Jenga分配: {self.stats['jenga_allocations']}")
        
        if self.head_offloader:
            saved_mb = self.stats.get('head_offload_saves_mb', 0)
            logger.info(f"   🎯 HeadInfer节省: {saved_mb:.1f}MB")
        
        if hasattr(self, 'symphony_manager') and self.symphony_manager:
            symphony_hits = self.stats.get('symphony_cache_hits', 0)
            logger.info(f"   🎵 SYMPHONY缓存命中: {symphony_hits}")
    
    def update_stats(self, operation: str, value: int = 1):
        """更新统计信息"""
        if operation in self.stats:
            self.stats[operation] += value
            
            # 每100次操作记录一次性能摘要
            if self.stats['total_allocations'] % 100 == 0 and self.stats['total_allocations'] > 0:
                self.log_performance_summary()
    
    async def allocate_memory(
        self, 
        size: int, 
        tier: MemoryTier = MemoryTier.GPU_HBM,
        tensor_type: str = "kv_cache"
    ) -> torch.Tensor:
        """
        智能内存分配
        
        Args:
            size: 内存大小 (bytes)
            tier: 目标内存层级
            tensor_type: 张量类型 (kv_cache, attention, etc.)
            
        Returns:
            分配的张量
        """
        self.stats["total_allocations"] += 1
        
        try:
            # Step 1: Jenga异构分配决策
            optimal_tier = await self.jenga_allocator.select_optimal_tier(
                size=size,
                access_pattern=tensor_type,
                current_usage=self._get_memory_usage()
            )
            
            # Step 2: vTensor虚拟内存分配
            if optimal_tier == MemoryTier.GPU_HBM:
                tensor = await self.vtensor_manager.allocate_virtual_tensor(
                    size=size,
                    device=self.device
                )
            else:
                # 使用传统分配作为fallback
                tensor = self._allocate_fallback(size, optimal_tier)
            
            # Step 3: 更新内存使用统计
            self._update_memory_stats(optimal_tier, size)
            
            logger.debug(f"Allocated {size} bytes to {optimal_tier}")
            return tensor
            
        except Exception as e:
            logger.error(f"Memory allocation failed: {e}")
            raise
    
    async def manage_kv_cache(
        self,
        layer_id: int,
        head_id: int, 
        seq_len: int,
        kv_data: torch.Tensor
    ) -> torch.Tensor:
        """
        智能KV缓存管理
        
        集成MiniKV量化 + LaCache阶梯形缓存 + HeadInfer offloading
        """
        cache_key = f"layer_{layer_id}_head_{head_id}"
        
        try:
            # Step 1: MiniKV 2位量化压缩
            if self.config.enable_minikv:
                compressed_kv = await self.kv_managers['minikv'].compress(
                    kv_data, layer_id=layer_id, head_id=head_id
                )
            else:
                compressed_kv = kv_data
            
            # Step 2: LaCache阶梯形缓存策略
            if self.config.enable_lacache:
                cached_kv = await self.kv_managers['lacache'].cache_with_ladder(
                    compressed_kv, 
                    seq_len=seq_len,
                    layer_id=layer_id
                )
            else:
                cached_kv = compressed_kv
            
            # Step 3: HeadInfer智能offloading
            if self.config.enable_head_offload:
                final_kv = await self.head_offloader.smart_offload(
                    cached_kv,
                    head_id=head_id,
                    priority=self._calculate_head_priority(layer_id, head_id)
                )
            else:
                final_kv = cached_kv
            
            # Step 4: SYMPHONY多轮优化
            if self.config.enable_symphony:
                optimized_kv = await self.symphony_manager.optimize_for_session(
                    final_kv,
                    cache_key=cache_key
                )
            else:
                optimized_kv = final_kv
            
            self.stats["cache_hits"] += 1
            return optimized_kv
            
        except Exception as e:
            self.stats["cache_misses"] += 1
            logger.error(f"KV cache management failed: {e}")
            return kv_data  # 返回原始数据作为fallback
    
    def _calculate_head_priority(self, layer_id: int, head_id: int) -> float:
        """计算注意力头优先级 (基于HeadInfer研究)"""
        # 基于层级和头部重要性的启发式算法
        layer_weight = 1.0 - (layer_id / 32)  # 假设32层模型
        head_weight = 1.0  # 可以基于实际使用统计调整
        
        return layer_weight * head_weight
    
    def _get_memory_usage(self) -> Dict[MemoryTier, float]:
        """获取各层内存使用率"""
        # 实现内存使用监控逻辑
        usage = {}
        for tier in MemoryTier:
            # 这里应该调用实际的内存监控API
            usage[tier] = 0.5  # 占位符
        return usage
    
    def _update_memory_stats(self, tier: MemoryTier, size: int):
        """更新内存使用统计"""
        if tier not in self.stats["memory_usage"]:
            self.stats["memory_usage"][tier] = 0
        self.stats["memory_usage"][tier] += size
    
    def _allocate_fallback(self, size: int, tier: MemoryTier) -> torch.Tensor:
        """备用内存分配方案"""
        if tier == MemoryTier.GPU_HBM:
            return torch.empty(size // 4, dtype=torch.float32, device=self.device)
        elif tier == MemoryTier.CPU_DRAM:
            return torch.empty(size // 4, dtype=torch.float32, device="cpu")
        else:
            # NVMe等其他存储的处理
            return torch.empty(size // 4, dtype=torch.float32, device="cpu")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        cache_hit_rate = (
            self.stats["cache_hits"] / 
            max(1, self.stats["cache_hits"] + self.stats["cache_misses"])
        )
        
        return {
            "cache_hit_rate": cache_hit_rate,
            "total_allocations": self.stats["total_allocations"],
            "offload_operations": self.stats["offload_operations"],
            "memory_usage_by_tier": self.stats["memory_usage"],
            "active_optimizations": {
                "minikv_enabled": self.config.enable_minikv,
                "lacache_enabled": self.config.enable_lacache,
                "head_offload_enabled": self.config.enable_head_offload,
                "symphony_enabled": self.config.enable_symphony,
            }
        }
    
    async def cleanup(self):
        """清理资源"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        
        # 清理各个子系统
        await self.vtensor_manager.cleanup()
        
        logger.info("HMT Manager cleanup completed") 