"""
WiCore主引擎
协调所有子系统运作的核心控制器

核心职责:
- 系统初始化和生命周期管理
- 各子系统协调和通信
- 资源分配和调度
- 性能监控和优化
- 故障检测和恢复
"""

import torch
import logging
import asyncio
import threading
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import signal
import sys

from .config import WiCoreConfig, get_config_manager
from .device_manager import DeviceManager
from .inference_engine import InferenceEngine
from ..memory.hmt_manager import HMTManager
from ..routing.mor_router import MoRRouter
from ..memory.kv_cache import get_kv_cache_manager

logger = logging.getLogger(__name__)


class EngineState(Enum):
    """引擎状态"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class EngineStats:
    """引擎统计信息"""
    start_time: float
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_latency: float = 0.0
    peak_memory_usage: int = 0
    total_tokens_processed: int = 0
    current_qps: float = 0.0


class WiCoreEngine:
    """WiCore主引擎"""
    
    def __init__(self, config: Optional[WiCoreConfig] = None):
        self.config = config or WiCoreConfig()
        self.state = EngineState.UNINITIALIZED
        
        # 核心组件
        self.device_manager: Optional[DeviceManager] = None
        self.inference_engine: Optional[InferenceEngine] = None
        self.hmt_manager: Optional[HMTManager] = None
        self.mor_router: Optional[MoRRouter] = None
        self.kv_cache_manager = None
        
        # 运行时状态
        self.stats = EngineStats(start_time=time.time())
        self.is_shutdown = False
        self.background_tasks: List[asyncio.Task] = []
        
        # 锁和事件
        self.state_lock = threading.RLock()
        self.shutdown_event = threading.Event()
        
        # 监控线程
        self.monitor_thread: Optional[threading.Thread] = None
        
        logger.info("WiCore引擎实例已创建")
    
    async def initialize(self) -> bool:
        """初始化引擎"""
        with self.state_lock:
            if self.state != EngineState.UNINITIALIZED:
                logger.warning(f"引擎状态错误，无法初始化: {self.state}")
                return False
            
            self.state = EngineState.INITIALIZING
        
        try:
            logger.info("🚀 开始初始化WiCore引擎...")
            
            # 初始化设备管理器
            await self._initialize_device_manager()
            
            # 初始化推理引擎（包含模型加载、HMT、MoR等）
            await self._initialize_inference_engine()
            
            # 初始化KV缓存管理
            await self._initialize_kv_cache()
            
            # 启动监控系统
            await self._start_monitoring()
            
            # 注册信号处理
            self._register_signal_handlers()
            
            with self.state_lock:
                self.state = EngineState.READY
            
            logger.info("✅ WiCore引擎初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 引擎初始化失败: {e}")
            with self.state_lock:
                self.state = EngineState.ERROR
            return False
    
    async def _initialize_device_manager(self):
        """初始化设备管理器"""
        logger.info("📱 初始化设备管理器...")
        
        self.device_manager = DeviceManager()
        
        # 启动设备监控
        if self.config.performance.enable_gpu_monitoring:
            self.device_manager.start_monitoring()
        
        # 获取系统统计
        stats = self.device_manager.get_system_stats()
        logger.info(f"设备发现完成: {stats['gpu_count']}个GPU, {stats['total_memory_gb']:.1f}GB总内存")
    
    async def _initialize_inference_engine(self):
        """初始化推理引擎（整合所有组件）"""
        logger.info("🚀 初始化推理引擎...")
        
        # 创建推理引擎，它会自动初始化HMT、MoR等组件
        self.inference_engine = InferenceEngine(self.config)
        
        # 初始化推理引擎
        success = await self.inference_engine.initialize()
        if not success:
            raise RuntimeError("推理引擎初始化失败")
        
        # 从推理引擎获取子组件引用
        self.hmt_manager = self.inference_engine.hmt_manager
        self.mor_router = self.inference_engine.mor_router
        
        logger.info("推理引擎初始化完成")
    
    async def _initialize_kv_cache(self):
        """初始化KV缓存管理"""
        if not self.config.attention.enable_kv_cache:
            logger.info("⏭️ KV缓存已禁用")
            return
        
        logger.info("💾 初始化KV缓存管理器...")
        
        # 获取全局KV缓存管理器
        self.kv_cache_manager = get_kv_cache_manager()
        
        logger.info("KV缓存管理器初始化完成")
    
    async def _start_monitoring(self):
        """启动监控系统"""
        if not self.config.performance.enable_monitoring:
            logger.info("⏭️ 性能监控已禁用")
            return
        
        logger.info("📊 启动性能监控...")
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(
            target=self._monitor_performance,
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info("性能监控已启动")
    
    def _register_signal_handlers(self):
        """注册信号处理器"""
        def signal_handler(signum, frame):
            logger.info(f"收到信号 {signum}，开始优雅关闭...")
            asyncio.create_task(self.shutdown())
        
        # 只在主线程中注册信号处理器
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
    
    def _monitor_performance(self):
        """性能监控线程"""
        logger.info("性能监控线程已启动")
        
        while not self.shutdown_event.is_set():
            try:
                # 收集性能指标
                self._collect_performance_metrics()
                
                # 等待下次监控
                self.shutdown_event.wait(self.config.performance.metrics_interval)
                
            except Exception as e:
                logger.error(f"性能监控异常: {e}")
                time.sleep(1.0)
    
    def _collect_performance_metrics(self):
        """收集性能指标"""
        try:
            # 系统内存使用
            if self.device_manager:
                stats = self.device_manager.get_system_stats()
                current_memory = stats.get('total_memory_gb', 0) - stats.get('available_memory_gb', 0)
                self.stats.peak_memory_usage = max(self.stats.peak_memory_usage, int(current_memory * 1024 * 1024 * 1024))
            
            # HMT内存统计
            if self.hmt_manager:
                hmt_stats = self.hmt_manager.get_memory_stats()
                logger.debug(f"HMT统计: {hmt_stats}")
            
            # KV缓存统计
            if self.kv_cache_manager:
                cache_stats = self.kv_cache_manager.get_cache_stats()
                logger.debug(f"KV缓存统计: 命中率 {cache_stats['global_hit_rate']:.2%}")
            
            # 计算QPS
            current_time = time.time()
            elapsed_time = current_time - self.stats.start_time
            if elapsed_time > 0:
                self.stats.current_qps = self.stats.total_requests / elapsed_time
            
        except Exception as e:
            logger.warning(f"性能指标收集失败: {e}")
    
    async def start(self):
        """启动引擎"""
        with self.state_lock:
            if self.state != EngineState.READY:
                logger.error(f"引擎状态错误，无法启动: {self.state}")
                return False
            
            self.state = EngineState.RUNNING
        
        logger.info("🎯 WiCore引擎已启动")
        return True
    
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理推理请求"""
        if self.state != EngineState.RUNNING:
            raise RuntimeError(f"引擎未运行，当前状态: {self.state}")
        
        if not self.inference_engine or not self.inference_engine.is_ready():
            raise RuntimeError("推理引擎未就绪")
        
        request_start_time = time.time()
        self.stats.total_requests += 1
        
        try:
            # 使用推理引擎处理请求
            from .inference_engine import InferenceRequest
            
            # 构建推理请求
            inference_request = InferenceRequest(
                request_id=request_data.get("request_id", "unknown"),
                messages=request_data.get("messages", []),
                max_tokens=request_data.get("max_tokens", 512),
                temperature=request_data.get("temperature", 0.7),
                top_p=request_data.get("top_p", 0.9),
                top_k=request_data.get("top_k", 50),
                stream=request_data.get("stream", False)
            )
            
            # 执行推理
            inference_response = await self.inference_engine.generate_text(inference_request)
            
            response = {
                "status": "success",
                "request_id": inference_response.request_id,
                "response": inference_response.text,
                "finish_reason": inference_response.finish_reason,
                "tokens_generated": inference_response.tokens_generated,
                "processing_time": inference_response.processing_time,
                "model_info": inference_response.model_info
            }
            
            self.stats.successful_requests += 1
            self.stats.total_tokens_processed += inference_response.tokens_generated
            
            # 更新平均延迟
            total_time = time.time() - self.stats.start_time
            if self.stats.successful_requests > 0:
                self.stats.average_latency = total_time / self.stats.successful_requests
            
            return response
            
        except Exception as e:
            self.stats.failed_requests += 1
            logger.error(f"请求处理失败: {e}")
            
            return {
                "status": "error",
                "request_id": request_data.get("request_id", "unknown"),
                "error": str(e),
                "processing_time": time.time() - request_start_time
            }
    
    async def shutdown(self):
        """关闭引擎"""
        with self.state_lock:
            if self.state in [EngineState.STOPPING, EngineState.STOPPED]:
                logger.info("引擎已在关闭过程中")
                return
            
            self.state = EngineState.STOPPING
        
        logger.info("🛑 开始关闭WiCore引擎...")
        
        try:
            # 设置关闭标志
            self.is_shutdown = True
            self.shutdown_event.set()
            
            # 取消后台任务
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # 停止监控线程
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=2.0)
            
            # 关闭推理引擎
            if self.inference_engine:
                await self.inference_engine.shutdown()
            
            # 关闭设备管理器
            if self.device_manager:
                self.device_manager.stop_monitoring()
            
            # 清理其他资源
            # 推理引擎会自动清理HMT和MoR等子组件
            
            with self.state_lock:
                self.state = EngineState.STOPPED
            
            logger.info("✅ WiCore引擎已关闭")
            
        except Exception as e:
            logger.error(f"❌ 引擎关闭异常: {e}")
            with self.state_lock:
                self.state = EngineState.ERROR
    
    def get_status(self) -> Dict[str, Any]:
        """获取引擎状态"""
        return {
            "state": self.state.value,
            "uptime": time.time() - self.stats.start_time,
            "stats": {
                "total_requests": self.stats.total_requests,
                "successful_requests": self.stats.successful_requests,
                "failed_requests": self.stats.failed_requests,
                "success_rate": self.stats.successful_requests / max(self.stats.total_requests, 1),
                "average_latency": self.stats.average_latency,
                "current_qps": self.stats.current_qps,
                "peak_memory_usage_gb": self.stats.peak_memory_usage / 1024 / 1024 / 1024,
                "total_tokens_processed": self.stats.total_tokens_processed
            },
            "components": {
                "device_manager": self.device_manager is not None,
                "inference_engine": self.inference_engine is not None and self.inference_engine.is_ready(),
                "hmt_manager": self.hmt_manager is not None,
                "mor_router": self.mor_router is not None,
                "kv_cache_manager": self.kv_cache_manager is not None,
            }
        }
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """获取详细统计信息"""
        stats = self.get_status()
        
        # 添加设备统计
        if self.device_manager:
            stats["device_stats"] = self.device_manager.get_system_stats()
        
        # 添加HMT统计
        if self.hmt_manager:
            stats["hmt_stats"] = self.hmt_manager.get_memory_stats()
        
        # 添加推理引擎统计
        if self.inference_engine:
            stats["inference_stats"] = self.inference_engine.get_stats()
        
        # 添加KV缓存统计
        if self.kv_cache_manager:
            stats["kv_cache_stats"] = self.kv_cache_manager.get_cache_stats()
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_status = {
            "healthy": True,
            "timestamp": time.time(),
            "checks": {}
        }
        
        # 检查引擎状态
        health_status["checks"]["engine_state"] = {
            "healthy": self.state == EngineState.RUNNING,
            "state": self.state.value
        }
        
        # 检查设备管理器
        if self.device_manager:
            available_devices = len(self.device_manager.get_available_devices())
            health_status["checks"]["devices"] = {
                "healthy": available_devices > 0,
                "available_devices": available_devices
            }
        
        # 检查内存使用
        if self.device_manager:
            stats = self.device_manager.get_system_stats()
            memory_usage = 1 - (stats.get('available_memory_gb', 0) / max(stats.get('total_memory_gb', 1), 1))
            health_status["checks"]["memory"] = {
                "healthy": memory_usage < 0.9,  # 90%阈值
                "usage": memory_usage
            }
        
        # 综合健康状态
        health_status["healthy"] = all(
            check.get("healthy", False) for check in health_status["checks"].values()
        )
        
        return health_status
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        # 在同步上下文中，我们只能设置关闭标志
        self.is_shutdown = True
        self.shutdown_event.set()
        
        # 如果在异步环境中，需要用户手动调用 await engine.shutdown()
        if exc_type:
            logger.error(f"引擎上下文异常: {exc_type.__name__}: {exc_val}")
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.shutdown()
        if exc_type:
            logger.error(f"引擎异步上下文异常: {exc_type.__name__}: {exc_val}") 