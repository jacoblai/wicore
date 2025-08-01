"""
WiCore配置管理系统
支持多环境配置、动态更新和验证

核心功能:
- 分层配置管理（默认、环境、用户）
- 配置验证和类型检查
- 动态配置更新
- 配置版本管理
- 敏感信息保护
"""

import os
import json
import yaml
import logging
from typing import Dict, List, Optional, Any, Union, Type, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import threading
from copy import deepcopy

logger = logging.getLogger(__name__)


@dataclass
class HMTConfig:
    """HMT内存管理配置"""
    enable_hmt: bool = True
    
    # 🧠 分层内存管理配置
    memory_pools: Optional[Dict[str, Dict[str, float]]] = field(default_factory=lambda: {
        "gpu": {"initial_size_gb": 8.0, "max_size_gb": 14.0, "growth_factor": 1.5},
        "cpu": {"initial_size_gb": 4.0, "max_size_gb": 8.0, "growth_factor": 1.2},
        "nvme": {"initial_size_gb": 2.0, "max_size_gb": 16.0, "growth_factor": 1.0}
    })
    memory_tiers: List[str] = field(default_factory=lambda: ["gpu", "cpu", "nvme"])
    tier_ratios: List[float] = field(default_factory=lambda: [0.6, 0.3, 0.1])
    
    # 🔄 MiniKV: 2位量化KV缓存 (ArXiv 2411.18077)
    enable_minikv: bool = True
    minikv_quantization_bits: int = 2
    minikv_compression_ratio: float = 0.25
    
    # 🏗️ LaCache: 阶梯形缓存结构 (ArXiv 2507.14204)
    enable_lacache: bool = True
    lacache_levels: int = 3
    lacache_l1_size_mb: int = 512  # GPU HBM
    lacache_l2_size_mb: int = 2048  # CPU DRAM
    lacache_l3_size_mb: int = 8192  # NVMe SSD
    
    # 🎯 HeadInfer: 头级别KV缓存offloading (ArXiv 2502.12574)
    enable_head_offload: bool = True
    head_offload_ratio: float = 0.3  # 30%的头放在CPU
    head_offload_threshold_mb: int = 1024
    
    # 🎵 SYMPHONY: 多轮交互优化 (ArXiv 2412.16434)
    enable_symphony: bool = True
    symphony_window_size: int = 8
    symphony_cache_size_mb: int = 1024
    symphony_prefetch_rounds: int = 3
    
    # 📦 vTensor: GPU虚拟内存管理 (ArXiv 2407.15309)
    enable_vtensor: bool = True
    vtensor_page_size_mb: int = 64
    vtensor_swap_threshold: float = 0.8
    vtensor_prefetch_pages: int = 4
    
    # 🧩 Jenga: 异构嵌入内存分配 (ArXiv 2503.18292)
    enable_jenga: bool = True
    jenga_embedding_cache_mb: int = 512
    jenga_allocation_strategy: str = "heterogeneous"
    jenga_gpu_embedding_ratio: float = 0.7
    
    # 💾 缓存策略
    cache_strategy: str = "lacache"  # 使用LaCache策略
    max_cache_entries: int = 2048
    cache_warmup_enabled: bool = True
    
    # ⚡ 预取设置
    enable_prefetch: bool = True
    prefetch_size: int = 4
    prefetch_threads: int = 2
    
    # 📊 内存监控和优化
    memory_monitoring_interval_ms: int = 100
    auto_memory_optimization: bool = True
    memory_pressure_threshold: float = 0.85
    emergency_offload_threshold: float = 0.95
    
    # 🔧 线程池配置
    memory_pool_threads: int = 4
    offload_threads: int = 2
    
    # 📈 性能调优
    enable_async_offload: bool = True
    enable_memory_prefusion: bool = True
    enable_gradient_checkpointing: bool = True
    
    # 向后兼容的旧配置
    swap_threshold: float = 0.8
    compression_threshold: float = 0.9
    prefetch_window: int = 4
    enable_compression: bool = True
    gpu_memory_pool_size: int = 8 * 1024 * 1024 * 1024  # 8GB
    tier_bandwidth: Optional[Dict[str, float]] = None
    async_offload: bool = True


@dataclass
class MoRConfig:
    """MoR动态路由配置"""
    enable_mor: bool = True
    num_experts: int = 8
    routing_strategy: str = "dynamic_topk"
    max_experts_per_token: int = 2
    load_balance_weight: float = 0.01
    evolution_enabled: bool = True
    evolution_steps: int = 100
    
    # EvoMoE配置
    enable_expert_evolution: bool = True
    mutation_rate: float = 0.1
    diversity_weight: float = 0.3
    
    # RMoE配置
    enable_recursive_routing: bool = True
    gru_hidden_size: int = 256
    max_layer_memory: int = 1000
    
    # 动态选择配置
    enable_dynamic_selection: bool = True
    min_experts: int = 1
    max_experts: int = 8
    complexity_threshold: float = 0.5
    
    # 推理动态配置
    enable_inference_dynamics: bool = True
    capability_dim: int = 128
    knowledge_dim: int = 256
    
    # 路由缓存配置
    cache_routing_decisions: bool = True


@dataclass
class AttentionConfig:
    """注意力机制配置"""
    attention_type: str = "flashinfer"
    enable_kv_cache: bool = True
    kv_cache_quantization: bool = True
    quantization_bits: int = 2
    max_sequence_length: int = 128 * 1024
    attention_dropout: float = 0.0


@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str = ""
    model_path: str = ""
    model_type: str = "llama"  # 支持: llama, gemma, qwen等
    max_batch_size: int = 8
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    
    # Gemma特定配置
    use_flash_attention: bool = True
    torch_dtype: str = "auto"  # auto, fp16, bf16, fp32
    device_map: str = "auto"  # auto, balanced, sequential
    
    # 模型加载配置
    trust_remote_code: bool = False
    low_cpu_mem_usage: bool = True
    use_safetensors: bool = True
    
    # 推理优化配置
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    do_sample: bool = True
    num_beams: int = 1
    early_stopping: bool = True


@dataclass
class ServerConfig:
    """服务器配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    max_concurrent_requests: int = 64
    request_timeout: float = 300.0
    enable_cors: bool = True
    enable_metrics: bool = True


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: str = "10MB"
    backup_count: int = 5
    enable_console: bool = True


@dataclass
class PerformanceConfig:
    """性能配置"""
    enable_monitoring: bool = True
    metrics_interval: float = 5.0
    enable_profiling: bool = False
    profile_output_dir: str = "./profiles"
    enable_memory_tracking: bool = True
    enable_gpu_monitoring: bool = True


@dataclass
class WiCoreConfig:
    """WiCore主配置"""
    # 核心组件配置
    hmt: HMTConfig = field(default_factory=HMTConfig)
    mor: MoRConfig = field(default_factory=MoRConfig)  
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # 全局配置
    environment: str = "development"
    debug: bool = False
    seed: int = 42
    device: str = "auto"
    precision: str = "fp16"
    
    # 元数据
    version: str = "1.0.0"
    config_path: Optional[str] = None
    last_modified: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WiCoreConfig':
        """从字典创建配置"""
        # 处理嵌套的dataclass
        if 'hmt' in data and isinstance(data['hmt'], dict):
            data['hmt'] = HMTConfig(**data['hmt'])
        if 'mor' in data and isinstance(data['mor'], dict):
            data['mor'] = MoRConfig(**data['mor'])
        if 'attention' in data and isinstance(data['attention'], dict):
            data['attention'] = AttentionConfig(**data['attention'])
        if 'model' in data and isinstance(data['model'], dict):
            data['model'] = ModelConfig(**data['model'])
        if 'server' in data and isinstance(data['server'], dict):
            data['server'] = ServerConfig(**data['server'])
        if 'logging' in data and isinstance(data['logging'], dict):
            data['logging'] = LoggingConfig(**data['logging'])
        if 'performance' in data and isinstance(data['performance'], dict):
            data['performance'] = PerformanceConfig(**data['performance'])
        
        return cls(**data)


class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_config(config: WiCoreConfig) -> List[str]:
        """验证配置，返回错误列表"""
        errors = []
        
        # 验证HMT配置
        if config.hmt.enable_hmt:
            if len(config.hmt.memory_tiers) != len(config.hmt.tier_ratios):
                errors.append("HMT内存层级数量与比例数量不匹配")
            
            if abs(sum(config.hmt.tier_ratios) - 1.0) > 0.01:
                errors.append("HMT内存层级比例总和应为1.0")
            
            if not 0 < config.hmt.swap_threshold < 1:
                errors.append("HMT交换阈值应在0-1之间")
        
        # 验证MoR配置
        if config.mor.enable_mor:
            if config.mor.num_experts <= 0:
                errors.append("MoR专家数量应大于0")
            
            if config.mor.max_experts_per_token > config.mor.num_experts:
                errors.append("每token最大专家数不能超过总专家数")
        
        # 验证注意力配置
        if config.attention.kv_cache_quantization:
            if config.attention.quantization_bits not in [1, 2, 4, 8]:
                errors.append("KV缓存量化位数应为1, 2, 4, 8之一")
        
        # 验证模型配置
        if not config.model.model_name:
            errors.append("模型名称不能为空")
        
        if config.model.max_batch_size <= 0:
            errors.append("最大批次大小应大于0")
        
        # 验证服务器配置
        if not 1 <= config.server.port <= 65535:
            errors.append("服务器端口应在1-65535之间")
        
        if config.server.max_concurrent_requests <= 0:
            errors.append("最大并发请求数应大于0")
        
        return errors
    
    @staticmethod
    def validate_and_fix(config: WiCoreConfig) -> Tuple[WiCoreConfig, List[str]]:
        """验证并修复配置"""
        errors = ConfigValidator.validate_config(config)
        
        # 自动修复一些常见问题
        if config.hmt.enable_hmt and len(config.hmt.tier_ratios) > 0:
            # 归一化内存层级比例
            total_ratio = sum(config.hmt.tier_ratios)
            if abs(total_ratio - 1.0) > 0.01:
                config.hmt.tier_ratios = [r / total_ratio for r in config.hmt.tier_ratios]
                errors = [e for e in errors if "层级比例总和" not in e]
        
        # 修复端口范围
        if not 1 <= config.server.port <= 65535:
            config.server.port = max(1, min(65535, config.server.port))
            errors = [e for e in errors if "端口" not in e]
        
        return config, errors


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str = "./configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        self.current_config: Optional[WiCoreConfig] = None
        self.config_history: List[WiCoreConfig] = []
        self.lock = threading.RLock()
        
        # 配置监听器
        self.listeners: List[callable] = []
        
        logger.info(f"配置管理器初始化: {self.config_dir}")
    
    def load_config(
        self, 
        config_path: Optional[str] = None,
        environment: str = "development"
    ) -> WiCoreConfig:
        """加载配置"""
        with self.lock:
            if config_path:
                config = self._load_from_file(config_path)
            else:
                config = self._load_default_config(environment)
            
            # 验证配置
            config, errors = ConfigValidator.validate_and_fix(config)
            
            if errors:
                logger.warning(f"配置验证发现问题: {errors}")
            
            # 保存到历史
            if self.current_config:
                self.config_history.append(deepcopy(self.current_config))
                # 限制历史记录数量
                if len(self.config_history) > 10:
                    self.config_history = self.config_history[-10:]
            
            self.current_config = config
            config.config_path = config_path
            config.environment = environment
            
            # 通知监听器
            self._notify_listeners(config)
            
            logger.info(f"配置加载完成: {environment}环境")
            return config
    
    def _load_from_file(self, config_path: str) -> WiCoreConfig:
        """从文件加载配置"""
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            if config_file.suffix.lower() == '.yaml' or config_file.suffix.lower() == '.yml':
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        return WiCoreConfig.from_dict(data)
    
    def _load_default_config(self, environment: str) -> WiCoreConfig:
        """加载默认配置"""
        # 尝试加载环境特定配置
        env_config_path = self.config_dir / f"{environment}.yaml"
        if env_config_path.exists():
            return self._load_from_file(str(env_config_path))
        
        # 尝试加载通用配置
        default_config_path = self.config_dir / "default.yaml"
        if default_config_path.exists():
            return self._load_from_file(str(default_config_path))
        
        # 使用内置默认配置
        config = WiCoreConfig()
        config.environment = environment
        
        # 根据环境调整配置
        if environment == "production":
            config.debug = False
            config.logging.level = "WARNING"
            config.performance.enable_profiling = False
        elif environment == "development":
            config.debug = True
            config.logging.level = "DEBUG"
            config.performance.enable_profiling = True
        
        return config
    
    def save_config(self, config_path: Optional[str] = None) -> str:
        """保存配置"""
        with self.lock:
            if self.current_config is None:
                raise ValueError("没有当前配置可保存")
            
            if config_path is None:
                config_path = self.config_dir / f"{self.current_config.environment}.yaml"
            
            config_file = Path(config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 导出配置
            config_dict = self.current_config.to_dict()
            
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"配置已保存: {config_file}")
            return str(config_file)
    
    def update_config(self, updates: Dict[str, Any], validate: bool = True) -> WiCoreConfig:
        """更新配置"""
        with self.lock:
            if self.current_config is None:
                raise ValueError("没有当前配置可更新")
            
            # 深拷贝当前配置
            new_config = deepcopy(self.current_config)
            
            # 应用更新
            self._apply_updates(new_config, updates)
            
            if validate:
                # 验证新配置
                new_config, errors = ConfigValidator.validate_and_fix(new_config)
                if errors:
                    logger.warning(f"配置更新验证发现问题: {errors}")
            
            # 保存旧配置到历史
            self.config_history.append(deepcopy(self.current_config))
            if len(self.config_history) > 10:
                self.config_history = self.config_history[-10:]
            
            # 更新当前配置
            self.current_config = new_config
            
            # 通知监听器
            self._notify_listeners(new_config)
            
            logger.info(f"配置已更新: {list(updates.keys())}")
            return new_config
    
    def _apply_updates(self, config: WiCoreConfig, updates: Dict[str, Any]):
        """应用配置更新"""
        for key, value in updates.items():
            if '.' in key:
                # 支持嵌套键，如 "hmt.enable_hmt"
                self._set_nested_value(config, key, value)
            else:
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    logger.warning(f"未知配置键: {key}")
    
    def _set_nested_value(self, obj: Any, key: str, value: Any):
        """设置嵌套值"""
        keys = key.split('.')
        for k in keys[:-1]:
            if hasattr(obj, k):
                obj = getattr(obj, k)
            else:
                logger.warning(f"未知嵌套配置键: {k}")
                return
        
        final_key = keys[-1]
        if hasattr(obj, final_key):
            setattr(obj, final_key, value)
        else:
            logger.warning(f"未知配置属性: {final_key}")
    
    def get_config(self) -> Optional[WiCoreConfig]:
        """获取当前配置"""
        with self.lock:
            return self.current_config
    
    def add_listener(self, listener: callable):
        """添加配置变更监听器"""
        with self.lock:
            self.listeners.append(listener)
    
    def remove_listener(self, listener: callable):
        """移除配置变更监听器"""
        with self.lock:
            if listener in self.listeners:
                self.listeners.remove(listener)
    
    def _notify_listeners(self, config: WiCoreConfig):
        """通知监听器"""
        for listener in self.listeners:
            try:
                listener(config)
            except Exception as e:
                logger.error(f"配置监听器错误: {e}")
    
    def rollback_config(self, steps: int = 1) -> Optional[WiCoreConfig]:
        """回滚配置"""
        with self.lock:
            if len(self.config_history) < steps:
                logger.warning(f"历史配置不足，无法回滚{steps}步")
                return None
            
            # 获取回滚目标
            target_config = self.config_history[-(steps)]
            
            # 移除回滚的配置
            self.config_history = self.config_history[:-steps]
            
            # 保存当前配置到历史
            if self.current_config:
                self.config_history.append(deepcopy(self.current_config))
            
            # 设置新的当前配置
            self.current_config = deepcopy(target_config)
            
            # 通知监听器
            self._notify_listeners(self.current_config)
            
            logger.info(f"配置已回滚{steps}步")
            return self.current_config
    
    def export_config_template(self, output_path: str):
        """导出配置模板"""
        template_config = WiCoreConfig()
        config_dict = template_config.to_dict()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"配置模板已导出: {output_path}")


# 全局配置管理器实例
_global_config_manager: Optional[ConfigManager] = None

def get_config_manager() -> ConfigManager:
    """获取全局配置管理器"""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = ConfigManager()
    return _global_config_manager

def get_config() -> Optional[WiCoreConfig]:
    """获取当前配置"""
    return get_config_manager().get_config() 