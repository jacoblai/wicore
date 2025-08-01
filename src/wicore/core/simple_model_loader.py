"""
简化的模型加载器

专注于单GPU部署，支持：
- 自动量化以适应16GB内存限制
- 多种模型架构（Qwen、Llama、Gemma、Mistral等）
- 高效的内存管理
"""

import torch
import logging
from typing import Optional, Dict, Any, Union
from pathlib import Path
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoConfig,
    BitsAndBytesConfig,
    GenerationConfig
)
from accelerate import init_empty_weights
import gc

logger = logging.getLogger(__name__)


class SimpleModelLoader:
    """简化的模型加载器
    
    特点：
    - 单GPU部署，自动选择最佳设备
    - 自动量化支持（INT4/INT8）
    - 内存优化加载
    - 支持多种模型架构
    """
    
    def __init__(self, config):
        """初始化模型加载器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        self.device = self._select_device()
        self.model_path = config.model_path
        self.enable_quantization = getattr(config, 'enable_quantization', True)
        
        # 内存监控
        self.memory_used = 0.0
        
        logger.info(f"简化模型加载器初始化完成")
        logger.info(f"目标设备: {self.device}")
        logger.info(f"模型路径: {self.model_path}")
        logger.info(f"量化支持: {self.enable_quantization}")
    
    def _select_device(self) -> torch.device:
        """选择最佳设备"""
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"检测到GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU内存: {gpu_memory:.1f}GB")
            return device
        else:
            logger.warning("未检测到CUDA设备，使用CPU")
            return torch.device('cpu')
    
    def _determine_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """确定量化配置"""
        if not self.enable_quantization or self.device.type == 'cpu':
            return None
        
        # 检查是否有bitsandbytes
        try:
            import bitsandbytes
        except ImportError:
            logger.warning("bitsandbytes未安装，禁用量化。建议: pip install bitsandbytes")
            return None
        
        # 检查GPU内存
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            if gpu_memory_gb <= 16:
                # 16GB及以下：使用INT4量化
                logger.info("使用INT4量化以适应16GB内存限制")
                try:
                    return BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16
                    )
                except Exception as e:
                    logger.warning(f"量化配置失败，使用FP16: {e}")
                    return None
            elif gpu_memory_gb <= 24:
                # 24GB：使用INT8量化
                logger.info("使用INT8量化")
                try:
                    return BitsAndBytesConfig(
                        load_in_8bit=True
                    )
                except Exception as e:
                    logger.warning(f"量化配置失败，使用FP16: {e}")
                    return None
            else:
                # 24GB以上：不使用量化
                logger.info("GPU内存充足，不使用量化")
                return None
        
        return None
    
    def load_model(self):
        """加载模型"""
        logger.info(f"开始加载模型: {self.model_path}")
        
        try:
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # 加载配置
            self.model_config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
            logger.info(f"模型架构: {self.model_config.model_type}")
            logger.info(f"词汇表大小: {self.model_config.vocab_size}")
            
            # 确定量化配置
            quantization_config = self._determine_quantization_config()
            
            # 加载分词器
            logger.info("加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=True
            )
            
            # 设置padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            logger.info("加载模型...")
            model_kwargs = {
                'pretrained_model_name_or_path': self.model_path,
                'config': self.model_config,
                'torch_dtype': torch.float16 if self.device.type == 'cuda' else torch.float32,
                'device_map': self.device,
                'trust_remote_code': True,
                'low_cpu_mem_usage': True,
            }
            
            if quantization_config is not None:
                model_kwargs['quantization_config'] = quantization_config
                # 量化模型不能指定device_map
                del model_kwargs['device_map']
            
            self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
            
            # 如果没有使用量化，手动移动到设备
            if quantization_config is None:
                self.model = self.model.to(self.device)
            
            # 设置为评估模式
            self.model.eval()
            
            # 设置生成配置
            self._setup_generation_config()
            
            # 计算内存使用
            self._calculate_memory_usage()
            
            logger.info(f"✅ 模型加载完成")
            logger.info(f"模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
            logger.info(f"内存使用: {self.memory_used:.2f}GB")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def _setup_generation_config(self):
        """设置生成配置"""
        try:
            self.generation_config = GenerationConfig.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
        except:
            # 如果没有找到生成配置，创建默认配置
            self.generation_config = GenerationConfig(
                max_length=2048,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        logger.info("生成配置设置完成")
    
    def _calculate_memory_usage(self):
        """计算内存使用量"""
        if self.device.type == 'cuda':
            # GPU内存
            self.memory_used = torch.cuda.memory_allocated() / 1024**3
        else:
            # 估算CPU内存
            param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
            self.memory_used = param_size / 1024**3
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "model_type": self.model_config.model_type,
            "vocab_size": self.model_config.vocab_size,
            "hidden_size": getattr(self.model_config, 'hidden_size', 'unknown'),
            "num_layers": getattr(self.model_config, 'num_hidden_layers', 'unknown'),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "memory_usage_gb": self.memory_used,
            "device": str(self.device),
            "quantized": hasattr(self.model, 'quantization_config')
        }
    
    def unload_model(self):
        """卸载模型"""
        if self.model is not None:
            logger.info("🗑️ 卸载模型...")
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if self.generation_config is not None:
            del self.generation_config
            self.generation_config = None
        
        # 清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.memory_used = 0.0
        logger.info("模型卸载完成")
    
    @property
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None and self.tokenizer is not None 