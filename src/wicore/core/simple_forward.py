"""
简化的单GPU前向传播引擎

移除多设备复杂性，专注于单GPU高性能推理
支持多种模型：Qwen、Llama、Gemma、Mistral等
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Union
import logging
from transformers import AutoModelForCausalLM, AutoConfig

logger = logging.getLogger(__name__)


class SimpleForward:
    """简化的单GPU前向传播引擎
    
    特点：
    - 单GPU部署，无设备映射复杂性
    - 支持多种模型架构
    - 保持API兼容性
    - 高性能推理
    """
    
    def __init__(self, model_loader):
        """初始化简化前向传播引擎
        
        Args:
            model_loader: 模型加载器实例
        """
        self.model_loader = model_loader
        self.model = model_loader.model
        self.device = next(self.model.parameters()).device
        self.model_config = model_loader.config
        
        # 统计信息
        self.total_forwards = 0
        self.total_forward_time = 0.0
        
        # 模型类型检测
        self.model_type = self._detect_model_type()
        
        logger.info(f"简化前向传播引擎初始化完成")
        logger.info(f"模型类型: {self.model_type}")
        logger.info(f"设备: {self.device}")
        logger.info(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _detect_model_type(self) -> str:
        """自动检测模型类型"""
        model_name = self.model_config.model_type.lower() if hasattr(self.model_config, 'model_type') else 'unknown'
        
        if 'qwen' in model_name:
            return 'qwen'
        elif 'llama' in model_name:
            return 'llama'
        elif 'gemma' in model_name:
            return 'gemma'
        elif 'mistral' in model_name:
            return 'mistral'
        else:
            logger.warning(f"未识别的模型类型: {model_name}，使用通用处理")
            return 'generic'
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                use_cache: bool = True,
                **kwargs) -> torch.Tensor:
        """执行前向传播
        
        Args:
            input_ids: 输入token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            use_cache: 是否使用KV缓存
            **kwargs: 其他参数
            
        Returns:
            torch.Tensor: 输出logits [batch_size, seq_len, vocab_size]
        """
        import time
        start_time = time.time()
        
        try:
            # 确保输入在正确设备上
            input_ids = input_ids.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            # 记录输入信息
            logger.debug(f"前向传播输入: input_ids={input_ids.shape}, device={input_ids.device}")
            if attention_mask is not None:
                logger.debug(f"attention_mask={attention_mask.shape}")
            
            # 准备模型输入
            model_inputs = {
                'input_ids': input_ids,
                'use_cache': use_cache,
            }
            
            if attention_mask is not None:
                model_inputs['attention_mask'] = attention_mask
            
            # 添加模型特定参数
            model_inputs.update(self._prepare_model_specific_inputs(**kwargs))
            
            # 执行前向传播
            with torch.no_grad():
                outputs = self.model(**model_inputs)
            
            # 提取logits
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                # 处理可能的元组输出
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            # 更新统计
            self.total_forwards += 1
            forward_time = time.time() - start_time
            self.total_forward_time += forward_time
            
            logger.debug(f"前向传播完成: 输出shape={logits.shape}, 耗时={forward_time:.3f}s")
            
            return logits
            
        except Exception as e:
            logger.error(f"前向传播失败: {e}")
            logger.error(f"输入信息: input_ids={input_ids.shape if torch.is_tensor(input_ids) else type(input_ids)}")
            if attention_mask is not None:
                logger.error(f"attention_mask={attention_mask.shape if torch.is_tensor(attention_mask) else type(attention_mask)}")
            raise
    
    def _prepare_model_specific_inputs(self, **kwargs) -> Dict[str, Any]:
        """准备模型特定的输入参数
        
        不同模型可能需要不同的参数格式
        """
        model_inputs = {}
        
        if self.model_type == 'gemma':
            # Gemma模型特殊处理
            # 注意：简化版本中我们依赖transformers的内部处理
            pass
        elif self.model_type == 'qwen':
            # Qwen模型特殊处理
            pass
        elif self.model_type == 'llama':
            # Llama模型特殊处理
            pass
        elif self.model_type == 'mistral':
            # Mistral模型特殊处理
            pass
        
        # 添加其他支持的参数
        supported_params = ['position_ids', 'past_key_values', 'head_mask', 'inputs_embeds']
        for param in supported_params:
            if param in kwargs:
                model_inputs[param] = kwargs[param]
        
        return model_inputs
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        avg_time = self.total_forward_time / max(1, self.total_forwards)
        return {
            'total_forwards': self.total_forwards,
            'total_time': self.total_forward_time,
            'avg_forward_time': avg_time,
            'model_type': self.model_type,
            'device': str(self.device)
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.total_forwards = 0
        self.total_forward_time = 0.0
    
    def warmup(self, input_shape=(1, 32)):
        """预热模型"""
        logger.info("开始模型预热...")
        vocab_size = getattr(self.model_loader.model.config, 'vocab_size', 32000)
        dummy_input = torch.randint(0, vocab_size, input_shape, device=self.device)
        dummy_mask = torch.ones(input_shape, device=self.device)
        
        try:
            with torch.no_grad():
                _ = self.forward(dummy_input, dummy_mask, use_cache=False)
            logger.info("模型预热完成")
        except Exception as e:
            logger.warning(f"模型预热失败: {e}") 