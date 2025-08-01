"""
简化的文本生成器

使用标准transformers生成方法，移除多设备复杂性
支持多种模型架构的高效文本生成
"""

import torch
import logging
from typing import Optional, List, Dict, Any, Union
from transformers import GenerationConfig
import time

logger = logging.getLogger(__name__)


class SimpleGenerator:
    """简化的文本生成器
    
    特点：
    - 使用标准transformers生成方法
    - 单GPU高效推理
    - 支持多种生成策略
    - 保持API兼容性
    """
    
    def __init__(self, model_loader, simple_forward=None):
        """初始化简化生成器
        
        Args:
            model_loader: 简化模型加载器实例
            simple_forward: 简化前向传播引擎（可选）
        """
        self.model_loader = model_loader
        self.model = model_loader.model
        self.tokenizer = model_loader.tokenizer
        self.device = model_loader.device
        self.generation_config = model_loader.generation_config
        self.simple_forward = simple_forward
        
        # 统计信息
        self.total_generations = 0
        self.total_tokens_generated = 0
        self.total_generation_time = 0.0
        
        logger.info(f"简化生成器初始化完成")
        logger.info(f"设备: {self.device}")
    
    def generate(self, 
                 input_ids: torch.Tensor,
                 attention_mask: Optional[torch.Tensor] = None,
                 max_new_tokens: int = 100,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 top_k: int = 50,
                 do_sample: bool = True,
                 repetition_penalty: float = 1.1,
                 use_cache: bool = True,
                 **kwargs) -> torch.Tensor:
        """生成文本
        
        Args:
            input_ids: 输入token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            top_p: nucleus采样参数
            top_k: top-k采样参数
            do_sample: 是否使用采样
            repetition_penalty: 重复惩罚
            use_cache: 是否使用KV缓存
            **kwargs: 其他生成参数
            
        Returns:
            torch.Tensor: 生成的token IDs [batch_size, seq_len + new_tokens]
        """
        start_time = time.time()
        
        try:
            # 确保输入在正确设备上
            input_ids = input_ids.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            # 记录输入信息
            batch_size, input_length = input_ids.shape
            logger.debug(f"生成请求: batch_size={batch_size}, input_length={input_length}")
            logger.debug(f"生成参数: max_new_tokens={max_new_tokens}, temperature={temperature}")
            
            # 准备生成配置
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                use_cache=use_cache,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=False,
                output_scores=False,
                **kwargs
            )
            
            # 准备模型输入
            model_inputs = {
                'input_ids': input_ids,
                'generation_config': generation_config,
                'return_dict_in_generate': False,
            }
            
            if attention_mask is not None:
                model_inputs['attention_mask'] = attention_mask
            
            # 执行生成
            with torch.no_grad():
                generated_ids = self.model.generate(**model_inputs)
            
            # 更新统计
            self.total_generations += 1
            new_tokens = generated_ids.shape[1] - input_length
            self.total_tokens_generated += new_tokens * batch_size
            generation_time = time.time() - start_time
            self.total_generation_time += generation_time
            
            # 计算速度
            tokens_per_second = (new_tokens * batch_size) / generation_time
            
            logger.debug(f"生成完成: 新增tokens={new_tokens}, 耗时={generation_time:.3f}s")
            logger.debug(f"生成速度: {tokens_per_second:.1f} tokens/s")
            
            return generated_ids
            
        except Exception as e:
            logger.error(f"文本生成失败: {e}")
            logger.error(f"输入信息: input_ids={input_ids.shape if torch.is_tensor(input_ids) else type(input_ids)}")
            if attention_mask is not None:
                logger.error(f"attention_mask={attention_mask.shape if torch.is_tensor(attention_mask) else type(attention_mask)}")
            raise
    
    def generate_text(self,
                      text: str,
                      max_new_tokens: int = 100,
                      temperature: float = 0.7,
                      top_p: float = 0.9,
                      top_k: int = 50,
                      do_sample: bool = True,
                      repetition_penalty: float = 1.1,
                      **kwargs) -> str:
        """从文本生成新文本
        
        Args:
            text: 输入文本
            其他参数同generate方法
            
        Returns:
            str: 生成的文本
        """
        try:
            # 编码输入文本
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=min(2048, max(512, self.generation_config.max_length - max_new_tokens)) if hasattr(self.generation_config, 'max_length') else 1024
            )
            
            input_ids = inputs['input_ids']
            attention_mask = inputs.get('attention_mask', None)
            
            # 生成
            generated_ids = self.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                **kwargs
            )
            
            # 解码生成的文本
            # 只返回新生成的部分
            new_tokens = generated_ids[:, input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(
                new_tokens[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            return generated_text
            
        except Exception as e:
            logger.error(f"文本生成失败: {e}")
            raise
    
    def chat_generate(self,
                      messages: List[Dict[str, str]],
                      max_new_tokens: int = 100,
                      temperature: float = 0.7,
                      top_p: float = 0.9,
                      **kwargs) -> str:
        """对话生成
        
        Args:
            messages: 对话消息列表 [{"role": "user", "content": "..."}, ...]
            其他参数同generate方法
            
        Returns:
            str: 生成的回复
        """
        try:
            # 将对话转换为文本
            if hasattr(self.tokenizer, 'apply_chat_template'):
                # 使用模型的聊天模板
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # 简单的文本拼接
                text = ""
                for msg in messages:
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    if role == 'user':
                        text += f"User: {content}\n"
                    elif role == 'assistant':
                        text += f"Assistant: {content}\n"
                text += "Assistant: "
            
            # 生成回复
            generated_text = self.generate_text(
                text=text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
            
            return generated_text
            
        except Exception as e:
            logger.error(f"对话生成失败: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """获取生成统计"""
        avg_time = self.total_generation_time / max(1, self.total_generations)
        avg_tokens_per_gen = self.total_tokens_generated / max(1, self.total_generations)
        overall_speed = self.total_tokens_generated / max(0.001, self.total_generation_time)
        
        return {
            'total_generations': self.total_generations,
            'total_tokens_generated': self.total_tokens_generated,
            'total_time': self.total_generation_time,
            'avg_time_per_generation': avg_time,
            'avg_tokens_per_generation': avg_tokens_per_gen,
            'overall_tokens_per_second': overall_speed,
            'device': str(self.device)
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.total_generations = 0
        self.total_tokens_generated = 0
        self.total_generation_time = 0.0
    
    def warmup(self, warmup_text: str = "Hello world"):
        """预热生成器"""
        logger.info("开始生成器预热...")
        try:
            _ = self.generate_text(
                text=warmup_text,
                max_new_tokens=5,
                do_sample=False
            )
            logger.info("生成器预热完成")
        except Exception as e:
            logger.warning(f"生成器预热失败: {e}") 