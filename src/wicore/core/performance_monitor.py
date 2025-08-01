"""
WiCore性能监控器
实时监控和优化系统性能

核心功能:
- 实时性能指标收集
- 系统瓶颈识别
- 自动优化建议
- 性能趋势分析
"""

import torch
import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import psutil

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, monitor_interval: float = 5.0):
        self.monitor_interval = monitor_interval
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # 性能历史
        self.metrics_history = deque(maxlen=1000)
        self.alerts = deque(maxlen=100)
        
        # 性能阈值
        self.thresholds = {
            "cpu_usage": 80.0,      # CPU使用率阈值
            "memory_usage": 85.0,   # 内存使用率阈值
            "gpu_usage": 90.0,      # GPU使用率阈值
            "latency": 1.0,         # 延迟阈值（秒）
        }
        
        logger.info(f"性能监控器初始化: 监控间隔{monitor_interval}秒")
    
    def start_monitoring(self):
        """启动性能监控"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("性能监控已启动")
    
    def stop_monitoring(self):
        """停止性能监控"""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        logger.info("性能监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 收集性能指标
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # 检查告警
                self._check_alerts(metrics)
                
                # 等待下次监控
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"性能监控异常: {e}")
                time.sleep(self.monitor_interval)
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """收集性能指标"""
        timestamp = time.time()
        
        # CPU指标
        cpu_percent = psutil.cpu_percent()
        cpu_count = psutil.cpu_count()
        
        # 内存指标
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available = memory.available
        
        # GPU指标（如果可用）
        gpu_metrics = self._collect_gpu_metrics()
        
        metrics = {
            "timestamp": timestamp,
            "cpu": {
                "usage_percent": cpu_percent,
                "core_count": cpu_count,
            },
            "memory": {
                "usage_percent": memory_percent,
                "available_gb": memory_available / 1024 / 1024 / 1024,
                "total_gb": memory.total / 1024 / 1024 / 1024,
            },
            "gpu": gpu_metrics,
            "process": {
                "memory_mb": psutil.Process().memory_info().rss / 1024 / 1024,
                "cpu_percent": psutil.Process().cpu_percent(),
            }
        }
        
        return metrics
    
    def _collect_gpu_metrics(self) -> Dict[str, Any]:
        """收集GPU指标"""
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_metrics = {
                    "available": True,
                    "device_count": gpu_count,
                    "devices": []
                }
                
                for i in range(gpu_count):
                    device_props = torch.cuda.get_device_properties(i)
                    memory_allocated = torch.cuda.memory_allocated(i)
                    memory_reserved = torch.cuda.memory_reserved(i)
                    
                    device_info = {
                        "device_id": i,
                        "name": device_props.name,
                        "memory_allocated_mb": memory_allocated / 1024 / 1024,
                        "memory_reserved_mb": memory_reserved / 1024 / 1024,
                        "memory_total_mb": device_props.total_memory / 1024 / 1024,
                        "usage_percent": memory_allocated / device_props.total_memory * 100
                    }
                    
                    gpu_metrics["devices"].append(device_info)
                
                return gpu_metrics
            else:
                return {"available": False, "device_count": 0, "devices": []}
                
        except Exception as e:
            logger.warning(f"GPU指标收集失败: {e}")
            return {"available": False, "error": str(e)}
    
    def _check_alerts(self, metrics: Dict[str, Any]):
        """检查性能告警"""
        alerts = []
        
        # CPU告警
        if metrics["cpu"]["usage_percent"] > self.thresholds["cpu_usage"]:
            alerts.append({
                "type": "cpu_high",
                "severity": "warning",
                "message": f"CPU使用率过高: {metrics['cpu']['usage_percent']:.1f}%",
                "threshold": self.thresholds["cpu_usage"],
                "current": metrics["cpu"]["usage_percent"]
            })
        
        # 内存告警
        if metrics["memory"]["usage_percent"] > self.thresholds["memory_usage"]:
            alerts.append({
                "type": "memory_high",
                "severity": "warning",
                "message": f"内存使用率过高: {metrics['memory']['usage_percent']:.1f}%",
                "threshold": self.thresholds["memory_usage"],
                "current": metrics["memory"]["usage_percent"]
            })
        
        # GPU告警
        if metrics["gpu"]["available"]:
            for device in metrics["gpu"]["devices"]:
                if device["usage_percent"] > self.thresholds["gpu_usage"]:
                    alerts.append({
                        "type": "gpu_high",
                        "severity": "warning",
                        "message": f"GPU {device['device_id']} 使用率过高: {device['usage_percent']:.1f}%",
                        "threshold": self.thresholds["gpu_usage"],
                        "current": device["usage_percent"]
                    })
        
        # 记录告警
        for alert in alerts:
            alert["timestamp"] = time.time()
            self.alerts.append(alert)
            logger.warning(alert["message"])
    
    def get_current_metrics(self) -> Optional[Dict[str, Any]]:
        """获取当前性能指标"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_metrics_history(self, duration_seconds: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取性能指标历史"""
        if duration_seconds is None:
            return list(self.metrics_history)
        
        cutoff_time = time.time() - duration_seconds
        return [
            metrics for metrics in self.metrics_history
            if metrics["timestamp"] >= cutoff_time
        ]
    
    def get_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取告警信息"""
        alerts = list(self.alerts)
        
        if severity:
            alerts = [alert for alert in alerts if alert["severity"] == severity]
        
        return alerts
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        current_metrics = self.get_current_metrics()
        if not current_metrics:
            return {"status": "no_data"}
        
        recent_alerts = self.get_alerts()
        warning_count = len([a for a in recent_alerts if a["severity"] == "warning"])
        
        # 计算性能评分
        performance_score = self._calculate_performance_score(current_metrics)
        
        return {
            "status": "active" if self.monitoring_active else "inactive",
            "current_metrics": current_metrics,
            "performance_score": performance_score,
            "alert_count": len(recent_alerts),
            "warning_count": warning_count,
            "recommendations": self._generate_recommendations(current_metrics)
        }
    
    def _calculate_performance_score(self, metrics: Dict[str, Any]) -> float:
        """计算性能评分（0-100）"""
        score = 100.0
        
        # CPU评分
        cpu_usage = metrics["cpu"]["usage_percent"]
        if cpu_usage > 80:
            score -= (cpu_usage - 80) * 2
        
        # 内存评分
        memory_usage = metrics["memory"]["usage_percent"]
        if memory_usage > 80:
            score -= (memory_usage - 80) * 2
        
        # GPU评分
        if metrics["gpu"]["available"]:
            for device in metrics["gpu"]["devices"]:
                gpu_usage = device["usage_percent"]
                if gpu_usage > 90:
                    score -= (gpu_usage - 90)
        
        return max(0.0, score)
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # CPU建议
        if metrics["cpu"]["usage_percent"] > 80:
            recommendations.append("考虑减少并发请求数量或增加CPU资源")
        
        # 内存建议
        if metrics["memory"]["usage_percent"] > 80:
            recommendations.append("考虑启用内存压缩或增加系统内存")
        
        # GPU建议
        if metrics["gpu"]["available"]:
            for device in metrics["gpu"]["devices"]:
                if device["usage_percent"] > 90:
                    recommendations.append(f"GPU {device['device_id']} 使用率过高，考虑启用模型量化或分布式推理")
        
        if not recommendations:
            recommendations.append("系统性能良好，无需特殊优化")
        
        return recommendations 