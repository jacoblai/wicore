"""
WiCore服务器模块
提供HTTP API服务和WebSocket支持

核心组件:
- FastAPI服务器：RESTful API接口
- 请求调度器：智能请求调度和负载均衡
- 监控接口：性能监控和管理接口
"""

from .api_server import create_app, start_server

__all__ = [
    "create_app",
    "start_server",
] 