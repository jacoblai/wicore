"""
WiCore主入口文件
支持命令行启动和配置管理

使用方法:
    python -m wicore --config configs/default.yaml
    python -m wicore --help
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
import signal

# from .core.engine import WiCoreEngine  # 简化架构暂不使用
from .core.config import get_config_manager, WiCoreConfig
from .server.api_server import start_server


def setup_logging(config: WiCoreConfig):
    """设置日志系统"""
    logging.basicConfig(
        level=getattr(logging, config.logging.level.upper()),
        format=config.logging.format,
        handlers=[]
    )
    
    # 控制台日志
    if config.logging.enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(config.logging.format))
        logging.getLogger().addHandler(console_handler)
    
    # 文件日志
    if config.logging.file_path:
        file_handler = logging.FileHandler(config.logging.file_path)
        file_handler.setFormatter(logging.Formatter(config.logging.format))
        logging.getLogger().addHandler(file_handler)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="WiCore: 世界级高性能LLM推理引擎",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="配置文件路径"
    )
    
    parser.add_argument(
        "--environment", "-e",
        type=str,
        default="development",
        choices=["development", "production", "test"],
        help="运行环境"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        help="服务器主机地址"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        help="服务器端口"
    )
    
    parser.add_argument(
        "--workers", "-w",
        type=int,
        help="工作进程数"
    )
    
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="启用调试模式"
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="WiCore 1.0.0"
    )
    
    return parser.parse_args()


async def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    config_manager = get_config_manager()
    
    if args.config:
        config = config_manager.load_config(args.config, args.environment)
    else:
        config = config_manager.load_config(environment=args.environment)
    
    # 应用命令行参数覆盖
    config_updates = {}
    if args.host:
        config_updates["server.host"] = args.host
    if args.port:
        config_updates["server.port"] = args.port
    if args.workers:
        config_updates["server.workers"] = args.workers
    if args.debug:
        config_updates["debug"] = True
        config_updates["logging.level"] = "DEBUG"
    
    if config_updates:
        config = config_manager.update_config(config_updates)
    
    # 设置日志
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 启动WiCore推理引擎...")
    logger.info(f"环境: {config.environment}")
    logger.info(f"配置: {config.config_path or '默认配置'}")
    
    # 启动API服务器
    try:
        logger.info("🌐 启动WiCore API服务器...")
        
        # 启动API服务器（自动集成推理引擎）
        await start_server(config)
            
    except KeyboardInterrupt:
        logger.info("用户中断，正在关闭...")
    except Exception as e:
        logger.error(f"引擎运行异常: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("👋 WiCore推理引擎已关闭")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"启动失败: {e}")
        sys.exit(1) 