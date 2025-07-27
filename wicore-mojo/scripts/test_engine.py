#!/usr/bin/env python3
"""
WiCore Mojo 推理引擎测试脚本
验证所有核心组件的功能和集成
"""

import sys
import os
import time
import json
import subprocess
from typing import Dict, Any

# 添加模拟环境到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulation'))

def test_environment():
    """测试基础环境"""
    print("🔧 测试环境配置...")
    
    # 测试 Python 版本
    print(f"Python 版本: {sys.version}")
    
    # 测试基础包
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU 数量: {torch.cuda.device_count()}")
    except ImportError:
        print("❌ PyTorch 未安装")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers 未安装")
        return False
    
    return True

def test_modular_integration():
    """测试 Modular 集成"""
    print("\n🔧 测试 Modular 集成...")
    
    try:
        # 尝试导入模拟的 max 模块
        import max_simulation as max_sim
        engine = max_sim.engine
        
        print("✅ MAX Engine 模拟环境导入成功")
        
        # 测试设备发现
        devices = engine.discover_devices()
        print(f"✅ 发现 {len(devices)} 个设备")
        for device in devices:
            print(f"   - {device.type}: {device.id}")
        
        return True
    except Exception as e:
        print(f"❌ Modular 集成测试失败: {e}")
        return False

def test_configuration():
    """测试配置文件"""
    print("\n📋 测试配置文件...")
    
    try:
        with open('configs/production.json', 'r') as f:
            prod_config = json.load(f)
        print("✅ 生产配置文件加载成功")
        
        with open('configs/development.json', 'r') as f:
            dev_config = json.load(f)
        print("✅ 开发配置文件加载成功")
        
        return True
    except Exception as e:
        print(f"❌ 配置文件测试失败: {e}")
        return False

def test_pixi_environment():
    """测试 Pixi 环境"""
    print("\n📦 测试 Pixi 环境...")
    
    try:
        # 检查 pixi.toml 文件
        if os.path.exists('pixi.toml'):
            print("✅ pixi.toml 文件存在")
        else:
            print("⚠️  pixi.toml 文件不存在")
        
        # 测试 pixi 命令
        result = subprocess.run(['pixi', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Pixi 版本: {result.stdout.strip()}")
            return True
        else:
            print("❌ Pixi 命令不可用")
            return False
    except Exception as e:
        print(f"❌ Pixi 环境测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始 WiCore Mojo 推理引擎测试...")
    print("=" * 50)
    
    tests = [
        ("环境配置", test_environment),
        ("Modular 集成", test_modular_integration),
        ("配置文件", test_configuration),
        ("Pixi 环境", test_pixi_environment),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 测试通过")
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"🎯 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！WiCore 环境配置成功")
        return 0
    else:
        print("⚠️  部分测试失败，请检查环境配置")
        return 1

if __name__ == "__main__":
    sys.exit(main())
