#!/usr/bin/env python3
"""
简单的Qwen2.5-7B下载脚本
从魔塔(ModelScope)下载，适合直接执行
"""

import os
import sys
from pathlib import Path

def download_qwen25():
    """下载Qwen2.5-7B模型"""
    
    print("🚀 开始下载Qwen2.5-7B-Instruct...")
    print("📋 来源: 魔塔 ModelScope")
    print("📦 大小: ~15GB")
    print("⏰ 预计时间: 10-30分钟")
    print("=" * 50)
    
    try:
        # 导入ModelScope
        from modelscope import snapshot_download
        
        # 创建目录
        model_dir = Path("models/Qwen2.5-7B-Instruct")
        model_dir.parent.mkdir(exist_ok=True)
        
        # 如果目录已存在且有文件，询问是否重新下载
        if model_dir.exists() and any(model_dir.iterdir()):
            print(f"⚠️  目录已存在: {model_dir}")
            choice = input("是否重新下载? (y/n): ").lower()
            if choice != 'y':
                print("❌ 取消下载")
                return False
            
            # 清理目录
            import shutil
            shutil.rmtree(model_dir)
            print("✅ 已清理旧文件")
        
        print(f"📁 下载到: {model_dir}")
        print("⏬ 开始下载...")
        
        # 执行下载
        downloaded_path = snapshot_download(
            model_id='qwen/Qwen2.5-7B-Instruct',
            cache_dir=str(model_dir.parent),
            local_dir=str(model_dir)
        )
        
        print(f"\n✅ 下载完成!")
        print(f"📁 模型位置: {downloaded_path}")
        
        # 验证文件
        verify_files(model_dir)
        
        # 更新配置
        update_config(model_dir)
        
        print("\n🎉 Qwen2.5-7B下载并配置完成!")
        print("📝 下一步:")
        print("   python3 test_qwen25_modelscope.py  # 测试模型")
        
        return True
        
    except ImportError:
        print("❌ ModelScope未安装!")
        print("💡 请先运行: pip3 install modelscope")
        return False
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("💡 建议:")
        print("   1. 检查网络连接")
        print("   2. 重新运行此脚本")
        print("   3. 或者尝试手动下载")
        return False

def verify_files(model_dir):
    """验证下载的文件"""
    print("\n🔍 验证文件...")
    
    # 必需文件
    required_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json"
    ]
    
    for file_name in required_files:
        file_path = model_dir / file_name
        if file_path.exists():
            size = file_path.stat().st_size / 1024
            print(f"✅ {file_name} ({size:.1f}KB)")
        else:
            print(f"❌ {file_name} 缺失")
    
    # 检查模型文件
    model_files = list(model_dir.glob("*.safetensors"))
    if model_files:
        total_size = sum(f.stat().st_size for f in model_files) / (1024**3)
        print(f"✅ {len(model_files)} 个模型文件 ({total_size:.1f}GB)")
    else:
        print("❌ 模型文件缺失")

def update_config(model_dir):
    """更新配置文件"""
    config_file = Path("configs/qwen25_7b.yaml")
    
    if not config_file.exists():
        print(f"⚠️  配置文件不存在: {config_file}")
        return
    
    try:
        # 读取配置
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 更新路径
        old_path = 'model_path: "/path/to/Qwen2.5-7B-Instruct"'
        new_path = f'model_path: "{model_dir}"'
        
        if old_path in content:
            content = content.replace(old_path, new_path)
            
            # 写回文件
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ 配置文件已更新: {config_file}")
        else:
            print("⚠️  配置文件格式可能已变化")
            
    except Exception as e:
        print(f"⚠️  配置文件更新失败: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("Qwen2.5-7B 模型下载器")
    print("专为WiCore简化架构设计")
    print("=" * 60)
    
    success = download_qwen25()
    
    if success:
        print("\n🎊 下载成功!")
    else:
        print("\n😞 下载失败")
        print("💬 如有问题，请检查网络或重试")
    
    print("\n👋 脚本结束") 