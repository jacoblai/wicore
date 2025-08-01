"""
WiCore Python3 推理引擎安装脚本
"""

from setuptools import setup, find_packages
import os

# 读取 README
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "WiCore: 世界级高性能LLM推理引擎"

# 读取 requirements
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return [
            "torch>=2.1.0",
            "transformers>=4.36.0",
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
            "numpy>=1.24.0",
            "psutil>=5.9.0",
            "pyyaml>=6.0",
            "loguru>=0.7.0",
        ]

setup(
    name="wicore",
    version="1.0.0",
    author="WiCore Team",
    author_email="team@wicore.ai",
    description="世界级高性能LLM推理引擎",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/wicore/wicore",
    
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    python_requires=">=3.8",
    install_requires=read_requirements(),
    
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "gpu": [
            "nvidia-ml-py>=12.0.0",
            "cupy-cuda12x>=12.0.0",
        ],
        "quantization": [
            "bitsandbytes>=0.41.0",
            "auto-gptq>=0.5.0",
        ],
        "compression": [
            "lz4>=4.0.0",
        ]
    },
    
    entry_points={
        "console_scripts": [
            "wicore=wicore.__main__:main",
        ],
    },
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    keywords="llm, inference, ai, machine-learning, deep-learning, pytorch, transformers",
    
    project_urls={
        "Bug Reports": "https://github.com/wicore/wicore/issues",
        "Source": "https://github.com/wicore/wicore",
        "Documentation": "https://wicore.readthedocs.io/",
    },
    
    include_package_data=True,
    zip_safe=False,
) 