# WiCore Mojo 生产环境部署指南

## 概述

本文档详细描述了 WiCore Mojo 推理引擎在生产环境中的部署流程，包括系统准备、软件安装、配置优化和运维管理。

## 🛠️ 系统要求

### 硬件要求

#### 最低配置 (开发/测试)
- **CPU**: 8 核心 Intel/AMD 处理器
- **内存**: 32GB DDR4
- **GPU**: 单张 NVIDIA RTX 4090 或同等性能
- **存储**: 500GB NVMe SSD
- **网络**: 1Gbps 以太网

#### 推荐配置 (生产环境)
- **CPU**: 32 核心 Intel Xeon 或 AMD EPYC
- **内存**: 128GB DDR4/DDR5
- **GPU**: 2×NVIDIA T10 (32GB) 或 A100 (80GB)
- **存储**: 2TB NVMe SSD (PCIe 4.0)
- **网络**: 10Gbps 以太网，InfiniBand 可选

#### 高性能配置 (大规模部署)
- **CPU**: 64 核心服务器处理器
- **内存**: 512GB DDR5
- **GPU**: 4×NVIDIA H100 (80GB)
- **存储**: 10TB NVMe SSD 阵列
- **网络**: 100Gbps InfiniBand

### 软件要求

#### 操作系统
- **Ubuntu 20.04/22.04 LTS** (推荐)
- **CentOS 8/Rocky Linux 8**
- **RHEL 8/9**

#### 必需软件
- **Python 3.8+**
- **NVIDIA Driver 525+**
- **CUDA 11.8+**
- **Docker 20.10+** (可选)
- **Modular SDK** (最新版)

## 📋 部署前准备

### 1. 系统环境检查

```bash
# 检查操作系统
cat /etc/os-release

# 检查 CPU 信息
lscpu | grep -E "Architecture|CPU|Thread|Core|Socket"

# 检查内存
free -h

# 检查存储
lsblk
df -h

# 检查网络
ip addr show
```

### 2. GPU 环境验证

```bash
# 检查 GPU 硬件
lspci | grep -i nvidia

# 安装 NVIDIA 驱动
sudo apt update
sudo apt install nvidia-driver-525 nvidia-dkms-525

# 验证驱动安装
nvidia-smi

# 检查 CUDA 版本
nvcc --version
```

### 3. 网络配置

```bash
# 配置防火墙
sudo ufw allow 8000/tcp
sudo ufw enable

# 优化网络参数
echo 'net.core.rmem_max = 268435456' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 268435456' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_rmem = 4096 65536 16777216' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_wmem = 4096 65536 16777216' >> /etc/sysctl.conf
sysctl -p
```

## 🚀 自动化部署

### 使用部署脚本

```bash
# 1. 克隆代码仓库
git clone <repository_url>
cd wicore-mojo

# 2. 运行自动化部署
sudo ./scripts/deploy.sh

# 3. 检查部署状态
systemctl status wicore-engine
```

### 使用 Docker 部署 (推荐)

#### 构建 Docker 镜像

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# 设置工作目录
WORKDIR /opt/wicore

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3 python3-pip curl wget \
    && rm -rf /var/lib/apt/lists/*

# 安装 Modular SDK
RUN curl -s https://get.modular.com | sh - \
    && /root/.modular/bin/modular install max

# 复制应用文件
COPY build/ ./bin/
COPY configs/ ./configs/
COPY simulation/ ./simulation/

# 设置环境变量
ENV PATH="/root/.modular/bin:$PATH"
ENV WICORE_CONFIG_PATH="/opt/wicore/configs/production.json"

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["./bin/start_wicore.sh", "configs/production.json"]
```

#### 使用 docker-compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  wicore-engine:
    build: .
    container_name: wicore-mojo
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - ./models:/opt/wicore/models:ro
      - ./logs:/opt/wicore/logs
      - ./cache:/opt/wicore/cache
    environment:
      - NVIDIA_VISIBLE_DEVICES=0,1
      - CUDA_VISIBLE_DEVICES=0,1
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  nginx:
    image: nginx:alpine
    container_name: wicore-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - wicore-engine
```

## ⚙️ 配置管理

### 生产环境配置

```json
{
  "model_path": "/opt/wicore/models/gemma-3-27b-it",
  "server_port": 8000,
  "max_batch_size": 16,
  "max_context_length": 131072,
  "gpu_memory_limit_gb": 15.0,
  "enable_multi_gpu": true,
  "target_devices": ["gpu:0", "gpu:1"],
  "simulation_mode": false,
  
  "hmt_config": {
    "enable_a2cr": true,
    "nvme_cache_path": "/opt/wicore/cache",
    "time_decay_factor": 0.05,
    "attention_weight": 0.4,
    "frequency_weight": 0.3,
    "recency_weight": 0.3,
    "max_cache_size_gb": 100
  },
  
  "performance_config": {
    "enable_batching": true,
    "batch_timeout_ms": 50,
    "max_concurrent_requests": 32,
    "request_timeout_seconds": 30,
    "enable_streaming": true
  },
  
  "security_config": {
    "enable_auth": true,
    "api_key_required": true,
    "rate_limit_per_minute": 100,
    "cors_enabled": true,
    "allowed_origins": ["*"]
  },
  
  "logging_config": {
    "level": "INFO",
    "enable_request_logging": true,
    "enable_performance_logging": true,
    "log_file": "/opt/wicore/logs/wicore.log",
    "max_log_size_mb": 100,
    "log_retention_days": 30
  },
  
  "monitoring_config": {
    "enable_metrics": true,
    "metrics_port": 9090,
    "health_check_interval": 30,
    "alert_thresholds": {
      "memory_usage": 0.9,
      "gpu_utilization": 0.95,
      "error_rate": 0.05,
      "latency_p95": 1000
    }
  }
}
```

### 环境变量配置

```bash
# /etc/environment
WICORE_CONFIG_PATH="/opt/wicore/configs/production.json"
WICORE_MODEL_PATH="/opt/wicore/models"
WICORE_CACHE_PATH="/opt/wicore/cache"
WICORE_LOG_LEVEL="INFO"
CUDA_VISIBLE_DEVICES="0,1"
NVIDIA_VISIBLE_DEVICES="0,1"
```

## 🔧 模型部署

### Gemma-3-27B 模型下载

```bash
# 1. 创建模型目录
sudo mkdir -p /opt/wicore/models
sudo chown wicore:wicore /opt/wicore/models

# 2. 下载模型 (需要 HuggingFace 账户)
cd /opt/wicore/models
wget https://huggingface.co/google/gemma-2-27b-it/resolve/main/config.json
wget https://huggingface.co/google/gemma-2-27b-it/resolve/main/tokenizer.json
# ... 下载所有模型文件

# 3. 验证模型完整性
ls -la gemma-3-27b-it/
du -sh gemma-3-27b-it/
```

### 模型格式转换 (如需要)

```bash
# 转换为 Mojo 优化格式
/opt/wicore/bin/model_converter \
  --input /opt/wicore/models/gemma-3-27b-it \
  --output /opt/wicore/models/gemma-3-27b-it-optimized \
  --format mojo \
  --precision fp16 \
  --enable_quantization
```

## 🚦 服务管理

### Systemd 服务配置

```ini
# /etc/systemd/system/wicore-engine.service
[Unit]
Description=WiCore Mojo AI Inference Engine
After=network.target nvidia-persistenced.service

[Service]
Type=simple
User=wicore
Group=wicore
WorkingDirectory=/opt/wicore/bin
ExecStart=/opt/wicore/bin/start_wicore.sh /opt/wicore/configs/production.json
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=5
TimeoutStopSec=30

# 环境变量
Environment=PATH=/root/.modular/bin:/usr/local/bin:/usr/bin:/bin
Environment=CUDA_VISIBLE_DEVICES=0,1
Environment=NVIDIA_VISIBLE_DEVICES=0,1

# 资源限制
LimitNOFILE=65536
LimitNPROC=32768

# 日志配置
StandardOutput=append:/opt/wicore/logs/wicore.log
StandardError=append:/opt/wicore/logs/wicore.error
SyslogIdentifier=wicore-engine

# 安全设置
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/wicore/logs /opt/wicore/cache

[Install]
WantedBy=multi-user.target
```

### 服务操作

```bash
# 启用服务
sudo systemctl enable wicore-engine

# 启动服务
sudo systemctl start wicore-engine

# 查看状态
sudo systemctl status wicore-engine

# 查看日志
sudo journalctl -u wicore-engine -f

# 重启服务
sudo systemctl restart wicore-engine

# 停止服务
sudo systemctl stop wicore-engine
```

## 📊 监控和告警

### Prometheus 监控

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'wicore-engine'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: /metrics
    scrape_interval: 10s
```

### Grafana 仪表板

```json
{
  "dashboard": {
    "title": "WiCore Mojo 监控面板",
    "panels": [
      {
        "title": "请求量 (QPS)",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(wicore_requests_total[5m])"
          }
        ]
      },
      {
        "title": "响应延迟",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, wicore_request_duration_seconds)"
          }
        ]
      },
      {
        "title": "GPU 利用率",
        "type": "graph",
        "targets": [
          {
            "expr": "nvidia_gpu_utilization_gpu"
          }
        ]
      },
      {
        "title": "内存使用率",
        "type": "graph",
        "targets": [
          {
            "expr": "wicore_memory_usage_ratio"
          }
        ]
      }
    ]
  }
}
```

### 告警规则

```yaml
# alerts.yml
groups:
  - name: wicore-alerts
    rules:
      - alert: WiCoreHighLatency
        expr: histogram_quantile(0.95, wicore_request_duration_seconds) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "WiCore 响应延迟过高"
          
      - alert: WiCoreHighErrorRate  
        expr: rate(wicore_requests_failed_total[5m]) / rate(wicore_requests_total[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "WiCore 错误率过高"
          
      - alert: WiCoreServiceDown
        expr: up{job="wicore-engine"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "WiCore 服务不可用"
```

## 🔍 故障排除

### 常见问题诊断

#### 1. 服务启动失败

```bash
# 检查服务状态
systemctl status wicore-engine

# 查看详细日志
journalctl -u wicore-engine --no-pager

# 检查配置文件
/opt/wicore/bin/wicore_engine --config /opt/wicore/configs/production.json --validate

# 检查端口占用
ss -tlnp | grep 8000

# 检查文件权限
ls -la /opt/wicore/
```

#### 2. 模型加载失败

```bash
# 检查模型文件
ls -la /opt/wicore/models/gemma-3-27b-it/

# 检查 GPU 内存
nvidia-smi

# 检查存储空间
df -h /opt/wicore/

# 验证模型格式
python3 -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('/opt/wicore/models/gemma-3-27b-it')
print('Model loaded successfully')
"
```

#### 3. 性能问题

```bash
# 检查 GPU 利用率
nvidia-smi dmon -s u

# 检查 CPU 使用率
top -p $(pgrep wicore_engine)

# 检查内存使用
cat /proc/$(pgrep wicore_engine)/status | grep VmRSS

# 检查网络连接
ss -tuln | grep 8000
```

### 日志分析

```bash
# 分析错误日志
grep ERROR /opt/wicore/logs/wicore.log | tail -100

# 分析性能日志
grep "latency\|throughput" /opt/wicore/logs/wicore.log

# 分析请求模式
awk '/request_id/ {print $1, $4, $7}' /opt/wicore/logs/wicore.log | tail -50
```

## 📈 扩展部署

### 负载均衡配置

```nginx
# nginx.conf
upstream wicore_backend {
    least_conn;
    server 10.0.1.10:8000 weight=1 max_fails=3 fail_timeout=30s;
    server 10.0.1.11:8000 weight=1 max_fails=3 fail_timeout=30s;
    server 10.0.1.12:8000 weight=1 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name api.wicore.example.com;
    
    location / {
        proxy_pass http://wicore_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

### Kubernetes 部署

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: wicore-engine
spec:
  replicas: 3
  selector:
    matchLabels:
      app: wicore-engine
  template:
    metadata:
      labels:
        app: wicore-engine
    spec:
      containers:
      - name: wicore-engine
        image: wicore/mojo-engine:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            nvidia.com/gpu: 2
            memory: 32Gi
            cpu: 8
          limits:
            nvidia.com/gpu: 2
            memory: 64Gi
            cpu: 16
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0,1"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: wicore-service
spec:
  selector:
    app: wicore-engine
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

## 🛡️ 安全配置

### SSL/TLS 配置

```nginx
server {
    listen 443 ssl http2;
    server_name api.wicore.example.com;
    
    ssl_certificate /etc/nginx/ssl/wicore.crt;
    ssl_certificate_key /etc/nginx/ssl/wicore.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    
    location / {
        proxy_pass http://wicore_backend;
        proxy_ssl_verify off;
    }
}
```

### API 认证

```json
{
  "security_config": {
    "enable_auth": true,
    "auth_type": "api_key",
    "api_keys": [
      {
        "key": "sk-1234567890abcdef",
        "name": "production-key",
        "permissions": ["chat", "models"],
        "rate_limit": 1000
      }
    ]
  }
}
```

通过以上部署指南，您可以在生产环境中成功部署和运维 WiCore Mojo 推理引擎，实现高性能、高可用的 AI 推理服务。 