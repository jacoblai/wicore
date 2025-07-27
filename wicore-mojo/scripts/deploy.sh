#!/bin/bash

# WiCore Mojo 推理引擎部署脚本
# 用于在生产服务器上自动化部署

set -e

# 配置参数
DEPLOY_USER=${DEPLOY_USER:-"wicore"}
DEPLOY_PATH=${DEPLOY_PATH:-"/opt/wicore"}
SERVICE_NAME="wicore-engine"
CONFIG_FILE="production.json"

echo "🚀 WiCore Mojo 推理引擎生产环境部署"
echo "=" * 50

# 检查运行权限
if [ "$EUID" -ne 0 ]; then
    echo "❌ 请使用 root 权限运行部署脚本"
    exit 1
fi

# 检查系统环境
echo "🔍 检查系统环境..."

# 检查操作系统
if [ ! -f /etc/os-release ]; then
    echo "❌ 不支持的操作系统"
    exit 1
fi

OS_NAME=$(grep ^NAME /etc/os-release | cut -d= -f2 | tr -d '"')
echo "✅ 操作系统: $OS_NAME"

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU 驱动已安装"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  未检测到 NVIDIA GPU，请确保驱动已正确安装"
fi

# 创建部署用户
echo "👤 创建部署用户..."
if ! id "$DEPLOY_USER" &>/dev/null; then
    useradd -r -s /bin/bash -d "$DEPLOY_PATH" "$DEPLOY_USER"
    echo "✅ 用户 $DEPLOY_USER 创建成功"
else
    echo "✅ 用户 $DEPLOY_USER 已存在"
fi

# 创建部署目录
echo "📁 创建部署目录..."
mkdir -p "$DEPLOY_PATH"/{bin,configs,models,logs,cache}
chown -R "$DEPLOY_USER:$DEPLOY_USER" "$DEPLOY_PATH"
chmod 755 "$DEPLOY_PATH"

echo "✅ 部署目录: $DEPLOY_PATH"

# 安装系统依赖
echo "📦 安装系统依赖..."
if command -v apt-get &> /dev/null; then
    # Ubuntu/Debian
    apt-get update
    apt-get install -y python3 python3-pip curl wget htop
elif command -v yum &> /dev/null; then
    # CentOS/RHEL
    yum update -y
    yum install -y python3 python3-pip curl wget htop
else
    echo "⚠️  不支持的包管理器，请手动安装依赖"
fi

# 安装 Modular SDK
echo "🔧 安装 Modular SDK..."
if ! command -v modular &> /dev/null; then
    echo "正在下载 Modular SDK..."
    curl -s https://get.modular.com | sh -
    
    # 添加到系统 PATH
    echo 'export PATH="/root/.modular/bin:$PATH"' >> /etc/profile
    source /etc/profile
    
    # 安装 MAX Engine
    /root/.modular/bin/modular install max
    
    echo "✅ Modular SDK 安装完成"
else
    echo "✅ Modular SDK 已安装"
fi

# 复制应用文件
echo "📋 部署应用文件..."
if [ -d "build" ]; then
    cp -r build/* "$DEPLOY_PATH/bin/"
    cp -r configs/* "$DEPLOY_PATH/configs/"
    
    # 如果有模拟环境，也复制过去
    if [ -d "simulation" ]; then
        cp -r simulation "$DEPLOY_PATH/"
    fi
    
    echo "✅ 应用文件部署完成"
else
    echo "❌ 构建目录不存在，请先运行 build.sh"
    exit 1
fi

# 设置文件权限
chown -R "$DEPLOY_USER:$DEPLOY_USER" "$DEPLOY_PATH"
chmod +x "$DEPLOY_PATH/bin/wicore_engine"
chmod +x "$DEPLOY_PATH/bin/start_wicore.sh"

# 创建系统服务
echo "⚙️  创建系统服务..."
cat > /etc/systemd/system/$SERVICE_NAME.service << EOF
[Unit]
Description=WiCore Mojo AI Inference Engine
After=network.target

[Service]
Type=simple
User=$DEPLOY_USER
WorkingDirectory=$DEPLOY_PATH/bin
ExecStart=$DEPLOY_PATH/bin/start_wicore.sh $DEPLOY_PATH/configs/$CONFIG_FILE
Restart=always
RestartSec=5
Environment=PATH=/root/.modular/bin:/usr/local/bin:/usr/bin:/bin
StandardOutput=append:$DEPLOY_PATH/logs/wicore.log
StandardError=append:$DEPLOY_PATH/logs/wicore.error

[Install]
WantedBy=multi-user.target
EOF

# 重载 systemd 配置
systemctl daemon-reload
systemctl enable $SERVICE_NAME

echo "✅ 系统服务 $SERVICE_NAME 创建完成"

# 创建配置模板
echo "📄 创建生产配置..."
cat > "$DEPLOY_PATH/configs/production.json" << EOF
{
    "model_path": "$DEPLOY_PATH/models/gemma-3-27b-it",
    "server_port": 8000,
    "max_batch_size": 16,
    "max_context_length": 131072,
    "gpu_memory_limit_gb": 15.0,
    "enable_multi_gpu": true,
    "target_devices": ["gpu:0", "gpu:1"],
    "simulation_mode": false,
    "hmt_config": {
        "enable_a2cr": true,
        "nvme_cache_path": "$DEPLOY_PATH/cache",
        "time_decay_factor": 0.05,
        "attention_weight": 0.4,
        "frequency_weight": 0.3,
        "recency_weight": 0.3
    },
    "logging": {
        "level": "INFO",
        "file": "$DEPLOY_PATH/logs/wicore.log"
    }
}
EOF

# 创建管理脚本
echo "🛠️  创建管理脚本..."
cat > "$DEPLOY_PATH/manage.sh" << 'EOF'
#!/bin/bash

# WiCore 引擎管理脚本

SERVICE_NAME="wicore-engine"
DEPLOY_PATH="/opt/wicore"

case "$1" in
    start)
        echo "🚀 启动 WiCore 引擎..."
        systemctl start $SERVICE_NAME
        ;;
    stop)
        echo "🛑 停止 WiCore 引擎..."
        systemctl stop $SERVICE_NAME
        ;;
    restart)
        echo "🔄 重启 WiCore 引擎..."
        systemctl restart $SERVICE_NAME
        ;;
    status)
        echo "📊 WiCore 引擎状态:"
        systemctl status $SERVICE_NAME
        ;;
    logs)
        echo "📝 查看实时日志:"
        journalctl -u $SERVICE_NAME -f
        ;;
    health)
        echo "🏥 健康检查:"
        curl -s http://localhost:8000/health | python3 -m json.tool
        ;;
    test)
        echo "🧪 API 测试:"
        curl -X POST http://localhost:8000/v1/chat/completions \
          -H "Content-Type: application/json" \
          -d '{
            "model": "gemma-3-27b-it",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10
          }' | python3 -m json.tool
        ;;
    *)
        echo "使用方法: $0 {start|stop|restart|status|logs|health|test}"
        exit 1
        ;;
esac
EOF

chmod +x "$DEPLOY_PATH/manage.sh"
chown "$DEPLOY_USER:$DEPLOY_USER" "$DEPLOY_PATH/manage.sh"

# 设置防火墙
echo "🔥 配置防火墙..."
if command -v ufw &> /dev/null; then
    ufw allow 8000/tcp
    echo "✅ UFW 防火墙规则已添加"
elif command -v firewall-cmd &> /dev/null; then
    firewall-cmd --permanent --add-port=8000/tcp
    firewall-cmd --reload
    echo "✅ Firewalld 防火墙规则已添加"
fi

# 完成部署
echo ""
echo "🎉 WiCore Mojo 推理引擎部署完成！"
echo "=" * 50
echo "📍 部署路径: $DEPLOY_PATH"
echo "👤 运行用户: $DEPLOY_USER"
echo "🌐 服务端口: 8000"
echo "⚙️  系统服务: $SERVICE_NAME"
echo ""
echo "📝 下一步操作:"
echo "  1. 下载 Gemma-3-27B 模型到 $DEPLOY_PATH/models/"
echo "  2. 编辑配置文件: $DEPLOY_PATH/configs/production.json"
echo "  3. 启动服务: $DEPLOY_PATH/manage.sh start"
echo "  4. 检查状态: $DEPLOY_PATH/manage.sh status"
echo "  5. 测试 API: $DEPLOY_PATH/manage.sh test"
echo ""
echo "🛠️  管理命令:"
echo "  启动: $DEPLOY_PATH/manage.sh start"
echo "  停止: $DEPLOY_PATH/manage.sh stop"
echo "  重启: $DEPLOY_PATH/manage.sh restart"
echo "  状态: $DEPLOY_PATH/manage.sh status"
echo "  日志: $DEPLOY_PATH/manage.sh logs"
echo "  健康检查: $DEPLOY_PATH/manage.sh health"
echo "  API 测试: $DEPLOY_PATH/manage.sh test" 