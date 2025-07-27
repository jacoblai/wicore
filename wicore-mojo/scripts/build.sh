#!/bin/bash

# WiCore Mojo 推理引擎构建脚本
# 编译 Mojo 源代码和生成生产环境可执行文件

set -e

echo "🔨 开始构建 WiCore Mojo 推理引擎..."

# 检查 Mojo 环境
if ! command -v mojo &> /dev/null; then
    echo "❌ Mojo 编译器未找到，请确保 Modular SDK 已正确安装"
    exit 1
fi

echo "✅ Mojo 编译器版本: $(mojo --version)"

# 创建构建目录
BUILD_DIR="build"
mkdir -p $BUILD_DIR

echo "📁 创建构建目录: $BUILD_DIR"

# 编译核心组件
echo "🔧 编译核心组件..."

# 1. 设备管理器
echo "  编译设备管理器..."
mojo build src/device_manager.mojo -o $BUILD_DIR/device_manager

# 2. HMT 内存管理器
echo "  编译 HMT 内存管理器..."
mojo build src/hmt_memory_manager.mojo -o $BUILD_DIR/hmt_memory_manager

# 3. 模型执行器
echo "  编译模型执行器..."
mojo build src/model_executor.mojo -o $BUILD_DIR/model_executor

# 4. 请求调度器
echo "  编译请求调度器..."
mojo build src/request_scheduler.mojo -o $BUILD_DIR/request_scheduler

# 5. Web 服务器
echo "  编译 Web 服务器..."
mojo build src/web_server.mojo -o $BUILD_DIR/web_server

# 6. 主引擎
echo "  编译主引擎..."
mojo build src/wicore_engine.mojo -o $BUILD_DIR/wicore_engine

# 创建启动脚本
echo "📜 创建启动脚本..."
cat > $BUILD_DIR/start_wicore.sh << 'EOF'
#!/bin/bash

# WiCore Mojo 推理引擎启动脚本

CONFIG_FILE=${1:-"../configs/production.json"}
LOG_FILE="wicore_engine.log"

echo "🚀 启动 WiCore Mojo 推理引擎..."
echo "📄 配置文件: $CONFIG_FILE"
echo "📝 日志文件: $LOG_FILE"

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 启动引擎
./wicore_engine --config "$CONFIG_FILE" 2>&1 | tee "$LOG_FILE"
EOF

chmod +x $BUILD_DIR/start_wicore.sh

# 复制配置文件
echo "📋 复制配置文件..."
cp -r configs $BUILD_DIR/

# 复制模拟环境（如果需要）
if [ -d "simulation" ]; then
    echo "🎭 复制模拟环境..."
    cp -r simulation $BUILD_DIR/
fi

# 生成部署包
echo "📦 生成部署包..."
DEPLOY_PACKAGE="wicore-mojo-$(date +%Y%m%d-%H%M%S).tar.gz"
tar -czf $DEPLOY_PACKAGE $BUILD_DIR

echo "✅ 构建完成！"
echo ""
echo "📊 构建输出:"
echo "  构建目录: $BUILD_DIR/"
echo "  主执行文件: $BUILD_DIR/wicore_engine"
echo "  启动脚本: $BUILD_DIR/start_wicore.sh"
echo "  部署包: $DEPLOY_PACKAGE"
echo ""
echo "🚀 使用方法:"
echo "  cd $BUILD_DIR"
echo "  ./start_wicore.sh [配置文件路径]"
echo ""
echo "🎯 生产环境部署:"
echo "  1. 解压部署包到目标服务器"
echo "  2. 确保 Modular SDK 和 MAX Engine 已安装"
echo "  3. 下载 Gemma-3-27B 模型到 models/ 目录"
echo "  4. 编辑 configs/production.json 设置 GPU 配置"
echo "  5. 运行 ./start_wicore.sh configs/production.json" 