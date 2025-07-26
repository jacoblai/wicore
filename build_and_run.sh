#!/bin/bash
# build_and_run.sh - WiCore构建和运行脚本

set -e  # 遇到错误时退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

echo_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

echo_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 打印WiCore标题
print_banner() {
    echo -e "${BLUE}"
    cat << 'EOF'
████╗    ██╗██╗ ██████╗ ██████╗ ██████╗ ███████╗
██╔═██╗  ██║██║██╔════╝██╔═══██╗██╔══██╗██╔════╝
██║██╔██╗██║██║██║     ██║   ██║██████╔╝█████╗  
██║╚═╝ ██╗██║██║██║     ██║   ██║██╔══██╗██╔══╝  
██║    ╚═╝██║██║╚██████╗╚██████╔╝██║  ██║███████╗
╚═╝       ╚═╝╚═╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝

WiCore C++推理引擎构建脚本
面向Gemma-3-27B-IT的极致性能实现
EOF
    echo -e "${NC}"
}

# 检查系统依赖
check_dependencies() {
    echo_info "检查系统依赖..."
    
    # 检查CUDA
    if ! command -v nvcc &> /dev/null; then
        echo_error "CUDA未安装或未添加到PATH"
        echo "请安装CUDA 12.0或更高版本"
        exit 1
    fi
    
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
    echo_success "CUDA版本: $CUDA_VERSION"
    
    # 检查TensorRT
    if [ ! -d "/usr/local/TensorRT" ] && [ ! -d "/opt/TensorRT" ]; then
        echo_error "TensorRT未找到"
        echo "请安装TensorRT 8.5或更高版本"
        exit 1
    fi
    
    # 检查CMake
    if ! command -v cmake &> /dev/null; then
        echo_error "CMake未安装"
        echo "请安装CMake 3.18或更高版本"
        exit 1
    fi
    
    CMAKE_VERSION=$(cmake --version | head -n1 | sed 's/cmake version \([0-9.]*\).*/\1/')
    echo_success "CMake版本: $CMAKE_VERSION"
    
    # 检查其他依赖
    MISSING_DEPS=()
    
    if ! pkg-config --exists opencv4; then
        MISSING_DEPS+=("opencv4")
    fi
    
    if ! pkg-config --exists jsoncpp; then
        MISSING_DEPS+=("jsoncpp")
    fi
    
    if ! pkg-config --exists evhtp; then
        MISSING_DEPS+=("evhtp")
    fi
    
    if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
        echo_error "缺少依赖: ${MISSING_DEPS[*]}"
        echo "Ubuntu/Debian安装命令:"
        echo "  sudo apt-get install libopencv-dev libjsoncpp-dev libevhtp-dev"
        echo "CentOS/RHEL安装命令:"
        echo "  sudo yum install opencv-devel jsoncpp-devel libevhtp-devel"
        exit 1
    fi
    
    echo_success "所有依赖检查通过"
}

# 检查GPU设备
check_gpu() {
    echo_info "检查GPU设备..."
    
    if ! command -v nvidia-smi &> /dev/null; then
        echo_error "nvidia-smi未找到，请安装NVIDIA驱动"
        exit 1
    fi
    
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    if [ $GPU_COUNT -eq 0 ]; then
        echo_error "未检测到GPU设备"
        exit 1
    fi
    
    echo_success "检测到 $GPU_COUNT 个GPU设备"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | while read line; do
        echo "  $line"
    done
}

# 创建目录结构
setup_directories() {
    echo_info "创建目录结构..."
    
    mkdir -p build
    mkdir -p models
    mkdir -p cache/nvme
    mkdir -p logs
    mkdir -p src
    mkdir -p include
    
    echo_success "目录结构创建完成"
}

# 下载Gemma-3模型
download_model() {
    echo_info "检查Gemma-3-27B-IT模型..."
    
    MODEL_DIR="./models/gemma-3-27b-it"
    
    if [ -f "$MODEL_DIR/model.onnx" ]; then
        echo_success "模型已存在，跳过下载"
        return
    fi
    
    echo_warning "模型不存在，需要下载"
    echo "请确保已安装huggingface-hub:"
    echo "  pip install huggingface-hub"
    echo ""
    echo "然后运行以下命令下载模型:"
    echo "  huggingface-cli download google/gemma-3-27b-it --local-dir $MODEL_DIR"
    echo ""
    echo "由于模型较大(约54GB)，下载可能需要较长时间"
    echo ""
    
    read -p "是否现在下载模型? (y/n): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if command -v huggingface-cli &> /dev/null; then
            echo_info "开始下载Gemma-3-27B-IT模型..."
            huggingface-cli download google/gemma-3-27b-it --local-dir $MODEL_DIR
            echo_success "模型下载完成"
        else
            echo_error "huggingface-cli未安装"
            echo "请先安装: pip install huggingface-hub"
            exit 1
        fi
    else
        echo_warning "跳过模型下载，请手动下载后再运行"
    fi
}

# 构建项目
build_project() {
    echo_info "构建WiCore项目..."
    
    cd build
    
    # 配置CMake
    echo_info "配置CMake..."
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_STANDARD=17 \
        -DCUDA_ARCHITECTURES="75;80;86;89" \
        -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89"
    
    # 编译
    echo_info "开始编译..."
    make -j$(nproc)
    
    cd ..
    
    echo_success "构建完成"
}

# 生成配置文件
generate_config() {
    echo_info "生成配置文件..."
    
    if [ ! -f "config.json" ]; then
        cp config_template.json config.json
        echo_success "配置文件已生成: config.json"
        echo "你可以根据需要修改配置参数"
    else
        echo_warning "配置文件已存在，跳过生成"
    fi
}

# 运行性能测试
run_benchmark() {
    echo_info "运行性能基准测试..."
    
    # 创建测试脚本
    cat > test_inference.py << 'EOF'
import asyncio
import aiohttp
import time
import json
import base64
from PIL import Image
import io

async def test_single_inference():
    # 创建测试图像
    test_image = Image.new('RGB', (896, 896), color=(255, 100, 100))
    img_buffer = io.BytesIO()
    test_image.save(img_buffer, format='JPEG')
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    
    request_data = {
        "prompt": "请详细描述这张图片的内容，包括颜色、形状等特征。",
        "images": [img_base64],
        "max_tokens": 512,
        "temperature": 0.7
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            
            async with session.post(
                "http://localhost:8080/v1/inference",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    end_time = time.time()
                    
                    print(f"✅ 推理成功")
                    print(f"延迟: {(end_time - start_time) * 1000:.1f}ms")
                    print(f"生成内容: {result.get('content', '无内容')[:100]}...")
                    print(f"Token数量: {result.get('token_count', 0)}")
                    print(f"服务器延迟: {result.get('latency_ms', 0):.1f}ms")
                    return True
                else:
                    print(f"❌ 请求失败: HTTP {response.status}")
                    error_text = await response.text()
                    print(f"错误信息: {error_text}")
                    return False
                    
    except Exception as e:
        print(f"❌ 连接错误: {e}")
        return False

async def main():
    print("=== WiCore推理测试 ===")
    success = await test_single_inference()
    
    if success:
        print("\n🎉 测试通过！WiCore运行正常")
    else:
        print("\n💥 测试失败，请检查服务状态")

if __name__ == "__main__":
    asyncio.run(main())
EOF
    
    echo_success "测试脚本已生成: test_inference.py"
    echo "启动WiCore服务后，运行: python test_inference.py"
}

# 启动服务
start_service() {
    echo_info "启动WiCore服务..."
    
    if [ ! -f "build/wicore_server" ]; then
        echo_error "可执行文件不存在，请先构建项目"
        exit 1
    fi
    
    if [ ! -f "config.json" ]; then
        echo_error "配置文件不存在"
        exit 1
    fi
    
    echo_info "使用配置文件: config.json"
    echo_info "启动服务器..."
    echo_warning "按 Ctrl+C 停止服务"
    echo ""
    
    # 启动服务
    ./build/wicore_server config.json
}

# 主菜单
show_menu() {
    echo ""
    echo "选择操作:"
    echo "1) 检查环境"
    echo "2) 下载模型"  
    echo "3) 构建项目"
    echo "4) 生成配置"
    echo "5) 创建测试"
    echo "6) 启动服务"
    echo "7) 一键完整构建"
    echo "0) 退出"
    echo ""
}

# 主函数
main() {
    print_banner
    
    while true; do
        show_menu
        read -p "请选择 (0-7): " choice
        
        case $choice in
            1)
                check_dependencies
                check_gpu
                ;;
            2)
                download_model
                ;;
            3)
                setup_directories
                build_project
                ;;
            4)
                generate_config
                ;;
            5)
                run_benchmark
                ;;
            6)
                start_service
                ;;
            7)
                echo_info "开始一键构建..."
                check_dependencies
                check_gpu
                setup_directories
                download_model
                build_project
                generate_config
                run_benchmark
                echo_success "构建完成！现在可以启动服务了"
                ;;
            0)
                echo_info "退出构建脚本"
                exit 0
                ;;
            *)
                echo_error "无效选择，请重新输入"
                ;;
        esac
        
        echo ""
        read -p "按Enter继续..."
    done
}

# 运行主函数
main 