"""
WiCore Mojo 推理引擎 - 主引擎文件
基于 Mojo 和 MAX Engine 的自主可控 AI 推理平台
支持异构硬件统一调度和千亿参数模型推理
"""

from max import engine, graph
from python import Python

# 导入自定义组件
from device_manager import DeviceManager
from hmt_memory_manager import HMTMemoryManager
from model_executor import ModelExecutor
from request_scheduler import RequestScheduler
from web_server import WebServer

struct WiCoreConfig:
    var model_path: String
    var server_port: Int
    var max_batch_size: Int
    var max_context_length: Int
    var gpu_memory_limit_gb: Float64
    var enable_multi_gpu: Bool
    var target_devices: List[String]
    
    fn __init__(inout self, config_file_path: String):
        """从配置文件加载配置"""
        # 使用 Python 读取 JSON 配置
        var python = Python.import_module("builtins")
        var json = Python.import_module("json")
        
        with open(config_file_path, 'r') as f:
            config_data = json.load(f)
        
        self.model_path = str(config_data["model_path"])
        self.server_port = int(config_data["server_port"])
        self.max_batch_size = int(config_data["max_batch_size"])
        self.max_context_length = int(config_data["max_context_length"])
        self.gpu_memory_limit_gb = float(config_data["gpu_memory_limit_gb"])
        self.enable_multi_gpu = bool(config_data["enable_multi_gpu"])
        
        # 转换设备列表
        self.target_devices = List[String]()
        for device in config_data["target_devices"]:
            self.target_devices.append(str(device))

struct WiCoreEngine:
    var config: WiCoreConfig
    var device_manager: DeviceManager
    var memory_manager: HMTMemoryManager
    var model_executor: ModelExecutor
    var request_scheduler: RequestScheduler
    var web_server: WebServer
    var running: Bool
    
    fn __init__(inout self, config: WiCoreConfig):
        """初始化 WiCore 引擎"""
        print("🚀 初始化 WiCore Mojo 推理引擎...")
        
        self.config = config
        self.running = False
        
        # 初始化各个组件
        print("📱 初始化设备管理器...")
        self.device_manager = DeviceManager(config.target_devices)
        
        print("🧠 初始化 HMT 内存管理器...")
        self.memory_manager = HMTMemoryManager(self.device_manager)
        
        print("🤖 初始化模型执行器...")
        self.model_executor = ModelExecutor(config.model_path, self.memory_manager)
        
        print("📋 初始化请求调度器...")
        self.request_scheduler = RequestScheduler(self.model_executor, config)
        
        print("🌐 初始化 Web 服务器...")
        self.web_server = WebServer(self.request_scheduler, config.server_port)
        
        print("✅ WiCore 引擎初始化完成")
    
    fn start(self) -> Bool:
        """启动推理引擎"""
        print("🔥 启动 WiCore 推理引擎...")
        
        # 按顺序启动各个组件
        if not self.device_manager.initialize():
            print("❌ 设备管理器初始化失败")
            return False
        
        if not self.memory_manager.initialize():
            print("❌ 内存管理器初始化失败")
            return False
        
        if not self.model_executor.load_model():
            print("❌ 模型加载失败")
            return False
        
        if not self.request_scheduler.start():
            print("❌ 请求调度器启动失败")
            return False
        
        if not self.web_server.start():
            print("❌ Web 服务器启动失败")
            return False
        
        self.running = True
        print("🎉 WiCore 推理引擎启动成功！")
        print(f"🌐 API 服务地址: http://localhost:{self.config.server_port}")
        return True
    
    fn shutdown(self):
        """优雅关闭引擎"""
        if self.running:
            print("🛑 正在关闭 WiCore 推理引擎...")
            
            self.web_server.stop()
            self.request_scheduler.stop()
            self.model_executor.unload_model()
            self.memory_manager.cleanup()
            self.device_manager.cleanup()
            
            self.running = False
            print("✅ WiCore 引擎已安全关闭")
    
    fn get_status(self) -> String:
        """获取引擎状态"""
        if not self.running:
            return "stopped"
        
        # 组合各组件状态
        var status = "WiCore Engine Status:\n"
        status += f"  Running: {self.running}\n"
        status += f"  Port: {self.config.server_port}\n"
        status += f"  Model: {self.config.model_path}\n"
        status += f"  Multi-GPU: {self.config.enable_multi_gpu}\n"
        status += "  Components:\n"
        status += "    - Device Manager: Active\n"
        status += "    - Memory Manager: Active\n"
        status += "    - Model Executor: Active\n"
        status += "    - Request Scheduler: Active\n"
        status += "    - Web Server: Active\n"
        
        return status

fn main():
    """主函数"""
    try:
        print("🌟 WiCore Mojo 推理引擎启动中...")
        
        # 加载配置（默认生产配置）
        var config = WiCoreConfig("configs/production.json")
        
        # 创建并启动引擎
        var engine = WiCoreEngine(config)
        
        if engine.start():
            print("💫 引擎运行中，按 Ctrl+C 退出...")
            
            # 保持运行直到接收到中断信号
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n📡 接收到关闭信号...")
        
        # 优雅关闭
        engine.shutdown()
        
    except Exception as e:
        print(f"❌ 引擎启动失败: {e}")
        exit(1) 