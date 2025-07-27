"""
请求调度器 - WiCore Mojo 推理引擎
实现智能批处理调度、优先级管理和负载均衡
支持异步处理和流式输出
"""

from collections import Dict, List, Deque
from python import Python
from .model_executor import ModelExecutor
import time
import threading

enum RequestPriority:
    LOW = 0
    NORMAL = 1 
    HIGH = 2
    URGENT = 3

enum RequestStatus:
    PENDING = 0
    PROCESSING = 1
    COMPLETED = 2
    FAILED = 3
    CANCELLED = 4

struct InferenceRequest:
    """推理请求结构体"""
    var request_id: String
    var input_text: String
    var max_tokens: Int
    var temperature: Float64
    var priority: RequestPriority
    var created_time: Float64
    var timeout_seconds: Int
    var stream: Bool
    var status: RequestStatus
    var result: String
    var error_message: String
    var processing_start_time: Float64
    var processing_end_time: Float64
    
    fn __init__(inout self):
        """默认初始化"""
        self.request_id = ""
        self.input_text = ""
        self.max_tokens = 512
        self.temperature = 0.7
        self.priority = RequestPriority.NORMAL
        self.created_time = time.time_ns() / 1e9
        self.timeout_seconds = 30
        self.stream = False
        self.status = RequestStatus.PENDING
        self.result = ""
        self.error_message = ""
        self.processing_start_time = 0.0
        self.processing_end_time = 0.0
    
    fn __init__(inout self, request_id: String, input_text: String):
        """基础初始化"""
        self.request_id = request_id
        self.input_text = input_text
        self.max_tokens = 512
        self.temperature = 0.7
        self.priority = RequestPriority.NORMAL
        self.created_time = time.time_ns() / 1e9
        self.timeout_seconds = 30
        self.stream = False
        self.status = RequestStatus.PENDING
        self.result = ""
        self.error_message = ""
        self.processing_start_time = 0.0
        self.processing_end_time = 0.0
    
    fn is_expired(self) -> Bool:
        """检查请求是否过期"""
        current_time = time.time_ns() / 1e9
        return (current_time - self.created_time) > Float64(self.timeout_seconds)
    
    fn get_wait_time(self) -> Float64:
        """获取等待时间"""
        current_time = time.time_ns() / 1e9
        if self.processing_start_time > 0:
            return self.processing_start_time - self.created_time
        else:
            return current_time - self.created_time
    
    fn get_processing_time(self) -> Float64:
        """获取处理时间"""
        if self.processing_start_time == 0:
            return 0.0
        
        end_time = self.processing_end_time if self.processing_end_time > 0 else time.time_ns() / 1e9
        return end_time - self.processing_start_time
    
    fn mark_processing(inout self):
        """标记为处理中"""
        self.status = RequestStatus.PROCESSING
        self.processing_start_time = time.time_ns() / 1e9
    
    fn mark_completed(inout self, result: String):
        """标记为完成"""
        self.status = RequestStatus.COMPLETED
        self.result = result
        self.processing_end_time = time.time_ns() / 1e9
    
    fn mark_failed(inout self, error: String):
        """标记为失败"""
        self.status = RequestStatus.FAILED
        self.error_message = error
        self.processing_end_time = time.time_ns() / 1e9


struct BatchProcessor:
    """批处理器"""
    var max_batch_size: Int
    var max_wait_time: Float64      # 最大等待时间(秒)
    var current_batch: List[InferenceRequest]
    var batch_start_time: Float64
    
    fn __init__(inout self, max_batch_size: Int, max_wait_time: Float64 = 0.1):
        """初始化批处理器"""
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.current_batch = List[InferenceRequest]()
        self.batch_start_time = 0.0
    
    fn add_request(inout self, request: InferenceRequest) -> Bool:
        """添加请求到当前批次"""
        if len(self.current_batch) >= self.max_batch_size:
            return False
        
        if len(self.current_batch) == 0:
            self.batch_start_time = time.time_ns() / 1e9
        
        self.current_batch.append(request)
        return True
    
    fn should_process_batch(self) -> Bool:
        """判断是否应该处理当前批次"""
        if len(self.current_batch) == 0:
            return False
        
        # 批次已满
        if len(self.current_batch) >= self.max_batch_size:
            return True
        
        # 等待时间超过阈值
        current_time = time.time_ns() / 1e9
        if (current_time - self.batch_start_time) > self.max_wait_time:
            return True
        
        return False
    
    fn get_batch(inout self) -> List[InferenceRequest]:
        """获取当前批次并重置"""
        batch = self.current_batch
        self.current_batch = List[InferenceRequest]()
        self.batch_start_time = 0.0
        return batch


struct SchedulerStats:
    """调度器统计信息"""
    var total_requests: Int
    var completed_requests: Int
    var failed_requests: Int
    var cancelled_requests: Int
    var total_processing_time: Float64
    var total_wait_time: Float64
    var max_queue_length: Int
    var current_queue_length: Int
    
    fn __init__(inout self):
        """初始化统计信息"""
        self.total_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0
        self.cancelled_requests = 0
        self.total_processing_time = 0.0
        self.total_wait_time = 0.0
        self.max_queue_length = 0
        self.current_queue_length = 0
    
    fn update_request_completed(inout self, request: InferenceRequest):
        """更新请求完成统计"""
        self.completed_requests += 1
        self.total_processing_time += request.get_processing_time()
        self.total_wait_time += request.get_wait_time()
    
    fn update_request_failed(inout self):
        """更新请求失败统计"""
        self.failed_requests += 1
    
    fn update_queue_length(inout self, length: Int):
        """更新队列长度统计"""
        self.current_queue_length = length
        if length > self.max_queue_length:
            self.max_queue_length = length
    
    fn get_average_processing_time(self) -> Float64:
        """获取平均处理时间"""
        if self.completed_requests == 0:
            return 0.0
        return self.total_processing_time / Float64(self.completed_requests)
    
    fn get_average_wait_time(self) -> Float64:
        """获取平均等待时间"""
        if self.completed_requests == 0:
            return 0.0
        return self.total_wait_time / Float64(self.completed_requests)
    
    fn get_success_rate(self) -> Float64:
        """获取成功率"""
        if self.total_requests == 0:
            return 0.0
        return Float64(self.completed_requests) / Float64(self.total_requests) * 100.0


struct RequestScheduler:
    """请求调度器主类"""
    var model_executor: ModelExecutor
    var config: WiCoreConfig
    var request_queues: List[Deque[InferenceRequest]]  # 按优先级分队列
    var active_requests: Dict[String, InferenceRequest]
    var batch_processor: BatchProcessor
    var stats: SchedulerStats
    var running: Bool
    var max_concurrent_requests: Int
    
    fn __init__(inout self, model_executor: ModelExecutor, config: WiCoreConfig):
        """初始化请求调度器"""
        print("📋 初始化请求调度器...")
        
        self.model_executor = model_executor
        self.config = config
        self.running = False
        self.max_concurrent_requests = config.max_batch_size * 2
        
        # 初始化优先级队列
        self.request_queues = List[Deque[InferenceRequest]]()
        for _ in range(4):  # 4个优先级
            self.request_queues.append(Deque[InferenceRequest]())
        
        self.active_requests = Dict[String, InferenceRequest]()
        
        # 初始化批处理器
        self.batch_processor = BatchProcessor(config.max_batch_size, 0.05)  # 50ms等待
        
        # 初始化统计
        self.stats = SchedulerStats()
        
        print("✅ 请求调度器初始化完成")
    
    fn start(inout self) -> Bool:
        """启动调度器"""
        if self.running:
            print("⚠️  调度器已在运行")
            return True
        
        print("🚀 启动请求调度器...")
        
        self.running = True
        
        # 在真实环境中，这里会启动调度线程
        # 简化实现：标记为运行状态
        print("✅ 请求调度器启动成功")
        return True
    
    fn stop(self):
        """停止调度器"""
        if not self.running:
            return
        
        print("🛑 停止请求调度器...")
        
        self.running = False
        
        # 处理剩余请求
        self._process_remaining_requests()
        
        print("✅ 请求调度器已停止")
    
    fn submit_request(inout self, request: InferenceRequest) -> String:
        """提交推理请求"""
        if not self.running:
            return "Error: Scheduler not running"
        
        # 检查并发限制
        if len(self.active_requests) >= self.max_concurrent_requests:
            return "Error: Too many concurrent requests"
        
        # 生成请求ID
        if request.request_id == "":
            request.request_id = self._generate_request_id()
        
        # 更新统计
        self.stats.total_requests += 1
        
        # 根据优先级加入相应队列
        priority_index = int(request.priority)
        self.request_queues[priority_index].append(request)
        
        # 更新队列长度统计
        total_queue_length = self._get_total_queue_length()
        self.stats.update_queue_length(total_queue_length)
        
        print(f"📝 提交请求: {request.request_id} (优先级: {priority_index})")
        
        # 尝试立即处理
        self._try_process_requests()
        
        return request.request_id
    
    fn get_request_status(self, request_id: String) -> Optional[InferenceRequest]:
        """获取请求状态"""
        # 检查活跃请求
        if request_id in self.active_requests:
            return self.active_requests[request_id]
        
        # 检查队列中的请求
        for queue in self.request_queues:
            for request in queue:
                if request[].request_id == request_id:
                    return request[]
        
        return None
    
    fn cancel_request(inout self, request_id: String) -> Bool:
        """取消请求"""
        # 从活跃请求中移除
        if request_id in self.active_requests:
            request = self.active_requests[request_id]
            request.status = RequestStatus.CANCELLED
            del self.active_requests[request_id]
            self.stats.cancelled_requests += 1
            print(f"❌ 取消活跃请求: {request_id}")
            return True
        
        # 从队列中移除
        for queue in self.request_queues:
            for i in range(len(queue)):
                if queue[i].request_id == request_id:
                    request = queue[i]
                    request.status = RequestStatus.CANCELLED
                    # 这里需要从队列中删除元素
                    # 简化实现：标记为取消
                    self.stats.cancelled_requests += 1
                    print(f"❌ 取消队列请求: {request_id}")
                    return True
        
        return False
    
    fn _try_process_requests(inout self):
        """尝试处理请求"""
        if not self.running:
            return
        
        # 创建批次
        batch = self._create_optimal_batch()
        if len(batch) == 0:
            return
        
        # 处理批次
        self._process_batch(batch)
    
    fn _create_optimal_batch(inout self) -> List[InferenceRequest]:
        """创建最优批次"""
        batch = List[InferenceRequest]()
        
        # 从高优先级到低优先级选择请求
        for priority in range(3, -1, -1):
            queue = self.request_queues[priority]
            
            while len(queue) > 0 and len(batch) < self.config.max_batch_size:
                request = queue.popleft()
                
                # 检查请求是否过期
                if request.is_expired():
                    request.status = RequestStatus.FAILED
                    request.error_message = "Request timeout"
                    self.stats.update_request_failed()
                    continue
                
                batch.append(request)
        
        return batch
    
    fn _process_batch(inout self, batch: List[InferenceRequest]):
        """处理批次"""
        if len(batch) == 0:
            return
        
        print(f"⚡ 处理批次: {len(batch)} 个请求")
        
        # 标记请求为处理中
        for request in batch:
            request[].mark_processing()
            self.active_requests[request[].request_id] = request[]
        
        # 执行批量推理
        batch_inputs = List[String]()
        for request in batch:
            batch_inputs.append(request[].input_text)
        
        try:
            # 调用模型执行器
            batch_results = self.model_executor.infer_batch(batch_inputs)
            
            # 处理结果
            for i in range(len(batch)):
                request = batch[i]
                
                if i < len(batch_results):
                    result = batch_results[i]
                    request[].mark_completed(result)
                    self.stats.update_request_completed(request[])
                    print(f"✅ 完成请求: {request[].request_id}")
                else:
                    request[].mark_failed("Inference failed")
                    self.stats.update_request_failed()
                    print(f"❌ 请求失败: {request[].request_id}")
                
                # 从活跃请求中移除
                if request[].request_id in self.active_requests:
                    del self.active_requests[request[].request_id]
        
        except Exception as e:
            # 处理批次失败
            print("❌ 批次处理失败:", str(e))
            
            for request in batch:
                request[].mark_failed(str(e))
                self.stats.update_request_failed()
                
                if request[].request_id in self.active_requests:
                    del self.active_requests[request[].request_id]
    
    fn _process_remaining_requests(inout self):
        """处理剩余请求"""
        print("🔄 处理剩余请求...")
        
        # 取消所有队列中的请求
        for queue in self.request_queues:
            while len(queue) > 0:
                request = queue.popleft()
                request.status = RequestStatus.CANCELLED
                request.error_message = "Scheduler shutdown"
                self.stats.cancelled_requests += 1
        
        # 取消所有活跃请求
        for request_id in self.active_requests:
            request = self.active_requests[request_id]
            request.status = RequestStatus.CANCELLED
            request.error_message = "Scheduler shutdown"
            self.stats.cancelled_requests += 1
        
        self.active_requests.clear()
    
    fn _get_total_queue_length(self) -> Int:
        """获取总队列长度"""
        total = 0
        for queue in self.request_queues:
            total += len(queue)
        return total
    
    fn _generate_request_id(self) -> String:
        """生成请求ID"""
        timestamp = time.time_ns()
        return "req_" + str(timestamp)
    
    fn get_scheduler_status(self) -> String:
        """获取调度器状态"""
        status = f"调度器状态:\\n"
        status += f"  运行中: {'是' if self.running else '否'}\\n"
        status += f"  总请求数: {self.stats.total_requests}\\n"
        status += f"  已完成: {self.stats.completed_requests}\\n"
        status += f"  失败: {self.stats.failed_requests}\\n"
        status += f"  取消: {self.stats.cancelled_requests}\\n"
        status += f"  活跃请求: {len(self.active_requests)}\\n"
        status += f"  队列长度: {self._get_total_queue_length()}\\n"
        status += f"  成功率: {self.stats.get_success_rate():.1f}%\\n"
        status += f"  平均处理时间: {self.stats.get_average_processing_time():.3f}s\\n"
        status += f"  平均等待时间: {self.stats.get_average_wait_time():.3f}s"
        
        return status
    
    fn get_queue_summary(self) -> String:
        """获取队列摘要"""
        summary = "队列状态:\\n"
        
        priority_names = ["低", "普通", "高", "紧急"]
        for i in range(4):
            queue_length = len(self.request_queues[i])
            summary += f"  {priority_names[i]}优先级: {queue_length} 个请求\\n"
        
        summary += f"  活跃处理: {len(self.active_requests)} 个请求"
        return summary 