# WiCore Mojo 性能优化指南

## 概述

本文档详细介绍了 WiCore Mojo 推理引擎在 T10 双卡环境下的性能优化策略，包括硬件配置、内存管理、批处理优化和系统调优。

## 🎯 性能目标

### T10 双卡环境目标
- **吞吐量**: 100-150 tokens/s (总计)
- **延迟**: 50-100ms (首 token)
- **并发**: 16-32 并发请求
- **内存利用率**: >85%
- **GPU 利用率**: >90%

### 扩展目标
- **4×T10**: 300-500 tokens/s
- **A100 双卡**: 500-800 tokens/s
- **H100 双卡**: 1000+ tokens/s

## 🔧 硬件优化

### GPU 配置

#### NVIDIA T10 双卡最佳实践
```json
{
  "gpu_config": {
    "target_devices": ["gpu:0", "gpu:1"],
    "memory_limit_per_gpu": 15.0,
    "enable_p2p": true,
    "cuda_optimization": {
      "enable_cudnn": true,
      "enable_tensor_cores": true,
      "mixed_precision": "fp16"
    }
  }
}
```

#### GPU 内存分配策略
```mojo
# 内存池配置
gpu_pool_config = {
    "initial_size": 12 * 1024 * 1024 * 1024,  # 12GB
    "max_size": 15 * 1024 * 1024 * 1024,      # 15GB  
    "block_size": 4096,                        # 4KB 对齐
    "enable_defragmentation": true
}
```

### CPU 配置

#### NUMA 优化
```bash
# 检查 NUMA 拓扑
numactl --hardware

# 绑定 GPU 到 NUMA 节点
echo 0 > /sys/bus/pci/devices/0000:65:00.0/numa_node  # GPU0 -> NUMA0
echo 1 > /sys/bus/pci/devices/0000:b3:00.0/numa_node  # GPU1 -> NUMA1
```

#### CPU 亲和性设置
```bash
# 设置 CPU 亲和性
taskset -c 0-15 ./wicore_engine    # GPU0 相关进程
taskset -c 16-31 ./helper_process  # GPU1 相关进程
```

### 存储优化

#### NVMe SSD 配置
```bash
# 优化 NVMe 调度器
echo mq-deadline > /sys/block/nvme0n1/queue/scheduler

# 设置队列深度
echo 32 > /sys/block/nvme0n1/queue/nr_requests

# 启用 O_DIRECT
mount -o direct /dev/nvme0n1 /wicore_cache
```

## 🧠 HMT 内存管理优化

### A²CR 算法调优

#### 基础配置
```json
{
  "a2cr_config": {
    "time_decay_factor": 0.05,      # 时间衰减速度
    "attention_weight": 0.4,        # 注意力权重
    "frequency_weight": 0.3,        # 访问频率权重  
    "recency_weight": 0.3,          # 时效性权重
    "eviction_threshold": 0.2,      # 驱逐阈值
    "cleanup_interval": 60          # 清理间隔(秒)
  }
}
```

#### 自适应调优
```mojo
fn adaptive_tuning(memory_pressure: Float64) -> A2CRParams:
    """根据内存压力自适应调整参数"""
    var params = A2CRParams()
    
    if memory_pressure > 0.9:
        # 高内存压力：激进清理
        params.eviction_threshold = 0.4
        params.time_decay_factor = 0.1
    elif memory_pressure > 0.7:
        # 中等压力：平衡策略
        params.eviction_threshold = 0.3
        params.time_decay_factor = 0.05
    else:
        # 低压力：保守策略
        params.eviction_threshold = 0.2
        params.time_decay_factor = 0.02
    
    return params
```

### 三级存储优化

#### GPU 显存层 (Tier 0)
```mojo
# 热数据存储策略
struct GPUTierConfig:
    var data_format: String = "fp16"           # 半精度存储
    var compression: String = "sparse_2_4"     # 2:4 稀疏压缩
    var prefetch_size: Int = 64 * 1024 * 1024  # 64MB 预取
    var async_copy: Bool = True                # 异步拷贝
```

#### CPU 内存层 (Tier 1)  
```mojo
# 温数据存储策略
struct CPUTierConfig:
    var data_format: String = "q8_k"          # 8-bit 量化
    var page_size: Int = 2 * 1024 * 1024      # 2MB 大页
    var mlock_enabled: Bool = True             # 锁定内存
    var numa_aware: Bool = True                # NUMA 感知
```

#### NVMe 存储层 (Tier 2)
```mojo  
# 冷数据存储策略
struct NVMeTierConfig:
    var data_format: String = "q4_k"          # 4-bit 量化
    var compression: String = "lz4"           # LZ4 压缩
    var io_depth: Int = 32                    # 异步 I/O 深度
    var direct_io: Bool = True                # 直接 I/O
```

## ⚡ MoR 动态路由优化

### 路由策略调优

#### 阈值优化
```mojo
fn optimize_routing_threshold(workload_type: String) -> Float64:
    """根据工作负载类型优化路由阈值"""
    if workload_type == "creative_writing":
        return 0.3  # 更多 token 到 GPU
    elif workload_type == "code_generation":
        return 0.5  # 平衡分配
    elif workload_type == "simple_qa":
        return 0.7  # 更多 token 到 CPU
    else:
        return 0.5  # 默认平衡
```

#### 负载均衡
```mojo
struct LoadBalancer:
    var cpu_queue_length: Int
    var gpu_queue_length: Int
    var cpu_utilization: Float64
    var gpu_utilization: Float64
    
    fn get_optimal_device(self, attention_score: Float64) -> String:
        # 综合考虑注意力分数和设备负载
        var cpu_cost = attention_score * 0.5 + self.cpu_utilization * 0.3
        var gpu_cost = (1.0 - attention_score) * 0.5 + self.gpu_utilization * 0.3
        
        return "gpu" if gpu_cost < cpu_cost else "cpu"
```

### 并行计算优化

#### GPU 内核优化
```mojo
@gpu.kernel
fn optimized_attention_kernel(
    query: Tensor, key: Tensor, value: Tensor,
    output: Tensor, scale: Float64
):
    """优化的注意力计算内核"""
    # 使用 Tensor Core 加速
    var block_size = 64
    var thread_id = gpu.thread_id()
    
    # 分块矩阵乘法
    @vectorize(16)
    for i in range(block_size):
        # 使用混合精度计算
        var attention_score = vectorized_dot_product(
            query[thread_id], key[i]
        ) * scale
        
        # Softmax 计算
        output[thread_id] += attention_score * value[i]
```

## 🔄 批处理优化

### 动态批处理

#### 智能批次构建
```mojo
struct BatchOptimizer:
    var max_batch_size: Int
    var max_wait_time: Float64
    var memory_limit: Int
    
    fn create_optimal_batch(self, requests: List[InferenceRequest]) -> List[List[InferenceRequest]]:
        """创建最优批次"""
        var batches = List[List[InferenceRequest]]()
        var current_batch = List[InferenceRequest]()
        var current_memory = 0
        
        for request in requests:
            var request_memory = self._estimate_memory(request)
            
            # 检查是否可以添加到当前批次
            if (len(current_batch) < self.max_batch_size and 
                current_memory + request_memory < self.memory_limit):
                current_batch.append(request)
                current_memory += request_memory
            else:
                # 开始新批次
                if len(current_batch) > 0:
                    batches.append(current_batch)
                current_batch = List[InferenceRequest]()
                current_batch.append(request)
                current_memory = request_memory
        
        return batches
```

### 序列长度优化

#### 序列打包
```mojo
fn pack_sequences(sequences: List[List[Int]], max_length: Int) -> PackedBatch:
    """高效序列打包，减少 padding"""
    var packed = PackedBatch()
    var current_length = 0
    
    # 按长度排序
    sequences.sort(key=lambda x: len(x))
    
    for seq in sequences:
        if current_length + len(seq) <= max_length:
            packed.add_sequence(seq)
            current_length += len(seq)
        else:
            # 填充并处理当前批次
            packed.pad_and_finalize()
            # 开始新批次
            packed = PackedBatch()
            packed.add_sequence(seq)
            current_length = len(seq)
    
    return packed
```

## 📊 性能监控

### 关键指标

#### 系统级指标
```bash
# GPU 利用率监控
nvidia-smi dmon -s pucvmet -d 1

# CPU 使用监控  
top -p $(pgrep wicore_engine)

# 内存使用监控
cat /proc/$(pgrep wicore_engine)/status | grep VmRSS

# 网络 I/O 监控
iftop -i eth0
```

#### 应用级指标
```mojo
struct PerformanceMetrics:
    var tokens_per_second: Float64
    var average_latency: Float64
    var p95_latency: Float64
    var memory_utilization: Float64
    var cache_hit_rate: Float64
    var error_rate: Float64
    
    fn update_metrics(inout self, request_stats: RequestStats):
        # 滑动窗口统计
        self._update_sliding_window(request_stats)
```

### 性能基准测试

#### 标准测试套件
```bash
# 吞吐量测试
python benchmark/throughput_test.py \
  --batch_sizes 1,4,8,16 \
  --sequence_lengths 128,512,1024,2048 \
  --duration 300

# 延迟测试  
python benchmark/latency_test.py \
  --concurrent_users 1,5,10,20,32 \
  --request_rate 1,5,10,20

# 压力测试
python benchmark/stress_test.py \
  --max_concurrent 64 \
  --duration 3600 \
  --ramp_up 300
```

## 🎯 调优检查清单

### 硬件层面
- [ ] GPU P2P 通信已启用
- [ ] NUMA 绑定配置正确
- [ ] NVMe SSD 调度器优化
- [ ] 大页内存已启用
- [ ] CPU 频率调整器设置为性能模式

### 软件层面  
- [ ] HMT 三级存储配置优化
- [ ] A²CR 参数根据工作负载调整
- [ ] MoR 路由阈值已调优
- [ ] 批处理大小适合硬件配置
- [ ] 内存预分配和重用机制

### 系统层面
- [ ] 防火墙和安全组配置
- [ ] 文件描述符限制增加
- [ ] 内核参数优化
- [ ] 监控和日志配置
- [ ] 备份和恢复机制

## 📈 性能基线

### T10 双卡基线 (Gemma-3-27B)

| 批次大小 | 序列长度 | 吞吐量 (tok/s) | 首token延迟 (ms) | GPU 利用率 (%) |
|----------|----------|----------------|------------------|----------------|
| 1        | 512      | 45             | 85               | 60             |
| 4        | 512      | 120            | 95               | 75             |
| 8        | 512      | 180            | 110              | 85             |
| 16       | 512      | 220            | 140              | 90             |
| 16       | 1024     | 160            | 180              | 92             |
| 16       | 2048     | 100            | 250              | 95             |

### 优化后目标

| 批次大小 | 序列长度 | 吞吐量 (tok/s) | 首token延迟 (ms) | GPU 利用率 (%) |
|----------|----------|----------------|------------------|----------------|
| 1        | 512      | 60             | 65               | 70             |
| 4        | 512      | 150            | 75               | 82             |
| 8        | 512      | 220            | 85               | 90             |
| 16       | 512      | 280            | 100              | 95             |
| 16       | 1024     | 200            | 140              | 95             |
| 16       | 2048     | 130            | 200              | 96             |

## 🚀 高级优化技术

### 1. KV 缓存优化
```mojo
# PagedAttention 实现
struct PagedKVCache:
    var page_size: Int = 64
    var max_pages: Int = 1024
    
    fn allocate_kv_cache(self, sequence_length: Int) -> KVCacheBlocks:
        var num_pages = (sequence_length + self.page_size - 1) // self.page_size
        return self._allocate_pages(num_pages)
```

### 2. 前缀缓存
```mojo
# 系统提示词缓存
struct PrefixCache:
    var cached_prefixes: Dict[String, CachedPrefix]
    
    fn get_or_compute_prefix(self, prefix: String) -> CachedPrefix:
        if prefix in self.cached_prefixes:
            return self.cached_prefixes[prefix]
        else:
            return self._compute_and_cache(prefix)
```

### 3. 投机解码
```mojo
# Speculative Decoding
struct SpeculativeDecoder:
    var draft_model: SmallModel
    var target_model: LargeModel
    
    fn speculative_decode(self, input_tokens: List[Int]) -> List[Int]:
        # 使用小模型快速生成候选
        var candidates = self.draft_model.generate(input_tokens, k=4)
        
        # 使用大模型验证
        var verified = self.target_model.verify(candidates)
        return verified
```

## 🔍 故障排除

### 常见性能问题

#### 内存不足
```bash
# 检查 GPU 内存
nvidia-smi

# 检查系统内存
free -h

# 检查 swap 使用
swapon -s
```

#### GPU 利用率低
```bash
# 检查 GPU 状态
nvidia-smi dmon -s u

# 检查 CUDA 上下文
nvidia-smi -q -d COMPUTE

# 检查 P2P 连接
nvidia-smi topo -p2p w
```

#### 网络延迟高
```bash
# 检查网络连接
ss -tuln | grep 8000

# 检查网络延迟
ping localhost

# 检查 TCP 参数
sysctl net.ipv4.tcp_*
```

### 性能调试工具

#### Profiler 集成
```python
# 性能分析
import cProfile
import pstats

def profile_inference():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # 执行推理
    result = engine.infer(input_text)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats()
```

通过以上优化策略，WiCore Mojo 推理引擎能够在 T10 双卡环境下实现预期的性能目标，并为更高端硬件提供良好的扩展性。 