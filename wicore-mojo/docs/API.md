# WiCore Mojo API 文档

## 概述

WiCore Mojo 推理引擎提供完全兼容 OpenAI API 的 RESTful 接口，支持聊天完成、模型列表、健康检查等功能。

**基础 URL**: `http://localhost:8000`

## 认证

当前版本不需要认证，后续版本将支持 API Key 认证。

## 接口列表

### 1. 聊天完成 (Chat Completions)

与 GPT 模型进行对话交互。

**端点**: `POST /v1/chat/completions`

**请求体**:
```json
{
  "model": "gemma-3-27b-it",
  "messages": [
    {
      "role": "user",
      "content": "解释量子计算的基本原理"
    }
  ],
  "max_tokens": 512,
  "temperature": 0.7,
  "stream": false,
  "top_p": 1.0,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0
}
```

**参数说明**:
- `model` (string): 模型名称，目前支持 "gemma-3-27b-it"
- `messages` (array): 对话消息数组
  - `role` (string): 消息角色 ("system", "user", "assistant")
  - `content` (string): 消息内容
- `max_tokens` (integer): 最大生成 token 数，默认 512
- `temperature` (float): 采样温度，0-2之间，默认 0.7
- `stream` (boolean): 是否启用流式输出，默认 false
- `top_p` (float): 核采样参数，0-1之间，默认 1.0
- `frequency_penalty` (float): 频率惩罚，-2到2之间，默认 0.0
- `presence_penalty` (float): 存在惩罚，-2到2之间，默认 0.0

**响应体**:
```json
{
  "id": "req_1704067200123",
  "object": "chat.completion",
  "created": 1704067200,
  "model": "gemma-3-27b-it",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "量子计算是基于量子力学原理的计算技术..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 128,
    "total_tokens": 143
  }
}
```

**流式响应** (当 `stream=true` 时):
```
data: {"id":"req_1704067200123","object":"chat.completion.chunk","created":1704067200,"model":"gemma-3-27b-it","choices":[{"index":0,"delta":{"content":"量子"},"finish_reason":null}]}

data: {"id":"req_1704067200123","object":"chat.completion.chunk","created":1704067200,"model":"gemma-3-27b-it","choices":[{"index":0,"delta":{"content":"计算"},"finish_reason":null}]}

data: [DONE]
```

### 2. 模型列表 (Models)

获取可用模型列表。

**端点**: `GET /v1/models`

**响应体**:
```json
{
  "object": "list",
  "data": [
    {
      "id": "gemma-3-27b-it",
      "object": "model",
      "created": 1704067200,
      "owned_by": "wicore"
    }
  ]
}
```

### 3. 健康检查 (Health Check)

检查引擎运行状态。

**端点**: `GET /health`

**响应体**:
```json
{
  "status": "healthy",
  "engine": "wicore-mojo",
  "version": "1.0.0",
  "uptime": "3600.5",
  "scheduler": "running",
  "total_requests": "1250",
  "error_count": "3",
  "error_rate": "0.2%"
}
```

### 4. 系统状态 (Status)

获取详细的系统状态信息。

**端点**: `GET /status`

**响应体**:
```json
{
  "server_running": "是",
  "port": "8000",
  "uptime": "3600.5s",
  "total_requests": "1250",
  "error_count": "3",
  "scheduler_status": "调度器状态:\n  运行中: 是\n  总请求数: 1250\n  已完成: 1240\n  失败: 3\n  取消: 7\n  活跃请求: 5\n  队列长度: 12\n  成功率: 99.2%\n  平均处理时间: 0.156s\n  平均等待时间: 0.023s",
  "queue_summary": "队列状态:\n  低优先级: 2 个请求\n  普通优先级: 8 个请求\n  高优先级: 2 个请求\n  紧急优先级: 0 个请求\n  活跃处理: 5 个请求"
}
```

### 5. 根路径 (Root)

引擎基本信息。

**端点**: `GET /`

**响应体**:
```json
{
  "message": "WiCore Mojo 推理引擎",
  "version": "1.0.0"
}
```

## 错误处理

### 错误响应格式

```json
{
  "error": {
    "message": "详细错误信息",
    "type": "invalid_request_error",
    "code": "400"
  }
}
```

### 常见错误码

- `400` - 请求参数错误
- `429` - 请求频率限制
- `500` - 服务器内部错误
- `503` - 服务不可用

### 错误类型

- `invalid_request_error` - 请求格式或参数错误
- `rate_limit_exceeded` - 超过请求频率限制
- `insufficient_quota` - 配额不足
- `server_error` - 服务器内部错误

## 使用示例

### cURL 示例

```bash
# 基本聊天
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-27b-it",
    "messages": [
      {"role": "user", "content": "你好，请介绍一下你自己"}
    ],
    "max_tokens": 512
  }'

# 流式输出
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-27b-it",
    "messages": [
      {"role": "user", "content": "写一个 Python 函数来计算斐波那契数列"}
    ],
    "max_tokens": 256,
    "stream": true
  }'

# 健康检查
curl http://localhost:8000/health

# 系统状态
curl http://localhost:8000/status
```

### Python 示例

```python
import requests
import json

# 基础配置
BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

def chat_completion(message, temperature=0.7, max_tokens=512):
    """发送聊天完成请求"""
    payload = {
        "model": "gemma-3-27b-it",
        "messages": [{"role": "user", "content": message}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers=HEADERS,
        data=json.dumps(payload)
    )
    
    return response.json()

def chat_stream(message, max_tokens=512):
    """发送流式聊天请求"""
    payload = {
        "model": "gemma-3-27b-it",
        "messages": [{"role": "user", "content": message}],
        "max_tokens": max_tokens,
        "stream": True
    }
    
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers=HEADERS,
        data=json.dumps(payload),
        stream=True
    )
    
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data = line[6:]
                if data != '[DONE]':
                    yield json.loads(data)

def health_check():
    """健康检查"""
    response = requests.get(f"{BASE_URL}/health")
    return response.json()

# 使用示例
if __name__ == "__main__":
    # 基本聊天
    result = chat_completion("解释深度学习的基本概念")
    print("助手回答:", result['choices'][0]['message']['content'])
    
    # 流式输出
    print("\n流式输出:")
    for chunk in chat_stream("写一个简单的机器学习模型"):
        if 'choices' in chunk and chunk['choices']:
            content = chunk['choices'][0].get('delta', {}).get('content', '')
            if content:
                print(content, end='', flush=True)
    
    # 健康检查
    health = health_check()
    print(f"\n\n引擎状态: {health['status']}")
```

### JavaScript 示例

```javascript
// 基本聊天完成
async function chatCompletion(message, options = {}) {
    const response = await fetch('http://localhost:8000/v1/chat/completions', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            model: 'gemma-3-27b-it',
            messages: [{role: 'user', content: message}],
            max_tokens: options.maxTokens || 512,
            temperature: options.temperature || 0.7,
            ...options
        })
    });
    
    return await response.json();
}

// 流式输出
async function* chatStream(message, options = {}) {
    const response = await fetch('http://localhost:8000/v1/chat/completions', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            model: 'gemma-3-27b-it',
            messages: [{role: 'user', content: message}],
            max_tokens: options.maxTokens || 512,
            stream: true,
            ...options
        })
    });
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    if (data !== '[DONE]') {
                        yield JSON.parse(data);
                    }
                }
            }
        }
    } finally {
        reader.releaseLock();
    }
}

// 使用示例
(async () => {
    // 基本聊天
    const result = await chatCompletion('介绍一下人工智能的发展历史');
    console.log('助手回答:', result.choices[0].message.content);
    
    // 流式输出
    console.log('\n流式输出:');
    for await (const chunk of chatStream('写一个 JavaScript 函数')) {
        if (chunk.choices && chunk.choices[0]?.delta?.content) {
            process.stdout.write(chunk.choices[0].delta.content);
        }
    }
})();
```

## 性能优化建议

1. **批量请求**: 使用批处理可以提高吞吐量
2. **合理设置参数**: 
   - `max_tokens`: 根据需要设置，避免过大
   - `temperature`: 0.7-1.0 平衡创造性和一致性
3. **流式输出**: 对于长文本生成，使用流式输出提升用户体验
4. **连接复用**: 使用 HTTP keep-alive 减少连接开销
5. **错误重试**: 实现指数退避重试机制

## 限制和配额

- **最大并发请求**: 32 个
- **最大序列长度**: 131072 tokens
- **最大批处理大小**: 16 个请求
- **请求超时**: 30 秒
- **速率限制**: 每分钟 100 请求 (可配置)

## 监控和调试

### 日志查看
```bash
# 实时日志
curl http://localhost:8000/status

# 系统日志 (如果部署为服务)
journalctl -u wicore-engine -f
```

### 性能监控
```bash
# 健康检查
curl http://localhost:8000/health

# 详细状态
curl http://localhost:8000/status | jq '.'
```

### 调试模式

在开发环境中，可以通过配置启用详细日志：

```json
{
  "logging": {
    "level": "DEBUG",
    "enable_request_logging": true,
    "enable_performance_logging": true
  }
}
``` 