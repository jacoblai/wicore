"""
WiCore API服务器
基于FastAPI实现OpenAI兼容的推理接口

核心功能:
- OpenAI兼容的API接口
- 流式和非流式文本生成
- 健康检查和状态监控
- 请求限流和负载均衡
- 完整的错误处理
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from ..core.inference_engine import InferenceEngine, InferenceRequest, InferenceResponse
from ..core.config import WiCoreConfig, get_config_manager

logger = logging.getLogger(__name__)


# ==================== 请求/响应模型 ====================

class ChatMessage(BaseModel):
    """聊天消息"""
    role: str = Field(..., description="消息角色: system, user, assistant")
    content: str = Field(..., description="消息内容")


class ChatCompletionRequest(BaseModel):
    """聊天完成请求"""
    model: str = Field(default="gemma-3-27b-it", description="模型名称")
    messages: List[ChatMessage] = Field(..., description="对话消息列表")
    max_tokens: Optional[int] = Field(default=512, ge=1, le=8192, description="最大生成token数")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0, description="核采样参数")
    top_k: Optional[int] = Field(default=50, ge=1, le=100, description="Top-K采样")
    stream: Optional[bool] = Field(default=False, description="是否流式返回")
    stop: Optional[List[str]] = Field(default=None, description="停止序列")


class ChatCompletionResponseChoice(BaseModel):
    """聊天完成响应选择"""
    index: int
    message: ChatMessage
    finish_reason: str  # stop, length, error


class ChatCompletionResponse(BaseModel):
    """聊天完成响应"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Dict[str, int]


class ChatCompletionStreamChunk(BaseModel):
    """流式聊天完成响应块"""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    timestamp: float
    version: str
    uptime: float


class StatusResponse(BaseModel):
    """系统状态响应"""
    engine_ready: bool
    model_loaded: bool
    stats: Dict[str, Any]
    config: Dict[str, Any]


# ==================== 全局状态 ====================

class APIServerState:
    """API服务器状态"""
    
    def __init__(self):
        self.inference_engine: Optional[InferenceEngine] = None
        self.config: Optional[WiCoreConfig] = None
        self.start_time = time.time()
        self.request_count = 0


# 全局状态实例
api_state = APIServerState()


# ==================== 依赖注入 ====================

async def get_inference_engine() -> InferenceEngine:
    """获取推理引擎依赖"""
    if not api_state.inference_engine or not api_state.inference_engine.is_ready():
        raise HTTPException(
            status_code=503,
            detail="推理引擎未就绪，请稍后重试"
        )
    return api_state.inference_engine


# ==================== 生命周期管理 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    logger.info("🚀 启动WiCore API服务器...")
    
    try:
        # 加载配置
        config_manager = get_config_manager()
        api_state.config = config_manager.get_config()
        
        if not api_state.config:
            # 如果没有配置，加载默认配置
            api_state.config = config_manager.load_config(environment="development")
        
        # 初始化推理引擎
        api_state.inference_engine = InferenceEngine(api_state.config)
        
        # 启动推理引擎
        success = await api_state.inference_engine.initialize()
        if not success:
            raise RuntimeError("推理引擎初始化失败")
        
        logger.info("✅ WiCore API服务器启动成功")
        
        yield  # 服务器运行期间
        
    except Exception as e:
        logger.error(f"❌ API服务器启动失败: {e}")
        raise
    
    finally:
        # 关闭时清理
        logger.info("🛑 关闭WiCore API服务器...")
        
        if api_state.inference_engine:
            await api_state.inference_engine.shutdown()
        
        logger.info("👋 WiCore API服务器已关闭")


# ==================== FastAPI应用 ====================

app = FastAPI(
    title="WiCore API",
    description="世界级高性能LLM推理引擎 - OpenAI兼容API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 中间件 ====================

@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """请求中间件"""
    start_time = time.time()
    api_state.request_count += 1
    
    # 添加请求ID
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # 记录请求
    logger.info(f"收到请求 {request_id}: {request.method} {request.url}")
    
    try:
        response = await call_next(request)
        
        # 记录响应时间
        process_time = time.time() - start_time
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        
        logger.info(f"请求完成 {request_id}: {response.status_code} ({process_time:.3f}s)")
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"请求失败 {request_id}: {e} ({process_time:.3f}s)")
        raise


# ==================== API路由 ====================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version="1.0.0",
        uptime=time.time() - api_state.start_time
    )


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """获取系统状态"""
    engine_ready = (
        api_state.inference_engine is not None and 
        api_state.inference_engine.is_ready()
    )
    
    model_loaded = False
    stats = {}
    config_info = {}
    
    if api_state.inference_engine:
        model_loaded = (
            api_state.inference_engine.model_loader and
            api_state.inference_engine.model_loader.is_model_loaded()
        )
        stats = api_state.inference_engine.get_stats()
    
    if api_state.config:
        config_info = {
            "model_name": api_state.config.model.model_name,
            "model_type": api_state.config.model.model_type,
            "max_batch_size": api_state.config.model.max_batch_size,
            "hmt_enabled": api_state.config.hmt.enable_hmt,
            "mor_enabled": api_state.config.mor.enable_mor,
        }
    
    return StatusResponse(
        engine_ready=engine_ready,
        model_loaded=model_loaded,
        stats=stats,
        config=config_info
    )


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    engine: InferenceEngine = Depends(get_inference_engine)
):
    """OpenAI兼容的聊天完成接口"""
    request_id = str(uuid.uuid4())
    
    try:
        # 构建推理请求
        inference_request = InferenceRequest(
            request_id=request_id,
            messages=[{"role": msg.role, "content": msg.content} for msg in request.messages],
            max_tokens=request.max_tokens or 512,
            temperature=request.temperature or 0.7,
            top_p=request.top_p or 0.9,
            top_k=request.top_k or 50,
            stream=request.stream or False,
            stop_sequences=request.stop
        )
        
        if request.stream:
            # 流式响应
            return StreamingResponse(
                stream_chat_completions(engine, inference_request, request.model),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Request-ID": request_id
                }
            )
        else:
            # 非流式响应
            response = await engine.generate_text(inference_request)
            
            return ChatCompletionResponse(
                id=request_id,
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=response.text),
                        finish_reason=response.finish_reason
                    )
                ],
                usage={
                    "prompt_tokens": len(inference_request.input_text.split()) if inference_request.input_text else 0,
                    "completion_tokens": response.tokens_generated,
                    "total_tokens": len(inference_request.input_text.split()) + response.tokens_generated
                }
            )
            
    except Exception as e:
        logger.error(f"聊天完成请求失败: {e}")
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")


async def stream_chat_completions(
    engine: InferenceEngine,
    inference_request: InferenceRequest, 
    model_name: str
) -> AsyncGenerator[str, None]:
    """流式聊天完成生成器"""
    request_id = inference_request.request_id
    created = int(time.time())
    
    try:
        # 发送开始事件
        start_chunk = ChatCompletionStreamChunk(
            id=request_id,
            created=created,
            model=model_name,
            choices=[{
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None
            }]
        )
        yield f"data: {start_chunk.model_dump_json()}\n\n"
        
        # 流式生成文本
        async for chunk in engine.generate_stream(inference_request):
            if chunk.strip():  # 忽略空块
                stream_chunk = ChatCompletionStreamChunk(
                    id=request_id,
                    created=created,
                    model=model_name,
                    choices=[{
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None
                    }]
                )
                yield f"data: {stream_chunk.model_dump_json()}\n\n"
        
        # 发送结束事件
        end_chunk = ChatCompletionStreamChunk(
            id=request_id,
            created=created,
            model=model_name,
            choices=[{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        )
        yield f"data: {end_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"流式生成失败: {e}")
        error_chunk = ChatCompletionStreamChunk(
            id=request_id,
            created=created,
            model=model_name,
            choices=[{
                "index": 0,
                "delta": {"content": f"[ERROR] {str(e)}"},
                "finish_reason": "error"
            }]
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"


@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    models = []
    
    if api_state.config and api_state.config.model.model_name:
        models.append({
            "id": api_state.config.model.model_name,
            "object": "model",
            "created": int(api_state.start_time),
            "owned_by": "wicore"
        })
    
    return {"object": "list", "data": models}


# ==================== 错误处理 ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "type": "http_error",
                "code": exc.status_code
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理"""
    logger.error(f"未处理异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "内部服务器错误",
                "type": "internal_error",
                "code": 500
            }
        }
    )


# ==================== 服务器启动函数 ====================

def create_app(config: Optional[WiCoreConfig] = None) -> FastAPI:
    """创建FastAPI应用"""
    if config:
        api_state.config = config
    return app


async def start_server(
    config: Optional[WiCoreConfig] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1
):
    """启动API服务器"""
    if config:
        api_state.config = config
        host = config.server.host
        port = config.server.port
        workers = config.server.workers
    
    # 配置uvicorn
    uvicorn_config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        access_log=True
    )
    
    server = uvicorn.Server(uvicorn_config)
    
    logger.info(f"🌐 启动API服务器: http://{host}:{port}")
    await server.serve()


if __name__ == "__main__":
    # 直接运行时的入口点
    asyncio.run(start_server()) 