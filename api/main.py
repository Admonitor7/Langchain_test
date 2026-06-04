"""FastAPI 代码库问答接口。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from code_qa_service import CodeQAService, to_dict
from config import load_settings
from storage.chat_store import ChatStore


app = FastAPI(title="代码库问答系统", version="1.0.0")
settings = load_settings()
service = CodeQAService(settings)
chat_store = ChatStore(settings.code_index_dir / "chats")


class IndexRequest(BaseModel):
    """索引构建请求。"""

    repo_path: str | None = Field(default=None, description="代码库目录，默认使用 .env 中的 INDEX_REPO_PATH")
    reset: bool = Field(default=True, description="是否清空后重建索引")


class ChatRequest(BaseModel):
    """聊天请求。"""

    question: str = Field(..., min_length=1, description="用户问题")
    top_k: int = Field(default=8, ge=1, le=20, description="检索片段数量")
    stream: bool = Field(default=True, description="是否使用 SSE 流式返回")


@app.get("/health")
def health() -> dict[str, str]:
    """健康检查。"""

    return {"status": "ok"}


@app.post("/index")
async def build_index(request: IndexRequest) -> dict[str, Any]:
    """触发代码库索引构建。"""

    result = await asyncio.to_thread(
        service.build_index,
        Path(request.repo_path) if request.repo_path else None,
        request.reset,
    )
    return to_dict(result)


@app.delete("/index")
async def clear_index() -> dict[str, str]:
    """清空代码索引。"""

    return await asyncio.to_thread(service.clear_index)


@app.post("/chat")
def chat(request: ChatRequest):
    """代码库问答。"""

    if request.stream:
        return StreamingResponse(
            _stream_chat(request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    result = service.ask(request.question, top_k=request.top_k)
    chat_id = chat_store.create_id()
    record = chat_store.save(chat_id, request.question, result.answer, result.sources)
    return {
        "chat_id": record.chat_id,
        "answer": result.answer,
        "sources": result.sources,
    }


@app.get("/chat/{chat_id}")
def get_chat(chat_id: str) -> dict[str, Any]:
    """获取历史对话。"""

    record = chat_store.get(chat_id)
    if record is None:
        raise HTTPException(status_code=404, detail="聊天记录不存在。")
    return {
        "chat_id": record.chat_id,
        "question": record.question,
        "answer": record.answer,
        "sources": record.sources,
        "created_at": record.created_at,
    }


def _stream_chat(request: ChatRequest):
    """SSE 流式聊天生成器。"""

    chat_id = chat_store.create_id()
    answer_parts: list[str] = []
    sources, _, token_generator = service.stream_answer(request.question, top_k=request.top_k)
    yield _sse("metadata", {"chat_id": chat_id, "sources": sources})

    try:
        for token in token_generator:
            answer_parts.append(token)
            yield _sse("delta", {"content": token})
    except Exception as error:
        yield _sse("error", {"message": str(error)})
        return

    answer = "".join(answer_parts)
    chat_store.save(chat_id, request.question, answer, sources)
    yield _sse("done", {"chat_id": chat_id})


def _sse(event: str, data: dict[str, Any]) -> str:
    """格式化 SSE 事件。"""

    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
