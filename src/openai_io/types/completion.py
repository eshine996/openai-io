"""completions（旧版文本补全）相关的响应类型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from .chat import CompletionUsage

__all__ = ["Completion", "CompletionChoice", "CompletionChunk", "CompletionChunkChoice"]


class CompletionChoice(BaseModel):
    """单个生成候选。"""

    text: str
    index: int
    finish_reason: str | None = None
    logprobs: dict[str, Any] | None = None


class Completion(BaseModel):
    """completions 的完整响应。"""

    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: CompletionUsage | None = None


class CompletionChunkChoice(BaseModel):
    """流式 chunk 中的单个候选增量。"""

    text: str
    index: int
    finish_reason: str | None = None
    logprobs: dict[str, Any] | None = None


class CompletionChunk(BaseModel):
    """SSE 流中的一条 chunk。"""

    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChunkChoice]
    usage: CompletionUsage | None = None
