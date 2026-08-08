"""completions（旧版文本补全）相关的响应类型。"""

from __future__ import annotations

from typing import Literal

from ._base import APIResponseModel
from .chat import CompletionUsage

__all__ = ["Completion", "CompletionChoice", "CompletionChunk", "CompletionChunkChoice", "CompletionLogprobs"]


type CompletionFinishReason = Literal["stop", "length", "content_filter"]


class CompletionLogprobs(APIResponseModel):
    text_offset: list[int] | None = None
    token_logprobs: list[float] | None = None
    tokens: list[str] | None = None
    top_logprobs: list[dict[str, float]] | None = None


class CompletionChoice(APIResponseModel):
    """单个生成候选。"""

    text: str
    index: int
    finish_reason: CompletionFinishReason | None = None
    logprobs: CompletionLogprobs | None = None


class Completion(APIResponseModel):
    """completions 的完整响应。"""

    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: CompletionUsage | None = None
    system_fingerprint: str | None = None


class CompletionChunkChoice(APIResponseModel):
    """流式 chunk 中的单个候选增量。"""

    text: str
    index: int
    finish_reason: CompletionFinishReason | None = None
    logprobs: CompletionLogprobs | None = None


class CompletionChunk(APIResponseModel):
    """SSE 流中的一条 chunk。"""

    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChunkChoice]
    usage: CompletionUsage | None = None
    system_fingerprint: str | None = None
