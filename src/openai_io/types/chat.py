"""chat.completions 相关的响应类型（字段与 openai SDK 一致）。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from ..messages import FunctionCall, ToolCall

__all__ = [
    "ChatCompletion",
    "ChatCompletionChunk",
    "ChatCompletionChunkChoice",
    "ChatCompletionChunkDelta",
    "ChatCompletionMessage",
    "Choice",
    "CompletionUsage",
]


class CompletionUsage(BaseModel):
    """token 用量统计。"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: dict[str, Any] | None = None
    completion_tokens_details: dict[str, Any] | None = None


class ChatCompletionMessage(BaseModel):
    """chat.completions 响应中的助手消息。"""

    role: Literal["assistant"] = "assistant"
    content: str | None = None
    refusal: str | None = None
    tool_calls: list[ToolCall] | None = None
    function_call: FunctionCall | None = None
    annotations: list[dict[str, Any]] | None = None
    audio: dict[str, Any] | None = None


class Choice(BaseModel):
    """单个生成候选。"""

    index: int
    message: ChatCompletionMessage
    finish_reason: str | None = None
    logprobs: dict[str, Any] | None = None


class ChatCompletion(BaseModel):
    """chat.completions 的完整响应。"""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: CompletionUsage | None = None
    system_fingerprint: str | None = None


class ChatCompletionChunkDelta(BaseModel):
    """流式增量内容。"""

    role: str | None = None
    content: str | None = None
    refusal: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    function_call: FunctionCall | None = None


class ChatCompletionChunkChoice(BaseModel):
    """流式 chunk 中的单个候选增量。"""

    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: str | None = None
    logprobs: dict[str, Any] | None = None


class ChatCompletionChunk(BaseModel):
    """SSE 流中的一条 chunk。"""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]
    system_fingerprint: str | None = None
    usage: CompletionUsage | None = None
