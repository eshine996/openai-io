"""Chat Completions 响应类型。"""

from __future__ import annotations

from typing import Literal

from ..messages import FunctionCall, MessageToolCall
from ._base import APIResponseModel

__all__ = [
    "Annotation",
    "ChatCompletion",
    "ChatCompletionAudio",
    "ChatCompletionChunk",
    "ChatCompletionChunkChoice",
    "ChatCompletionChunkDelta",
    "ChatCompletionMessage",
    "ChatCompletionTokenLogprob",
    "Choice",
    "ChoiceLogprobs",
    "CompletionTokensDetails",
    "CompletionUsage",
    "CustomToolDelta",
    "FunctionCallDelta",
    "PromptTokensDetails",
    "ToolCallDelta",
    "TopLogprob",
    "URLCitation",
]


type FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
type ServiceTier = Literal["auto", "default", "flex", "scale", "priority", "fast"]


class PromptTokensDetails(APIResponseModel):
    audio_tokens: int | None = None
    cache_write_tokens: int | None = None
    cached_tokens: int | None = None


class CompletionTokensDetails(APIResponseModel):
    accepted_prediction_tokens: int | None = None
    audio_tokens: int | None = None
    reasoning_tokens: int | None = None
    rejected_prediction_tokens: int | None = None


class CompletionUsage(APIResponseModel):
    """token 用量统计。"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: PromptTokensDetails | None = None
    completion_tokens_details: CompletionTokensDetails | None = None


class URLCitation(APIResponseModel):
    end_index: int
    start_index: int
    title: str
    url: str


class Annotation(APIResponseModel):
    type: Literal["url_citation"]
    url_citation: URLCitation


class ChatCompletionAudio(APIResponseModel):
    id: str
    data: str
    expires_at: int
    transcript: str


class TopLogprob(APIResponseModel):
    token: str
    bytes: list[int] | None = None
    logprob: float


class ChatCompletionTokenLogprob(TopLogprob):
    top_logprobs: list[TopLogprob]


class ChoiceLogprobs(APIResponseModel):
    content: list[ChatCompletionTokenLogprob] | None = None
    refusal: list[ChatCompletionTokenLogprob] | None = None


class ChatCompletionMessage(APIResponseModel):
    """chat.completions 响应中的助手消息。"""

    role: Literal["assistant"] = "assistant"
    content: str | None = None
    refusal: str | None = None
    tool_calls: list[MessageToolCall] | None = None
    function_call: FunctionCall | None = None
    annotations: list[Annotation] | None = None
    audio: ChatCompletionAudio | None = None


class Choice(APIResponseModel):
    """单个生成候选。"""

    index: int
    message: ChatCompletionMessage
    finish_reason: FinishReason | None = None
    logprobs: ChoiceLogprobs | None = None


class ChatCompletion(APIResponseModel):
    """chat.completions 的完整响应。"""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: CompletionUsage | None = None
    system_fingerprint: str | None = None
    service_tier: ServiceTier | None = None


class FunctionCallDelta(APIResponseModel):
    arguments: str | None = None
    name: str | None = None


class CustomToolDelta(APIResponseModel):
    input: str | None = None
    name: str | None = None


class ToolCallDelta(APIResponseModel):
    index: int
    id: str | None = None
    type: Literal["function", "custom"] | None = None
    function: FunctionCallDelta | None = None
    custom: CustomToolDelta | None = None


class ChatCompletionChunkDelta(APIResponseModel):
    """流式增量内容。"""

    role: Literal["assistant"] | None = None
    content: str | None = None
    refusal: str | None = None
    tool_calls: list[ToolCallDelta] | None = None
    function_call: FunctionCallDelta | None = None


class ChatCompletionChunkChoice(APIResponseModel):
    """流式 chunk 中的单个候选增量。"""

    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: FinishReason | None = None
    logprobs: ChoiceLogprobs | None = None


class ChatCompletionChunk(APIResponseModel):
    """SSE 流中的一条 chunk。"""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]
    system_fingerprint: str | None = None
    service_tier: ServiceTier | None = None
    usage: CompletionUsage | None = None
