"""Chat Completions 消息类型。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any, ClassVar, Literal, NotRequired, Self, TypedDict

from pydantic import BaseModel, Field, model_validator

__all__ = [
    "AIMessage",
    "AssistantMessageContent",
    "BaseMessage",
    "ChatMessage",
    "CustomTool",
    "CustomToolCall",
    "DeveloperMessage",
    "FileContentPart",
    "FileInput",
    "FunctionCall",
    "FunctionMessage",
    "HumanMessage",
    "ImageContentPart",
    "ImageURL",
    "InputAudio",
    "InputAudioContentPart",
    "MessageContent",
    "MessageContentPart",
    "MessageLike",
    "MessageToolCall",
    "PromptCacheBreakpoint",
    "RefusalContentPart",
    "SystemMessage",
    "TextContentPart",
    "TextMessageContent",
    "ToolCall",
    "ToolMessage",
    "UserMessageContent",
    "to_openai_messages",
]


class PromptCacheBreakpoint(TypedDict):
    """可复用提示词前缀的显式边界。"""

    mode: Literal["explicit"]


class TextContentPart(TypedDict):
    """文本内容 part。"""

    type: Literal["text"]
    text: str
    prompt_cache_breakpoint: NotRequired[PromptCacheBreakpoint]


class ImageURL(TypedDict):
    """图片 URL 或 base64 data URL。"""

    url: str
    detail: NotRequired[Literal["auto", "low", "high"]]


class ImageContentPart(TypedDict):
    """图片输入 part。"""

    type: Literal["image_url"]
    image_url: ImageURL
    prompt_cache_breakpoint: NotRequired[PromptCacheBreakpoint]


class InputAudio(TypedDict):
    """base64 编码的音频输入。"""

    data: str
    format: Literal["wav", "mp3"]


class InputAudioContentPart(TypedDict):
    """音频输入 part。"""

    type: Literal["input_audio"]
    input_audio: InputAudio
    prompt_cache_breakpoint: NotRequired[PromptCacheBreakpoint]


class FileInput(TypedDict):
    """文件输入；使用 file_id，或同时提供 file_data 与 filename。"""

    file_data: NotRequired[str]
    file_id: NotRequired[str]
    filename: NotRequired[str]


class FileContentPart(TypedDict):
    """文件输入 part。"""

    type: Literal["file"]
    file: FileInput
    prompt_cache_breakpoint: NotRequired[PromptCacheBreakpoint]


class RefusalContentPart(TypedDict):
    """助手拒绝回答的内容 part。"""

    type: Literal["refusal"]
    refusal: str


type MessageContentPart = (
    TextContentPart | ImageContentPart | InputAudioContentPart | FileContentPart | RefusalContentPart
)
type UserMessageContent = str | list[TextContentPart | ImageContentPart | InputAudioContentPart | FileContentPart]
type AssistantMessageContent = str | list[TextContentPart | RefusalContentPart] | None
type TextMessageContent = str | list[TextContentPart]
type MessageContent = str | list[MessageContentPart]

#: create 接口 ``messages`` 参数接受的元素类型：langchain 风格消息或原始 dict。
type MessageLike = BaseMessage[Any] | Mapping[str, Any]


class FunctionCall(BaseModel):
    """openai 格式的 function call。"""

    name: str
    arguments: str


class ToolCall(BaseModel):
    """函数工具调用。"""

    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class CustomTool(BaseModel):
    """自定义工具调用内容。"""

    name: str
    input: str


class CustomToolCall(BaseModel):
    """自定义工具调用。"""

    id: str
    type: Literal["custom"] = "custom"
    custom: CustomTool


type MessageToolCall = ToolCall | CustomToolCall


class BaseMessage[ContentT](BaseModel, ABC):
    """所有消息的抽象基类。"""

    content: ContentT
    additional_kwargs: dict[str, Any] = Field(default_factory=dict)
    response_metadata: dict[str, Any] = Field(default_factory=dict)
    name: str | None = None
    id: str | None = None

    type: ClassVar[str]

    @abstractmethod
    def _role(self) -> str:
        """对应的 openai role（"system" / "user" / "assistant" / "tool" / "function"）。"""

    def to_openai_dict(self) -> dict[str, Any]:
        """转换为 openai chat.completions 请求中的消息 dict。

        注意：``additional_kwargs`` / ``response_metadata`` / ``id`` 属于本地元数据，
        不会发送给 API。
        """
        data: dict[str, Any] = {"role": self._role(), "content": self.content}
        if self.name is not None:
            data["name"] = self.name
        return data

    def __str__(self) -> str:
        content = self.content if isinstance(self.content, str) else "[多模态内容]"
        return f"{self.type}: {content}"


class SystemMessage(BaseMessage[TextMessageContent]):
    """系统消息（role=system）。"""

    type: ClassVar[str] = "system"

    def _role(self) -> Literal["system"]:
        return "system"


class DeveloperMessage(BaseMessage[TextMessageContent]):
    """开发者消息（role=developer）。"""

    type: ClassVar[str] = "developer"

    def _role(self) -> Literal["developer"]:
        return "developer"


class HumanMessage(BaseMessage[UserMessageContent]):
    """用户消息（role=user）。"""

    type: ClassVar[str] = "human"

    def _role(self) -> Literal["user"]:
        return "user"


class AIMessage(BaseMessage[AssistantMessageContent]):
    """助手消息（role=assistant），可携带 tool_calls / function_call。"""

    content: AssistantMessageContent = None
    tool_calls: list[MessageToolCall] | None = None
    function_call: FunctionCall | None = None

    type: ClassVar[str] = "ai"

    def _role(self) -> Literal["assistant"]:
        return "assistant"

    def to_openai_dict(self) -> dict[str, Any]:
        data = super().to_openai_dict()
        if self.tool_calls is not None:
            data["tool_calls"] = [call.model_dump() for call in self.tool_calls]
        if self.function_call is not None:
            data["function_call"] = self.function_call.model_dump()
        return data


class ToolMessage(BaseMessage[TextMessageContent]):
    """工具执行结果消息（role=tool），需携带对应的 ``tool_call_id``。"""

    tool_call_id: str
    # 工具返回的原始产物只保留在本地。
    artifact: Any | None = None

    type: ClassVar[str] = "tool"

    def _role(self) -> Literal["tool"]:
        return "tool"

    def to_openai_dict(self) -> dict[str, Any]:
        data = super().to_openai_dict()
        data["tool_call_id"] = self.tool_call_id
        return data


class FunctionMessage(BaseMessage[str | None]):
    """旧版 function 调用结果消息（role=function）。"""

    type: ClassVar[str] = "function"

    @model_validator(mode="after")
    def _require_name(self) -> Self:
        if self.name is None:
            raise ValueError("function 消息必须提供 name")
        return self

    def _role(self) -> Literal["function"]:
        return "function"


class ChatMessage(BaseMessage[MessageContent]):
    """自定义 role 的消息，role 由调用方指定。"""

    role: str = Field(...)

    type: ClassVar[str] = "chat"

    def _role(self) -> str:
        return self.role


def to_openai_messages(messages: Iterable[MessageLike]) -> list[dict[str, Any]]:
    """把 ``BaseMessage`` 与原始 ``dict`` 混合的消息列表转为 openai 请求格式。

    Raises:
        TypeError: 列表中存在既不是 ``BaseMessage`` 也不是 ``dict`` 的元素。
    """
    result: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, BaseMessage):
            result.append(message.to_openai_dict())
        else:
            result.append(dict(message))
    return result
