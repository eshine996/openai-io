"""langchain 风格的 message 体系（不依赖 langchain-core）。

- :class:`BaseMessage` 抽象基类：``content`` / ``additional_kwargs`` /
  ``response_metadata`` / ``name`` / ``id``；``type`` 为 ``ClassVar``（消息类别，
  如 ``"human"`` / ``"ai"`` / ``"tool"``），``_role()`` 返回对应的 openai role。
- 具体类型：``SystemMessage`` / ``HumanMessage`` / ``AIMessage`` /
  ``ToolMessage`` / ``FunctionMessage`` / ``ChatMessage``。
- :meth:`BaseMessage.to_openai_dict` 转 openai 请求格式；:func:`to_openai_messages`
  接受 ``BaseMessage`` 与原始 ``dict`` 混合列表。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

__all__ = [
    "AIMessage",
    "BaseMessage",
    "ChatMessage",
    "FunctionCall",
    "FunctionMessage",
    "HumanMessage",
    "MessageContent",
    "MessageLike",
    "SystemMessage",
    "ToolCall",
    "ToolMessage",
    "to_openai_messages",
]

#: 消息内容：纯文本，或 openai 多模态 part 列表
#: （如 ``[{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {...}}]``）。
type MessageContent = str | list[dict[str, Any]]

#: create 接口 ``messages`` 参数接受的元素类型：langchain 风格消息或原始 dict。
type MessageLike = BaseMessage | Mapping[str, Any]


class FunctionCall(BaseModel):
    """openai 格式的 function call。"""

    name: str
    arguments: str


class ToolCall(BaseModel):
    """openai 格式的 tool call（chat.completions 响应与请求共用）。"""

    id: str
    type: str = "function"
    function: FunctionCall


class BaseMessage(BaseModel, ABC):
    """所有消息的抽象基类（langchain 风格）。

    实例化时 ``content`` 是第一个字段（pydantic 2.13 起构造参数为关键字形式）：
    ``HumanMessage(content="你好")``。
    """

    content: MessageContent
    additional_kwargs: dict[str, Any] = Field(default_factory=dict)
    response_metadata: dict[str, Any] = Field(default_factory=dict)
    name: str | None = None
    id: str | None = None

    #: 消息类别标识（如 "human" / "ai" / "tool"）。
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


class SystemMessage(BaseMessage):
    """系统消息（role=system）。"""

    type: ClassVar[str] = "system"

    def _role(self) -> Literal["system"]:
        return "system"


class HumanMessage(BaseMessage):
    """用户消息（role=user）。"""

    type: ClassVar[str] = "human"

    def _role(self) -> Literal["user"]:
        return "user"


class AIMessage(BaseMessage):
    """助手消息（role=assistant），可携带 tool_calls / function_call。"""

    tool_calls: list[ToolCall] | None = None
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


class ToolMessage(BaseMessage):
    """工具执行结果消息（role=tool），需携带对应的 ``tool_call_id``。"""

    tool_call_id: str | None = None
    #: 工具返回的原始产物，仅本地使用，不会发送给 API。
    artifact: Any | None = None

    type: ClassVar[str] = "tool"

    def _role(self) -> Literal["tool"]:
        return "tool"

    def to_openai_dict(self) -> dict[str, Any]:
        data = super().to_openai_dict()
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        return data


class FunctionMessage(BaseMessage):
    """旧版 function 调用结果消息（role=function）。"""

    type: ClassVar[str] = "function"

    def _role(self) -> Literal["function"]:
        return "function"


class ChatMessage(BaseMessage):
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
            # 直接转 dict：非法输入会在 dict() 时报错
            result.append(dict(message))
    return result
