"""message 体系测试：构造、转换、混合输入。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from openai_io import (
    AIMessage,
    ChatMessage,
    CustomTool,
    CustomToolCall,
    DeveloperMessage,
    FileContentPart,
    FunctionCall,
    FunctionMessage,
    HumanMessage,
    ImageContentPart,
    InputAudioContentPart,
    SystemMessage,
    TextContentPart,
    ToolCall,
    ToolMessage,
    to_openai_messages,
)


def test_human_message_to_openai_dict() -> None:
    message = HumanMessage(content="你好")
    assert message.type == "human"
    assert message.to_openai_dict() == {"role": "user", "content": "你好"}


def test_system_message() -> None:
    assert SystemMessage(content="你是助手").to_openai_dict() == {"role": "system", "content": "你是助手"}


def test_developer_message() -> None:
    assert DeveloperMessage(content="遵循项目约定").to_openai_dict() == {
        "role": "developer",
        "content": "遵循项目约定",
    }


def test_ai_message_with_tool_calls() -> None:
    message = AIMessage(
        content="让我查一下天气",
        tool_calls=[
            ToolCall(id="call_1", function=FunctionCall(name="get_weather", arguments='{"city": "北京"}')),
        ],
    )
    data = message.to_openai_dict()
    assert data["role"] == "assistant"
    assert data["tool_calls"][0]["id"] == "call_1"  # type: ignore[index]
    assert data["tool_calls"][0]["function"]["name"] == "get_weather"  # type: ignore[index]


def test_ai_message_with_tool_calls_allows_null_content() -> None:
    message = AIMessage(
        content=None,
        tool_calls=[ToolCall(id="call_1", function=FunctionCall(name="get_weather", arguments="{}"))],
    )
    assert message.to_openai_dict()["content"] is None


def test_ai_message_with_custom_tool_call() -> None:
    message = AIMessage(tool_calls=[CustomToolCall(id="call_1", custom=CustomTool(name="shell", input="pwd"))])
    assert message.to_openai_dict()["tool_calls"] == [
        {"id": "call_1", "type": "custom", "custom": {"name": "shell", "input": "pwd"}}
    ]


def test_tool_message() -> None:
    message = ToolMessage(content="晴，25°C", tool_call_id="call_1")
    assert message.to_openai_dict() == {"role": "tool", "content": "晴，25°C", "tool_call_id": "call_1"}


def test_tool_message_requires_tool_call_id() -> None:
    with pytest.raises(ValidationError):
        ToolMessage(content="ok")  # type: ignore[call-arg]


def test_tool_message_artifact_not_serialized() -> None:
    message = ToolMessage(content="ok", tool_call_id="call_1", artifact={"raw": [1, 2, 3]})
    assert "artifact" not in message.to_openai_dict()


def test_function_message() -> None:
    message = FunctionMessage(content="结果是 42", name="compute")
    data = message.to_openai_dict()
    assert data["role"] == "function"
    assert data["name"] == "compute"


def test_function_message_requires_name() -> None:
    with pytest.raises(ValidationError):
        FunctionMessage(content="ok")  # type: ignore[call-arg]


def test_chat_message_custom_role() -> None:
    message = ChatMessage(content="内容", role="user-voice")
    assert message.type == "chat"
    assert message.to_openai_dict() == {"role": "user-voice", "content": "内容"}


def test_message_with_name() -> None:
    message = HumanMessage(content="hi", name="alice")
    assert message.to_openai_dict()["name"] == "alice"


def test_multimodal_content() -> None:
    parts: list[TextContentPart | ImageContentPart | InputAudioContentPart | FileContentPart] = [
        {"type": "text", "text": "描述这张图"},
        {"type": "image_url", "image_url": {"url": "https://example.com/pic.png", "detail": "high"}},
    ]
    message = HumanMessage(content=parts)
    assert message.to_openai_dict()["content"] == parts


def test_multimodal_content_validates_part_structure() -> None:
    with pytest.raises(ValidationError):
        HumanMessage(content=[{"type": "image_url", "image_url": {"detail": "high"}}])  # type: ignore[list-item]


def test_to_openai_messages_mixed() -> None:
    result = to_openai_messages(
        [
            SystemMessage(content="你是助手"),
            {"role": "user", "content": "你好"},
            HumanMessage(content="在吗"),
        ]
    )
    assert result == [
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": "你好"},
        {"role": "user", "content": "在吗"},
    ]


def test_to_openai_messages_rejects_invalid_element() -> None:
    with pytest.raises(TypeError):
        to_openai_messages([123])  # type: ignore[list-item]


def test_str_representation() -> None:
    assert str(HumanMessage(content="你好")) == "human: 你好"
