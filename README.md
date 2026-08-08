# openai-io

一个只封装 `chat.completions`、`completions` 和 `embeddings` 的 OpenAI 客户端。
提供同步与异步接口，使用 `httpx` 发送请求，使用 Pydantic v2 解析响应。

## 为什么写这个库

我的项目只用到模型调用，不需要官方 SDK 覆盖的全部资源。这个库主要解决两个问题：

- 缩小接口范围，只维护模型调用需要的资源。
- 用消息对象组织多轮对话，同时允许直接传原始 `dict`。

资源路径和常用参数沿用 OpenAI API 的命名，但这不是官方 SDK 的完整替代品。

## 特性

- 运行时只依赖 `httpx` 和 `pydantic`
- 同步 `OpenAI` 与异步 `AsyncOpenAI` 双客户端
- 消息对象：`DeveloperMessage` / `SystemMessage` / `HumanMessage` / `AIMessage` /
  `ToolMessage` / `FunctionMessage` / `ChatMessage`（不依赖 langchain-core）
- 流式输出（SSE）：`stream=True` 返回可迭代的 `Stream` / `AsyncStream`
- 为常见 HTTP 状态和连接错误提供独立异常类型
- 对限流、服务端错误等状态执行有限次数的重试

## 安装

```bash
pip install -e .
```

要求 Python >= 3.12。

## 快速开始

### 同步

```python
from openai_io import OpenAI
from openai_io.messages import HumanMessage, SystemMessage

client = OpenAI()  # 或 OpenAI(api_key="sk-...")

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        SystemMessage(content="你是一个友好的助手"),
        HumanMessage(content="你好"),
    ],
)
print(resp.choices[0].message.content)

# embeddings
vectors = client.embeddings.create(model="text-embedding-3-small", input="你好")
print(vectors.data[0].embedding[:5])

# 旧版文本补全
comp = client.completions.create(model="gpt-3.5-turbo-instruct", prompt="1+1=?")
print(comp.choices[0].text)
```

### 异步

```python
import asyncio

from openai_io import AsyncOpenAI
from openai_io.messages import HumanMessage

async def main() -> None:
    client = AsyncOpenAI()
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[HumanMessage(content="你好")],
    )
    print(resp.choices[0].message.content)

asyncio.run(main())
```

### 流式输出

```python
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[HumanMessage(content="讲个故事")],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="")
```

异步流式使用 `async for chunk in await client.chat.completions.create(..., stream=True)`。

## Message 体系（与原生 openai 的差异）

原生 openai SDK 的 messages 参数是 TypedDict 风格（`{"role": ..., "content": ...}`），
本库改为 langchain 风格的类：

```python
from openai_io import AIMessage, FunctionCall, HumanMessage, SystemMessage, ToolCall, ToolMessage

messages = [
    SystemMessage(content="你是助手"),
    HumanMessage(content="北京天气如何？"),
    AIMessage(
        content=None,
        tool_calls=[ToolCall(id="call_1", function=FunctionCall(name="get_weather", arguments='{"city": "北京"}'))],
    ),
    ToolMessage(content="晴，25°C", tool_call_id="call_1"),
]
```

- `BaseMessage` 抽象基类：`content` / `additional_kwargs` / `response_metadata` /
  `name` / `id`，`type` 为类别标识（`"human"` / `"ai"` / `"tool"` …）
- `content` 支持结构化多模态 part 列表：`TextContentPart`、`ImageContentPart`、
  `InputAudioContentPart`、`FileContentPart` 和 `RefusalContentPart`；各 part 的字段有
  明确的类型提示。`HumanMessage` 支持文本、图片、音频和文件，`DeveloperMessage` /
  `SystemMessage` / `ToolMessage` 只支持文本，`AIMessage` 还支持拒答 part，并允许在工具调用时使用
  `content=None`。
- `messages` 可以混合传入 `BaseMessage` 与 `{"role": ..., "content": ...}`。
- 消息使用关键字参数构造，例如 `HumanMessage(content="你好")`。

## 与 openai SDK 的对应关系

| openai SDK | openai-io |
| --- | --- |
| `from openai import OpenAI` | `from openai_io import OpenAI` |
| `client.chat.completions.create(...)` | `client.chat.completions.create(...)` |
| `client.completions.create(...)` | `client.completions.create(...)` |
| `client.embeddings.create(...)` | `client.embeddings.create(...)` |
| `from openai import AsyncOpenAI` | `from openai_io import AsyncOpenAI` |
| 常见 API 异常 | `openai_io` 提供相近名称的异常类型 |
| 常用请求参数 | 沿用 OpenAI API 的字段名；未传参数不会写入请求体 |
| TypedDict messages | langchain 风格 `BaseMessage`（也可传 dict） |

本库只声明当前维护的常用参数，不保证与任一版本的官方 SDK 签名完全相同。未声明的
API 参数不能直接传给 `create`。

字段定义以 OpenAI 的 [Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)、
[Completions](https://developers.openai.com/api/reference/resources/completions/methods/create) 和
[Embeddings](https://developers.openai.com/api/reference/resources/embeddings/methods/create) 文档为准。

响应会解析为 Pydantic 模型。已声明的嵌套字段（例如 usage、logprobs、引用和工具调用）
也会继续解析为模型；API 新增但本库尚未声明的字段保留在 `model_extra`。Embeddings 默认
返回浮点数组；显式使用 `encoding_format="base64"` 时，`embedding` 保留 base64 字符串。

## 开发

```bash
uv sync --extra dev
uv run ruff check src tests && uv run ruff format --check src tests
uv run pyright
uv run pytest
```

测试使用 `httpx.MockTransport` 注入 mock 响应，无需真实 API key。

## 许可证

Apache-2.0
