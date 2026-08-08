# openai-io

轻量级的 OpenAI 大模型 IO 库：只保留 `chat.completions` / `completions` / `embeddings`
三件套，同步（`OpenAI`）与异步（`AsyncOpenAI`）双客户端，传输层基于 `httpx`，
数据模型基于 `pydantic` v2。

## 为什么不用官方 SDK

官方 SDK 功能全，但用起来有几处不太顺手：

- 太重。模型 IO 只占它的一小部分，`images`、`audio`、`files`、`batch`、`assistants`
  这些资源平时根本用不到，依赖和包体积都跟着上去了。
- `messages` 是 TypedDict，`{"role": "user", "content": "..."}` 全靠手写，没有对象、
  没有自动补全，多轮对话拼起来很啰嗦。
- 类型太绕：`NotGiven` 哨兵、一长串 Union、各种 `*_Param`，IDE 提示经常是几行
  联合类型，报错也不好读。

这个库只做 chat / completions / embeddings 三件事，`messages` 换成 langchain 风格的
对象，接口和官方 SDK 保持一致，迁移成本低。

## 特性

- 轻量：只依赖 `httpx` 和 `pydantic`
- 同步 `OpenAI` 与异步 `AsyncOpenAI` 双客户端
- langchain 风格 message：`SystemMessage` / `HumanMessage` / `AIMessage` /
  `ToolMessage` / `FunctionMessage` / `ChatMessage`（不依赖 langchain-core）
- 流式输出（SSE）：`stream=True` 返回可迭代的 `Stream` / `AsyncStream`
- 异常体系与 openai SDK 对齐，自动重试（指数退避 + 抖动）

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
- `content` 支持多模态 part 列表（`[{"type": "text", ...}, {"type": "image_url", ...}]`）
- **兼容原始 dict**：`messages` 可以混合传入 `BaseMessage` 与 `{"role": ..., "content": ...}`，
  迁移时无需一次性替换全部代码
- pydantic 2.13 起构造参数为关键字形式：`HumanMessage(content="你好")`

## 与 openai SDK 的对应关系

| openai SDK | openai-io |
| --- | --- |
| `from openai import OpenAI` | `from openai_io import OpenAI` |
| `client.chat.completions.create(...)` | `client.chat.completions.create(...)` |
| `client.completions.create(...)` | `client.completions.create(...)` |
| `client.embeddings.create(...)` | `client.embeddings.create(...)` |
| `from openai import AsyncOpenAI` | `from openai_io import AsyncOpenAI` |
| `openai.OpenAIError` 等异常 | `openai_io.OpenAIError` 等，类名一致 |
| 请求参数（`temperature` / `max_tokens` / `tools` / …） | 同名同语义；未传参不写入请求体（`NotGiven` 哨兵语义一致） |
| TypedDict messages | langchain 风格 `BaseMessage`（也可传 dict） |

`create` 的入口参数与 openai SDK 对齐，含 `stream` / `stream_options` / `tools` /
`tool_choice` / `response_format` / `seed` 等；未显式传参的字段不会出现在请求体中。

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
