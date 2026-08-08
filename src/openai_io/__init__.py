"""openai-io：轻量级的 OpenAI 大模型 IO 库。

只保留大模型 IO 三件套（chat completions / completions / embeddings），
同步（:class:`OpenAI`）与异步（:class:`AsyncOpenAI`）双客户端，基于 httpx。
消息体系采用 langchain 风格（见 :mod:`openai_io.messages`）。

快速开始::

    from openai_io import OpenAI
    from openai_io.messages import HumanMessage

    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[HumanMessage(content="你好")],
    )
    print(resp.choices[0].message.content)
"""

from __future__ import annotations

from ._exceptions import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)
from ._streaming import AsyncStream, Stream
from ._types import NOT_GIVEN, NotGiven
from .async_client import AsyncOpenAI
from .client import OpenAI
from .messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionCall,
    FunctionMessage,
    HumanMessage,
    MessageContent,
    MessageLike,
    SystemMessage,
    ToolCall,
    ToolMessage,
    to_openai_messages,
)
from .resources import (
    AsyncChat,
    AsyncChatCompletions,
    AsyncCompletions,
    AsyncEmbeddings,
    Chat,
    ChatCompletions,
    Completions,
    Embeddings,
)
from .types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionMessage,
    Choice,
    Completion,
    CompletionChoice,
    CompletionChunk,
    CompletionChunkChoice,
    CompletionUsage,
    CreateEmbeddingResponse,
    Embedding,
    EmbeddingUsage,
)

__version__ = "0.1.1"

__all__ = [
    "NOT_GIVEN",
    "AIMessage",
    "APIConnectionError",
    "APIError",
    "APIStatusError",
    "APITimeoutError",
    "AsyncChat",
    "AsyncChatCompletions",
    "AsyncCompletions",
    "AsyncEmbeddings",
    "AsyncOpenAI",
    "AsyncStream",
    "AuthenticationError",
    "BadRequestError",
    "BaseMessage",
    "Chat",
    "ChatCompletion",
    "ChatCompletionChunk",
    "ChatCompletionChunkChoice",
    "ChatCompletionChunkDelta",
    "ChatCompletionMessage",
    "ChatCompletions",
    "ChatMessage",
    "Choice",
    "Completion",
    "CompletionChoice",
    "CompletionChunk",
    "CompletionChunkChoice",
    "CompletionUsage",
    "Completions",
    "ConflictError",
    "CreateEmbeddingResponse",
    "Embedding",
    "EmbeddingUsage",
    "Embeddings",
    "FunctionCall",
    "FunctionMessage",
    "HumanMessage",
    "InternalServerError",
    "MessageContent",
    "MessageLike",
    "NotFoundError",
    "NotGiven",
    "OpenAI",
    "OpenAIError",
    "PermissionDeniedError",
    "RateLimitError",
    "Stream",
    "SystemMessage",
    "ToolCall",
    "ToolMessage",
    "UnprocessableEntityError",
    "__version__",
    "to_openai_messages",
]
