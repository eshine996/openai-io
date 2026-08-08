"""chat 资源（聚合 chat.completions）。"""

from __future__ import annotations

from ..._base_client import AsyncAPIClient, SyncAPIClient
from .completions import AsyncChatCompletions, ChatCompletions

__all__ = ["AsyncChat", "Chat"]


class Chat:
    """同步 chat 资源：``client.chat.completions``。"""

    def __init__(self, client: SyncAPIClient) -> None:
        self.completions = ChatCompletions(client)


class AsyncChat:
    """异步 chat 资源：``client.chat.completions``。"""

    def __init__(self, client: AsyncAPIClient) -> None:
        self.completions = AsyncChatCompletions(client)
