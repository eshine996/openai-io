"""embeddings 相关的响应类型。"""

from __future__ import annotations

from ._base import APIResponseModel

__all__ = ["CreateEmbeddingResponse", "Embedding", "EmbeddingUsage"]


class EmbeddingUsage(APIResponseModel):
    """token 用量统计。"""

    prompt_tokens: int
    total_tokens: int


class Embedding(APIResponseModel):
    """单个文本的向量。"""

    index: int
    object: str = "embedding"
    embedding: list[float] | str


class CreateEmbeddingResponse(APIResponseModel):
    """embeddings.create 的完整响应。"""

    object: str = "list"
    data: list[Embedding]
    model: str
    usage: EmbeddingUsage
