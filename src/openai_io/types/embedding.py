"""embeddings 相关的响应类型。"""

from __future__ import annotations

from pydantic import BaseModel

__all__ = ["CreateEmbeddingResponse", "Embedding", "EmbeddingUsage"]


class EmbeddingUsage(BaseModel):
    """token 用量统计。"""

    prompt_tokens: int
    total_tokens: int


class Embedding(BaseModel):
    """单个文本的向量。"""

    index: int
    object: str = "embedding"
    embedding: list[float]


class CreateEmbeddingResponse(BaseModel):
    """embeddings.create 的完整响应。"""

    object: str = "list"
    data: list[Embedding]
    model: str
    usage: EmbeddingUsage
