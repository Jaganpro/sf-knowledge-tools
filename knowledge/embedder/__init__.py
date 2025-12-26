"""Embedding module for sf-knowledge-tools."""

from .embedding_client import (
    EmbeddingClient,
    CachedEmbeddingClient,
    EmbeddingConfig,
    create_client
)

__all__ = [
    "EmbeddingClient",
    "CachedEmbeddingClient",
    "EmbeddingConfig",
    "create_client"
]
