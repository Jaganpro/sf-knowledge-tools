"""
Local embedding client using sentence-transformers.

Uses BAAI/bge-large-en-v1.5:
- 1024 dimensions
- 1.3GB model size
- Top-ranked on MTEB leaderboard for retrieval tasks
- 100% offline after initial download
"""

from typing import List, Union, Optional
from pathlib import Path
import numpy as np
from dataclasses import dataclass


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding model."""
    model_name: str = "BAAI/bge-large-en-v1.5"
    dimensions: int = 1024
    max_seq_length: int = 512
    batch_size: int = 32
    normalize: bool = True
    device: Optional[str] = None  # None = auto-detect (cuda/mps/cpu)
    cache_dir: Optional[str] = None  # Model cache directory


class EmbeddingClient:
    """
    Local embedding client using sentence-transformers.

    Features:
    - Batch processing for efficiency
    - Automatic device detection (GPU/MPS/CPU)
    - Query vs document prefixes (per BGE spec)
    - Progress callbacks for long operations
    """

    # BGE models use instruction prefixes for better retrieval
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
    DOCUMENT_PREFIX = ""  # Documents don't need prefix for BGE

    def __init__(self, config: EmbeddingConfig = None):
        """
        Initialize the embedding client.

        Args:
            config: EmbeddingConfig with model settings
        """
        self.config = config or EmbeddingConfig()
        self._model = None
        self._device = None

    @property
    def model(self):
        """Lazy-load the model on first use."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def device(self) -> str:
        """Get the device being used."""
        if self._device is None:
            self._detect_device()
        return self._device

    def _detect_device(self):
        """Detect best available device."""
        if self.config.device:
            self._device = self.config.device
            return

        import torch

        if torch.cuda.is_available():
            self._device = "cuda"
        elif torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"

    def _load_model(self):
        """Load the sentence-transformers model."""
        from sentence_transformers import SentenceTransformer

        self._detect_device()

        # Load model with optional cache directory
        kwargs = {"device": self._device}
        if self.config.cache_dir:
            kwargs["cache_folder"] = self.config.cache_dir

        self._model = SentenceTransformer(
            self.config.model_name,
            **kwargs
        )

        # Set max sequence length
        self._model.max_seq_length = self.config.max_seq_length

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a search query.

        Uses query prefix for BGE models to optimize retrieval.

        Args:
            query: Search query text

        Returns:
            1024-dimensional embedding as numpy array
        """
        # Add query instruction prefix for BGE models
        prefixed = self.QUERY_PREFIX + query
        embedding = self.model.encode(
            prefixed,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True
        )
        return embedding

    def embed_documents(
        self,
        texts: List[str],
        batch_size: int = None,
        show_progress: bool = True,
        progress_callback=None
    ) -> np.ndarray:
        """
        Embed multiple documents in batches.

        Args:
            texts: List of document texts to embed
            batch_size: Override default batch size
            show_progress: Show progress bar
            progress_callback: Optional callback(current, total) for progress

        Returns:
            Array of shape (len(texts), 1024)
        """
        if not texts:
            return np.array([])

        batch_size = batch_size or self.config.batch_size

        # BGE documents don't need prefix, but add if configured
        if self.DOCUMENT_PREFIX:
            texts = [self.DOCUMENT_PREFIX + t for t in texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )

        return embeddings

    def embed_batch(
        self,
        texts: List[str],
        is_query: bool = False
    ) -> List[np.ndarray]:
        """
        Embed a batch of texts.

        Args:
            texts: List of texts to embed
            is_query: Whether these are queries (adds prefix)

        Returns:
            List of embeddings
        """
        if is_query:
            texts = [self.QUERY_PREFIX + t for t in texts]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True
        )

        return list(embeddings)

    def similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and documents.

        Args:
            query_embedding: Single query embedding (1024,)
            document_embeddings: Document embeddings (N, 1024)

        Returns:
            Similarity scores (N,)
        """
        # If embeddings are normalized, dot product = cosine similarity
        if self.config.normalize:
            return np.dot(document_embeddings, query_embedding)

        # Otherwise compute full cosine similarity
        from numpy.linalg import norm
        query_norm = query_embedding / norm(query_embedding)
        doc_norms = document_embeddings / norm(document_embeddings, axis=1, keepdims=True)
        return np.dot(doc_norms, query_norm)

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.config.model_name,
            "dimensions": self.config.dimensions,
            "max_seq_length": self.config.max_seq_length,
            "device": self.device,
            "normalized": self.config.normalize
        }


class CachedEmbeddingClient(EmbeddingClient):
    """
    Embedding client with disk caching for repeated documents.

    Useful when re-processing PDFs or testing queries.
    """

    def __init__(self, config: EmbeddingConfig = None, cache_path: Path = None):
        """
        Initialize with optional embedding cache.

        Args:
            config: EmbeddingConfig
            cache_path: Path to cache file (SQLite or pickle)
        """
        super().__init__(config)
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache = {}

        if self.cache_path and self.cache_path.exists():
            self._load_cache()

    def _load_cache(self):
        """Load embedding cache from disk."""
        import pickle
        try:
            with open(self.cache_path, 'rb') as f:
                self._cache = pickle.load(f)
        except Exception:
            self._cache = {}

    def _save_cache(self):
        """Save embedding cache to disk."""
        if not self.cache_path:
            return

        import pickle
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self._cache, f)

    def _text_hash(self, text: str) -> str:
        """Get hash of text for cache key."""
        import hashlib
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def embed_documents(
        self,
        texts: List[str],
        batch_size: int = None,
        show_progress: bool = True,
        progress_callback=None
    ) -> np.ndarray:
        """Embed documents with caching."""
        results = []
        to_embed = []
        to_embed_indices = []

        # Check cache
        for i, text in enumerate(texts):
            key = self._text_hash(text)
            if key in self._cache:
                results.append((i, self._cache[key]))
            else:
                to_embed.append(text)
                to_embed_indices.append(i)

        # Embed uncached texts
        if to_embed:
            new_embeddings = super().embed_documents(
                to_embed,
                batch_size=batch_size,
                show_progress=show_progress,
                progress_callback=progress_callback
            )

            # Cache new embeddings
            for text, embedding in zip(to_embed, new_embeddings):
                key = self._text_hash(text)
                self._cache[key] = embedding

            # Add to results
            for i, embedding in zip(to_embed_indices, new_embeddings):
                results.append((i, embedding))

            # Persist cache
            self._save_cache()

        # Sort by original index and return
        results.sort(key=lambda x: x[0])
        return np.array([r[1] for r in results])


def create_client(
    model_name: str = "BAAI/bge-large-en-v1.5",
    cache_path: Path = None,
    **kwargs
) -> EmbeddingClient:
    """
    Factory function to create an embedding client.

    Args:
        model_name: HuggingFace model name
        cache_path: Optional path for embedding cache
        **kwargs: Additional config options

    Returns:
        EmbeddingClient or CachedEmbeddingClient
    """
    config = EmbeddingConfig(model_name=model_name, **kwargs)

    if cache_path:
        return CachedEmbeddingClient(config, cache_path)
    return EmbeddingClient(config)
