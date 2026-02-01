"""
Vector store implementation for semantic search of documentation.
Uses sentence transformers for embeddings and Qdrant/FAISS for storage.
Production-ready with async operations and batch processing.
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
import pickle

try:
    from sentence_transformers import SentenceTransformer

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logging.warning("Sentence transformers not available, using mock embeddings")

try:
    import qdrant_client
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct,
        Filter, FieldCondition, MatchValue
    )

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logging.warning("Qdrant client not available, using in-memory storage")

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available")

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    """Result from vector search."""
    id: str
    score: float
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "score": self.score,
            **self.payload
        }


class VectorStore:
    """
    Production-grade vector store for semantic search.
    Supports multiple backends (Qdrant, FAISS, in-memory).
    """

    def __init__(
            self,
            backend: str = "qdrant",  # "qdrant", "faiss", or "memory"
            model_name: str = "all-MiniLM-L6-v2",
            dimension: int = 384,
            collection_name: str = "documentation",
            cache_dir: str = "./data/vectors"
    ):
        self.backend = backend
        self.model_name = model_name
        self.dimension = dimension
        self.collection_name = collection_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.embedding_model = self._initialize_embedding_model()
        self.vector_store = self._initialize_vector_store()

        # Statistics
        self.stats = {
            "total_vectors": 0,
            "queries": 0,
            "cache_hits": 0,
            "last_indexed": None
        }

        # Cache for embeddings
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.cache_file = self.cache_dir / "embedding_cache.pkl"
        self._load_embedding_cache()

        logger.info(f"VectorStore initialized with backend: {backend}")

    def _initialize_embedding_model(self):
        """Initialize the embedding model."""
        if not EMBEDDINGS_AVAILABLE:
            logger.warning("Using mock embedding model")
            return MockEmbeddingModel(self.dimension)

        try:
            model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return MockEmbeddingModel(self.dimension)

    def _initialize_vector_store(self):
        """Initialize the vector store backend."""
        if self.backend == "qdrant" and QDRANT_AVAILABLE:
            return self._initialize_qdrant()
        elif self.backend == "faiss" and FAISS_AVAILABLE:
            return self._initialize_faiss()
        else:
            return self._initialize_memory_store()

    def _initialize_qdrant(self):
        """Initialize Qdrant client."""
        try:
            # Try to connect to Qdrant server
            client = QdrantClient(host="localhost", port=6333)

            # Create collection if it doesn't exist
            collections = client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name not in collection_names:
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")

            logger.info("Connected to Qdrant")
            return client

        except Exception as e:
            logger.warning(f"Failed to connect to Qdrant: {e}, falling back to memory")
            return self._initialize_memory_store()

    def _initialize_faiss(self):
        """Initialize FAISS index."""
        try:
            # Create FAISS index
            index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity

            # Load existing index if available
            index_file = self.cache_dir / f"{self.collection_name}.faiss"
            if index_file.exists():
                index = faiss.read_index(str(index_file))
                logger.info(f"Loaded FAISS index from {index_file}")

            logger.info(f"Initialized FAISS index with dimension {self.dimension}")
            return {
                "index": index,
                "ids": [],
                "payloads": []
            }

        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {e}")
            return self._initialize_memory_store()

    def _initialize_memory_store(self):
        """Initialize in-memory vector store."""
        logger.info("Using in-memory vector store")
        return {
            "vectors": [],
            "ids": [],
            "payloads": []
        }

    def _load_embedding_cache(self):
        """Load embedding cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Loaded embedding cache with {len(self.embedding_cache)} entries")
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {e}")
                self.embedding_cache = {}

    def _save_embedding_cache(self):
        """Save embedding cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            logger.debug(f"Saved embedding cache with {len(self.embedding_cache)} entries")
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")

    async def embed_text(self, text: str) -> np.ndarray:
        """Embed text using the model."""
        # Check cache first
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        # Generate embedding
        if hasattr(self.embedding_model, 'encode'):
            # SentenceTransformer
            embedding = self.embedding_model.encode(
                text,
                show_progress_bar=False,
                convert_to_numpy=True
            )
        else:
            # Mock model
            embedding = self.embedding_model.encode(text)

        # Cache the embedding
        self.embedding_cache[cache_key] = embedding

        # Save cache periodically
        if len(self.embedding_cache) % 100 == 0:
            await asyncio.to_thread(self._save_embedding_cache)

        return embedding

    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a batch of texts."""
        embeddings = []

        for text in texts:
            embedding = await self.embed_text(text)
            embeddings.append(embedding)

        return embeddings

    async def add_documents(
            self,
            documents: List[Dict[str, Any]],
            batch_size: int = 100
    ):
        """Add documents to the vector store."""
        if not documents:
            return

        logger.info(f"Adding {len(documents)} documents to vector store")

        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            await self._add_batch(batch)

            if (i // batch_size) % 10 == 0:
                logger.info(f"Processed {i + len(batch)}/{len(documents)} documents")

        # Update statistics
        self.stats["total_vectors"] += len(documents)
        self.stats["last_indexed"] = datetime.now().isoformat()

        logger.info(f"Added {len(documents)} documents to vector store")

    async def _add_batch(self, batch: List[Dict[str, Any]]):
        """Add a batch of documents."""
        # Extract texts for embedding
        texts = []
        for doc in batch:
            # Create text representation
            text_parts = []
            if "title" in doc:
                text_parts.append(doc["title"])
            if "content" in doc:
                # Use first 1000 chars of content
                text_parts.append(doc["content"][:1000])
            if "tags" in doc:
                text_parts.extend(doc["tags"])

            text = " ".join(text_parts)
            texts.append(text)

        # Generate embeddings
        embeddings = await self.embed_batch(texts)

        # Add to vector store based on backend
        if self.backend == "qdrant" and QDRANT_AVAILABLE:
            await self._add_to_qdrant(batch, embeddings)
        elif self.backend == "faiss" and FAISS_AVAILABLE:
            await self._add_to_faiss(batch, embeddings)
        else:
            await self._add_to_memory(batch, embeddings)

    async def _add_to_qdrant(self, batch: List[Dict[str, Any]], embeddings: List[np.ndarray]):
        """Add batch to Qdrant."""
        points = []

        for i, (doc, embedding) in enumerate(zip(batch, embeddings)):
            point_id = i + self.stats["total_vectors"]

            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload=doc
            )
            points.append(point)

        # Upload points
        self.vector_store.upload_points(
            collection_name=self.collection_name,
            points=points
        )

    async def _add_to_faiss(self, batch: List[Dict[str, Any]], embeddings: List[np.ndarray]):
        """Add batch to FAISS."""
        # Convert embeddings to numpy array
        embedding_array = np.array(embeddings).astype('float32')

        # Normalize for cosine similarity
        faiss.normalize_L2(embedding_array)

        # Add to index
        self.vector_store["index"].add(embedding_array)

        # Store IDs and payloads
        start_id = len(self.vector_store["ids"])
        for i, doc in enumerate(batch):
            self.vector_store["ids"].append(start_id + i)
            self.vector_store["payloads"].append(doc)

        # Save index to disk
        index_file = self.cache_dir / f"{self.collection_name}.faiss"
        faiss.write_index(self.vector_store["index"], str(index_file))

    async def _add_to_memory(self, batch: List[Dict[str, Any]], embeddings: List[np.ndarray]):
        """Add batch to in-memory store."""
        for doc, embedding in zip(batch, embeddings):
            self.vector_store["vectors"].append(embedding)
            self.vector_store["ids"].append(doc.get("id", len(self.vector_store["ids"])))
            self.vector_store["payloads"].append(doc)

    async def semantic_search(
            self,
            query: str,
            source: Optional[str] = None,
            doc_type: Optional[str] = None,
            language: Optional[str] = None,
            limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search with filters.

        Args:
            query: Search query
            source: Filter by source
            doc_type: Filter by document type
            language: Filter by language
            limit: Maximum number of results

        Returns:
            List of search results
        """
        self.stats["queries"] += 1

        # Embed the query
        query_embedding = await self.embed_text(query)

        # Perform search based on backend
        if self.backend == "qdrant" and QDRANT_AVAILABLE:
            results = await self._search_qdrant(
                query_embedding, source, doc_type, language, limit
            )
        elif self.backend == "faiss" and FAISS_AVAILABLE:
            results = await self._search_faiss(
                query_embedding, source, doc_type, language, limit
            )
        else:
            results = await self._search_memory(
                query_embedding, source, doc_type, language, limit
            )

        # Convert to standard format
        formatted_results = []
        for result in results:
            formatted_results.append(result.to_dict())

        return formatted_results

    async def _search_qdrant(
            self,
            query_embedding: np.ndarray,
            source: Optional[str],
            doc_type: Optional[str],
            language: Optional[str],
            limit: int
    ) -> List[VectorSearchResult]:
        """Search using Qdrant."""
        # Build filter
        filter_conditions = []

        if source:
            filter_conditions.append(
                FieldCondition(
                    key="source",
                    match=MatchValue(value=source)
                )
            )

        if doc_type:
            filter_conditions.append(
                FieldCondition(
                    key="doc_type",
                    match=MatchValue(value=doc_type)
                )
            )

        if language:
            filter_conditions.append(
                FieldCondition(
                    key="language",
                    match=MatchValue(value=language)
                )
            )

        search_filter = Filter(
            must=filter_conditions
        ) if filter_conditions else None

        # Perform search
        search_result = self.vector_store.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            query_filter=search_filter,
            limit=limit
        )

        # Convert to results
        results = []
        for point in search_result:
            results.append(
                VectorSearchResult(
                    id=str(point.id),
                    score=point.score,
                    payload=point.payload
                )
            )

        return results

    async def _search_faiss(
            self,
            query_embedding: np.ndarray,
            source: Optional[str],
            doc_type: Optional[str],
            language: Optional[str],
            limit: int
    ) -> List[VectorSearchResult]:
        """Search using FAISS."""
        # Prepare query vector
        query_vector = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_vector)

        # Search
        scores, indices = self.vector_store["index"].search(query_vector, limit * 2)

        # Filter results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= len(self.vector_store["payloads"]):
                continue

            payload = self.vector_store["payloads"][idx]

            # Apply filters
            if source and payload.get("source") != source:
                continue
            if doc_type and payload.get("doc_type") != doc_type:
                continue
            if language and payload.get("language") != language:
                continue

            results.append(
                VectorSearchResult(
                    id=str(idx),
                    score=float(score),
                    payload=payload
                )
            )

            if len(results) >= limit:
                break

        return results

    async def _search_memory(
            self,
            query_embedding: np.ndarray,
            source: Optional[str],
            doc_type: Optional[str],
            language: Optional[str],
            limit: int
    ) -> List[VectorSearchResult]:
        """Search using in-memory store."""
        # Calculate similarities
        similarities = []
        for i, vector in enumerate(self.vector_store["vectors"]):
            # Cosine similarity
            similarity = np.dot(query_embedding, vector) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(vector)
            )
            similarities.append((i, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Filter and collect results
        results = []
        for idx, score in similarities:
            if idx >= len(self.vector_store["payloads"]):
                continue

            payload = self.vector_store["payloads"][idx]

            # Apply filters
            if source and payload.get("source") != source:
                continue
            if doc_type and payload.get("doc_type") != doc_type:
                continue
            if language and payload.get("language") != language:
                continue

            results.append(
                VectorSearchResult(
                    id=str(idx),
                    score=float(score),
                    payload=payload
                )
            )

            if len(results) >= limit:
                break

        return results

    async def reindex_all(self):
        """Re-index all documents (for after updates)."""
        logger.info("Re-indexing all documents...")

        # Clear current index
        await self.clear()

        # Get all documents from loader (you'll need to inject this dependency)
        # For now, this is a placeholder
        documents = []  # Get from documentation loader

        if documents:
            await self.add_documents(documents)
            logger.info(f"Re-indexed {len(documents)} documents")

    async def clear(self):
        """Clear the vector store."""
        if self.backend == "qdrant" and QDRANT_AVAILABLE:
            self.vector_store.delete_collection(collection_name=self.collection_name)
            self.vector_store.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=Distance.COSINE
                )
            )
        elif self.backend == "faiss" and FAISS_AVAILABLE:
            self.vector_store["index"] = faiss.IndexFlatIP(self.dimension)
            self.vector_store["ids"] = []
            self.vector_store["payloads"] = []
        else:
            self.vector_store["vectors"] = []
            self.vector_store["ids"] = []
            self.vector_store["payloads"] = []

        self.stats["total_vectors"] = 0
        logger.info("Cleared vector store")

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            **self.stats,
            "backend": self.backend,
            "embedding_cache_size": len(self.embedding_cache),
            "dimension": self.dimension
        }


class MockEmbeddingModel:
    """Mock embedding model for testing."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        logger.warning("Using mock embedding model")

    def encode(self, text: str) -> np.ndarray:
        """Generate mock embedding."""
        # Create deterministic "embedding" based on text hash
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        np.random.seed(hash_val % (2 ** 32))
        return np.random.randn(self.dimension).astype(np.float32)