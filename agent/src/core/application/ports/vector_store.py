"""
Port definition for vector store.
Abstracts vector database operations for semantic search and storage.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class VectorDocument:
    """Document with vector embedding."""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    score: Optional[float] = None


class VectorStorePort(ABC):
    """
    Interface for vector store operations.
    Abstracts vector databases like Qdrant, Pinecone, Weaviate, etc.
    """
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the vector store.
        
        Returns:
            bool: True if connection successful
        """
        pass
    
    @abstractmethod
    async def close(self):
        """Close connection to the vector store."""
        pass
    
    @abstractmethod
    async def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance_metric: str = "cosine"
    ) -> bool:
        """
        Create a new collection in the vector store.
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimensionality of vectors
            distance_metric: Distance metric to use (cosine, euclidean, dot)
        
        Returns:
            bool: True if collection created successfully
        """
        pass
    
    @abstractmethod
    async def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists.
        
        Args:
            collection_name: Name of the collection
        
        Returns:
            bool: True if collection exists
        """
        pass
    
    @abstractmethod
    async def upsert(
        self,
        collection_name: str,
        documents: List[VectorDocument]
    ) -> bool:
        """
        Insert or update documents in the vector store.
        
        Args:
            collection_name: Name of the collection
            documents: List of documents with embeddings
        
        Returns:
            bool: True if upsert successful
        """
        pass
    
    @abstractmethod
    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[VectorDocument]:
        """
        Search for similar vectors in the collection.
        
        Args:
            collection_name: Name of the collection
            query_vector: Query vector embedding
            limit: Maximum number of results to return
            filter_conditions: Optional metadata filters
            score_threshold: Minimum similarity score threshold
        
        Returns:
            List of similar documents with scores
        """
        pass
    
    @abstractmethod
    async def search_batch(
        self,
        collection_name: str,
        query_vectors: List[List[float]],
        limit: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[List[VectorDocument]]:
        """
        Search for multiple query vectors in batch.
        
        Args:
            collection_name: Name of the collection
            query_vectors: List of query vector embeddings
            limit: Maximum number of results per query
            filter_conditions: Optional metadata filters
        
        Returns:
            List of result lists, one per query
        """
        pass
    
    @abstractmethod
    async def delete(
        self,
        collection_name: str,
        document_ids: List[str]
    ) -> bool:
        """
        Delete documents from the collection.
        
        Args:
            collection_name: Name of the collection
            document_ids: IDs of documents to delete
        
        Returns:
            bool: True if deletion successful
        """
        pass
    
    @abstractmethod
    async def get_by_id(
        self,
        collection_name: str,
        document_id: str
    ) -> Optional[VectorDocument]:
        """
        Get a document by its ID.
        
        Args:
            collection_name: Name of the collection
            document_id: ID of the document
        
        Returns:
            Document if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def count(self, collection_name: str) -> int:
        """
        Get the number of documents in a collection.
        
        Args:
            collection_name: Name of the collection
        
        Returns:
            Number of documents
        """
        pass
    
    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to the vector store."""
        pass
    
    @property
    @abstractmethod
    def collections(self) -> List[str]:
        """Get list of available collections."""
        pass
