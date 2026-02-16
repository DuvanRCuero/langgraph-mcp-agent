"""
Qdrant adapter implementing the Vector Store Port.
Provides integration with Qdrant vector database.
"""

import logging
from typing import Dict, Any, List, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)

from ...application.ports.vector_store import VectorStorePort, VectorDocument

logger = logging.getLogger(__name__)


class QdrantAdapter(VectorStorePort):
    """
    Qdrant adapter for vector store operations.
    Implements the VectorStorePort interface using Qdrant.
    """
    
    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "code_documentation",
        api_key: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize Qdrant adapter.
        
        Args:
            url: Qdrant server URL
            collection_name: Default collection name
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.url = url
        self.default_collection = collection_name
        self.api_key = api_key
        self.timeout = timeout
        
        # Initialize client
        self.client: Optional[AsyncQdrantClient] = None
        self._is_connected = False
        self._collections_cache: List[str] = []
        
        logger.info(f"Qdrant adapter initialized for: {url}")
    
    async def connect(self) -> bool:
        """Connect to Qdrant server."""
        
        try:
            self.client = AsyncQdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=self.timeout
            )
            
            # Test connection by getting collections
            collections = await self.client.get_collections()
            self._collections_cache = [col.name for col in collections.collections]
            
            self._is_connected = True
            logger.info(f"Connected to Qdrant. Collections: {self._collections_cache}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            self._is_connected = False
            return False
    
    async def close(self):
        """Close connection to Qdrant."""
        
        if self.client:
            try:
                await self.client.close()
                logger.info("Qdrant connection closed")
            except Exception as e:
                logger.warning(f"Error closing Qdrant connection: {e}")
            finally:
                self.client = None
                self._is_connected = False
    
    async def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance_metric: str = "cosine"
    ) -> bool:
        """Create a new collection in Qdrant."""
        
        if not self.client:
            raise ConnectionError("Not connected to Qdrant")
        
        try:
            # Map distance metric
            distance_map = {
                "cosine": Distance.COSINE,
                "euclidean": Distance.EUCLID,
                "dot": Distance.DOT
            }
            
            distance = distance_map.get(distance_metric.lower(), Distance.COSINE)
            
            # Create collection
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance
                )
            )
            
            # Update cache
            if collection_name not in self._collections_cache:
                self._collections_cache.append(collection_name)
            
            logger.info(f"Created collection: {collection_name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            return False
    
    async def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        
        if not self.client:
            raise ConnectionError("Not connected to Qdrant")
        
        try:
            collections = await self.client.get_collections()
            return collection_name in [col.name for col in collections.collections]
        
        except Exception as e:
            logger.error(f"Failed to check collection existence: {e}")
            return False
    
    async def upsert(
        self,
        collection_name: str,
        documents: List[VectorDocument]
    ) -> bool:
        """Insert or update documents in Qdrant."""
        
        if not self.client:
            raise ConnectionError("Not connected to Qdrant")
        
        try:
            # Ensure collection exists
            exists = await self.collection_exists(collection_name)
            if not exists:
                # Create with default vector size from first document
                if documents:
                    vector_size = len(documents[0].embedding)
                    await self.create_collection(collection_name, vector_size)
                else:
                    return False
            
            # Convert documents to Qdrant points
            points = []
            for doc in documents:
                point = PointStruct(
                    id=doc.id,
                    vector=doc.embedding,
                    payload={
                        "content": doc.content,
                        **doc.metadata
                    }
                )
                points.append(point)
            
            # Upsert points
            await self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            logger.info(f"Upserted {len(documents)} documents to {collection_name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to upsert documents: {e}")
            return False
    
    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[VectorDocument]:
        """Search for similar vectors in Qdrant."""
        
        if not self.client:
            raise ConnectionError("Not connected to Qdrant")
        
        try:
            # Build filter if provided
            query_filter = None
            if filter_conditions:
                query_filter = self._build_filter(filter_conditions)
            
            # Perform search
            results = await self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=query_filter,
                score_threshold=score_threshold
            )
            
            # Convert results to VectorDocument
            documents = []
            for result in results:
                doc = VectorDocument(
                    id=str(result.id),
                    content=result.payload.get("content", ""),
                    embedding=result.vector if result.vector else [],
                    metadata={k: v for k, v in result.payload.items() if k != "content"},
                    score=result.score
                )
                documents.append(doc)
            
            logger.info(f"Found {len(documents)} similar documents")
            return documents
        
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def search_batch(
        self,
        collection_name: str,
        query_vectors: List[List[float]],
        limit: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[List[VectorDocument]]:
        """Search for multiple query vectors in batch."""
        
        if not self.client:
            raise ConnectionError("Not connected to Qdrant")
        
        try:
            # Build filter if provided
            query_filter = None
            if filter_conditions:
                query_filter = self._build_filter(filter_conditions)
            
            # Perform batch search
            results = await self.client.search_batch(
                collection_name=collection_name,
                requests=[
                    {
                        "vector": query_vector,
                        "limit": limit,
                        "filter": query_filter
                    }
                    for query_vector in query_vectors
                ]
            )
            
            # Convert results
            all_documents = []
            for result_set in results:
                documents = []
                for result in result_set:
                    doc = VectorDocument(
                        id=str(result.id),
                        content=result.payload.get("content", ""),
                        embedding=result.vector if result.vector else [],
                        metadata={k: v for k, v in result.payload.items() if k != "content"},
                        score=result.score
                    )
                    documents.append(doc)
                all_documents.append(documents)
            
            logger.info(f"Batch search completed for {len(query_vectors)} queries")
            return all_documents
        
        except Exception as e:
            logger.error(f"Batch search failed: {e}")
            return [[] for _ in query_vectors]
    
    async def delete(
        self,
        collection_name: str,
        document_ids: List[str]
    ) -> bool:
        """Delete documents from Qdrant."""
        
        if not self.client:
            raise ConnectionError("Not connected to Qdrant")
        
        try:
            await self.client.delete(
                collection_name=collection_name,
                points_selector=document_ids
            )
            
            logger.info(f"Deleted {len(document_ids)} documents from {collection_name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False
    
    async def get_by_id(
        self,
        collection_name: str,
        document_id: str
    ) -> Optional[VectorDocument]:
        """Get a document by its ID."""
        
        if not self.client:
            raise ConnectionError("Not connected to Qdrant")
        
        try:
            result = await self.client.retrieve(
                collection_name=collection_name,
                ids=[document_id],
                with_vectors=True
            )
            
            if not result:
                return None
            
            point = result[0]
            
            return VectorDocument(
                id=str(point.id),
                content=point.payload.get("content", ""),
                embedding=point.vector if point.vector else [],
                metadata={k: v for k, v in point.payload.items() if k != "content"}
            )
        
        except Exception as e:
            logger.error(f"Failed to get document by ID: {e}")
            return None
    
    async def count(self, collection_name: str) -> int:
        """Get the number of documents in a collection."""
        
        if not self.client:
            raise ConnectionError("Not connected to Qdrant")
        
        try:
            collection_info = await self.client.get_collection(collection_name)
            return collection_info.points_count
        
        except Exception as e:
            logger.error(f"Failed to count documents: {e}")
            return 0
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to Qdrant."""
        return self._is_connected and self.client is not None
    
    @property
    def collections(self) -> List[str]:
        """Get list of available collections."""
        return self._collections_cache.copy()
    
    def _build_filter(self, conditions: Dict[str, Any]) -> Filter:
        """Build Qdrant filter from conditions."""
        
        # Simple implementation - can be extended for complex filters
        must_conditions = []
        
        for key, value in conditions.items():
            condition = FieldCondition(
                key=key,
                match=MatchValue(value=value)
            )
            must_conditions.append(condition)
        
        return Filter(must=must_conditions) if must_conditions else None
