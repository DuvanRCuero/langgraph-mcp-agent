"""
Documentation query use case.
Handles querying and retrieving documentation from various sources.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ...domain.entities import DocumentationReference
from ..ports.mcp_client import MCPClientPort
from ..ports.vector_store import VectorStorePort, VectorDocument
from ..ports.llm_gateway import LLMGatewayPort

logger = logging.getLogger(__name__)


@dataclass
class DocumentationQuery:
    """Request for documentation query."""
    query: str
    language: Optional[str] = None
    framework: Optional[str] = None
    max_results: int = 10
    include_examples: bool = True
    min_relevance_score: float = 0.5


@dataclass
class DocumentationResult:
    """Result from documentation query."""
    documents: List[DocumentationReference]
    total_found: int
    query: str
    sources: List[str]


class DocumentationQueryUseCase:
    """
    Use case for querying documentation.
    
    Coordinates between MCP client for documentation retrieval
    and vector store for semantic search.
    """
    
    def __init__(
        self,
        mcp_client: MCPClientPort,
        vector_store: Optional[VectorStorePort] = None,
        llm_gateway: Optional[LLMGatewayPort] = None
    ):
        """
        Initialize documentation query use case.
        
        Args:
            mcp_client: MCP client for documentation retrieval
            vector_store: Optional vector store for semantic search
            llm_gateway: Optional LLM gateway for query enhancement
        """
        self.mcp_client = mcp_client
        self.vector_store = vector_store
        self.llm_gateway = llm_gateway
    
    async def execute(self, query: DocumentationQuery) -> DocumentationResult:
        """
        Execute documentation query.
        
        Args:
            query: Documentation query parameters
        
        Returns:
            Documentation result with matched documents
        """
        
        logger.info(f"Executing documentation query: {query.query}")
        
        try:
            # Ensure MCP client is connected
            if not self.mcp_client.is_connected:
                await self.mcp_client.connect()
            
            # Search via MCP client
            mcp_docs = await self.mcp_client.search_documentation(
                query=query.query,
                language=query.language,
                framework=query.framework,
                limit=query.max_results
            )
            
            # Filter by relevance score
            filtered_docs = [
                doc for doc in mcp_docs
                if doc.relevance_score >= query.min_relevance_score
            ]
            
            # Get unique sources
            sources = list(set(doc.source for doc in filtered_docs))
            
            logger.info(f"Found {len(filtered_docs)} relevant documents")
            
            return DocumentationResult(
                documents=filtered_docs,
                total_found=len(filtered_docs),
                query=query.query,
                sources=sources
            )
        
        except Exception as e:
            logger.error(f"Documentation query failed: {e}")
            # Return empty result on error
            return DocumentationResult(
                documents=[],
                total_found=0,
                query=query.query,
                sources=[]
            )
    
    async def search_best_practices(
        self,
        language: str,
        framework: Optional[str] = None,
        topic: Optional[str] = None
    ) -> List[DocumentationReference]:
        """
        Search for best practices documentation.
        
        Args:
            language: Programming language
            framework: Optional framework
            topic: Optional specific topic
        
        Returns:
            List of best practice documents
        """
        
        logger.info(f"Searching best practices for {language}")
        
        try:
            # Ensure MCP client is connected
            if not self.mcp_client.is_connected:
                await self.mcp_client.connect()
            
            # Get best practices
            docs = await self.mcp_client.get_latest_best_practices(
                language=language,
                framework=framework,
                topic=topic
            )
            
            logger.info(f"Found {len(docs)} best practice documents")
            return docs
        
        except Exception as e:
            logger.error(f"Best practices search failed: {e}")
            return []
    
    async def search_framework_docs(
        self,
        framework: str,
        version: Optional[str] = None
    ) -> List[DocumentationReference]:
        """
        Search for framework-specific documentation.
        
        Args:
            framework: Framework name
            version: Optional version
        
        Returns:
            List of framework documents
        """
        
        logger.info(f"Searching framework docs for {framework}")
        
        try:
            # Ensure MCP client is connected
            if not self.mcp_client.is_connected:
                await self.mcp_client.connect()
            
            # Get framework docs
            docs = await self.mcp_client.get_framework_specific_docs(
                framework=framework,
                version=version
            )
            
            logger.info(f"Found {len(docs)} framework documents")
            return docs
        
        except Exception as e:
            logger.error(f"Framework docs search failed: {e}")
            return []
    
    async def semantic_search(
        self,
        query: str,
        collection_name: str = "documentation",
        limit: int = 10
    ) -> List[DocumentationReference]:
        """
        Perform semantic search using vector store.
        
        Args:
            query: Search query
            collection_name: Vector store collection
            limit: Maximum results
        
        Returns:
            List of semantically similar documents
        """
        
        if not self.vector_store or not self.llm_gateway:
            logger.warning("Vector store or LLM gateway not available for semantic search")
            return []
        
        try:
            # Generate query embedding
            query_embedding = await self.llm_gateway.embed_text(query)
            
            # Search vector store
            vector_docs = await self.vector_store.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=limit
            )
            
            # Convert to DocumentationReference
            docs = []
            for vec_doc in vector_docs:
                doc = DocumentationReference(
                    title=vec_doc.metadata.get("title", "Untitled"),
                    url=vec_doc.metadata.get("url", ""),
                    content=vec_doc.content,
                    source=vec_doc.metadata.get("source", "vector_store"),
                    relevance_score=vec_doc.score or 0.0,
                    doc_type=vec_doc.metadata.get("doc_type", "unknown"),
                    tags=vec_doc.metadata.get("tags", [])
                )
                docs.append(doc)
            
            logger.info(f"Semantic search found {len(docs)} documents")
            return docs
        
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    async def enhance_query(self, query: str) -> List[str]:
        """
        Enhance query with synonyms and related terms.
        
        Args:
            query: Original query
        
        Returns:
            List of enhanced queries
        """
        
        if not self.llm_gateway:
            return [query]
        
        try:
            from ..ports.llm_gateway import LLMMessage, LLMGenerationConfig
            
            # Ask LLM to generate related queries
            prompt = f"""Given this search query: "{query}"
            
Generate 3-5 related search queries that would help find relevant documentation.
Return as a JSON list of strings.

Example format: ["query 1", "query 2", "query 3"]
"""
            
            messages = [
                LLMMessage(role="system", content="You are a search query expert."),
                LLMMessage(role="user", content=prompt)
            ]
            
            config = LLMGenerationConfig(temperature=0.5, max_tokens=200)
            response = await self.llm_gateway.generate(messages, config)
            
            # Parse JSON response
            import json
            queries = json.loads(response.content)
            
            # Include original query
            return [query] + queries
        
        except Exception as e:
            logger.warning(f"Query enhancement failed: {e}")
            return [query]
