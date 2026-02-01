"""
Production-grade documentation tool for MCP server.
Provides access to latest coding documentation with semantic search,
caching, and real-time updates.
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from urllib.parse import urljoin, quote

from .base import BaseTool, ToolMetadata, ToolCategory, ToolSchema, ExecutionContext
from ..documentation.loader import DocumentationLoader
from ..documentation.vector_store import VectorStore
from ..documentation.cache import DocumentationCache
from ..knowledge.elite_practices import EliteEngineeringPractices

logger = logging.getLogger(__name__)


class DocumentationSource(str, Enum):
    """Documentation sources."""
    PYTHON_OFFICIAL = "python_official"
    PYTHON_PYPI = "python_pypi"
    TYPESCRIPT = "typescript"
    REACT = "react"
    VUE = "vue"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    LANGCHAIN = "langchain"
    LANGGRAPH = "langgraph"
    FASTAPI = "fastapi"
    DJANGO = "django"
    FLASK = "flask"
    POSTGRESQL = "postgresql"
    REDIS = "redis"
    KAFKA = "kafka"
    ELIXIR = "elixir"
    RUST = "rust"
    GO = "go"


class DocumentationType(str, Enum):
    """Types of documentation."""
    API_REFERENCE = "api_reference"
    GETTING_STARTED = "getting_started"
    TUTORIAL = "tutorial"
    BEST_PRACTICES = "best_practices"
    TROUBLESHOOTING = "troubleshooting"
    SECURITY = "security"
    PERFORMANCE = "performance"
    DEPLOYMENT = "deployment"


@dataclass
class DocumentationQuery:
    """Structured documentation query."""
    search_terms: List[str]
    source: Optional[DocumentationSource] = None
    doc_type: Optional[DocumentationType] = None
    language: Optional[str] = None
    framework: Optional[str] = None
    version: Optional[str] = None
    max_results: int = 10
    include_code_examples: bool = True
    include_api_refs: bool = True
    freshness_days: Optional[int] = 30

    def to_cache_key(self) -> str:
        """Generate cache key from query."""
        data = {
            "search_terms": self.search_terms,
            "source": self.source,
            "doc_type": self.doc_type,
            "language": self.language,
            "framework": self.framework,
            "version": self.version,
            "max_results": self.max_results
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()


@dataclass
class DocumentationResult:
    """Documentation search result."""
    title: str
    content: str
    url: str
    source: DocumentationSource
    doc_type: DocumentationType
    relevance_score: float
    last_updated: datetime
    language: Optional[str] = None
    framework: Optional[str] = None
    version: Optional[str] = None
    code_examples: List[str] = field(default_factory=list)
    related_topics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "content": self.content[:500] + "..." if len(self.content) > 500 else self.content,
            "url": self.url,
            "source": self.source.value,
            "doc_type": self.doc_type.value,
            "relevance_score": self.relevance_score,
            "last_updated": self.last_updated.isoformat(),
            "language": self.language,
            "framework": self.framework,
            "version": self.version,
            "has_code_examples": len(self.code_examples) > 0,
            "code_examples_count": len(self.code_examples),
            "related_topics": self.related_topics
        }


class DocumentationTool(BaseTool):
    """
    Elite documentation tool providing access to latest coding documentation.
    Features:
    - Semantic search across multiple sources
    - Real-time documentation updates
    - Intelligent caching
    - Code example extraction
    - Framework-specific guidance
    """

    def __init__(self):
        super().__init__()
        self.docs_loader = DocumentationLoader()
        self.vector_store = VectorStore()
        self.cache = DocumentationCache()
        self.elite_practices = EliteEngineeringPractices()
        self.source_urls = self._initialize_source_urls()

        # Background task for updating documentation
        self._update_task = None
        self._is_updating = False

        logger.info("DocumentationTool initialized")

    def get_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            version="2.0.0",
            author="MCP Documentation Team",
            category=ToolCategory.DOCUMENTATION,
            tags=["documentation", "code", "best-practices", "api"],
            timeout_seconds=60,
            max_concurrent=20,
            requires_approval=False
        )

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="documentation_search",
            description="Search latest coding documentation across multiple sources including Python, JavaScript, ML frameworks, and cloud services",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for documentation"
                    },
                    "source": {
                        "type": "string",
                        "description": "Documentation source (python, typescript, react, docker, kubernetes, aws, pytorch, etc.)",
                        "enum": [source.value for source in DocumentationSource]
                    },
                    "doc_type": {
                        "type": "string",
                        "description": "Type of documentation needed",
                        "enum": [doc_type.value for doc_type in DocumentationType]
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language"
                    },
                    "framework": {
                        "type": "string",
                        "description": "Framework or library"
                    },
                    "version": {
                        "type": "string",
                        "description": "Version constraint"
                    },
                    "include_code_examples": {
                        "type": "boolean",
                        "description": "Include code examples in results",
                        "default": True
                    },
                    "include_api_refs": {
                        "type": "boolean",
                        "description": "Include API references",
                        "default": True
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50
                    },
                    "freshness_days": {
                        "type": "integer",
                        "description": "Maximum age of documentation in days",
                        "default": 30
                    }
                },
                "required": ["query"]
            },
            rate_limit=100  # 100 requests per minute
        )

    async def _execute_impl(self, arguments: Dict[str, Any], context: ExecutionContext) -> Any:
        """
        Execute documentation search with advanced features:
        1. Query parsing and validation
        2. Cache lookup
        3. Multi-source search
        4. Semantic relevance scoring
        5. Elite practice integration
        6. Result formatting
        """

        # Parse query
        query = self._parse_query(arguments)

        # Check cache
        cache_key = query.to_cache_key()
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            logger.debug(f"Cache hit for query: {query.search_terms}")
            return self._format_results(cached_result, query)

        # Perform search
        results = await self._search_documentation(query)

        # Enhance with elite practices
        if self._should_enhance_with_practices(query):
            results = await self._enhance_with_elite_practices(results, query)

        # Cache results
        await self.cache.set(cache_key, results, ttl=timedelta(hours=1))

        # Format and return
        return self._format_results(results, query)

    def _parse_query(self, arguments: Dict[str, Any]) -> DocumentationQuery:
        """Parse and validate query arguments."""

        # Extract search terms
        raw_query = arguments.get("query", "")
        search_terms = self._extract_search_terms(raw_query)

        # Parse source
        source_str = arguments.get("source")
        source = None
        if source_str:
            try:
                source = DocumentationSource(source_str)
            except ValueError:
                logger.warning(f"Invalid source: {source_str}")

        # Parse doc type
        doc_type_str = arguments.get("doc_type")
        doc_type = None
        if doc_type_str:
            try:
                doc_type = DocumentationType(doc_type_str)
            except ValueError:
                logger.warning(f"Invalid doc type: {doc_type_str}")

        return DocumentationQuery(
            search_terms=search_terms,
            source=source,
            doc_type=doc_type,
            language=arguments.get("language"),
            framework=arguments.get("framework"),
            version=arguments.get("version"),
            max_results=arguments.get("max_results", 10),
            include_code_examples=arguments.get("include_code_examples", True),
            include_api_refs=arguments.get("include_api_refs", True),
            freshness_days=arguments.get("freshness_days", 30)
        )

    def _extract_search_terms(self, query: str) -> List[str]:
        """Extract meaningful search terms from query."""
        # Remove common words and split
        common_words = {"how", "to", "what", "is", "the", "a", "an", "in", "on", "at"}
        words = query.lower().split()
        terms = [word for word in words if word not in common_words and len(word) > 2]

        # Add stemming/synonyms in production
        return terms if terms else [query]

    async def _search_documentation(self, query: DocumentationQuery) -> List[DocumentationResult]:
        """
        Search documentation across multiple sources.
        Uses vector store for semantic search and falls back to traditional search.
        """

        all_results: List[DocumentationResult] = []

        # Determine which sources to search
        sources_to_search = self._determine_sources(query)

        # Search each source
        for source in sources_to_search:
            try:
                results = await self._search_source(source, query)
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Failed to search source {source}: {e}")
                continue

        # Sort by relevance
        all_results.sort(key=lambda x: x.relevance_score, reverse=True)

        # Apply freshness filter
        if query.freshness_days:
            cutoff_date = datetime.now() - timedelta(days=query.freshness_days)
            all_results = [
                result for result in all_results
                if result.last_updated >= cutoff_date
            ]

        # Limit results
        return all_results[:query.max_results]

    def _determine_sources(self, query: DocumentationQuery) -> List[DocumentationSource]:
        """Determine which documentation sources to search."""

        if query.source:
            return [query.source]

        # Auto-detect based on language/framework
        sources = []

        # Language-based sources
        if query.language:
            lang = query.language.lower()
            if lang in ["python", "py"]:
                sources.extend([
                    DocumentationSource.PYTHON_OFFICIAL,
                    DocumentationSource.PYTHON_PYPI
                ])
            elif lang in ["typescript", "ts", "javascript", "js"]:
                sources.append(DocumentationSource.TYPESCRIPT)
            elif lang == "rust":
                sources.append(DocumentationSource.RUST)
            elif lang == "go":
                sources.append(DocumentationSource.GO)
            elif lang == "elixir":
                sources.append(DocumentationSource.ELIXIR)

        # Framework-based sources
        if query.framework:
            framework = query.framework.lower()
            if framework in ["react", "reactjs"]:
                sources.append(DocumentationSource.REACT)
            elif framework == "vue":
                sources.append(DocumentationSource.VUE)
            elif framework in ["pytorch", "torch"]:
                sources.append(DocumentationSource.PYTORCH)
            elif framework == "tensorflow":
                sources.append(DocumentationSource.TENSORFLOW)
            elif framework == "langchain":
                sources.append(DocumentationSource.LANGCHAIN)
            elif framework == "langgraph":
                sources.append(DocumentationSource.LANGGRAPH)
            elif framework == "fastapi":
                sources.append(DocumentationSource.FASTAPI)
            elif framework == "django":
                sources.append(DocumentationSource.DJANGO)
            elif framework == "flask":
                sources.append(DocumentationSource.FLASK)

        # If no specific sources, use general ones
        if not sources:
            sources = [
                DocumentationSource.PYTHON_OFFICIAL,
                DocumentationSource.TYPESCRIPT,
                DocumentationSource.REACT,
                DocumentationSource.PYTORCH
            ]

        return list(set(sources))  # Remove duplicates

    async def _search_source(
            self,
            source: DocumentationSource,
            query: DocumentationQuery
    ) -> List[DocumentationResult]:
        """Search a specific documentation source."""

        # Try vector search first
        vector_results = await self.vector_store.semantic_search(
            query=" ".join(query.search_terms),
            source=source.value,
            limit=query.max_results * 2  # Get more for filtering
        )

        if vector_results:
            return await self._process_vector_results(vector_results, query)

        # Fallback to traditional search
        return await self._traditional_search(source, query)

    async def _process_vector_results(
            self,
            vector_results: List[Dict[str, Any]],
            query: DocumentationQuery
    ) -> List[DocumentationResult]:
        """Process vector search results."""

        results = []

        for item in vector_results:
            try:
                result = DocumentationResult(
                    title=item.get("title", "Untitled"),
                    content=item.get("content", ""),
                    url=item.get("url", ""),
                    source=DocumentationSource(item.get("source", "python_official")),
                    doc_type=DocumentationType(item.get("doc_type", "api_reference")),
                    relevance_score=item.get("score", 0.0),
                    last_updated=datetime.fromisoformat(item.get("last_updated", datetime.now().isoformat())),
                    language=item.get("language"),
                    framework=item.get("framework"),
                    version=item.get("version"),
                    code_examples=item.get("code_examples", [])
                )

                # Filter by doc type if specified
                if query.doc_type and result.doc_type != query.doc_type:
                    continue

                # Filter by freshness
                if query.freshness_days:
                    cutoff_date = datetime.now() - timedelta(days=query.freshness_days)
                    if result.last_updated < cutoff_date:
                        continue

                results.append(result)

            except Exception as e:
                logger.warning(f"Failed to process vector result: {e}")
                continue

        return results

    async def _traditional_search(
            self,
            source: DocumentationSource,
            query: DocumentationQuery
    ) -> List[DocumentationResult]:
        """Traditional documentation search (fallback)."""

        # Load documentation for this source
        docs = await self.docs_loader.load_source(source.value)

        # Simple keyword matching
        results = []
        search_terms = [term.lower() for term in query.search_terms]

        for doc in docs:
            content_lower = doc.get("content", "").lower()
            title_lower = doc.get("title", "").lower()

            # Calculate relevance score
            relevance = 0.0

            # Title matches are most important
            for term in search_terms:
                if term in title_lower:
                    relevance += 0.5

            # Content matches
            for term in search_terms:
                if term in content_lower:
                    relevance += 0.1 * (content_lower.count(term) / 10)  # Normalize

            if relevance > 0:
                result = DocumentationResult(
                    title=doc.get("title", "Untitled"),
                    content=doc.get("content", ""),
                    url=doc.get("url", ""),
                    source=source,
                    doc_type=DocumentationType(doc.get("doc_type", "api_reference")),
                    relevance_score=min(relevance, 1.0),
                    last_updated=datetime.fromisoformat(doc.get("last_updated", datetime.now().isoformat())),
                    language=doc.get("language"),
                    framework=doc.get("framework"),
                    version=doc.get("version")
                )
                results.append(result)

        return results

    def _should_enhance_with_practices(self, query: DocumentationQuery) -> bool:
        """Determine if results should be enhanced with elite practices."""
        practice_keywords = [
            "best practice", "best practices", "clean code", "production",
            "scalable", "maintainable", "performance", "security",
            "testing", "deployment", "monitoring", "observability"
        ]

        query_text = " ".join(query.search_terms).lower()
        return any(keyword in query_text for keyword in practice_keywords)

    async def _enhance_with_elite_practices(
            self,
            results: List[DocumentationResult],
            query: DocumentationQuery
    ) -> List[DocumentationResult]:
        """Enhance documentation results with elite engineering practices."""

        enhanced_results = []

        for result in results:
            # Get relevant elite practices
            practices = await self.elite_practices.get_relevant_practices(
                topic=result.title,
                language=result.language,
                framework=result.framework
            )

            if practices:
                # Enhance content with practices
                enhanced_content = self._combine_content_with_practices(
                    result.content,
                    practices
                )

                # Add related topics from practices
                related_topics = list(set(result.related_topics + practices.related_topics))

                # Create enhanced result
                enhanced_result = DocumentationResult(
                    title=f"🚀 {result.title} (Elite Practices)",
                    content=enhanced_content,
                    url=result.url,
                    source=result.source,
                    doc_type=DocumentationType.BEST_PRACTICES,
                    relevance_score=min(result.relevance_score * 1.1, 1.0),  # Boost score
                    last_updated=result.last_updated,
                    language=result.language,
                    framework=result.framework,
                    version=result.version,
                    code_examples=result.code_examples,
                    related_topics=related_topics
                )

                enhanced_results.append(enhanced_result)

            # Always include original result
            enhanced_results.append(result)

        return enhanced_results

    def _combine_content_with_practices(
            self,
            content: str,
            practices: Any  # ElitePracticeCollection
    ) -> str:
        """Combine documentation content with elite practices."""

        practice_sections = []

        for practice in practices.practices:
            practice_sections.append(
                f"\n\n### 🎯 Elite Practice: {practice.title}\n"
                f"{practice.description}\n\n"
                f"**Why it matters:** {practice.rationale}\n\n"
                f"**Implementation:**\n{practice.implementation}\n\n"
                f"**Example:**\n```{practices.language or 'python'}\n{practice.example}\n```"
            )

        return f"""{content}

## 🔥 Elite Engineering Practices

*These practices represent the top 0.1% of engineering standards. 
They've been battle-tested at scale and will help you write production-grade code.*

{''.join(practice_sections)}
"""

    def _format_results(
            self,
            results: List[DocumentationResult],
            query: DocumentationQuery
    ) -> Dict[str, Any]:
        """Format results for MCP response."""

        formatted_results = []

        for result in results:
            result_dict = result.to_dict()

            # Add code examples if requested
            if query.include_code_examples and result.code_examples:
                result_dict["code_examples"] = result.code_examples[:3]  # Limit to 3

            formatted_results.append(result_dict)

        return {
            "query": " ".join(query.search_terms),
            "total_results": len(results),
            "results": formatted_results,
            "sources_searched": [source.value for source in self._determine_sources(query)],
            "filters_applied": {
                "source": query.source.value if query.source else None,
                "doc_type": query.doc_type.value if query.doc_type else None,
                "language": query.language,
                "framework": query.framework,
                "freshness_days": query.freshness_days
            },
            "metadata": {
                "search_id": hashlib.md5(str(query).encode()).hexdigest()[:8],
                "timestamp": datetime.now().isoformat(),
                "cache_hit": len(results) > 0 and hasattr(self,
                                                          '_last_cache_key') and self._last_cache_key == query.to_cache_key()
            }
        }

    def _initialize_source_urls(self) -> Dict[DocumentationSource, str]:
        """Initialize URLs for documentation sources."""
        return {
            DocumentationSource.PYTHON_OFFICIAL: "https://docs.python.org/3/",
            DocumentationSource.PYTHON_PYPI: "https://pypi.org/",
            DocumentationSource.TYPESCRIPT: "https://www.typescriptlang.org/docs/",
            DocumentationSource.REACT: "https://react.dev/",
            DocumentationSource.VUE: "https://vuejs.org/guide/",
            DocumentationSource.DOCKER: "https://docs.docker.com/",
            DocumentationSource.KUBERNETES: "https://kubernetes.io/docs/",
            DocumentationSource.AWS: "https://docs.aws.amazon.com/",
            DocumentationSource.GCP: "https://cloud.google.com/docs",
            DocumentationSource.AZURE: "https://docs.microsoft.com/en-us/azure/",
            DocumentationSource.PYTORCH: "https://pytorch.org/docs/stable/",
            DocumentationSource.TENSORFLOW: "https://www.tensorflow.org/api_docs",
            DocumentationSource.LANGCHAIN: "https://python.langchain.com/docs/",
            DocumentationSource.LANGGRAPH: "https://langchain-ai.github.io/langgraph/",
            DocumentationSource.FASTAPI: "https://fastapi.tiangolo.com/",
            DocumentationSource.DJANGO: "https://docs.djangoproject.com/",
            DocumentationSource.FLASK: "https://flask.palletsprojects.com/",
            DocumentationSource.POSTGRESQL: "https://www.postgresql.org/docs/",
            DocumentationSource.REDIS: "https://redis.io/docs/",
            DocumentationSource.KAFKA: "https://kafka.apache.org/documentation/",
            DocumentationSource.ELIXIR: "https://elixir-lang.org/docs.html",
            DocumentationSource.RUST: "https://doc.rust-lang.org/",
            DocumentationSource.GO: "https://go.dev/doc/"
        }

    async def start_background_updates(self, interval_hours: int = 6):
        """Start background task for updating documentation."""
        if self._update_task is None:
            self._update_task = asyncio.create_task(
                self._update_documentation_loop(interval_hours)
            )
            logger.info(f"Started background documentation updates every {interval_hours} hours")

    async def stop_background_updates(self):
        """Stop background update task."""
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
            self._update_task = None
            logger.info("Stopped background documentation updates")

    async def _update_documentation_loop(self, interval_hours: int):
        """Background loop for updating documentation."""
        while True:
            try:
                await asyncio.sleep(interval_hours * 3600)
                await self._update_documentation()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Documentation update failed: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry

    async def _update_documentation(self):
        """Update documentation from all sources."""
        if self._is_updating:
            logger.warning("Documentation update already in progress")
            return

        self._is_updating = True
        logger.info("Starting documentation update...")

        try:
            # Update each source
            for source in DocumentationSource:
                try:
                    await self.docs_loader.update_source(source.value)
                    logger.info(f"Updated documentation for {source.value}")
                except Exception as e:
                    logger.error(f"Failed to update {source.value}: {e}")

            # Re-index in vector store
            await self.vector_store.reindex_all()

            # Clear cache since we have new data
            self.cache.clear()

            logger.info("Documentation update completed successfully")

        finally:
            self._is_updating = False