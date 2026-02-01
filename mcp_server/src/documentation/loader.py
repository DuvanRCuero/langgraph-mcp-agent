"""
Asynchronous documentation loader with support for multiple sources,
real-time updates, and efficient parsing.
Built with resilience and performance in mind.
"""

import asyncio
import aiohttp
import aiofiles
import json
import logging
import re
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin, urlparse
import gzip
import ssl
import certifi

from bs4 import BeautifulSoup
from markdownify import markdownify
import yaml

logger = logging.getLogger(__name__)


@dataclass
class DocumentationEntry:
    """A single documentation entry."""
    id: str
    title: str
    content: str
    url: str
    source: str
    doc_type: str
    language: Optional[str] = None
    framework: Optional[str] = None
    version: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "source": self.source,
            "doc_type": self.doc_type,
            "language": self.language,
            "framework": self.framework,
            "version": self.version,
            "tags": self.tags,
            "last_updated": self.last_updated.isoformat(),
            "metadata": self.metadata
        }


class AsyncHTTPClient:
    """Async HTTP client with retries, rate limiting, and caching."""

    def __init__(self, max_retries: int = 3, rate_limit: float = 10.0):
        self.max_retries = max_retries
        self.rate_limit = rate_limit  # requests per second
        self.semaphore = asyncio.Semaphore(int(rate_limit))
        self.session: Optional[aiohttp.ClientSession] = None

        # SSL context for secure connections
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        ssl_context.check_hostname = True
        ssl_context.verify_mode = ssl.CERT_REQUIRED

        self.ssl_context = ssl_context

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=self.ssl_context),
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get(self, url: str, **kwargs) -> Optional[str]:
        """GET request with retries and rate limiting."""
        async with self.semaphore:
            for attempt in range(self.max_retries):
                try:
                    async with self.session.get(url, **kwargs) as response:
                        if response.status == 200:
                            return await response.text()
                        elif response.status == 404:
                            logger.warning(f"404 Not Found: {url}")
                            return None
                        else:
                            logger.warning(f"HTTP {response.status} for {url}")
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        logger.error(f"Failed to fetch {url} after {self.max_retries} attempts")
            return None

    async def get_json(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """GET JSON with retries."""
        text = await self.get(url, **kwargs)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from {url}: {e}")
        return None


class DocumentationSourceConfig:
    """Configuration for a documentation source."""

    def __init__(
            self,
            name: str,
            base_url: str,
            parser_type: str = "html",
            update_interval: int = 86400,  # 24 hours
            priority: int = 1,
            enabled: bool = True
    ):
        self.name = name
        self.base_url = base_url
        self.parser_type = parser_type
        self.update_interval = update_interval
        self.priority = priority
        self.enabled = enabled
        self.last_updated: Optional[datetime] = None
        self.next_update: Optional[datetime] = None


class DocumentationLoader:
    """
    Asynchronous documentation loader that fetches, parses, and stores
    documentation from multiple sources.
    """

    def __init__(self, data_dir: str = "./data/documentation"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.sources: Dict[str, DocumentationSourceConfig] = {}
        self.entries: Dict[str, DocumentationEntry] = {}
        self.http_client = AsyncHTTPClient()

        self._initialized = False
        self._lock = asyncio.Lock()

        # Register default sources
        self._register_default_sources()

    def _register_default_sources(self):
        """Register default documentation sources."""
        sources = [
            DocumentationSourceConfig(
                name="python_official",
                base_url="https://docs.python.org/3/",
                parser_type="sphinx",
                update_interval=86400,
                priority=1
            ),
            DocumentationSourceConfig(
                name="python_pypi",
                base_url="https://pypi.org/",
                parser_type="html",
                update_interval=43200,
                priority=2
            ),
            DocumentationSourceConfig(
                name="typescript",
                base_url="https://www.typescriptlang.org/docs/",
                parser_type="docusaurus",
                update_interval=86400,
                priority=1
            ),
            DocumentationSourceConfig(
                name="react",
                base_url="https://react.dev/",
                parser_type="mdx",
                update_interval=43200,
                priority=1
            ),
            DocumentationSourceConfig(
                name="pytorch",
                base_url="https://pytorch.org/docs/stable/",
                parser_type="sphinx",
                update_interval=86400,
                priority=1
            ),
            DocumentationSourceConfig(
                name="docker",
                base_url="https://docs.docker.com/",
                parser_type="gitbook",
                update_interval=86400,
                priority=2
            ),
            DocumentationSourceConfig(
                name="kubernetes",
                base_url="https://kubernetes.io/docs/",
                parser_type="hugo",
                update_interval=86400,
                priority=2
            )
        ]

        for source in sources:
            self.sources[source.name] = source

    async def initialize(self):
        """Initialize the loader."""
        if self._initialized:
            return

        async with self._lock:
            # Load cached entries
            await self._load_cached_entries()

            # Update sources if needed
            await self._update_sources_if_needed()

            self._initialized = True
            logger.info(f"DocumentationLoader initialized with {len(self.entries)} entries")

    async def _load_cached_entries(self):
        """Load cached documentation entries from disk."""
        cache_file = self.data_dir / "cache.json"

        if cache_file.exists():
            try:
                async with aiofiles.open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.loads(await f.read())

                for entry_data in data.get("entries", []):
                    entry = DocumentationEntry(
                        id=entry_data["id"],
                        title=entry_data["title"],
                        content=entry_data["content"],
                        url=entry_data["url"],
                        source=entry_data["source"],
                        doc_type=entry_data["doc_type"],
                        language=entry_data.get("language"),
                        framework=entry_data.get("framework"),
                        version=entry_data.get("version"),
                        tags=entry_data.get("tags", []),
                        last_updated=datetime.fromisoformat(entry_data["last_updated"]),
                        metadata=entry_data.get("metadata", {})
                    )
                    self.entries[entry.id] = entry

                logger.info(f"Loaded {len(self.entries)} cached entries")

            except Exception as e:
                logger.error(f"Failed to load cache: {e}")

    async def _save_cached_entries(self):
        """Save documentation entries to disk."""
        cache_file = self.data_dir / "cache.json"

        try:
            data = {
                "entries": [entry.to_dict() for entry in self.entries.values()],
                "timestamp": datetime.now().isoformat(),
                "count": len(self.entries)
            }

            async with aiofiles.open(cache_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(data, indent=2, ensure_ascii=False))

            logger.info(f"Saved {len(self.entries)} entries to cache")

        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    async def _update_sources_if_needed(self):
        """Update sources that need updating."""
        update_tasks = []

        for source_name, source in self.sources.items():
            if not source.enabled:
                continue

            needs_update = (
                    source.next_update is None or
                    datetime.now() >= source.next_update
            )

            if needs_update:
                update_tasks.append(self._update_source(source_name))

        if update_tasks:
            logger.info(f"Updating {len(update_tasks)} sources...")
            await asyncio.gather(*update_tasks, return_exceptions=True)

    async def _update_source(self, source_name: str):
        """Update a single documentation source."""
        source = self.sources[source_name]

        logger.info(f"Updating source: {source_name}")

        try:
            async with self.http_client:
                # Get sitemap or index
                if source.parser_type == "sphinx":
                    entries = await self._parse_sphinx(source)
                elif source.parser_type == "docusaurus":
                    entries = await self._parse_docusaurus(source)
                elif source.parser_type == "mdx":
                    entries = await self._parse_mdx(source)
                elif source.parser_type == "gitbook":
                    entries = await self._parse_gitbook(source)
                elif source.parser_type == "hugo":
                    entries = await self._parse_hugo(source)
                else:
                    entries = await self._parse_html(source)

                # Update entries
                for entry in entries:
                    self.entries[entry.id] = entry

                # Update source metadata
                source.last_updated = datetime.now()
                source.next_update = source.last_updated + timedelta(seconds=source.update_interval)

                logger.info(f"Updated source {source_name} with {len(entries)} entries")

                # Save cache
                await self._save_cached_entries()

        except Exception as e:
            logger.error(f"Failed to update source {source_name}: {e}")

    async def _parse_sphinx(self, source: DocumentationSourceConfig) -> List[DocumentationEntry]:
        """Parse Sphinx-generated documentation."""
        entries = []

        # Try to find objects.inv (Sphinx inventory)
        inventory_url = urljoin(source.base_url, "objects.inv")
        inventory = await self._parse_sphinx_inventory(inventory_url)

        if inventory:
            # Parse each documented object
            for obj_name, obj_info in inventory.items():
                entry = await self._create_entry_from_sphinx_object(
                    source, obj_name, obj_info
                )
                if entry:
                    entries.append(entry)
        else:
            # Fallback: parse index.html
            index_url = urljoin(source.base_url, "index.html")
            html = await self.http_client.get(index_url)

            if html:
                entries.extend(await self._parse_sphinx_index(html, source))

        return entries

    async def _parse_sphinx_inventory(self, url: str) -> Optional[Dict[str, Any]]:
        """Parse Sphinx inventory file."""
        try:
            content = await self.http_client.get(url)
            if not content:
                return None

            # Parse inventory (simplified)
            # In production, use sphinx.util.inventory
            return self._decode_sphinx_inventory(content)

        except Exception as e:
            logger.warning(f"Failed to parse Sphinx inventory: {e}")
            return None

    def _decode_sphinx_inventory(self, content: str) -> Dict[str, Any]:
        """Decode Sphinx inventory file."""
        # Simplified implementation
        # In production, implement full Sphinx inventory parsing
        result = {}

        lines = content.split('\n')
        for line in lines:
            if line and not line.startswith('#'):
                parts = line.split(' ')
                if len(parts) >= 4:
                    name = parts[0]
                    result[name] = {
                        "type": parts[1],
                        "location": parts[2]
                    }

        return result

    async def _create_entry_from_sphinx_object(
            self,
            source: DocumentationSourceConfig,
            obj_name: str,
            obj_info: Dict[str, Any]
    ) -> Optional[DocumentationEntry]:
        """Create entry from Sphinx object."""
        try:
            # Build URL
            obj_url = urljoin(source.base_url, obj_info.get("location", ""))

            # Fetch and parse the page
            html = await self.http_client.get(obj_url)
            if not html:
                return None

            # Parse HTML
            soup = BeautifulSoup(html, 'html.parser')

            # Extract content
            content_div = soup.find('div', {'role': 'main'}) or soup.find('body')
            if not content_div:
                return None

            # Convert to markdown
            content = markdownify(str(content_div))

            # Extract title
            title = soup.find('h1')
            title_text = title.get_text() if title else obj_name

            # Determine doc type
            doc_type = self._determine_doc_type_from_sphinx(obj_info.get("type", ""))

            # Extract language/framework from source name
            language, framework = self._extract_metadata_from_source(source.name)

            # Generate ID
            entry_id = f"{source.name}:{obj_name}"

            return DocumentationEntry(
                id=entry_id,
                title=title_text,
                content=content,
                url=obj_url,
                source=source.name,
                doc_type=doc_type,
                language=language,
                framework=framework,
                version=self._extract_version_from_url(source.base_url),
                tags=self._extract_tags_from_content(content),
                last_updated=datetime.now()
            )

        except Exception as e:
            logger.warning(f"Failed to create entry for {obj_name}: {e}")
            return None

    def _determine_doc_type_from_sphinx(self, obj_type: str) -> str:
        """Determine documentation type from Sphinx object type."""
        type_map = {
            "function": "api_reference",
            "class": "api_reference",
            "method": "api_reference",
            "module": "api_reference",
            "attribute": "api_reference",
            "tutorial": "tutorial",
            "guide": "getting_started",
            "howto": "tutorial",
            "ref": "api_reference",
            "std:label": "api_reference"
        }
        return type_map.get(obj_type, "api_reference")

    async def _parse_docusaurus(self, source: DocumentationSourceConfig) -> List[DocumentationEntry]:
        """Parse Docusaurus documentation."""
        entries = []

        # Try to get sitemap
        sitemap_url = urljoin(source.base_url, "sitemap.xml")
        sitemap = await self.http_client.get(sitemap_url)

        if sitemap:
            # Parse sitemap XML
            urls = self._extract_urls_from_sitemap(sitemap)

            for url in urls[:50]:  # Limit for demo
                entry = await self._parse_docusaurus_page(url, source)
                if entry:
                    entries.append(entry)

        return entries

    async def _parse_docusaurus_page(
            self,
            url: str,
            source: DocumentationSourceConfig
    ) -> Optional[DocumentationEntry]:
        """Parse a single Docusaurus page."""
        try:
            html = await self.http_client.get(url)
            if not html:
                return None

            soup = BeautifulSoup(html, 'html.parser')

            # Find main content
            main_content = soup.find('article') or soup.find('main')
            if not main_content:
                return None

            # Extract title
            title = soup.find('h1') or soup.find('title')
            title_text = title.get_text() if title else url.split('/')[-1]

            # Convert to markdown
            content = markdownify(str(main_content))

            # Determine doc type from URL
            doc_type = self._determine_doc_type_from_url(url)

            # Generate ID
            entry_id = f"{source.name}:{url}"

            return DocumentationEntry(
                id=entry_id,
                title=title_text,
                content=content,
                url=url,
                source=source.name,
                doc_type=doc_type,
                language=self._extract_language_from_source(source.name),
                framework=self._extract_framework_from_source(source.name),
                tags=self._extract_tags_from_content(content),
                last_updated=datetime.now()
            )

        except Exception as e:
            logger.warning(f"Failed to parse Docusaurus page {url}: {e}")
            return None

    async def _parse_html(self, source: DocumentationSourceConfig) -> List[DocumentationEntry]:
        """Generic HTML parser."""
        entries = []

        # Fetch the main page
        html = await self.http_client.get(source.base_url)
        if not html:
            return entries

        soup = BeautifulSoup(html, 'html.parser')

        # Find all links that look like documentation
        links = soup.find_all('a', href=True)

        for link in links[:20]:  # Limit for demo
            href = link['href']

            # Skip external links, fragments, etc.
            if not href or href.startswith('#') or href.startswith('javascript:'):
                continue

            # Build absolute URL
            page_url = urljoin(source.base_url, href)

            # Only follow same-domain links
            if urlparse(page_url).netloc != urlparse(source.base_url).netloc:
                continue

            # Parse the page
            entry = await self._parse_generic_page(page_url, source)
            if entry:
                entries.append(entry)

        return entries

    async def _parse_generic_page(
            self,
            url: str,
            source: DocumentationSourceConfig
    ) -> Optional[DocumentationEntry]:
        """Parse a generic HTML page."""
        try:
            html = await self.http_client.get(url)
            if not html:
                return None

            soup = BeautifulSoup(html, 'html.parser')

            # Remove script and style elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()

            # Find main content
            main_content = soup.find('main') or soup.find('article') or soup.find('body')
            if not main_content:
                return None

            # Extract title
            title = soup.find('h1') or soup.find('title')
            title_text = title.get_text() if title else url.split('/')[-1]

            # Convert to markdown
            content = markdownify(str(main_content))

            # Clean up content
            content = self._clean_content(content)

            # Generate ID
            entry_id = f"{source.name}:{url}"

            return DocumentationEntry(
                id=entry_id,
                title=title_text,
                content=content,
                url=url,
                source=source.name,
                doc_type=self._determine_doc_type_from_url(url),
                language=self._extract_language_from_source(source.name),
                framework=self._extract_framework_from_source(source.name),
                tags=self._extract_tags_from_content(content),
                last_updated=datetime.now()
            )

        except Exception as e:
            logger.warning(f"Failed to parse page {url}: {e}")
            return None

    def _clean_content(self, content: str) -> str:
        """Clean up markdown content."""
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)

        # Remove empty lines at start/end
        content = content.strip()

        # Limit length (for demo)
        if len(content) > 10000:
            content = content[:10000] + "...\n\n*Content truncated for demonstration*"

        return content

    def _determine_doc_type_from_url(self, url: str) -> str:
        """Determine documentation type from URL."""
        url_lower = url.lower()

        if any(word in url_lower for word in ['tutorial', 'learn', 'getting-started']):
            return "tutorial"
        elif any(word in url_lower for word in ['api', 'reference', 'docs']):
            return "api_reference"
        elif any(word in url_lower for word in ['guide', 'how-to']):
            return "getting_started"
        elif any(word in url_lower for word in ['best-practice', 'best_practice']):
            return "best_practices"
        elif any(word in url_lower for word in ['troubleshoot', 'faq']):
            return "troubleshooting"
        elif any(word in url_lower for word in ['security', 'secure']):
            return "security"
        elif any(word in url_lower for word in ['performance', 'optimization']):
            return "performance"
        elif any(word in url_lower for word in ['deploy', 'production']):
            return "deployment"
        else:
            return "api_reference"

    def _extract_language_from_source(self, source_name: str) -> Optional[str]:
        """Extract language from source name."""
        if 'python' in source_name:
            return 'python'
        elif 'typescript' in source_name or 'react' in source_name:
            return 'typescript'
        elif 'rust' in source_name:
            return 'rust'
        elif 'go' in source_name:
            return 'go'
        elif 'elixir' in source_name:
            return 'elixir'
        return None

    def _extract_framework_from_source(self, source_name: str) -> Optional[str]:
        """Extract framework from source name."""
        if 'pytorch' in source_name:
            return 'pytorch'
        elif 'tensorflow' in source_name:
            return 'tensorflow'
        elif 'react' in source_name:
            return 'react'
        elif 'vue' in source_name:
            return 'vue'
        elif 'fastapi' in source_name:
            return 'fastapi'
        elif 'django' in source_name:
            return 'django'
        elif 'flask' in source_name:
            return 'flask'
        elif 'langchain' in source_name:
            return 'langchain'
        elif 'langgraph' in source_name:
            return 'langgraph'
        return None

    def _extract_version_from_url(self, url: str) -> Optional[str]:
        """Extract version from URL."""
        match = re.search(r'/(\d+\.\d+(?:\.\d+)?)/', url)
        return match.group(1) if match else None

    def _extract_tags_from_content(self, content: str) -> List[str]:
        """Extract tags from content."""
        tags = []

        # Extract potential tags from content (simplified)
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content[:1000])

        # Common programming terms to include as tags
        programming_terms = {
            'function', 'class', 'method', 'module', 'package',
            'async', 'await', 'decorator', 'generator', 'iterator',
            'context', 'manager', 'exception', 'error', 'handling',
            'database', 'query', 'orm', 'migration', 'schema',
            'api', 'endpoint', 'route', 'middleware', 'authentication',
            'testing', 'unit', 'integration', 'mock', 'fixture',
            'deployment', 'docker', 'kubernetes', 'ci', 'cd'
        }

        for word in words:
            if word.lower() in programming_terms and word.lower() not in tags:
                tags.append(word.lower())

        return list(set(tags))[:10]  # Limit to 10 tags

    def _extract_urls_from_sitemap(self, sitemap_xml: str) -> List[str]:
        """Extract URLs from sitemap XML."""
        urls = []

        # Simplified XML parsing
        # In production, use proper XML parser
        url_pattern = r'<loc>(.*?)</loc>'
        matches = re.findall(url_pattern, sitemap_xml)

        for match in matches:
            urls.append(match)

        return urls

    async def get_entries(
            self,
            source: Optional[str] = None,
            doc_type: Optional[str] = None,
            language: Optional[str] = None,
            limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get documentation entries with filters."""
        if not self._initialized:
            await self.initialize()

        filtered_entries = []

        for entry in self.entries.values():
            if source and entry.source != source:
                continue
            if doc_type and entry.doc_type != doc_type:
                continue
            if language and entry.language != language:
                continue

            filtered_entries.append(entry.to_dict())

            if len(filtered_entries) >= limit:
                break

        return filtered_entries

    async def search_entries(
            self,
            query: str,
            limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search documentation entries."""
        if not self._initialized:
            await self.initialize()

        # Simple keyword search (in production, use vector search)
        query_lower = query.lower()
        results = []

        for entry in self.entries.values():
            score = 0

            # Title match
            if query_lower in entry.title.lower():
                score += 3

            # Content match
            if query_lower in entry.content.lower():
                score += 1

            # Tag match
            for tag in entry.tags:
                if query_lower in tag.lower():
                    score += 2

            if score > 0:
                entry_dict = entry.to_dict()
                entry_dict["score"] = score
                results.append(entry_dict)

        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:limit]

    async def update_source(self, source_name: str):
        """Update a specific source."""
        if source_name in self.sources:
            await self._update_source(source_name)

    async def update_all_sources(self):
        """Update all sources."""
        update_tasks = []

        for source_name in self.sources:
            update_tasks.append(self._update_source(source_name))

        if update_tasks:
            logger.info(f"Updating all {len(update_tasks)} sources...")
            results = await asyncio.gather(*update_tasks, return_exceptions=True)

            # Log results
            for source_name, result in zip(self.sources.keys(), results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to update {source_name}: {result}")
                else:
                    logger.info(f"Successfully updated {source_name}")

    def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics."""
        stats = {
            "total_entries": len(self.entries),
            "sources": {},
            "by_language": {},
            "by_doc_type": {}
        }

        # Source stats
        for source_name, source in self.sources.items():
            source_entries = [
                e for e in self.entries.values()
                if e.source == source_name
            ]
            stats["sources"][source_name] = {
                "count": len(source_entries),
                "last_updated": source.last_updated.isoformat() if source.last_updated else None,
                "next_update": source.next_update.isoformat() if source.next_update else None,
                "enabled": source.enabled
            }

        # Language stats
        for entry in self.entries.values():
            lang = entry.language or "unknown"
            stats["by_language"][lang] = stats["by_language"].get(lang, 0) + 1

        # Doc type stats
        for entry in self.entries.values():
            doc_type = entry.doc_type
            stats["by_doc_type"][doc_type] = stats["by_doc_type"].get(doc_type, 0) + 1

        return stats