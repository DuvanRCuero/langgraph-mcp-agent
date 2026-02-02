"""
Repository pattern interfaces for entity persistence abstraction.
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional

from ...domain.entities import ProgrammingTask, CodeSnippet, ArchitectureDesign
from ...domain.result import Result
from ...domain.errors import DomainError

T = TypeVar('T')
ID = TypeVar('ID')


class Repository(ABC, Generic[T, ID]):
    """Base repository interface."""
    
    @abstractmethod
    async def find_by_id(self, id: ID) -> Result[Optional[T], DomainError]:
        """Find entity by ID."""
        pass
    
    @abstractmethod
    async def find_all(self, limit: int = 100, offset: int = 0) -> Result[List[T], DomainError]:
        """Find all entities with pagination."""
        pass
    
    @abstractmethod
    async def save(self, entity: T) -> Result[T, DomainError]:
        """Save entity (create or update)."""
        pass
    
    @abstractmethod
    async def delete(self, id: ID) -> Result[bool, DomainError]:
        """Delete entity by ID."""
        pass
    
    @abstractmethod
    async def exists(self, id: ID) -> Result[bool, DomainError]:
        """Check if entity exists."""
        pass


class TaskRepository(Repository[ProgrammingTask, str], ABC):
    """Repository for ProgrammingTask entities."""
    
    @abstractmethod
    async def find_by_status(self, status: str) -> Result[List[ProgrammingTask], DomainError]:
        """Find tasks by status."""
        pass
    
    @abstractmethod
    async def find_by_language(self, language: str) -> Result[List[ProgrammingTask], DomainError]:
        """Find tasks by programming language."""
        pass
    
    @abstractmethod
    async def find_recent(self, limit: int = 10) -> Result[List[ProgrammingTask], DomainError]:
        """Find most recent tasks."""
        pass
    
    @abstractmethod
    async def update_status(self, task_id: str, status: str) -> Result[ProgrammingTask, DomainError]:
        """Update task status."""
        pass


class CodeSnippetRepository(Repository[CodeSnippet, str], ABC):
    """Repository for CodeSnippet entities."""
    
    @abstractmethod
    async def find_by_task_id(self, task_id: str) -> Result[List[CodeSnippet], DomainError]:
        """Find code snippets by task ID."""
        pass
    
    @abstractmethod
    async def find_by_language(self, language: str) -> Result[List[CodeSnippet], DomainError]:
        """Find code snippets by language."""
        pass
    
    @abstractmethod
    async def search_by_content(self, query: str, limit: int = 10) -> Result[List[CodeSnippet], DomainError]:
        """Search code snippets by content."""
        pass


class ArchitectureRepository(Repository[ArchitectureDesign, str], ABC):
    """Repository for ArchitectureDesign entities."""
    
    @abstractmethod
    async def find_by_style(self, style: str) -> Result[List[ArchitectureDesign], DomainError]:
        """Find architectures by style."""
        pass
    
    @abstractmethod
    async def find_by_task_id(self, task_id: str) -> Result[Optional[ArchitectureDesign], DomainError]:
        """Find architecture by task ID."""
        pass
