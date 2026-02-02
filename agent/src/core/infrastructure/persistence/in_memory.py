"""
In-memory implementations of repositories and Unit of Work for testing.
"""

from typing import Dict, List, Optional
from copy import deepcopy

from ...application.ports.repositories import (
    TaskRepository, CodeSnippetRepository, ArchitectureRepository
)
from ...application.ports.unit_of_work import UnitOfWork, UnitOfWorkFactory
from ...domain.entities import ProgrammingTask, CodeSnippet, ArchitectureDesign, TaskStatus
from ...domain.result import Result, Success, Failure
from ...domain.errors import DomainError, NotFoundError, ErrorCode


class InMemoryTaskRepository(TaskRepository):
    """In-memory implementation of TaskRepository."""
    
    def __init__(self):
        self._storage: Dict[str, ProgrammingTask] = {}
    
    async def find_by_id(self, id: str) -> Result[Optional[ProgrammingTask], DomainError]:
        """Find entity by ID."""
        try:
            task = self._storage.get(id)
            return Success(deepcopy(task) if task else None)
        except Exception as e:
            return Failure(DomainError(
                code=ErrorCode.INTERNAL_ERROR,
                message=f"Failed to find task: {str(e)}"
            ))
    
    async def find_all(self, limit: int = 100, offset: int = 0) -> Result[List[ProgrammingTask], DomainError]:
        """Find all entities with pagination."""
        try:
            all_tasks = list(self._storage.values())
            paginated = all_tasks[offset:offset + limit]
            return Success([deepcopy(task) for task in paginated])
        except Exception as e:
            return Failure(DomainError(
                code=ErrorCode.INTERNAL_ERROR,
                message=f"Failed to find all tasks: {str(e)}"
            ))
    
    async def save(self, entity: ProgrammingTask) -> Result[ProgrammingTask, DomainError]:
        """Save entity (create or update)."""
        try:
            self._storage[entity.task_id] = deepcopy(entity)
            return Success(deepcopy(entity))
        except Exception as e:
            return Failure(DomainError(
                code=ErrorCode.INTERNAL_ERROR,
                message=f"Failed to save task: {str(e)}"
            ))
    
    async def delete(self, id: str) -> Result[bool, DomainError]:
        """Delete entity by ID."""
        try:
            if id in self._storage:
                del self._storage[id]
                return Success(True)
            return Success(False)
        except Exception as e:
            return Failure(DomainError(
                code=ErrorCode.INTERNAL_ERROR,
                message=f"Failed to delete task: {str(e)}"
            ))
    
    async def exists(self, id: str) -> Result[bool, DomainError]:
        """Check if entity exists."""
        try:
            return Success(id in self._storage)
        except Exception as e:
            return Failure(DomainError(
                code=ErrorCode.INTERNAL_ERROR,
                message=f"Failed to check task existence: {str(e)}"
            ))
    
    async def find_by_status(self, status: str) -> Result[List[ProgrammingTask], DomainError]:
        """Find tasks by status."""
        try:
            tasks = [
                deepcopy(task) for task in self._storage.values()
                if task.status.value == status
            ]
            return Success(tasks)
        except Exception as e:
            return Failure(DomainError(
                code=ErrorCode.INTERNAL_ERROR,
                message=f"Failed to find tasks by status: {str(e)}"
            ))
    
    async def find_by_language(self, language: str) -> Result[List[ProgrammingTask], DomainError]:
        """Find tasks by programming language."""
        try:
            tasks = [
                deepcopy(task) for task in self._storage.values()
                if task.language.value == language
            ]
            return Success(tasks)
        except Exception as e:
            return Failure(DomainError(
                code=ErrorCode.INTERNAL_ERROR,
                message=f"Failed to find tasks by language: {str(e)}"
            ))
    
    async def find_recent(self, limit: int = 10) -> Result[List[ProgrammingTask], DomainError]:
        """Find most recent tasks."""
        try:
            all_tasks = sorted(
                self._storage.values(),
                key=lambda t: t.created_at,
                reverse=True
            )
            recent = all_tasks[:limit]
            return Success([deepcopy(task) for task in recent])
        except Exception as e:
            return Failure(DomainError(
                code=ErrorCode.INTERNAL_ERROR,
                message=f"Failed to find recent tasks: {str(e)}"
            ))
    
    async def update_status(self, task_id: str, status: str) -> Result[ProgrammingTask, DomainError]:
        """Update task status."""
        try:
            if task_id not in self._storage:
                return Failure(NotFoundError(
                    code=ErrorCode.NOT_FOUND,
                    message=f"Task not found: {task_id}",
                    resource_type="ProgrammingTask",
                    resource_id=task_id
                ))
            
            task = self._storage[task_id]
            task.update_status(TaskStatus(status))
            return Success(deepcopy(task))
        except Exception as e:
            return Failure(DomainError(
                code=ErrorCode.INTERNAL_ERROR,
                message=f"Failed to update task status: {str(e)}"
            ))


class InMemoryCodeSnippetRepository(CodeSnippetRepository):
    """In-memory implementation of CodeSnippetRepository."""
    
    def __init__(self):
        self._storage: Dict[str, CodeSnippet] = {}
        self._task_index: Dict[str, List[str]] = {}  # task_id -> list of snippet ids
    
    def _get_snippet_id(self, snippet: CodeSnippet) -> str:
        """Generate a unique ID for a snippet."""
        return snippet.compute_hash()
    
    async def find_by_id(self, id: str) -> Result[Optional[CodeSnippet], DomainError]:
        """Find entity by ID."""
        try:
            snippet = self._storage.get(id)
            return Success(deepcopy(snippet) if snippet else None)
        except Exception as e:
            return Failure(DomainError(
                code=ErrorCode.INTERNAL_ERROR,
                message=f"Failed to find snippet: {str(e)}"
            ))
    
    async def find_all(self, limit: int = 100, offset: int = 0) -> Result[List[CodeSnippet], DomainError]:
        """Find all entities with pagination."""
        try:
            all_snippets = list(self._storage.values())
            paginated = all_snippets[offset:offset + limit]
            return Success([deepcopy(snippet) for snippet in paginated])
        except Exception as e:
            return Failure(DomainError(
                code=ErrorCode.INTERNAL_ERROR,
                message=f"Failed to find all snippets: {str(e)}"
            ))
    
    async def save(self, entity: CodeSnippet) -> Result[CodeSnippet, DomainError]:
        """Save entity (create or update)."""
        try:
            snippet_id = self._get_snippet_id(entity)
            self._storage[snippet_id] = deepcopy(entity)
            return Success(deepcopy(entity))
        except Exception as e:
            return Failure(DomainError(
                code=ErrorCode.INTERNAL_ERROR,
                message=f"Failed to save snippet: {str(e)}"
            ))
    
    async def delete(self, id: str) -> Result[bool, DomainError]:
        """Delete entity by ID."""
        try:
            if id in self._storage:
                del self._storage[id]
                return Success(True)
            return Success(False)
        except Exception as e:
            return Failure(DomainError(
                code=ErrorCode.INTERNAL_ERROR,
                message=f"Failed to delete snippet: {str(e)}"
            ))
    
    async def exists(self, id: str) -> Result[bool, DomainError]:
        """Check if entity exists."""
        try:
            return Success(id in self._storage)
        except Exception as e:
            return Failure(DomainError(
                code=ErrorCode.INTERNAL_ERROR,
                message=f"Failed to check snippet existence: {str(e)}"
            ))
    
    async def find_by_task_id(self, task_id: str) -> Result[List[CodeSnippet], DomainError]:
        """Find code snippets by task ID."""
        try:
            snippet_ids = self._task_index.get(task_id, [])
            snippets = [
                deepcopy(self._storage[sid])
                for sid in snippet_ids
                if sid in self._storage
            ]
            return Success(snippets)
        except Exception as e:
            return Failure(DomainError(
                code=ErrorCode.INTERNAL_ERROR,
                message=f"Failed to find snippets by task: {str(e)}"
            ))
    
    async def find_by_language(self, language: str) -> Result[List[CodeSnippet], DomainError]:
        """Find code snippets by language."""
        try:
            snippets = [
                deepcopy(snippet) for snippet in self._storage.values()
                if snippet.language.value == language
            ]
            return Success(snippets)
        except Exception as e:
            return Failure(DomainError(
                code=ErrorCode.INTERNAL_ERROR,
                message=f"Failed to find snippets by language: {str(e)}"
            ))
    
    async def search_by_content(self, query: str, limit: int = 10) -> Result[List[CodeSnippet], DomainError]:
        """Search code snippets by content."""
        try:
            matches = [
                deepcopy(snippet) for snippet in self._storage.values()
                if query.lower() in snippet.content.lower()
            ]
            return Success(matches[:limit])
        except Exception as e:
            return Failure(DomainError(
                code=ErrorCode.INTERNAL_ERROR,
                message=f"Failed to search snippets: {str(e)}"
            ))


class InMemoryArchitectureRepository(ArchitectureRepository):
    """In-memory implementation of ArchitectureRepository."""
    
    def __init__(self):
        self._storage: Dict[str, ArchitectureDesign] = {}
        self._task_index: Dict[str, str] = {}  # task_id -> architecture_id
        self._next_id: int = 1
    
    def _generate_id(self) -> str:
        """Generate a unique ID."""
        id_str = f"arch_{self._next_id}"
        self._next_id += 1
        return id_str
    
    async def find_by_id(self, id: str) -> Result[Optional[ArchitectureDesign], DomainError]:
        """Find entity by ID."""
        try:
            arch = self._storage.get(id)
            return Success(deepcopy(arch) if arch else None)
        except Exception as e:
            return Failure(DomainError(
                code=ErrorCode.INTERNAL_ERROR,
                message=f"Failed to find architecture: {str(e)}"
            ))
    
    async def find_all(self, limit: int = 100, offset: int = 0) -> Result[List[ArchitectureDesign], DomainError]:
        """Find all entities with pagination."""
        try:
            all_architectures = list(self._storage.values())
            paginated = all_architectures[offset:offset + limit]
            return Success([deepcopy(arch) for arch in paginated])
        except Exception as e:
            return Failure(DomainError(
                code=ErrorCode.INTERNAL_ERROR,
                message=f"Failed to find all architectures: {str(e)}"
            ))
    
    async def save(self, entity: ArchitectureDesign) -> Result[ArchitectureDesign, DomainError]:
        """Save entity (create or update)."""
        try:
            # Generate ID if not exists
            arch_id = self._generate_id()
            self._storage[arch_id] = deepcopy(entity)
            return Success(deepcopy(entity))
        except Exception as e:
            return Failure(DomainError(
                code=ErrorCode.INTERNAL_ERROR,
                message=f"Failed to save architecture: {str(e)}"
            ))
    
    async def delete(self, id: str) -> Result[bool, DomainError]:
        """Delete entity by ID."""
        try:
            if id in self._storage:
                del self._storage[id]
                return Success(True)
            return Success(False)
        except Exception as e:
            return Failure(DomainError(
                code=ErrorCode.INTERNAL_ERROR,
                message=f"Failed to delete architecture: {str(e)}"
            ))
    
    async def exists(self, id: str) -> Result[bool, DomainError]:
        """Check if entity exists."""
        try:
            return Success(id in self._storage)
        except Exception as e:
            return Failure(DomainError(
                code=ErrorCode.INTERNAL_ERROR,
                message=f"Failed to check architecture existence: {str(e)}"
            ))
    
    async def find_by_style(self, style: str) -> Result[List[ArchitectureDesign], DomainError]:
        """Find architectures by style."""
        try:
            architectures = [
                deepcopy(arch) for arch in self._storage.values()
                if arch.style.value == style
            ]
            return Success(architectures)
        except Exception as e:
            return Failure(DomainError(
                code=ErrorCode.INTERNAL_ERROR,
                message=f"Failed to find architectures by style: {str(e)}"
            ))
    
    async def find_by_task_id(self, task_id: str) -> Result[Optional[ArchitectureDesign], DomainError]:
        """Find architecture by task ID."""
        try:
            arch_id = self._task_index.get(task_id)
            if arch_id and arch_id in self._storage:
                return Success(deepcopy(self._storage[arch_id]))
            return Success(None)
        except Exception as e:
            return Failure(DomainError(
                code=ErrorCode.INTERNAL_ERROR,
                message=f"Failed to find architecture by task: {str(e)}"
            ))


class InMemoryUnitOfWork(UnitOfWork):
    """In-memory implementation of Unit of Work."""
    
    def __init__(
        self,
        task_repo: InMemoryTaskRepository,
        snippet_repo: InMemoryCodeSnippetRepository,
        arch_repo: InMemoryArchitectureRepository
    ):
        super().__init__()
        self._task_repo = task_repo
        self._snippet_repo = snippet_repo
        self._arch_repo = arch_repo
    
    @property
    def tasks(self) -> TaskRepository:
        """Get task repository."""
        return self._task_repo
    
    @property
    def code_snippets(self) -> CodeSnippetRepository:
        """Get code snippet repository."""
        return self._snippet_repo
    
    @property
    def architectures(self) -> ArchitectureRepository:
        """Get architecture repository."""
        return self._arch_repo
    
    async def _commit_changes(self) -> None:
        """Implementation-specific commit logic."""
        # In-memory storage is already updated, nothing to do
        pass
    
    async def _rollback_changes(self) -> None:
        """Implementation-specific rollback logic."""
        # For in-memory, we can't really rollback easily
        # In a real implementation, this would revert changes
        pass


class InMemoryUnitOfWorkFactory(UnitOfWorkFactory):
    """Factory for creating in-memory Unit of Work instances."""
    
    def __init__(self):
        # Shared repositories for all UoW instances
        self._task_repo = InMemoryTaskRepository()
        self._snippet_repo = InMemoryCodeSnippetRepository()
        self._arch_repo = InMemoryArchitectureRepository()
    
    def create(self) -> UnitOfWork:
        """Create a new Unit of Work instance."""
        return InMemoryUnitOfWork(
            self._task_repo,
            self._snippet_repo,
            self._arch_repo
        )
