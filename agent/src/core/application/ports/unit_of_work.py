"""
Unit of Work pattern for transaction management across operations.
"""

from abc import ABC, abstractmethod
from typing import List
from contextlib import asynccontextmanager
import logging

from .repositories import TaskRepository, CodeSnippetRepository, ArchitectureRepository
from ...domain.events import DomainEvent, get_event_bus

logger = logging.getLogger(__name__)


class UnitOfWork(ABC):
    """
    Unit of Work pattern for managing transactions across multiple repositories.
    Ensures atomicity of operations and handles domain event publishing.
    """
    
    def __init__(self):
        self._events: List[DomainEvent] = []
        self._committed: bool = False
        self._rolled_back: bool = False
    
    @property
    @abstractmethod
    def tasks(self) -> TaskRepository:
        """Get task repository."""
        pass
    
    @property
    @abstractmethod
    def code_snippets(self) -> CodeSnippetRepository:
        """Get code snippet repository."""
        pass
    
    @property
    @abstractmethod
    def architectures(self) -> ArchitectureRepository:
        """Get architecture repository."""
        pass
    
    def register_event(self, event: DomainEvent) -> None:
        """Register a domain event to be published on commit."""
        self._events.append(event)
        logger.debug(f"Registered event: {event.event_type}")
    
    def register_events(self, events: List[DomainEvent]) -> None:
        """Register multiple domain events."""
        self._events.extend(events)
    
    async def commit(self) -> None:
        """
        Commit the unit of work.
        Persists all changes and publishes domain events.
        """
        if self._committed:
            raise RuntimeError("Unit of work already committed")
        if self._rolled_back:
            raise RuntimeError("Unit of work was rolled back")
        
        try:
            # Persist changes
            await self._commit_changes()
            self._committed = True
            
            # Publish events after successful commit
            event_bus = get_event_bus()
            for event in self._events:
                await event_bus.publish(event)
            
            logger.info(f"Committed unit of work with {len(self._events)} events")
            
        except Exception as e:
            logger.error(f"Failed to commit unit of work: {e}")
            await self.rollback()
            raise
    
    async def rollback(self) -> None:
        """Rollback the unit of work."""
        if self._committed:
            raise RuntimeError("Cannot rollback committed unit of work")
        
        await self._rollback_changes()
        self._rolled_back = True
        self._events.clear()
        
        logger.info("Rolled back unit of work")
    
    @abstractmethod
    async def _commit_changes(self) -> None:
        """Implementation-specific commit logic."""
        pass
    
    @abstractmethod
    async def _rollback_changes(self) -> None:
        """Implementation-specific rollback logic."""
        pass
    
    async def __aenter__(self) -> 'UnitOfWork':
        """Enter async context."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context, rollback on exception."""
        if exc_type is not None:
            await self.rollback()
        elif not self._committed and not self._rolled_back:
            await self.rollback()


class UnitOfWorkFactory(ABC):
    """Factory for creating Unit of Work instances."""
    
    @abstractmethod
    def create(self) -> UnitOfWork:
        """Create a new Unit of Work instance."""
        pass


@asynccontextmanager
async def transactional(uow_factory: UnitOfWorkFactory):
    """
    Context manager for transactional operations.
    
    Usage:
        async with transactional(uow_factory) as uow:
            task = await uow.tasks.find_by_id("123")
            # ... modify task
            await uow.tasks.save(task)
            uow.register_event(TaskUpdatedEvent(...))
            await uow.commit()
    """
    uow = uow_factory.create()
    try:
        yield uow
        if not uow._committed and not uow._rolled_back:
            await uow.commit()
    except Exception:
        if not uow._rolled_back:
            await uow.rollback()
        raise
