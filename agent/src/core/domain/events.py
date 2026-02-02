"""
Domain events for cross-cutting concerns and decoupled communication.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Callable, Awaitable, Type, Optional
from enum import Enum
import uuid
import logging

logger = logging.getLogger(__name__)


class EventPriority(int, Enum):
    """Event handling priority."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class DomainEvent(ABC):
    """Base class for all domain events."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    occurred_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    @abstractmethod
    def event_type(self) -> str:
        """Return the event type name."""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "occurred_at": self.occurred_at.isoformat(),
            "metadata": self.metadata,
            "payload": self._payload_to_dict()
        }
    
    def _payload_to_dict(self) -> Dict[str, Any]:
        """Convert event-specific payload to dictionary."""
        return {}


# Concrete Domain Events

@dataclass
class TaskCreatedEvent(DomainEvent):
    """Event raised when a task is created."""
    task_id: str = ""
    title: str = ""
    language: str = ""
    requirements: List[str] = field(default_factory=list)
    
    @property
    def event_type(self) -> str:
        return "task.created"
    
    def _payload_to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "language": self.language,
            "requirements": self.requirements
        }


@dataclass
class TaskStatusChangedEvent(DomainEvent):
    """Event raised when task status changes."""
    task_id: str = ""
    old_status: str = ""
    new_status: str = ""
    
    @property
    def event_type(self) -> str:
        return "task.status_changed"
    
    def _payload_to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "old_status": self.old_status,
            "new_status": self.new_status
        }


@dataclass
class CodeGeneratedEvent(DomainEvent):
    """Event raised when code is generated."""
    task_id: str = ""
    component_name: str = ""
    language: str = ""
    lines_of_code: int = 0
    
    @property
    def event_type(self) -> str:
        return "code.generated"
    
    def _payload_to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "component_name": self.component_name,
            "language": self.language,
            "lines_of_code": self.lines_of_code
        }


@dataclass
class QualityAssessedEvent(DomainEvent):
    """Event raised when quality is assessed."""
    task_id: str = ""
    quality_level: str = ""
    overall_score: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    
    @property
    def event_type(self) -> str:
        return "quality.assessed"
    
    def _payload_to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "quality_level": self.quality_level,
            "overall_score": self.overall_score,
            "metrics": self.metrics
        }


@dataclass
class TaskCompletedEvent(DomainEvent):
    """Event raised when a task is completed."""
    task_id: str = ""
    execution_time_seconds: float = 0.0
    quality_level: str = ""
    total_lines_of_code: int = 0
    
    @property
    def event_type(self) -> str:
        return "task.completed"
    
    def _payload_to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "execution_time_seconds": self.execution_time_seconds,
            "quality_level": self.quality_level,
            "total_lines_of_code": self.total_lines_of_code
        }


@dataclass
class TaskFailedEvent(DomainEvent):
    """Event raised when a task fails."""
    task_id: str = ""
    error_message: str = ""
    error_code: str = ""
    phase: str = ""
    
    @property
    def event_type(self) -> str:
        return "task.failed"
    
    def _payload_to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "error_message": self.error_message,
            "error_code": self.error_code,
            "phase": self.phase
        }


# Event Handler Types
EventHandler = Callable[[DomainEvent], Awaitable[None]]


@dataclass
class EventSubscription:
    """Represents an event subscription."""
    handler: EventHandler
    priority: EventPriority = EventPriority.NORMAL
    filter_func: Optional[Callable[[DomainEvent], bool]] = None


class EventBus:
    """
    Central event bus for publishing and subscribing to domain events.
    Implements the publish-subscribe pattern.
    """
    
    _instance: Optional['EventBus'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._subscribers: Dict[str, List[EventSubscription]] = {}
        self._global_subscribers: List[EventSubscription] = []
        self._event_history: List[DomainEvent] = []
        self._max_history_size: int = 1000
        self._initialized = True
    
    def subscribe(
        self,
        event_type: Type[DomainEvent],
        handler: EventHandler,
        priority: EventPriority = EventPriority.NORMAL,
        filter_func: Optional[Callable[[DomainEvent], bool]] = None
    ) -> None:
        """Subscribe to a specific event type."""
        type_name = event_type.__name__
        if type_name not in self._subscribers:
            self._subscribers[type_name] = []
        
        subscription = EventSubscription(
            handler=handler,
            priority=priority,
            filter_func=filter_func
        )
        self._subscribers[type_name].append(subscription)
        
        # Sort by priority (higher first)
        self._subscribers[type_name].sort(key=lambda s: s.priority.value, reverse=True)
        
        logger.debug(f"Subscribed to {type_name} with priority {priority}")
    
    def subscribe_all(
        self,
        handler: EventHandler,
        priority: EventPriority = EventPriority.NORMAL,
        filter_func: Optional[Callable[[DomainEvent], bool]] = None
    ) -> None:
        """Subscribe to all events."""
        subscription = EventSubscription(
            handler=handler,
            priority=priority,
            filter_func=filter_func
        )
        self._global_subscribers.append(subscription)
        self._global_subscribers.sort(key=lambda s: s.priority.value, reverse=True)
    
    def unsubscribe(self, event_type: Type[DomainEvent], handler: EventHandler) -> None:
        """Unsubscribe from an event type."""
        type_name = event_type.__name__
        if type_name in self._subscribers:
            self._subscribers[type_name] = [
                s for s in self._subscribers[type_name] if s.handler != handler
            ]
    
    async def publish(self, event: DomainEvent) -> None:
        """Publish an event to all subscribers."""
        type_name = type(event).__name__
        
        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history_size:
            self._event_history = self._event_history[-self._max_history_size:]
        
        logger.info(f"Publishing event: {event.event_type} (ID: {event.event_id})")
        
        # Gather all applicable subscriptions
        subscriptions = []
        
        # Type-specific subscribers
        if type_name in self._subscribers:
            subscriptions.extend(self._subscribers[type_name])
        
        # Global subscribers
        subscriptions.extend(self._global_subscribers)
        
        # Sort all by priority
        subscriptions.sort(key=lambda s: s.priority.value, reverse=True)
        
        # Execute handlers
        for subscription in subscriptions:
            try:
                # Apply filter if present
                if subscription.filter_func and not subscription.filter_func(event):
                    continue
                
                await subscription.handler(event)
            except Exception as e:
                logger.error(f"Error in event handler for {type_name}: {e}", exc_info=True)
    
    async def publish_batch(self, events: List[DomainEvent]) -> None:
        """Publish multiple events."""
        for event in events:
            await self.publish(event)
    
    def get_event_history(
        self,
        event_type: Optional[Type[DomainEvent]] = None,
        limit: int = 100
    ) -> List[DomainEvent]:
        """Get event history, optionally filtered by type."""
        if event_type:
            type_name = event_type.__name__
            filtered = [e for e in self._event_history if type(e).__name__ == type_name]
            return filtered[-limit:]
        return self._event_history[-limit:]
    
    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()
    
    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None


# Convenience function
def get_event_bus() -> EventBus:
    """Get the singleton EventBus instance."""
    return EventBus()
