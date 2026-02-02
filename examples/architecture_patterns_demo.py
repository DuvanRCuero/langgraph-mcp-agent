"""
Example demonstrating the advanced clean architecture patterns.
This script shows how to use Result, Events, Repositories, and UnitOfWork patterns.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.src.core.domain.result import Success, Failure, ResultHelper
from agent.src.core.domain.errors import DomainError, ValidationError, ErrorCode
from agent.src.core.domain.events import (
    TaskCreatedEvent, TaskStatusChangedEvent, get_event_bus, EventPriority
)
from agent.src.core.domain.entities import (
    ProgrammingTask, ProgrammingLanguage, TaskStatus, CodeSnippet
)
from agent.src.core.infrastructure.persistence.in_memory import (
    InMemoryUnitOfWorkFactory
)


async def event_logger(event):
    """Simple event handler that logs events."""
    print(f"[EVENT] {event.event_type}: {event.to_dict()['payload']}")


async def demo_result_pattern():
    """Demonstrate Result pattern usage."""
    print("\n=== Result Pattern Demo ===")
    
    # Success case
    success_result = Success("Hello, World!")
    print(f"Success: {success_result.value}, is_success: {success_result.is_success()}")
    
    # Failure case
    failure_result = Failure(DomainError(
        code=ErrorCode.VALIDATION_ERROR,
        message="Invalid input"
    ))
    print(f"Failure: {failure_result.error.message}, is_failure: {failure_result.is_failure()}")
    
    # Using ResultHelper
    mapped = ResultHelper.map(success_result, lambda x: x.upper())
    print(f"Mapped success: {mapped.value if isinstance(mapped, Success) else 'N/A'}")
    
    # Get or else
    value1 = ResultHelper.get_or_else(success_result, "default")
    value2 = ResultHelper.get_or_else(failure_result, "default")
    print(f"Get or else: success={value1}, failure={value2}")


async def demo_events():
    """Demonstrate domain events and EventBus."""
    print("\n=== Domain Events Demo ===")
    
    event_bus = get_event_bus()
    
    # Subscribe to specific event type
    event_bus.subscribe(TaskCreatedEvent, event_logger, priority=EventPriority.HIGH)
    event_bus.subscribe(TaskStatusChangedEvent, event_logger)
    
    # Publish events
    await event_bus.publish(TaskCreatedEvent(
        task_id="task-123",
        title="Build REST API",
        language="python",
        requirements=["FastAPI", "PostgreSQL", "Docker"]
    ))
    
    await event_bus.publish(TaskStatusChangedEvent(
        task_id="task-123",
        old_status="pending",
        new_status="analyzing"
    ))
    
    # Check event history
    history = event_bus.get_event_history(limit=10)
    print(f"\nEvent history count: {len(history)}")
    
    # Clear for next demo
    event_bus.clear_history()


async def demo_repositories():
    """Demonstrate Repository pattern."""
    print("\n=== Repository Pattern Demo ===")
    
    # Create UnitOfWork factory
    uow_factory = InMemoryUnitOfWorkFactory()
    uow = uow_factory.create()
    
    # Create a task
    task = ProgrammingTask(
        title="Build Authentication Service",
        description="Implement JWT-based authentication",
        requirements=["JWT tokens", "Password hashing", "OAuth2"],
        language=ProgrammingLanguage.PYTHON,
        framework="FastAPI"
    )
    
    # Save task using repository
    save_result = await uow.tasks.save(task)
    if isinstance(save_result, Success):
        print(f"Task saved: {save_result.value.task_id}")
    
    # Find by ID
    find_result = await uow.tasks.find_by_id(task.task_id)
    if isinstance(find_result, Success) and find_result.value:
        print(f"Task found: {find_result.value.title}")
    
    # Find by language
    lang_result = await uow.tasks.find_by_language("python")
    if isinstance(lang_result, Success):
        print(f"Tasks with Python: {len(lang_result.value)}")
    
    # Update status
    update_result = await uow.tasks.update_status(task.task_id, TaskStatus.ANALYZING.value)
    if isinstance(update_result, Success):
        print(f"Task status updated: {update_result.value.status}")


async def demo_unit_of_work():
    """Demonstrate Unit of Work pattern with events."""
    print("\n=== Unit of Work Pattern Demo ===")
    
    event_bus = get_event_bus()
    event_bus.subscribe_all(event_logger)
    
    # Create UnitOfWork factory
    uow_factory = InMemoryUnitOfWorkFactory()
    
    async with uow_factory.create() as uow:
        # Create entities
        task = ProgrammingTask(
            title="Data Processing Pipeline",
            description="Build ETL pipeline",
            requirements=["Apache Spark", "Kafka", "S3"],
            language=ProgrammingLanguage.PYTHON
        )
        
        code_snippet = CodeSnippet(
            language=ProgrammingLanguage.PYTHON,
            content="def process_data(df):\n    return df.filter(df.value > 0)",
            framework="PySpark"
        )
        
        # Save entities
        await uow.tasks.save(task)
        await uow.code_snippets.save(code_snippet)
        
        # Register events
        uow.register_event(TaskCreatedEvent(
            task_id=task.task_id,
            title=task.title,
            language=task.language.value,
            requirements=task.requirements
        ))
        
        # Commit - this will save changes and publish events
        await uow.commit()
        print("Unit of work committed successfully")
    
    print(f"Event history after UoW: {len(event_bus.get_event_history())}")


async def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("Advanced Clean Architecture Patterns Demo")
    print("=" * 60)
    
    try:
        await demo_result_pattern()
        await demo_events()
        await demo_repositories()
        await demo_unit_of_work()
        
        print("\n" + "=" * 60)
        print("All demonstrations completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
