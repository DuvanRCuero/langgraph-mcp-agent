# Advanced Clean Architecture Patterns

This directory contains the implementation of advanced clean architecture patterns that enhance error handling, data access abstraction, cross-cutting concerns, and transaction management.

## Patterns Implemented

### 1. Result Pattern (`result.py`)

The Result pattern makes success/failure explicit in the type system, eliminating the need for exception handling for expected errors.

**Key Components:**
- `Success[T]`: Represents a successful result containing a value of type T
- `Failure[E]`: Represents a failed result containing an error of type E
- `ResultHelper`: Utility class with operations like `map`, `flat_map`, `get_or_else`, `get_or_raise`

**Usage Example:**
```python
from agent.src.core.domain.result import Success, Failure, ResultHelper
from agent.src.core.domain.errors import ValidationError, ErrorCode

def validate_input(value: str) -> Result[str, ValidationError]:
    if not value.strip():
        return Failure(ValidationError(
            code=ErrorCode.VALIDATION_ERROR,
            message="Value cannot be empty",
            field="value"
        ))
    return Success(value.strip())

# Use the result
result = validate_input("  test  ")
if isinstance(result, Success):
    print(f"Valid: {result.value}")
else:
    print(f"Error: {result.error.message}")

# Or use helpers
value = ResultHelper.get_or_else(result, "default")
mapped = ResultHelper.map(result, lambda x: x.upper())
```

### 2. Domain Errors (`errors.py`)

Structured error types that provide rich context about failures.

**Key Components:**
- `ErrorCode`: Enum with standard error codes (VALIDATION_ERROR, NOT_FOUND, INTERNAL_ERROR, etc.)
- `DomainError`: Base error class with code, message, details, and causes
- `ValidationError`: Specialized error for validation failures
- `NotFoundError`: Specialized error for missing resources

**Usage Example:**
```python
from agent.src.core.domain.errors import ValidationError, NotFoundError, ErrorCode

# Validation error with field information
error = ValidationError(
    code=ErrorCode.VALIDATION_ERROR,
    message="Email format is invalid",
    field="email",
    details={"provided": "invalid@", "expected": "user@domain.com"}
)

# Not found error with resource information
error = NotFoundError(
    code=ErrorCode.NOT_FOUND,
    message="Task not found",
    resource_type="ProgrammingTask",
    resource_id="task-123"
)

# Convert to dictionary for API responses
error_dict = error.to_dict()
```

### 3. Domain Events (`events.py`)

Event-driven architecture support with a publish-subscribe pattern for decoupled communication.

**Key Components:**
- `DomainEvent`: Abstract base class for all domain events
- `EventBus`: Central event bus implementing the pub/sub pattern
- Concrete events: `TaskCreatedEvent`, `TaskStatusChangedEvent`, `CodeGeneratedEvent`, etc.
- `EventPriority`: Prioritize event handler execution (LOW, NORMAL, HIGH, CRITICAL)

**Usage Example:**
```python
from agent.src.core.domain.events import (
    TaskCreatedEvent, get_event_bus, EventPriority
)

# Get the event bus singleton
event_bus = get_event_bus()

# Define an event handler
async def on_task_created(event: TaskCreatedEvent):
    print(f"Task created: {event.task_id}")
    # Send notification, update analytics, etc.

# Subscribe to events
event_bus.subscribe(
    TaskCreatedEvent, 
    on_task_created,
    priority=EventPriority.HIGH
)

# Publish events
await event_bus.publish(TaskCreatedEvent(
    task_id="task-123",
    title="Build REST API",
    language="python",
    requirements=["FastAPI", "PostgreSQL"]
))

# View event history
history = event_bus.get_event_history(TaskCreatedEvent, limit=10)
```

**Available Domain Events:**
- `TaskCreatedEvent`: When a new task is created
- `TaskStatusChangedEvent`: When task status changes
- `CodeGeneratedEvent`: When code is generated
- `QualityAssessedEvent`: When quality metrics are assessed
- `TaskCompletedEvent`: When a task completes successfully
- `TaskFailedEvent`: When a task fails

### 4. Repository Pattern (`repositories.py`)

Abstract interfaces for data persistence, separating domain logic from infrastructure concerns.

**Key Components:**
- `Repository[T, ID]`: Generic base repository interface
- `TaskRepository`: Repository for ProgrammingTask entities
- `CodeSnippetRepository`: Repository for CodeSnippet entities
- `ArchitectureRepository`: Repository for ArchitectureDesign entities

**Usage Example:**
```python
from agent.src.core.application.ports.repositories import TaskRepository
from agent.src.core.domain.result import Success, Failure

# Use repository (implementation injected via DI)
async def get_recent_tasks(repo: TaskRepository):
    result = await repo.find_recent(limit=10)
    
    if isinstance(result, Success):
        tasks = result.value
        for task in tasks:
            print(f"Task: {task.title}")
    else:
        print(f"Error: {result.error.message}")

# Find by status
result = await repo.find_by_status("implementing")

# Update status
result = await repo.update_status("task-123", "completed")
```

### 5. Unit of Work Pattern (`unit_of_work.py`)

Transaction management across multiple repository operations with integrated domain event publishing.

**Key Components:**
- `UnitOfWork`: Abstract base class managing transactions
- `UnitOfWorkFactory`: Factory for creating UoW instances
- `transactional`: Context manager for automatic transaction handling

**Usage Example:**
```python
from agent.src.core.application.ports.unit_of_work import UnitOfWorkFactory
from agent.src.core.domain.events import TaskCreatedEvent

# Using Unit of Work
async with uow_factory.create() as uow:
    # Perform operations
    task = ProgrammingTask(...)
    await uow.tasks.save(task)
    
    # Register events to publish on commit
    uow.register_event(TaskCreatedEvent(
        task_id=task.task_id,
        title=task.title,
        language=task.language.value,
        requirements=task.requirements
    ))
    
    # Commit - saves changes and publishes events
    await uow.commit()

# Using transactional context manager
async with transactional(uow_factory) as uow:
    # Operations here
    # Automatic commit on success, rollback on error
    pass
```

### 6. In-Memory Implementations (`infrastructure/persistence/in_memory.py`)

Complete in-memory implementations for testing and development.

**Key Components:**
- `InMemoryTaskRepository`
- `InMemoryCodeSnippetRepository`
- `InMemoryArchitectureRepository`
- `InMemoryUnitOfWork`
- `InMemoryUnitOfWorkFactory`

**Usage Example:**
```python
from agent.src.core.infrastructure.persistence.in_memory import (
    InMemoryUnitOfWorkFactory
)

# Create factory
uow_factory = InMemoryUnitOfWorkFactory()

# Use in tests or development
async with uow_factory.create() as uow:
    task = ProgrammingTask(...)
    result = await uow.tasks.save(task)
    await uow.commit()
```

## Integration Example

See `code_generation.py` for a complete integration example showing how all patterns work together:

```python
class CodeGenerationUseCase:
    def __init__(
        self,
        mcp_client: MCPClientPort,
        llm_gateway: LLMGatewayPort,
        vector_store: VectorStorePort,
        orchestration_service: OrchestrationService,
        uow_factory: Optional[UnitOfWorkFactory] = None
    ):
        # Dependencies injected
        self.uow_factory = uow_factory
    
    async def execute(self, request: CodeGenerationRequest) -> CodeGenerationResponse:
        event_bus = get_event_bus()
        
        # Validate with Result pattern
        validation_result = self._validate_request(request)
        if isinstance(validation_result, Failure):
            raise ValueError(f"Validation failed: {validation_result.error.message}")
        
        # Create task
        task = ProgrammingTask(...)
        
        # Publish domain events
        await event_bus.publish(TaskCreatedEvent(
            task_id=task.task_id,
            title=task.title,
            language=task.language.value,
            requirements=task.requirements
        ))
        
        # ... business logic ...
        
        # On completion
        await event_bus.publish(TaskCompletedEvent(
            task_id=task.task_id,
            execution_time_seconds=execution_time,
            quality_level=task.quality_level.value,
            total_lines_of_code=task.total_lines_of_code
        ))
        
        return response
```

## Benefits

1. **Explicit Error Handling**: Result pattern makes errors visible in type signatures
2. **Decoupled Communication**: Events enable loosely coupled components
3. **Testability**: Repository pattern makes testing easier with in-memory implementations
4. **Transaction Safety**: Unit of Work ensures atomic operations
5. **Type Safety**: Strong typing with generics throughout
6. **Maintainability**: Clear separation of concerns

## Testing

Run the demo to see all patterns in action:

```bash
python3 examples/architecture_patterns_demo.py
```

Or run the validation test:

```bash
cd /home/runner/work/langgraph-mcp-agent/langgraph-mcp-agent
python3 << 'EOF'
import sys
sys.path.insert(0, '.')

from agent.src.core.domain.result import Success, Failure, ResultHelper
from agent.src.core.domain.errors import DomainError, ErrorCode
from agent.src.core.domain.events import TaskCreatedEvent, get_event_bus

# Test Result pattern
success = Success("value")
assert success.is_success()
print("✓ Result pattern works")

# Test Events
event_bus = get_event_bus()
event = TaskCreatedEvent(task_id="123", title="Test", language="python")
print("✓ Events work")

print("All patterns validated!")
EOF
```

## Architecture

```
agent/src/core/
├── domain/
│   ├── entities.py         # Domain entities
│   ├── exceptions.py       # Domain exceptions
│   ├── result.py          # ✨ Result pattern
│   ├── errors.py          # ✨ Structured errors
│   └── events.py          # ✨ Domain events
├── application/
│   ├── ports/
│   │   ├── mcp_client.py
│   │   ├── llm_gateway.py
│   │   ├── repositories.py # ✨ Repository interfaces
│   │   └── unit_of_work.py # ✨ UoW pattern
│   └── use_cases/
│       └── code_generation.py # Updated to use patterns
└── infrastructure/
    └── persistence/
        └── in_memory.py    # ✨ In-memory implementations
```

## Future Enhancements

- Add database-backed repository implementations (PostgreSQL, MongoDB)
- Implement event sourcing for audit trails
- Add saga pattern for distributed transactions
- Create repository decorators for caching
- Add retry policies for repository operations
