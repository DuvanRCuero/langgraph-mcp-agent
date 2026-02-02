"""
Result pattern for explicit error handling.
Makes success/failure explicit in the type system.
"""

from typing import TypeVar, Generic, Union, Callable
from dataclasses import dataclass

T = TypeVar('T')  # Success type
E = TypeVar('E')  # Error type
U = TypeVar('U')  # Mapped type


@dataclass(frozen=True)
class Success(Generic[T]):
    """Represents a successful result."""
    value: T
    
    def is_success(self) -> bool:
        return True
    
    def is_failure(self) -> bool:
        return False


@dataclass(frozen=True)
class Failure(Generic[E]):
    """Represents a failed result."""
    error: E
    
    def is_success(self) -> bool:
        return False
    
    def is_failure(self) -> bool:
        return True


Result = Union[Success[T], Failure[E]]


class ResultHelper:
    """Helper class for Result operations."""
    
    @staticmethod
    def success(value: T) -> Success[T]:
        return Success(value)
    
    @staticmethod
    def failure(error: E) -> Failure[E]:
        return Failure(error)
    
    @staticmethod
    def map(result: Result[T, E], func: Callable[[T], U]) -> Result[U, E]:
        """Map over success value."""
        if isinstance(result, Success):
            return Success(func(result.value))
        return result
    
    @staticmethod
    def flat_map(result: Result[T, E], func: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """Flat map over success value."""
        if isinstance(result, Success):
            return func(result.value)
        return result
    
    @staticmethod
    def get_or_else(result: Result[T, E], default: T) -> T:
        """Get value or default."""
        if isinstance(result, Success):
            return result.value
        return default
    
    @staticmethod
    def get_or_raise(result: Result[T, E]) -> T:
        """Get value or raise exception."""
        if isinstance(result, Success):
            return result.value
        raise ValueError(f"Result is a failure: {result.error}")
