"""
Domain-specific error types with structured information.
"""

from dataclasses import dataclass
from typing import Optional, List
from enum import Enum


class ErrorCode(str, Enum):
    """Standard error codes."""
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    CONFLICT = "CONFLICT"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"
    TIMEOUT = "TIMEOUT"
    RATE_LIMIT = "RATE_LIMIT"


@dataclass(frozen=True)
class DomainError:
    """Base domain error with structured information."""
    code: ErrorCode
    message: str
    details: Optional[dict] = None
    causes: Optional[List['DomainError']] = None
    
    def to_dict(self) -> dict:
        return {
            "code": self.code.value,
            "message": self.message,
            "details": self.details,
            "causes": [c.to_dict() for c in self.causes] if self.causes else None
        }


@dataclass(frozen=True)
class ValidationError(DomainError):
    """Validation-specific error."""
    field: Optional[str] = None
    
    def __post_init__(self):
        object.__setattr__(self, 'code', ErrorCode.VALIDATION_ERROR)


@dataclass(frozen=True)
class NotFoundError(DomainError):
    """Resource not found error."""
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    
    def __post_init__(self):
        object.__setattr__(self, 'code', ErrorCode.NOT_FOUND)
