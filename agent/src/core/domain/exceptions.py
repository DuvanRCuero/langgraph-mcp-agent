"""
Domain-specific exceptions following clean architecture principles.
"""

class DomainException(Exception):
    """Base domain exception."""
    pass


class InvalidCodeException(DomainException):
    """Raised when code validation fails."""
    pass


class InvalidArchitectureException(DomainException):
    """Raised when architecture design is invalid."""
    pass


class TaskTransitionException(DomainException):
    """Raised when invalid task status transition is attempted."""
    pass


class DocumentationNotFoundException(DomainException):
    """Raised when required documentation is not found."""
    pass


class QualityThresholdException(DomainException):
    """Raised when quality thresholds are not met."""
    pass


class ConfigurationException(DomainException):
    """Raised when configuration is invalid."""
    pass


class ToolExecutionException(DomainException):
    """Raised when tool execution fails."""
    pass


class LLMException(DomainException):
    """Raised when LLM operations fail."""
    pass


class VectorStoreException(DomainException):
    """Raised when vector store operations fail."""
    pass


class FileSystemException(DomainException):
    """Raised when file system operations fail."""
    pass