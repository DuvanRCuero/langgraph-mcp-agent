"""
Base tool framework for MCP server.
Provides abstract base classes, decorators, and utilities for creating tools.
Follows SOLID principles and elite engineering practices.
"""

import asyncio
import inspect
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Awaitable, Type, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json
from functools import wraps

from pydantic import BaseModel, ValidationError, validator, root_validator
from ..protocols.mcp_protocol import ToolSchema

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class ToolCategory(str, Enum):
    """Tool categories for organization."""
    DOCUMENTATION = "documentation"
    CODE_ANALYSIS = "code_analysis"
    BEST_PRACTICES = "best_practices"
    FRAMEWORK_SPECIFIC = "framework_specific"
    UTILITY = "utility"
    VALIDATION = "validation"


class ToolPermission(str, Enum):
    """Tool permission levels."""
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    ADMIN = "admin"


@dataclass
class ToolMetadata:
    """Metadata for tools."""
    version: str = "1.0.0"
    author: str = "MCP Server"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    category: ToolCategory = ToolCategory.DOCUMENTATION
    permission: ToolPermission = ToolPermission.PUBLIC
    timeout_seconds: int = 30
    max_concurrent: int = 10
    requires_approval: bool = False


@dataclass
class ExecutionContext:
    """Context for tool execution."""
    connection_id: str
    client_info: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: str = field(default_factory=lambda: str(hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]))
    start_time: datetime = field(default_factory=datetime.now)

    @property
    def execution_time(self) -> float:
        """Get execution time in seconds."""
        return (datetime.now() - self.start_time).total_seconds()


class BaseTool(ABC):
    """
    Abstract base class for all MCP tools.
    Implements common functionality, validation, and error handling.
    """

    def __init__(self):
        self.metadata = self.get_metadata()
        self.schema = self.get_schema()
        self.metrics = {
            "calls": 0,
            "errors": 0,
            "total_execution_time": 0.0,
            "last_called": None,
            "concurrent_calls": 0
        }
        self._semaphore = asyncio.Semaphore(self.metadata.max_concurrent)

        # Cache for tool results
        self._cache: Dict[str, tuple[Any, datetime]] = {}
        self._cache_ttl = timedelta(minutes=5)

        logger.info(f"Initialized tool: {self.schema.name}")

    @abstractmethod
    def get_metadata(self) -> ToolMetadata:
        """Get tool metadata."""
        pass

    @abstractmethod
    def get_schema(self) -> ToolSchema:
        """Get tool schema for MCP."""
        pass

    @abstractmethod
    async def _execute_impl(self, arguments: Dict[str, Any], context: ExecutionContext) -> Any:
        """Tool implementation method to be overridden."""
        pass

    async def execute(self, arguments: Dict[str, Any], connection_id: str) -> Any:
        """
        Execute tool with proper error handling, validation, and metrics.

        Args:
            arguments: Tool arguments
            connection_id: Client connection ID

        Returns:
            Tool execution result

        Raises:
            ValidationError: If arguments are invalid
            ToolExecutionError: If execution fails
            TimeoutError: If execution times out
        """
        start_time = datetime.now()
        context = ExecutionContext(connection_id=connection_id)

        # Check cache
        cache_key = self._generate_cache_key(arguments)
        if cache_key in self._cache:
            cached_result, cached_time = self._cache[cache_key]
            if datetime.now() - cached_time < self._cache_ttl:
                logger.debug(f"Cache hit for tool {self.schema.name}")
                return cached_result

        # Acquire semaphore for concurrent execution limit
        async with self._semaphore:
            try:
                # Validate arguments
                await self.validate_arguments(arguments)

                # Update metrics
                self.metrics["calls"] += 1
                self.metrics["concurrent_calls"] += 1

                # Execute with timeout
                try:
                    result = await asyncio.wait_for(
                        self._execute_impl(arguments, context),
                        timeout=self.metadata.timeout_seconds
                    )
                except asyncio.TimeoutError:
                    raise TimeoutError(
                        f"Tool {self.schema.name} timed out after {self.metadata.timeout_seconds} seconds"
                    )

                # Update metrics
                execution_time = (datetime.now() - start_time).total_seconds()
                self.metrics["total_execution_time"] += execution_time
                self.metrics["last_called"] = datetime.now()
                self.metrics["concurrent_calls"] -= 1

                # Cache result
                self._cache[cache_key] = (result, datetime.now())

                # Log execution
                logger.info(
                    f"Tool {self.schema.name} executed successfully in {execution_time:.2f}s"
                )

                return result

            except ValidationError as e:
                logger.warning(f"Validation failed for tool {self.schema.name}: {e}")
                self.metrics["errors"] += 1
                raise
            except Exception as e:
                logger.error(f"Tool {self.schema.name} execution failed: {e}")
                self.metrics["errors"] += 1
                raise ToolExecutionError(
                    f"Tool {self.schema.name} failed: {str(e)}"
                )
            finally:
                if self.metrics["concurrent_calls"] > 0:
                    self.metrics["concurrent_calls"] -= 1

    async def validate_arguments(self, arguments: Dict[str, Any]):
        """
        Validate tool arguments.
        Can be overridden for custom validation.
        """
        # Basic validation against schema
        if not isinstance(arguments, dict):
            raise ValidationError("Arguments must be a dictionary")

        schema = self.schema.inputSchema
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # Check required fields
        for field_name in required:
            if field_name not in arguments:
                raise ValidationError(f"Missing required field: {field_name}")

        # Type checking (simplified)
        for field_name, value in arguments.items():
            if field_name in properties:
                prop_schema = properties[field_name]
                expected_type = prop_schema.get("type")

                if expected_type == "string" and not isinstance(value, str):
                    raise ValidationError(f"Field {field_name} must be a string")
                elif expected_type == "number" and not isinstance(value, (int, float)):
                    raise ValidationError(f"Field {field_name} must be a number")
                elif expected_type == "boolean" and not isinstance(value, bool):
                    raise ValidationError(f"Field {field_name} must be a boolean")
                elif expected_type == "array" and not isinstance(value, list):
                    raise ValidationError(f"Field {field_name} must be an array")
                elif expected_type == "object" and not isinstance(value, dict):
                    raise ValidationError(f"Field {field_name} must be an object")

    def _generate_cache_key(self, arguments: Dict[str, Any]) -> str:
        """Generate cache key from arguments."""
        sorted_args = json.dumps(arguments, sort_keys=True)
        return hashlib.md5(sorted_args.encode()).hexdigest()

    def clear_cache(self):
        """Clear tool cache."""
        self._cache.clear()

    def get_health_status(self) -> Dict[str, Any]:
        """Get tool health status."""
        return {
            "name": self.schema.name,
            "status": "healthy" if self.metrics["errors"] < 10 else "degraded",
            "metrics": self.metrics,
            "cache_size": len(self._cache),
            "semaphore_value": self._semaphore._value,
            "last_updated": datetime.now().isoformat()
        }


class ToolExecutionError(Exception):
    """Custom exception for tool execution errors."""
    pass


def tool_decorator(
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        category: ToolCategory = ToolCategory.UTILITY,
        rate_limit: Optional[int] = None,
        timeout: int = 30
):
    """
    Decorator for creating tool functions.
    Converts async functions into BaseTool instances.
    """

    def decorator(func: Callable[..., Awaitable[Any]]):

        class DecoratedTool(BaseTool):

            def get_metadata(self) -> ToolMetadata:
                return ToolMetadata(
                    version="1.0.0",
                    author="Decorated Tool",
                    category=category,
                    timeout_seconds=timeout
                )

            def get_schema(self) -> ToolSchema:
                return ToolSchema(
                    name=name,
                    description=description,
                    inputSchema=input_schema,
                    rate_limit=rate_limit
                )

            async def _execute_impl(self, arguments: Dict[str, Any], context: ExecutionContext) -> Any:
                # Call the decorated function
                if inspect.iscoroutinefunction(func):
                    return await func(arguments, context)
                else:
                    return func(arguments, context)

        return DecoratedTool

    return decorator


class ToolFactory:
    """
    Factory for creating and managing tools.
    Implements singleton pattern for tool instances.
    """

    _instance = None
    _tools: Dict[str, BaseTool] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register_tool(cls, tool_class: Type[BaseTool]) -> str:
        """Register a tool class."""
        tool_instance = tool_class()
        tool_name = tool_instance.schema.name

        if tool_name in cls._tools:
            logger.warning(f"Tool {tool_name} already registered, overwriting")

        cls._tools[tool_name] = tool_instance
        logger.info(f"Registered tool: {tool_name}")

        return tool_name

    @classmethod
    def get_tool(cls, name: str) -> Optional[BaseTool]:
        """Get tool by name."""
        return cls._tools.get(name)

    @classmethod
    def get_all_tools(cls) -> Dict[str, BaseTool]:
        """Get all registered tools."""
        return cls._tools.copy()

    @classmethod
    def get_tool_metrics(cls) -> Dict[str, Any]:
        """Get metrics for all tools."""
        return {
            name: tool.get_health_status()
            for name, tool in cls._tools.items()
        }

    @classmethod
    async def health_check(cls) -> Dict[str, Any]:
        """Perform health check on all tools."""
        results = {}

        for name, tool in cls._tools.items():
            try:
                # Quick validation check
                await tool.validate_arguments({})
                results[name] = {
                    "status": "healthy",
                    "message": "Tool is operational"
                }
            except Exception as e:
                results[name] = {
                    "status": "unhealthy",
                    "message": str(e)
                }

        return results