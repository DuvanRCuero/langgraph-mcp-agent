"""
Port definition for MCP client.
Following clean architecture - this is an interface that infrastructure will implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime

from ...domain.entities import DocumentationReference


@dataclass
class MCPToolCall:
    """MCP tool call request."""
    tool_name: str
    arguments: Dict[str, Any]
    timeout_seconds: int = 30


@dataclass
class MCPToolResult:
    """MCP tool call result."""
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None


class MCPClientPort(ABC):
    """
    Interface for MCP client.
    Abstracts communication with MCP server.
    """

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to MCP server."""
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect from MCP server."""
        pass

    @abstractmethod
    async def call_tool(self, tool_call: MCPToolCall) -> MCPToolResult:
        """Call a tool on MCP server."""
        pass

    @abstractmethod
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools on MCP server."""
        pass

    @abstractmethod
    async def search_documentation(
            self,
            query: str,
            language: Optional[str] = None,
            framework: Optional[str] = None,
            limit: int = 10
    ) -> List[DocumentationReference]:
        """Search documentation via MCP server."""
        pass

    @abstractmethod
    async def get_latest_best_practices(
            self,
            language: str,
            framework: Optional[str] = None,
            topic: Optional[str] = None
    ) -> List[DocumentationReference]:
        """Get latest best practices for language/framework."""
        pass

    @abstractmethod
    async def analyze_code_quality(
            self,
            code: str,
            language: str,
            framework: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze code quality using MCP tools."""
        pass

    @abstractmethod
    async def get_framework_specific_docs(
            self,
            framework: str,
            version: Optional[str] = None
    ) -> List[DocumentationReference]:
        """Get framework-specific documentation."""
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to MCP server."""
        pass

    @property
    @abstractmethod
    def connection_metrics(self) -> Dict[str, Any]:
        """Get connection metrics."""
        pass


class AsyncMCPStream(ABC):
    """Interface for streaming MCP responses."""

    @abstractmethod
    async def stream_tool_call(
            self,
            tool_call: MCPToolCall
    ) -> AsyncGenerator[MCPToolResult, None]:
        """Stream tool call results."""
        pass