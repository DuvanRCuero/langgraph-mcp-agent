"""
MCP client adapter implementing the MCPClientPort interface.
Communicates with our MCP server via WebSocket.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
import aiohttp
from aiohttp import ClientSession, ClientWebSocketResponse

from ....core.domain.entities import DocumentationReference
from ....core.application.ports.mcp_client import (
    MCPClientPort, MCPToolCall, MCPToolResult, AsyncMCPStream
)

logger = logging.getLogger(__name__)


@dataclass
class MCPConnectionConfig:
    """Configuration for MCP connection."""
    server_url: str = "ws://localhost:8000/mcp"
    reconnect_attempts: int = 3
    reconnect_delay: float = 1.0
    connection_timeout: float = 30.0
    ping_interval: float = 30.0
    ping_timeout: float = 10.0


class MCPClientAdapter(MCPClientPort, AsyncMCPStream):
    """
    Production-grade MCP client adapter.
    Implements WebSocket communication with MCP server.
    """

    def __init__(self, config: Optional[MCPConnectionConfig] = None):
        self.config = config or MCPConnectionConfig()
        self.session: Optional[ClientSession] = None
        self.websocket: Optional[ClientWebSocketResponse] = None
        self.message_id = 0
        self.is_connected_flag = False
        self.connection_metrics_data = {
            "connection_count": 0,
            "total_messages_sent": 0,
            "total_messages_received": 0,
            "last_connected": None,
            "last_disconnected": None,
            "total_connection_time": 0.0
        }
        self._connection_start_time: Optional[float] = None

        # Message queue for async communication
        self._response_futures: Dict[str, asyncio.Future] = {}
        self._message_queue: asyncio.Queue = asyncio.Queue()

        # Background tasks
        self._receive_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None

        logger.info(f"MCP Client initialized for server: {self.config.server_url}")

    async def connect(self) -> bool:
        """Connect to MCP server with retry logic."""

        if self.is_connected_flag:
            logger.warning("Already connected to MCP server")
            return True

        for attempt in range(self.config.reconnect_attempts):
            try:
                logger.info(f"Connecting to MCP server (attempt {attempt + 1})...")

                # Create session if needed
                if self.session is None or self.session.closed:
                    self.session = ClientSession()

                # Connect WebSocket
                self.websocket = await self.session.ws_connect(
                    self.config.server_url,
                    timeout=self.config.connection_timeout
                )

                # Initialize connection
                await self._initialize_connection()

                # Start background tasks
                self._receive_task = asyncio.create_task(self._receive_messages())
                self._ping_task = asyncio.create_task(self._send_pings())

                self.is_connected_flag = True
                self._connection_start_time = asyncio.get_event_loop().time()
                self.connection_metrics_data["connection_count"] += 1
                self.connection_metrics_data["last_connected"] = datetime.now().isoformat()

                logger.info("Successfully connected to MCP server")
                return True

            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")

                if attempt < self.config.reconnect_attempts - 1:
                    await asyncio.sleep(self.config.reconnect_delay * (2 ** attempt))
                else:
                    logger.error(f"Failed to connect after {self.config.reconnect_attempts} attempts")
                    return False

    async def disconnect(self):
        """Disconnect from MCP server gracefully."""

        if not self.is_connected_flag:
            return

        logger.info("Disconnecting from MCP server...")

        # Cancel background tasks
        if self._ping_task:
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        # Close WebSocket
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

        # Close session
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

        # Update metrics
        self.is_connected_flag = False
        self.connection_metrics_data["last_disconnected"] = datetime.now().isoformat()

        if self._connection_start_time:
            connection_duration = asyncio.get_event_loop().time() - self._connection_start_time
            self.connection_metrics_data["total_connection_time"] += connection_duration

        # Clear response futures
        for future in self._response_futures.values():
            if not future.done():
                future.set_exception(ConnectionError("Disconnected from MCP server"))
        self._response_futures.clear()

        logger.info("Disconnected from MCP server")

    async def call_tool(self, tool_call: MCPToolCall) -> MCPToolResult:
        """Call a tool on MCP server."""

        if not self.is_connected_flag:
            raise ConnectionError("Not connected to MCP server")

        start_time = asyncio.get_event_loop().time()

        try:
            # Prepare message
            self.message_id += 1
            message_id = str(self.message_id)

            message = {
                "jsonrpc": "2.0",
                "id": message_id,
                "method": "tools/call",
                "params": {
                    "tool": tool_call.tool_name,
                    "arguments": tool_call.arguments
                }
            }

            # Create future for response
            future = asyncio.Future()
            self._response_futures[message_id] = future

            # Send message
            await self.websocket.send_json(message)
            self.connection_metrics_data["total_messages_sent"] += 1

            # Wait for response with timeout
            try:
                response = await asyncio.wait_for(
                    future,
                    timeout=tool_call.timeout_seconds
                )
            except asyncio.TimeoutError:
                del self._response_futures[message_id]
                raise TimeoutError(f"Tool call timed out after {tool_call.timeout_seconds}s")

            # Process response
            execution_time = asyncio.get_event_loop().time() - start_time

            if "error" in response:
                return MCPToolResult(
                    success=False,
                    error=response["error"].get("message", "Unknown error"),
                    execution_time=execution_time
                )

            return MCPToolResult(
                success=True,
                result=response.get("result"),
                execution_time=execution_time
            )

        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            return MCPToolResult(
                success=False,
                error=str(e),
                execution_time=asyncio.get_event_loop().time() - start_time
            )

    async def stream_tool_call(
            self,
            tool_call: MCPToolCall
    ) -> AsyncGenerator[MCPToolResult, None]:
        """Stream tool call results."""

        if not self.is_connected_flag:
            raise ConnectionError("Not connected to MCP server")

        # For now, implement as non-streaming (MCP protocol needs extension for streaming)
        # In production, implement proper streaming support
        result = await self.call_tool(tool_call)
        yield result

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools on MCP server."""

        if not self.is_connected_flag:
            raise ConnectionError("Not connected to MCP server")

        try:
            self.message_id += 1
            message_id = str(self.message_id)

            message = {
                "jsonrpc": "2.0",
                "id": message_id,
                "method": "tools/list"
            }

            future = asyncio.Future()
            self._response_futures[message_id] = future

            await self.websocket.send_json(message)
            self.connection_metrics_data["total_messages_sent"] += 1

            response = await future

            if "error" in response:
                logger.error(f"Failed to list tools: {response['error']}")
                return []

            tools = response.get("result", {}).get("tools", [])
            logger.info(f"Retrieved {len(tools)} tools from MCP server")
            return tools

        except Exception as e:
            logger.error(f"Failed to list tools: {e}")
            return []

    async def search_documentation(
            self,
            query: str,
            language: Optional[str] = None,
            framework: Optional[str] = None,
            limit: int = 10
    ) -> List[DocumentationReference]:
        """Search documentation via MCP server."""

        arguments = {
            "query": query,
            "max_results": limit
        }

        if language:
            arguments["language"] = language
        if framework:
            arguments["framework"] = framework

        tool_call = MCPToolCall(
            tool_name="documentation_search",
            arguments=arguments,
            timeout_seconds=30
        )

        result = await self.call_tool(tool_call)

        if not result.success:
            logger.warning(f"Documentation search failed: {result.error}")
            return []

        # Parse result into DocumentationReference objects
        docs = []
        for item in result.result.get("results", []):
            try:
                doc = DocumentationReference(
                    title=item.get("title", "Untitled"),
                    url=item.get("url", ""),
                    content=item.get("content", ""),
                    source=item.get("source", "unknown"),
                    relevance_score=item.get("relevance_score", 0.0),
                    last_updated=datetime.fromisoformat(item.get("last_updated", datetime.now().isoformat())),
                    doc_type=item.get("doc_type", "api_reference"),
                    tags=item.get("tags", [])
                )
                docs.append(doc)
            except Exception as e:
                logger.warning(f"Failed to parse documentation result: {e}")
                continue

        logger.info(f"Found {len(docs)} documentation results for query: {query}")
        return docs

    async def get_latest_best_practices(
            self,
            language: str,
            framework: Optional[str] = None,
            topic: Optional[str] = None
    ) -> List[DocumentationReference]:
        """Get latest best practices for language/framework."""

        query = f"{language} best practices"
        if framework:
            query = f"{framework} {query}"
        if topic:
            query = f"{query} {topic}"

        return await self.search_documentation(
            query=query,
            language=language,
            framework=framework,
            limit=5
        )

    async def analyze_code_quality(
            self,
            code: str,
            language: str,
            framework: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze code quality using MCP tools."""

        arguments = {
            "code": code,
            "language": language,
            "level": "elite"
        }

        if framework:
            arguments["framework"] = framework

        tool_call = MCPToolCall(
            tool_name="code_analysis",
            arguments=arguments,
            timeout_seconds=45
        )

        result = await self.call_tool(tool_call)

        if not result.success:
            logger.warning(f"Code analysis failed: {result.error}")
            return {
                "success": False,
                "error": result.error,
                "complexity_score": 0.0,
                "security_score": 0.0,
                "issues_count": 0
            }

        return {
            "success": True,
            **result.result
        }

    async def get_framework_specific_docs(
            self,
            framework: str,
            version: Optional[str] = None
    ) -> List[DocumentationReference]:
        """Get framework-specific documentation."""

        query = f"{framework} documentation"
        if version:
            query = f"{query} version {version}"

        arguments = {
            "query": query,
            "framework": framework,
            "max_results": 10
        }

        if version:
            arguments["version"] = version

        tool_call = MCPToolCall(
            tool_name="framework_specific",
            arguments=arguments,
            timeout_seconds=30
        )

        result = await self.call_tool(tool_call)

        if not result.success:
            logger.warning(f"Framework docs retrieval failed: {result.error}")
            return []

        # Parse results
        docs = []
        for item in result.result.get("results", []):
            try:
                doc = DocumentationReference(
                    title=item.get("title", "Untitled"),
                    url=item.get("url", ""),
                    content=item.get("content", ""),
                    source=item.get("source", framework),
                    relevance_score=item.get("relevance_score", 0.0),
                    last_updated=datetime.fromisoformat(item.get("last_updated", datetime.now().isoformat())),
                    doc_type=item.get("doc_type", "api_reference"),
                    tags=item.get("tags", [])
                )
                docs.append(doc)
            except Exception as e:
                logger.warning(f"Failed to parse framework doc: {e}")
                continue

        return docs

    @property
    def is_connected(self) -> bool:
        """Check if connected to MCP server."""
        return self.is_connected_flag and self.websocket and not self.websocket.closed

    @property
    def connection_metrics(self) -> Dict[str, Any]:
        """Get connection metrics."""
        return {
            **self.connection_metrics_data,
            "is_connected": self.is_connected_flag,
            "current_message_id": self.message_id,
            "pending_requests": len(self._response_futures)
        }

    async def _initialize_connection(self):
        """Initialize MCP connection."""

        self.message_id += 1
        message_id = str(self.message_id)

        message = {
            "jsonrpc": "2.0",
            "id": message_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "0.1.0",
                "clientInfo": {
                    "name": "langgraph-programming-agent",
                    "version": "2.0.0"
                },
                "capabilities": {
                    "tools": {},
                    "resources": {
                        "listChanged": True,
                        "subscribe": False
                    }
                }
            }
        }

        future = asyncio.Future()
        self._response_futures[message_id] = future

        await self.websocket.send_json(message)
        self.connection_metrics_data["total_messages_sent"] += 1

        response = await future

        if "error" in response:
            error_msg = response["error"].get("message", "Unknown error")
            raise ConnectionError(f"Failed to initialize MCP connection: {error_msg}")

        server_info = response.get("result", {}).get("serverInfo", {})
        logger.info(
            f"Connected to MCP server: {server_info.get('name', 'Unknown')} v{server_info.get('version', 'Unknown')}")

    async def _receive_messages(self):
        """Receive and process messages from MCP server."""

        try:
            async for msg in self.websocket:
                self.connection_metrics_data["total_messages_received"] += 1

                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._process_message(json.loads(msg.data))
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {self.websocket.exception()}")
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSE:
                    logger.info("WebSocket connection closed by server")
                    break

        except Exception as e:
            logger.error(f"Error in message receiver: {e}")
        finally:
            self.is_connected_flag = False
            # Notify all pending futures
            for future in self._response_futures.values():
                if not future.done():
                    future.set_exception(ConnectionError("Connection lost"))
            self._response_futures.clear()

    async def _process_message(self, message: Dict[str, Any]):
        """Process incoming message."""

        message_id = message.get("id")

        if message_id and message_id in self._response_futures:
            future = self._response_futures.pop(message_id)
            if not future.done():
                future.set_result(message)
        else:
            # Handle notifications or other messages
            method = message.get("method")
            if method == "notification":
                await self._handle_notification(message.get("params", {}))
            else:
                logger.debug(f"Unhandled message: {message}")

    async def _handle_notification(self, params: Dict[str, Any]):
        """Handle MCP notifications."""

        notification_type = params.get("type")
        data = params.get("data", {})

        if notification_type == "tools_changed":
            logger.info("Tools changed on MCP server")
            # In production, update local tool cache
        elif notification_type == "resources_changed":
            logger.info("Resources changed on MCP server")
        else:
            logger.debug(f"Received notification: {notification_type}")

    async def _send_pings(self):
        """Send periodic ping messages to keep connection alive."""

        try:
            while self.is_connected_flag and self.websocket and not self.websocket.closed:
                await asyncio.sleep(self.config.ping_interval)

                self.message_id += 1
                message_id = str(self.message_id)

                message = {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "method": "ping"
                }

                future = asyncio.Future()
                self._response_futures[message_id] = future

                await self.websocket.send_json(message)
                self.connection_metrics_data["total_messages_sent"] += 1

                # Wait for pong with timeout
                try:
                    await asyncio.wait_for(future, timeout=self.config.ping_timeout)
                except asyncio.TimeoutError:
                    logger.warning("Ping timeout, connection may be dead")
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in ping task: {e}")