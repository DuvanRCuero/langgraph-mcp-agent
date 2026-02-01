"""
Production-grade MCP server for coding documentation.
Implements full MCP protocol with WebSocket support, tool registration,
and comprehensive monitoring.
"""

import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any, List
from datetime import datetime
import traceback

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .protocols.mcp_protocol import MCPProtocolHandler, ResourceSchema
from .tools.base import ToolFactory
from .tools.documentation_tool import DocumentationTool
from .tools.code_standards_tool import CodeStandardsTool
from .tools.best_practices_tool import BestPracticesTool
from .tools.framework_specific import FrameworkSpecificTool
from .utils.monitoring import MetricsCollector, HealthCheck
from .config.settings import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mcp_server.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class MCPServer:
    """
    Main MCP server class.
    Manages WebSocket connections, tool registration, and server lifecycle.
    """

    def __init__(self):
        self.settings = Settings()
        self.app = FastAPI(
            title="Code Documentation MCP Server",
            description="Production-grade MCP server providing latest coding documentation and best practices",
            version="2.0.0",
            docs_url="/docs" if self.settings.debug else None,
            redoc_url="/redoc" if self.settings.debug else None
        )

        # Initialize components
        self.metrics = MetricsCollector()
        self.health = HealthCheck()
        self.protocol_handler: Optional[MCPProtocolHandler] = None

        # Server state
        self.is_running = False
        self.start_time = datetime.now()

        # Setup
        self._setup_middleware()
        self._setup_routes()
        self._setup_signal_handlers()

        logger.info("MCP Server initialized")

    def _setup_middleware(self):
        """Setup middleware."""
        # CORS middleware
        if self.settings.cors_origins:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=self.settings.cors_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        # Add custom middleware for monitoring
        @self.app.middleware("http")
        async def add_process_time_header(request, call_next):
            start_time = datetime.now()
            response = await call_next(request)
            process_time = (datetime.now() - start_time).total_seconds()
            response.headers["X-Process-Time"] = str(process_time)
            self.metrics.record_request(request.method, request.url.path, process_time)
            return response

    def _setup_routes(self):
        """Setup HTTP and WebSocket routes."""

        # Health check
        @self.app.get("/health")
        async def health_check():
            return await self.health.get_status()

        # Metrics endpoint
        @self.app.get("/metrics")
        async def get_metrics():
            return self.metrics.get_metrics()

        # Tool information
        @self.app.get("/tools")
        async def list_tools():
            tools = ToolFactory.get_all_tools()
            return {
                "tools": [
                    {
                        "name": tool.schema.name,
                        "description": tool.schema.description,
                        "category": tool.metadata.category.value,
                        "rate_limit": tool.schema.rate_limit
                    }
                    for tool in tools.values()
                ]
            }

        # WebSocket endpoint for MCP protocol
        @self.app.websocket("/mcp")
        async def websocket_endpoint(websocket: WebSocket):
            await self.handle_websocket_connection(websocket)

        # HTTP endpoint for tool calls (fallback)
        @self.app.post("/tools/{tool_name}")
        async def call_tool_http(tool_name: str, arguments: Dict[str, Any]):
            tool = ToolFactory.get_tool(tool_name)
            if not tool:
                raise HTTPException(status_code=404, detail=f"Tool not found: {tool_name}")

            try:
                result = await tool.execute(arguments, "http")
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""

        def shutdown_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self.is_running = False

        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)

    async def initialize(self):
        """Initialize the server and all components."""
        logger.info("Initializing MCP Server...")

        try:
            # Register tools
            await self._register_tools()

            # Initialize protocol handler
            await self._initialize_protocol_handler()

            # Run health checks
            await self.health.run_checks()

            logger.info("MCP Server initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize server: {e}")
            raise

    async def _register_tools(self):
        """Register all available tools."""
        logger.info("Registering tools...")

        tools = [
            DocumentationTool,
            CodeStandardsTool,
            BestPracticesTool,
            FrameworkSpecificTool
        ]

        for tool_class in tools:
            try:
                tool_name = ToolFactory.register_tool(tool_class)
                logger.info(f"Registered tool: {tool_name}")
            except Exception as e:
                logger.error(f"Failed to register tool {tool_class.__name__}: {e}")

        logger.info(f"Registered {len(ToolFactory.get_all_tools())} tools")

    async def _initialize_protocol_handler(self):
        """Initialize the MCP protocol handler."""
        tools = ToolFactory.get_all_tools()

        # Create resources
        resources = {
            "docs:overview": ResourceSchema(
                uri="docs://overview",
                name="Documentation Overview",
                description="Overview of available documentation",
                mimeType="text/markdown"
            )
        }

        self.protocol_handler = MCPProtocolHandler(tools, resources)
        logger.info("Protocol handler initialized")

    async def handle_websocket_connection(self, websocket: WebSocket):
        """Handle incoming WebSocket connection."""
        if not self.protocol_handler:
            await websocket.close(code=1011, reason="Server not initialized")
            return

        await self.protocol_handler.handle_connection(websocket)

    async def start(self):
        """Start the server."""
        if self.is_running:
            logger.warning("Server is already running")
            return

        logger.info("Starting MCP Server...")

        try:
            # Initialize
            await self.initialize()

            # Start background tasks
            await self._start_background_tasks()

            # Update server state
            self.is_running = True
            self.start_time = datetime.now()

            logger.info(f"MCP Server started on {self.settings.host}:{self.settings.port}")
            logger.info(f"WebSocket endpoint: ws://{self.settings.host}:{self.settings.port}/mcp")

        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            raise

    async def stop(self):
        """Stop the server gracefully."""
        if not self.is_running:
            return

        logger.info("Stopping MCP Server...")

        # Stop background tasks
        await self._stop_background_tasks()

        # Update server state
        self.is_running = False

        logger.info("MCP Server stopped")

    async def _start_background_tasks(self):
        """Start background tasks."""
        # Start documentation update task
        doc_tool = ToolFactory.get_tool("documentation_search")
        if doc_tool and hasattr(doc_tool, 'start_background_updates'):
            await doc_tool.start_background_updates(interval_hours=6)

        # Start metrics collection
        self.metrics.start_collection()

        logger.info("Background tasks started")

    async def _stop_background_tasks(self):
        """Stop background tasks."""
        # Stop documentation updates
        doc_tool = ToolFactory.get_tool("documentation_search")
        if doc_tool and hasattr(doc_tool, 'stop_background_updates'):
            await doc_tool.stop_background_updates()

        # Stop metrics collection
        self.metrics.stop_collection()

        logger.info("Background tasks stopped")

    def run(self):
        """Run the server using uvicorn."""
        config = uvicorn.Config(
            app=self.app,
            host=self.settings.host,
            port=self.settings.port,
            log_level=self.settings.log_level.lower(),
            access_log=True,
            timeout_keep_alive=30,
            workers=self.settings.workers if not self.settings.debug else 1
        )

        server = uvicorn.Server(config)

        # Run in async context
        async def run_server():
            await self.start()
            await server.serve()
            await self.stop()

        try:
            asyncio.run(run_server())
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        except Exception as e:
            logger.error(f"Server crashed: {e}")
            sys.exit(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI."""
    server = MCPServer()
    await server.start()
    yield
    await server.stop()


# Create the app with lifespan
app = FastAPI(lifespan=lifespan)

if __name__ == "__main__":
    server = MCPServer()
    server.run()