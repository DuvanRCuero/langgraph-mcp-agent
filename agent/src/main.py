"""
Main entry point for the LangGraph Programming Agent.
Production-ready FastAPI application with comprehensive features.
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
import uvicorn
from pydantic import BaseModel, Field

from core.infrastructure.config.settings import Settings
from core.infrastructure.adapters.mcp_adapter import MCPClientAdapter, MCPConnectionConfig
from core.infrastructure.adapters.openai_adapter import OpenAIAdapter
from core.infrastructure.adapters.qdrant_adapter import QdrantAdapter
from core.infrastructure.langgraph.agent_graph import AdvancedProgrammingAgent
from api.routes import router as api_router
from api.middleware import RequestLoggingMiddleware, MetricsMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class AgentManager:
    """Manages agent lifecycle and dependencies."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.agent: Optional[AdvancedProgrammingAgent] = None
        self.is_initialized = False

        # Dependencies
        self.mcp_client: Optional[MCPClientAdapter] = None
        self.llm_gateway: Optional[OpenAIAdapter] = None
        self.vector_store: Optional[QdrantAdapter] = None

    async def initialize(self):
        """Initialize the agent and all dependencies."""

        if self.is_initialized:
            logger.warning("Agent already initialized")
            return

        logger.info("Initializing Agent Manager...")

        try:
            # Initialize MCP client
            mcp_config = MCPConnectionConfig(
                server_url=self.settings.mcp_server_url,
                reconnect_attempts=3,
                connection_timeout=30.0
            )
            self.mcp_client = MCPClientAdapter(mcp_config)

            # Initialize LLM gateway
            self.llm_gateway = OpenAIAdapter(
                api_key=self.settings.openai_api_key,
                model=self.settings.llm_model,
                temperature=self.settings.llm_temperature,
                max_tokens=self.settings.llm_max_tokens
            )

            # Initialize vector store
            self.vector_store = QdrantAdapter(
                url=self.settings.qdrant_url,
                collection_name=self.settings.qdrant_collection
            )

            # Initialize agent
            agent_config = {
                "persist_checkpoints": self.settings.persist_checkpoints,
                "checkpoint_db": self.settings.checkpoint_db,
                "debug": self.settings.debug,
                "enable_learning": self.settings.enable_learning,
                "enable_monitoring": self.settings.enable_monitoring
            }

            self.agent = AdvancedProgrammingAgent(
                llm_gateway=self.llm_gateway,
                mcp_client=self.mcp_client,
                vector_store=self.vector_store,
                config=agent_config
            )

            self.is_initialized = True
            logger.info("Agent Manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Agent Manager: {e}")
            raise

    async def shutdown(self):
        """Shutdown agent and clean up resources."""

        logger.info("Shutting down Agent Manager...")

        # Disconnect MCP client
        if self.mcp_client:
            await self.mcp_client.disconnect()

        # Close vector store connections
        if self.vector_store:
            await self.vector_store.close()

        self.is_initialized = False
        logger.info("Agent Manager shutdown complete")

    async def execute_task(
            self,
            task_description: str,
            requirements: List[str],
            language: str,
            **kwargs
    ) -> Dict[str, Any]:
        """Execute a programming task."""

        if not self.is_initialized or not self.agent:
            raise RuntimeError("Agent not initialized")

        return await self.agent.execute_with_monitoring(
            task_description=task_description,
            requirements=requirements,
            language=language,
            **kwargs
        )

    async def get_agent_status(self) -> Dict[str, Any]:
        """Get agent status and metrics."""

        if not self.agent:
            return {"status": "not_initialized"}

        try:
            performance_report = await self.agent.get_performance_report()

            return {
                "status": "ready",
                "initialized": self.is_initialized,
                "mcp_connected": self.mcp_client.is_connected if self.mcp_client else False,
                "performance_report": performance_report,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get agent status: {e}")
            return {
                "status": "error",
                "error": str(e)
            }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI."""

    # Load settings
    settings = Settings()

    # Initialize agent manager
    app.state.agent_manager = AgentManager(settings)
    await app.state.agent_manager.initialize()

    logger.info("Agent application started")

    yield

    # Shutdown
    await app.state.agent_manager.shutdown()
    logger.info("Agent application shutdown")


# Create FastAPI app
app = FastAPI(
    title="LangGraph Programming Agent",
    description="Production-grade AI agent for elite code generation with MCP integration",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(MetricsMiddleware)

# Include API routes
app.include_router(api_router, prefix="/api/v1")


# Pydantic models for request/response
class TaskRequest(BaseModel):
    """Request model for task execution."""
    task_description: str = Field(..., description="Description of the programming task")
    requirements: List[str] = Field(..., description="List of requirements")
    language: str = Field(default="python", description="Target programming language")
    framework: Optional[str] = Field(None, description="Framework to use")
    quality_level: str = Field(default="excellent", description="Desired quality level")
    max_iterations: int = Field(default=3, description="Maximum optimization iterations")

    class Config:
        schema_extra = {
            "example": {
                "task_description": "Create a REST API for user management",
                "requirements": [
                    "User registration with email verification",
                    "JWT authentication",
                    "Role-based access control",
                    "Password reset functionality"
                ],
                "language": "python",
                "framework": "fastapi",
                "quality_level": "elite",
                "max_iterations": 3
            }
        }


class TaskResponse(BaseModel):
    """Response model for task execution."""
    success: bool
    task_id: str
    generated_artifacts: Dict[str, Any]
    quality_report: Dict[str, Any]
    execution_metrics: Dict[str, Any]
    workflow_info: Dict[str, Any]
    metadata: Dict[str, Any]


# Health endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }


@app.get("/health/agent")
async def agent_health_check():
    """Agent-specific health check."""
    status = await app.state.agent_manager.get_agent_status()
    return status


# Task execution endpoint
@app.post("/tasks", response_model=TaskResponse)
async def create_task(request: TaskRequest, background_tasks: BackgroundTasks):
    """
    Create and execute a programming task.

    This endpoint initiates the agent workflow for generating code
    with elite engineering standards.
    """

    try:
        # Execute task
        result = await app.state.agent_manager.execute_task(
            task_description=request.task_description,
            requirements=request.requirements,
            language=request.language,
            framework=request.framework,
            quality_level=request.quality_level,
            max_iterations=request.max_iterations
        )

        # Schedule cleanup or monitoring tasks
        background_tasks.add_task(
            _post_task_cleanup,
            result.get("task_id")
        )

        return TaskResponse(**result)

    except Exception as e:
        logger.error(f"Task execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Task status endpoint
@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """
    Get status of a task.

    In production, this would query a database or checkpoint store.
    """

    # For now, return basic status
    # In production, implement proper task tracking

    return {
        "task_id": task_id,
        "status": "completed",  # Simplified
        "timestamp": datetime.now().isoformat()
    }


# Agent management endpoints
@app.post("/agent/optimize")
async def optimize_agent(optimization_targets: List[str]):
    """
    Optimize agent performance.

    Targets can include:
    - prompt_efficiency
    - tool_selection
    - error_recovery
    """

    try:
        result = await app.state.agent_manager.agent.optimize_agent(
            optimization_targets
        )

        return {
            "success": True,
            "optimizations": result,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Agent optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent/metrics")
async def get_agent_metrics():
    """Get comprehensive agent metrics."""

    try:
        report = await app.state.agent_manager.agent.get_performance_report()

        return {
            "success": True,
            "metrics": report,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to get agent metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Utility endpoints
@app.get("/docs/custom")
async def custom_docs():
    """Custom API documentation."""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="LangGraph Agent API",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css"
    )


@app.get("/openapi.json")
async def get_openapi():
    """Get OpenAPI specification."""
    return app.openapi()


async def _post_task_cleanup(task_id: str):
    """Cleanup tasks after task completion."""

    try:
        # In production, implement cleanup logic
        # - Remove temporary files
        # - Update analytics
        # - Send notifications
        logger.info(f"Post-task cleanup for {task_id}")

    except Exception as e:
        logger.warning(f"Post-task cleanup failed: {e}")


def run_server():
    """Run the FastAPI server."""

    # Load settings
    settings = Settings()

    # Server configuration
    server_config = uvicorn.Config(
        app=app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        access_log=True,
        timeout_keep_alive=30,
        workers=settings.workers,
        reload=settings.debug
    )

    server = uvicorn.Server(server_config)

    # Run server
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_server()