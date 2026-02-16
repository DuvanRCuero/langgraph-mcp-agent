"""
API routes for the LangGraph Programming Agent.
Defines REST endpoints for agent operations.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field

router = APIRouter()


# Request/Response Models
class DocumentationSearchRequest(BaseModel):
    """Request model for documentation search."""
    query: str = Field(..., description="Search query")
    language: Optional[str] = Field(None, description="Programming language filter")
    framework: Optional[str] = Field(None, description="Framework filter")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum results")


class DocumentationSearchResponse(BaseModel):
    """Response model for documentation search."""
    results: List[Dict[str, Any]]
    total: int
    query: str
    timestamp: str


class AgentConfigUpdate(BaseModel):
    """Request model for updating agent configuration."""
    llm_temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    llm_max_tokens: Optional[int] = Field(None, gt=0)
    enable_learning: Optional[bool] = None
    enable_monitoring: Optional[bool] = None
    debug: Optional[bool] = None


class AgentConfigResponse(BaseModel):
    """Response model for agent configuration."""
    config: Dict[str, Any]
    timestamp: str


class TaskStatusResponse(BaseModel):
    """Response model for task status."""
    task_id: str
    status: str
    phase: Optional[str] = None
    progress: float
    created_at: str
    updated_at: str
    metadata: Dict[str, Any]


# Documentation endpoints
@router.get("/documentation/search", response_model=DocumentationSearchResponse)
async def search_documentation(
    query: str,
    language: Optional[str] = None,
    framework: Optional[str] = None,
    limit: int = 10,
    request: Request = None
):
    """
    Search documentation using the MCP server.
    
    This endpoint queries the MCP server for relevant documentation
    based on the provided search parameters.
    """
    
    try:
        agent_manager = request.app.state.agent_manager
        mcp_client = agent_manager.mcp_client
        
        # Ensure MCP client is connected
        if not mcp_client.is_connected:
            await mcp_client.connect()
        
        # Search documentation
        docs = await mcp_client.search_documentation(
            query=query,
            language=language,
            framework=framework,
            limit=limit
        )
        
        # Convert to dict format
        results = []
        for doc in docs:
            results.append({
                "title": doc.title,
                "url": doc.url,
                "content": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                "source": doc.source,
                "relevance_score": doc.relevance_score,
                "last_updated": doc.last_updated.isoformat(),
                "doc_type": doc.doc_type,
                "tags": doc.tags
            })
        
        return DocumentationSearchResponse(
            results=results,
            total=len(results),
            query=query,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Documentation search failed: {str(e)}")


@router.get("/documentation/best-practices")
async def get_best_practices(
    language: str,
    framework: Optional[str] = None,
    topic: Optional[str] = None,
    request: Request = None
):
    """
    Get best practices for a language/framework.
    
    Returns curated best practices and guidelines from the
    documentation database.
    """
    
    try:
        agent_manager = request.app.state.agent_manager
        mcp_client = agent_manager.mcp_client
        
        # Ensure MCP client is connected
        if not mcp_client.is_connected:
            await mcp_client.connect()
        
        # Get best practices
        docs = await mcp_client.get_latest_best_practices(
            language=language,
            framework=framework,
            topic=topic
        )
        
        # Convert to dict format
        results = []
        for doc in docs:
            results.append({
                "title": doc.title,
                "url": doc.url,
                "content": doc.content,
                "source": doc.source,
                "relevance_score": doc.relevance_score
            })
        
        return {
            "best_practices": results,
            "language": language,
            "framework": framework,
            "topic": topic,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get best practices: {str(e)}")


# Agent configuration endpoints
@router.get("/agent/config", response_model=AgentConfigResponse)
async def get_agent_config(request: Request):
    """
    Get current agent configuration.
    
    Returns the current configuration settings for the agent.
    """
    
    try:
        agent_manager = request.app.state.agent_manager
        settings = agent_manager.settings
        
        config = {
            "llm_model": settings.llm_model,
            "llm_temperature": settings.llm_temperature,
            "llm_max_tokens": settings.llm_max_tokens,
            "enable_learning": settings.enable_learning,
            "enable_monitoring": settings.enable_monitoring,
            "persist_checkpoints": settings.persist_checkpoints,
            "debug": settings.debug
        }
        
        return AgentConfigResponse(
            config=config,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get config: {str(e)}")


@router.patch("/agent/config", response_model=AgentConfigResponse)
async def update_agent_config(
    config_update: AgentConfigUpdate,
    request: Request
):
    """
    Update agent configuration.
    
    Allows partial updates to the agent configuration.
    Changes take effect immediately.
    """
    
    try:
        agent_manager = request.app.state.agent_manager
        settings = agent_manager.settings
        
        # Update settings that were provided
        update_data = config_update.model_dump(exclude_unset=True)
        
        for key, value in update_data.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
        
        # Return updated config
        config = {
            "llm_model": settings.llm_model,
            "llm_temperature": settings.llm_temperature,
            "llm_max_tokens": settings.llm_max_tokens,
            "enable_learning": settings.enable_learning,
            "enable_monitoring": settings.enable_monitoring,
            "persist_checkpoints": settings.persist_checkpoints,
            "debug": settings.debug
        }
        
        return AgentConfigResponse(
            config=config,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")


# Task management endpoints
@router.get("/tasks", response_model=List[TaskStatusResponse])
async def list_tasks(
    limit: int = 20,
    offset: int = 0,
    status: Optional[str] = None
):
    """
    List recent tasks.
    
    Returns a paginated list of recent tasks with their status.
    In a production system, this would query a database.
    """
    
    # Placeholder implementation
    # In production, query from database/checkpoint store
    return []


@router.delete("/tasks/{task_id}")
async def cancel_task(task_id: str, request: Request):
    """
    Cancel a running task.
    
    Attempts to gracefully stop a task that is in progress.
    """
    
    try:
        # In production, implement task cancellation logic
        return {
            "task_id": task_id,
            "status": "cancelled",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel task: {str(e)}")


# Code analysis endpoints
@router.post("/code/analyze")
async def analyze_code(
    code: str,
    language: str,
    framework: Optional[str] = None,
    request: Request = None
):
    """
    Analyze code quality using MCP tools.
    
    Performs comprehensive code analysis including complexity,
    security, and best practices checks.
    """
    
    try:
        agent_manager = request.app.state.agent_manager
        mcp_client = agent_manager.mcp_client
        
        # Ensure MCP client is connected
        if not mcp_client.is_connected:
            await mcp_client.connect()
        
        # Analyze code
        analysis = await mcp_client.analyze_code_quality(
            code=code,
            language=language,
            framework=framework
        )
        
        return {
            "analysis": analysis,
            "language": language,
            "framework": framework,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Code analysis failed: {str(e)}")


# System endpoints
@router.get("/system/metrics")
async def get_system_metrics(request: Request):
    """
    Get system metrics and statistics.
    
    Returns performance metrics, resource usage, and
    operational statistics for the agent.
    """
    
    try:
        agent_manager = request.app.state.agent_manager
        
        metrics = {
            "agent_initialized": agent_manager.is_initialized,
            "mcp_connected": agent_manager.mcp_client.is_connected if agent_manager.mcp_client else False,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add MCP connection metrics if available
        if agent_manager.mcp_client and agent_manager.mcp_client.is_connected:
            metrics["mcp_metrics"] = agent_manager.mcp_client.connection_metrics
        
        return metrics
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.post("/system/reset")
async def reset_system(request: Request, background_tasks: BackgroundTasks):
    """
    Reset the agent system.
    
    Reinitializes the agent and clears any cached state.
    Use with caution as this will interrupt any running tasks.
    """
    
    try:
        agent_manager = request.app.state.agent_manager
        
        # Schedule reset in background
        async def perform_reset():
            await agent_manager.shutdown()
            await agent_manager.initialize()
        
        background_tasks.add_task(perform_reset)
        
        return {
            "status": "reset_scheduled",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")
