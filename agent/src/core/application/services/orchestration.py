"""
Orchestration Service for coordinating agent components.
Manages workflow coordination and component lifecycle.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


logger = logging.getLogger(__name__)


class WorkflowStage(str, Enum):
    """Workflow execution stages."""
    INITIALIZATION = "initialization"
    ANALYSIS = "analysis"
    RESEARCH = "research"
    DESIGN = "design"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    OPTIMIZATION = "optimization"
    VALIDATION = "validation"
    COMPLETION = "completion"
    FAILED = "failed"


@dataclass
class WorkflowContext:
    """Context for workflow execution."""
    workflow_id: str
    stage: WorkflowStage = WorkflowStage.INITIALIZATION
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class OrchestrationService:
    """
    Service for orchestrating agent workflows and components.
    
    Responsibilities:
    - Coordinate workflow execution
    - Manage component lifecycle
    - Handle error recovery
    - Track execution metrics
    """
    
    def __init__(self):
        """Initialize orchestration service."""
        self.active_workflows: Dict[str, WorkflowContext] = {}
        self.workflow_history: List[WorkflowContext] = []
        self.component_registry: Dict[str, Any] = {}
        
        # Metrics
        self.metrics = {
            "total_workflows": 0,
            "successful_workflows": 0,
            "failed_workflows": 0,
            "average_duration": 0.0
        }
        
        logger.info("Orchestration service initialized")
    
    async def execute_workflow(
        self,
        workflow_id: str,
        stages: List[Callable],
        initial_data: Dict[str, Any]
    ) -> WorkflowContext:
        """
        Execute a workflow with multiple stages.
        
        Args:
            workflow_id: Unique identifier for the workflow
            stages: List of async functions to execute in order
            initial_data: Initial data for the workflow
        
        Returns:
            WorkflowContext with results and status
        """
        
        # Create workflow context
        context = WorkflowContext(
            workflow_id=workflow_id,
            data=initial_data.copy()
        )
        
        # Register active workflow
        self.active_workflows[workflow_id] = context
        self.metrics["total_workflows"] += 1
        
        logger.info(f"Starting workflow: {workflow_id}")
        
        try:
            # Execute stages sequentially
            for i, stage_func in enumerate(stages):
                stage_name = stage_func.__name__ if hasattr(stage_func, '__name__') else f"stage_{i}"
                
                logger.info(f"Workflow {workflow_id}: Executing {stage_name}")
                
                try:
                    # Execute stage
                    result = await stage_func(context)
                    
                    # Update context with result
                    if isinstance(result, dict):
                        context.data.update(result)
                    
                    logger.info(f"Workflow {workflow_id}: Completed {stage_name}")
                
                except Exception as e:
                    error_msg = f"Stage {stage_name} failed: {str(e)}"
                    logger.error(f"Workflow {workflow_id}: {error_msg}")
                    context.errors.append(error_msg)
                    context.stage = WorkflowStage.FAILED
                    
                    # Decide whether to continue or abort
                    if self._is_critical_error(e):
                        raise
            
            # Mark as completed
            context.stage = WorkflowStage.COMPLETION
            context.completed_at = datetime.now()
            
            # Update metrics
            self.metrics["successful_workflows"] += 1
            self._update_average_duration(context)
            
            logger.info(f"Workflow {workflow_id} completed successfully")
        
        except Exception as e:
            # Mark as failed
            context.stage = WorkflowStage.FAILED
            context.completed_at = datetime.now()
            context.errors.append(str(e))
            
            # Update metrics
            self.metrics["failed_workflows"] += 1
            self._update_average_duration(context)
            
            logger.error(f"Workflow {workflow_id} failed: {e}")
        
        finally:
            # Move to history
            self.workflow_history.append(context)
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
        
        return context
    
    async def coordinate_parallel_tasks(
        self,
        tasks: List[Callable],
        context: WorkflowContext
    ) -> List[Any]:
        """
        Execute multiple tasks in parallel.
        
        Args:
            tasks: List of async functions to execute in parallel
            context: Workflow context
        
        Returns:
            List of results from each task
        """
        
        logger.info(f"Executing {len(tasks)} parallel tasks")
        
        try:
            # Execute tasks concurrently
            results = await asyncio.gather(
                *[task(context) for task in tasks],
                return_exceptions=True
            )
            
            # Check for errors
            errors = [r for r in results if isinstance(r, Exception)]
            if errors:
                logger.warning(f"Parallel tasks completed with {len(errors)} errors")
                for error in errors:
                    context.errors.append(str(error))
            
            return results
        
        except Exception as e:
            logger.error(f"Parallel task execution failed: {e}")
            context.errors.append(str(e))
            raise
    
    def register_component(
        self,
        component_name: str,
        component: Any
    ):
        """
        Register a component for lifecycle management.
        
        Args:
            component_name: Unique name for the component
            component: Component instance
        """
        
        self.component_registry[component_name] = component
        logger.info(f"Registered component: {component_name}")
    
    def get_component(self, component_name: str) -> Optional[Any]:
        """
        Get a registered component.
        
        Args:
            component_name: Name of the component
        
        Returns:
            Component instance if found, None otherwise
        """
        
        return self.component_registry.get(component_name)
    
    async def shutdown_components(self):
        """Shutdown all registered components."""
        
        logger.info("Shutting down components...")
        
        for name, component in self.component_registry.items():
            try:
                # Try to call shutdown method if available
                if hasattr(component, 'shutdown'):
                    await component.shutdown()
                elif hasattr(component, 'close'):
                    await component.close()
                
                logger.info(f"Component {name} shut down")
            
            except Exception as e:
                logger.warning(f"Failed to shutdown component {name}: {e}")
        
        self.component_registry.clear()
        logger.info("All components shut down")
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a workflow.
        
        Args:
            workflow_id: Workflow identifier
        
        Returns:
            Status dictionary if workflow found
        """
        
        # Check active workflows
        if workflow_id in self.active_workflows:
            context = self.active_workflows[workflow_id]
            return self._context_to_status(context)
        
        # Check history
        for context in reversed(self.workflow_history):
            if context.workflow_id == workflow_id:
                return self._context_to_status(context)
        
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get orchestration metrics.
        
        Returns:
            Dictionary of metrics
        """
        
        return {
            **self.metrics,
            "active_workflows": len(self.active_workflows),
            "workflow_history_size": len(self.workflow_history),
            "registered_components": len(self.component_registry)
        }
    
    def _context_to_status(self, context: WorkflowContext) -> Dict[str, Any]:
        """Convert workflow context to status dictionary."""
        
        duration = None
        if context.completed_at:
            duration = (context.completed_at - context.started_at).total_seconds()
        
        return {
            "workflow_id": context.workflow_id,
            "stage": context.stage.value,
            "started_at": context.started_at.isoformat(),
            "completed_at": context.completed_at.isoformat() if context.completed_at else None,
            "duration_seconds": duration,
            "errors": context.errors,
            "metadata": context.metadata
        }
    
    def _is_critical_error(self, error: Exception) -> bool:
        """Determine if an error is critical and should abort workflow."""
        
        # Define critical error types
        critical_errors = (
            ConnectionError,
            TimeoutError,
            MemoryError,
            KeyboardInterrupt
        )
        
        return isinstance(error, critical_errors)
    
    def _update_average_duration(self, context: WorkflowContext):
        """Update average workflow duration metric."""
        
        if context.completed_at:
            duration = (context.completed_at - context.started_at).total_seconds()
            
            total_workflows = self.metrics["successful_workflows"] + self.metrics["failed_workflows"]
            if total_workflows > 0:
                current_avg = self.metrics["average_duration"]
                self.metrics["average_duration"] = (
                    (current_avg * (total_workflows - 1) + duration) / total_workflows
                )
