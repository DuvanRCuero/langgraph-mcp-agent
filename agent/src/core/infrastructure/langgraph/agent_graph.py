"""
Main LangGraph implementation for the programming agent.
Stateful workflow with self-correction and human-in-the-loop capabilities.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver, AsyncSqliteSaver
from langgraph.prebuilt import ToolNode

from .state import AgentState, AgentPhase, StateManager
from .nodes import (
    AnalysisNode,
    ResearchNode,
    DesignNode,
    CodeGenNode,
    TestGenNode,
    ReviewNode,
    OptimizationNode,
    HumanReviewNode
)
from .tools import AgentTools

logger = logging.getLogger(__name__)


class ProgrammingAgentGraph:
    """
    Production-grade LangGraph agent for programming tasks.
    Implements sophisticated workflow with error handling, self-correction,
    and human oversight.
    """

    def __init__(
        self,
        llm_gateway,
        mcp_client,
        vector_store,
        config: Optional[Dict[str, Any]] = None
    ):
        self.llm_gateway = llm_gateway
        self.mcp_client = mcp_client
        self.vector_store = vector_store
        self.config = config or {}

        # Initialize components
        self.agent_tools = AgentTools(mcp_client, llm_gateway, vector_store)
        self.state_manager = StateManager()

        # Initialize nodes
        self.nodes = self._initialize_nodes()

        # Build graph
        self.graph = self._build_graph()
        self.app = self._compile_graph()

        logger.info("ProgrammingAgentGraph initialized")

    def _initialize_nodes(self) -> Dict[str, Any]:
        """Initialize all graph nodes."""

        return {
            "analyze": AnalysisNode(self.llm_gateway),
            "research": ResearchNode(self.mcp_client),
            "design": DesignNode(self.llm_gateway),
            "generate_code": CodeGenNode(self.llm_gateway, self.agent_tools),
            "generate_tests": TestGenNode(self.llm_gateway),
            "review": ReviewNode(self.llm_gateway, self.agent_tools),
            "optimize": OptimizationNode(self.llm_gateway, self.agent_tools),
            "human_review": HumanReviewNode(),
        }

    def _build_graph(self) -> StateGraph:
        """Build the state graph with conditional edges."""

        workflow = StateGraph(AgentState)

        # Add all nodes
        for name, node in self.nodes.items():
            workflow.add_node(name, node.execute)

        # Define workflow with conditional routing
        workflow.set_entry_point("analyze")

        # Main sequential flow
        workflow.add_edge("analyze", "research")
        workflow.add_edge("research", "design")
        workflow.add_edge("design", "generate_code")
        workflow.add_edge("generate_code", "generate_tests")
        workflow.add_edge("generate_tests", "review")

        # Conditional routing after review
        workflow.add_conditional_edges(
            "review",
            self._route_after_review,
            {
                "optimize": "optimize",
                "human_review": "human_review",
                "complete": END
            }
        )

        # Routing after optimization
        workflow.add_edge("optimize", "review")

        # Routing after human review
        workflow.add_conditional_edges(
            "human_review",
            self._route_after_human_review,
            {
                "optimize": "optimize",
                "complete": END,
                "redo": "analyze"  # Restart from analysis
            }
        )

        # Error handling edges
        workflow.add_conditional_edges(
            "analyze",
            self._check_for_errors,
            {"continue": "research", "fail": END}
        )

        workflow.add_conditional_edges(
            "research",
            self._check_for_errors,
            {"continue": "design", "fail": END}
        )

        workflow.add_conditional_edges(
            "design",
            self._check_for_errors,
            {"continue": "generate_code", "fail": END}
        )

        workflow.add_conditional_edges(
            "generate_code",
            self._check_for_errors,
            {"continue": "generate_tests", "fail": END}
        )

        workflow.add_conditional_edges(
            "generate_tests",
            self._check_for_errors,
            {"continue": "review", "fail": END}
        )

        return workflow

    def _compile_graph(self):
        """Compile graph with checkpointing."""

        # Choose checkpointer based on config
        if self.config.get("persist_checkpoints", False):
            checkpointer = AsyncSqliteSaver.from_conn_string(
                self.config.get("checkpoint_db", "checkpoints.db")
            )
        else:
            checkpointer = MemorySaver()

        return self.graph.compile(
            checkpointer=checkpointer,
            interrupt_before=["human_review"],  # Allow human intervention
            debug=self.config.get("debug", False)
        )

    def _route_after_review(self, state: AgentState) -> str:
        """Route after automated code review."""

        review_result = state.get("review_result", {})

        if review_result.get("needs_human_review", False):
            return "human_review"
        elif review_result.get("needs_optimization", False):
            return "optimize"
        else:
            return "complete"

    def _route_after_human_review(self, state: AgentState) -> str:
        """Route after human review."""

        human_feedback = state.get("human_feedback", {})

        if human_feedback.get("approved", False):
            return "complete"
        elif human_feedback.get("needs_optimization", False):
            return "optimize"
        else:
            # Major issues require restart
            return "redo"

    def _check_for_errors(self, state: AgentState) -> str:
        """Check for errors and decide routing."""

        if state.get("error"):
            logger.error(f"Node failed with error: {state['error']}")

            # Check if we should retry
            if state.get("retry_count", 0) < state.get("max_retries", 3):
                state["retry_count"] = state.get("retry_count", 0) + 1
                logger.info(f"Retrying (attempt {state['retry_count']})")
                return "continue"
            else:
                logger.error("Max retries exceeded, failing task")
                return "fail"

        return "continue"

    async def execute(
        self,
        task_description: str,
        requirements: List[str],
        language: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a programming task through the graph.

        Args:
            task_description: Description of the programming task
            requirements: List of requirements
            language: Target programming language
            **kwargs: Additional parameters

        Returns:
            Complete task result
        """

        # Prepare initial state
        initial_state = self._prepare_initial_state(
            task_description, requirements, language, kwargs
        )

        # Generate thread ID
        thread_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(task_description) % 10000:04d}"

        try:
            # Execute graph
            final_state = await self.app.ainvoke(
                initial_state,
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "user_id": kwargs.get("user_id", "anonymous")
                    }
                }
            )

            # Prepare result
            result = self._prepare_result(final_state, thread_id)

            logger.info(
                f"Task completed: {thread_id}, "
                f"Quality: {result.get('quality_level', 'unknown')}, "
                f"Time: {result.get('execution_metrics', {}).get('total_time_seconds', 0):.1f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Task execution failed: {e}", exc_info=True)

            return {
                "success": False,
                "task_id": thread_id,
                "error": str(e),
                "final_state": "failed",
                "execution_time": (datetime.now().timestamp() - initial_state.get("start_time", 0))
            }

    def _prepare_initial_state(
        self,
        task_description: str,
        requirements: List[str],
        language: str,
        kwargs: Dict[str, Any]
    ) -> AgentState:
        """Prepare initial state for graph execution."""

        return {
            "task_id": kwargs.get("task_id", f"task_{datetime.now().timestamp()}"),
            "task_description": task_description,
            "requirements": requirements,
            "language": language,
            "framework": kwargs.get("framework"),
            "quality_level": kwargs.get("quality_level", "excellent"),

            "current_phase": AgentPhase.INITIALIZED,
            "iteration": 0,
            "max_iterations": kwargs.get("max_iterations", 3),

            "start_time": datetime.now().timestamp(),
            "token_usage": {"prompt": 0, "completion": 0, "total": 0},
            "tool_calls": 0,

            "error": None,
            "retry_count": 0,
            "max_retries": 3,

            "agent_config": self.config,
            "llm_config": {
                "model": kwargs.get("llm_model", "gpt-4-turbo-preview"),
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 4000)
            }
        }

    def _prepare_result(self, state: AgentState, thread_id: str) -> Dict[str, Any]:
        """Prepare final result from state."""

        # Calculate quality metrics
        quality_metrics = state.get("quality_metrics", {})
        overall_score = quality_metrics.get("overall_score", 0)

        # Determine quality level
        if overall_score >= 95:
            quality_level = "elite"
        elif overall_score >= 85:
            quality_level = "excellent"
        elif overall_score >= 75:
            quality_level = "good"
        elif overall_score >= 60:
            quality_level = "fair"
        else:
            quality_level = "poor"

        # Calculate total code lines
        total_lines = sum(item.get("lines", 0) for item in state.get("generated_code", []))
        total_test_lines = sum(item.get("test_lines", 0) for item in state.get("generated_tests", []))

        return {
            "success": True,
            "task_id": thread_id,
            "task_description": state.get("task_description"),
            "language": state.get("language"),
            "framework": state.get("framework"),

            "generated_artifacts": {
                "code_files": state.get("generated_code", []),
                "test_files": state.get("generated_tests", []),
                "architecture": state.get("architecture_design"),
                "documentation": state.get("researched_docs", [])[:10]  # Top 10 docs
            },

            "quality_report": {
                "level": quality_level,
                "score": overall_score,
                "detailed_metrics": quality_metrics,
                "test_coverage": quality_metrics.get("test_coverage", 0),
                "complexity": quality_metrics.get("complexity_score", 0)
            },

            "execution_metrics": {
                "total_time_seconds": state.get("execution_time", 0),
                "iterations": state.get("iteration", 0),
                "token_usage": state.get("token_usage", {}),
                "tool_calls": state.get("tool_calls", 0),
                "total_code_lines": total_lines,
                "total_test_lines": total_test_lines,
                "test_to_code_ratio": (total_test_lines / total_lines * 100) if total_lines > 0 else 0
            },

            "workflow_info": {
                "final_phase": state.get("current_phase", AgentPhase.COMPLETED).value,
                "phase_history": state.get("phase_history", []),
                "needed_human_review": state.get("needs_human_review", False),
                "human_feedback": state.get("human_feedback")
            },

            "metadata": {
                "completed_at": datetime.now().isoformat(),
                "agent_version": "2.0.0",
                "config_used": state.get("agent_config", {})
            }
        }

    async def execute_with_collaboration(
        self,
        task_description: str,
        requirements: List[str],
        language: str,
        collaborators: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute task with multi-agent collaboration.

        Args:
            collaborators: List of collaborator agent IDs

        Returns:
            Combined result from all collaborators
        """

        if not collaborators:
            return await self.execute(task_description, requirements, language, **kwargs)

        # Divide task among collaborators
        subtasks = await self._decompose_task(task_description, requirements, len(collaborators))

        # Execute subtasks in parallel
        subtask_results = []
        for i, subtask in enumerate(subtasks):
            # In production, distribute to different agent instances
            # For now, execute sequentially
            result = await self.execute(
                task_description=subtask["description"],
                requirements=subtask["requirements"],
                language=language,
                **{**kwargs, "task_id": f"{kwargs.get('task_id', 'task')}_subtask_{i}"}
            )
            subtask_results.append(result)

        # Integrate results
        integrated_result = await self._integrate_subtask_results(subtask_results)

        return {
            "success": True,
            "collaboration_mode": "multi_agent",
            "collaborators": collaborators,
            "subtask_count": len(subtasks),
            "integrated_result": integrated_result,
            "subtask_results": subtask_results
        }

    async def _decompose_task(
        self,
        task_description: str,
        requirements: List[str],
        num_collaborators: int
    ) -> List[Dict[str, Any]]:
        """Decompose task into subtasks for collaborators."""

        decomposition_prompt = f"""
        Decompose this programming task into {num_collaborators} independent subtasks:
        
        Task: {task_description}
        Requirements: {requirements}
        
        Each subtask should be self-contained and can be worked on independently.
        Provide subtask descriptions and specific requirements.
        """

        # In production, use LLM to decompose task
        # For now, return simple decomposition
        subtasks = []

        # Split requirements among subtasks
        requirements_per_subtask = len(requirements) // num_collaborators
        remainder = len(requirements) % num_collaborators

        start_idx = 0
        for i in range(num_collaborators):
            end_idx = start_idx + requirements_per_subtask + (1 if i < remainder else 0)
            subtask_reqs = requirements[start_idx:end_idx]

            subtasks.append({
                "description": f"{task_description} - Part {i + 1}",
                "requirements": subtask_reqs
            })

            start_idx = end_idx

        return subtasks

    async def _integrate_subtask_results(
        self,
        subtask_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Integrate results from multiple subtasks."""

        # Combine all generated code
        all_code = []
        all_tests = []

        for result in subtask_results:
            artifacts = result.get("generated_artifacts", {})
            all_code.extend(artifacts.get("code_files", []))
            all_tests.extend(artifacts.get("test_files", []))

        # Calculate aggregate metrics
        total_score = sum(
            r.get("quality_report", {}).get("score", 0)
            for r in subtask_results
        )
        avg_score = total_score / len(subtask_results) if subtask_results else 0

        total_time = sum(
            r.get("execution_metrics", {}).get("total_time_seconds", 0)
            for r in subtask_results
        )

        total_tokens = {
            "prompt": sum(r.get("execution_metrics", {}).get("token_usage", {}).get("prompt", 0)
                         for r in subtask_results),
            "completion": sum(r.get("execution_metrics", {}).get("token_usage", {}).get("completion", 0)
                            for r in subtask_results),
            "total": sum(r.get("execution_metrics", {}).get("token_usage", {}).get("total", 0)
                        for r in subtask_results)
        }

        return {
            "integrated_code": all_code,
            "integrated_tests": all_tests,
            "aggregate_quality_score": avg_score,
            "total_execution_time": total_time,
            "total_token_usage": total_tokens,
            "subtask_count": len(subtask_results)
        }

    def visualize(self, output_path: Optional[str] = None):
        """Generate visualization of the graph."""

        try:
            from langgraph.graph import draw

            if output_path:
                draw(self.graph).save(output_path)
                logger.info(f"Graph visualization saved to {output_path}")
                return output_path
            else:
                # Return SVG string
                return draw(self.graph)._repr_svg_()

        except Exception as e:
            logger.warning(f"Failed to visualize graph: {e}")
            return None

    async def get_execution_history(self, thread_id: str) -> List[Dict[str, Any]]:
        """Get execution history for a thread."""

        try:
            # Get checkpoints from the compiled app
            checkpoints = await self.app.get_state_history(thread_id)

            history = []
            for checkpoint in checkpoints:
                history.append({
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "state": checkpoint.values,
                    "metadata": checkpoint.metadata,
                    "timestamp": checkpoint.timestamp.isoformat() if checkpoint.timestamp else None
                })

            return history

        except Exception as e:
            logger.warning(f"Failed to get execution history: {e}")
            return []

    def get_graph_config(self) -> Dict[str, Any]:
        """Get graph configuration."""

        return {
            "nodes": list(self.nodes.keys()),
            "edges": self._get_graph_edges(),
            "config": self.config,
            "version": "2.0.0"
        }

    def _get_graph_edges(self) -> List[Dict[str, str]]:
        """Get graph edges for documentation."""

        edges = []

        # Main flow edges
        edges.append({"from": "analyze", "to": "research"})
        edges.append({"from": "research", "to": "design"})
        edges.append({"from": "design", "to": "generate_code"})
        edges.append({"from": "generate_code", "to": "generate_tests"})
        edges.append({"from": "generate_tests", "to": "review"})

        # Conditional edges (documentation purposes)
        edges.append({"from": "review", "to": "optimize", "condition": "needs_optimization"})
        edges.append({"from": "review", "to": "human_review", "condition": "needs_human_review"})
        edges.append({"from": "review", "to": "END", "condition": "complete"})

        edges.append({"from": "optimize", "to": "review"})

        edges.append({"from": "human_review", "to": "optimize", "condition": "needs_optimization"})
        edges.append({"from": "human_review", "to": "END", "condition": "complete"})
        edges.append({"from": "human_review", "to": "analyze", "condition": "redo"})

        return edges


class AdvancedProgrammingAgent(ProgrammingAgentGraph):
    """
    Extended agent with additional capabilities:
    - Advanced error recovery
    - Performance monitoring
    - Adaptive learning
    - Multi-modal support
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Additional components
        self.performance_monitor = PerformanceMonitor()
        self.adaptive_learner = AdaptiveLearner()
        self.error_recovery = ErrorRecoverySystem()

        # Extended configuration
        self.advanced_config = {
            "enable_learning": kwargs.get("enable_learning", True),
            "enable_monitoring": kwargs.get("enable_monitoring", True),
            "max_recovery_attempts": kwargs.get("max_recovery_attempts", 5),
            "learning_window": kwargs.get("learning_window", 100)
        }

    async def execute_with_monitoring(
        self,
        task_description: str,
        requirements: List[str],
        language: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute task with comprehensive monitoring."""

        # Start monitoring
        monitor_id = self.performance_monitor.start_monitoring()

        try:
            # Execute task
            result = await super().execute(
                task_description, requirements, language, **kwargs
            )

            # Collect monitoring data
            monitoring_data = self.performance_monitor.stop_monitoring(monitor_id)

            # Update adaptive learning
            if self.advanced_config["enable_learning"]:
                await self.adaptive_learner.record_execution(
                    task_description, requirements, language, result, monitoring_data
                )

            # Add monitoring to result
            result["monitoring"] = monitoring_data
            result["learning_insights"] = await self.adaptive_learner.get_insights(
                task_description, language
            )

            return result

        except Exception as e:
            # Attempt error recovery
            if self.advanced_config["enable_monitoring"]:
                recovery_result = await self.error_recovery.attempt_recovery(
                    task_description, requirements, language, str(e)
                )

                if recovery_result["success"]:
                    logger.info(f"Error recovered: {recovery_result['method']}")
                    # Retry with recovered parameters
                    return await self.execute_with_monitoring(
                        task_description, requirements, language,
                        **{**kwargs, **recovery_result.get("adjusted_params", {})}
                    )

            # If recovery fails, re-raise
            raise

    async def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report for the agent."""

        return {
            "monitoring_stats": self.performance_monitor.get_statistics(),
            "learning_stats": await self.adaptive_learner.get_statistics(),
            "recovery_stats": self.error_recovery.get_statistics(),
            "agent_config": self.advanced_config
        }

    async def optimize_agent(self, optimization_targets: List[str]) -> Dict[str, Any]:
        """Optimize agent based on specified targets."""

        optimizations = []

        for target in optimization_targets:
            if target == "prompt_efficiency":
                optimization = await self.adaptive_learner.optimize_prompts()
                optimizations.append(optimization)

            elif target == "tool_selection":
                optimization = await self.adaptive_learner.optimize_tool_selection()
                optimizations.append(optimization)

            elif target == "error_recovery":
                optimization = self.error_recovery.optimize_recovery_strategies()
                optimizations.append(optimization)

        return {
            "optimizations_applied": optimizations,
            "optimization_targets": optimization_targets,
            "timestamp": datetime.now().isoformat()
        }


# Supporting classes for advanced features

class PerformanceMonitor:
    """Monitors agent performance metrics."""

    def __init__(self):
        self.active_monitors = {}
        self.history = []

    def start_monitoring(self) -> str:
        """Start a new monitoring session."""
        import uuid
        monitor_id = str(uuid.uuid4())

        self.active_monitors[monitor_id] = {
            "start_time": datetime.now().timestamp(),
            "metrics": {
                "llm_calls": 0,
                "tool_calls": 0,
                "tokens_used": 0,
                "execution_time": 0,
                "memory_usage": 0
            }
        }

        return monitor_id

    def stop_monitoring(self, monitor_id: str) -> Dict[str, Any]:
        """Stop monitoring and return metrics."""
        if monitor_id not in self.active_monitors:
            return {}

        monitor_data = self.active_monitors.pop(monitor_id)
        end_time = datetime.now().timestamp()

        monitor_data["metrics"]["execution_time"] = end_time - monitor_data["start_time"]

        # Calculate memory usage
        import psutil
        process = psutil.Process()
        monitor_data["metrics"]["memory_usage_mb"] = process.memory_info().rss / 1024 / 1024

        # Store in history
        self.history.append(monitor_data)
        if len(self.history) > 1000:  # Keep last 1000 executions
            self.history = self.history[-1000:]

        return monitor_data["metrics"]

    def record_llm_call(self, monitor_id: str, tokens: int):
        """Record LLM call."""
        if monitor_id in self.active_monitors:
            self.active_monitors[monitor_id]["metrics"]["llm_calls"] += 1
            self.active_monitors[monitor_id]["metrics"]["tokens_used"] += tokens

    def record_tool_call(self, monitor_id: str):
        """Record tool call."""
        if monitor_id in self.active_monitors:
            self.active_monitors[monitor_id]["metrics"]["tool_calls"] += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics."""

        if not self.history:
            return {}

        total_executions = len(self.history)

        # Calculate averages
        avg_metrics = {}
        for metric in ["llm_calls", "tool_calls", "tokens_used", "execution_time"]:
            values = [h["metrics"].get(metric, 0) for h in self.history]
            avg_metrics[f"avg_{metric}"] = sum(values) / len(values) if values else 0

        return {
            "total_executions": total_executions,
            "average_metrics": avg_metrics,
            "recent_executions": len(self.history)
        }


class AdaptiveLearner:
    """Learns from executions to improve performance."""

    def __init__(self):
        self.execution_history = []
        self.patterns = {}
        self.optimizations = {}

    async def record_execution(
        self,
        task_description: str,
        requirements: List[str],
        language: str,
        result: Dict[str, Any],
        monitoring_data: Dict[str, Any]
    ):
        """Record execution for learning."""

        execution_record = {
            "task_description": task_description,
            "language": language,
            "result_quality": result.get("quality_report", {}).get("score", 0),
            "execution_time": monitoring_data.get("execution_time", 0),
            "tokens_used": monitoring_data.get("tokens_used", 0),
            "llm_calls": monitoring_data.get("llm_calls", 0),
            "tool_calls": monitoring_data.get("tool_calls", 0),
            "timestamp": datetime.now().isoformat()
        }

        self.execution_history.append(execution_record)

        # Keep history manageable
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]

        # Update patterns
        await self._update_patterns(execution_record)

    async def _update_patterns(self, execution_record: Dict[str, Any]):
        """Update learned patterns."""

        language = execution_record["language"]

        if language not in self.patterns:
            self.patterns[language] = {
                "total_executions": 0,
                "avg_quality": 0,
                "avg_time": 0,
                "avg_tokens": 0
            }

        pattern = self.patterns[language]
        pattern["total_executions"] += 1

        # Update moving averages
        n = pattern["total_executions"]
        pattern["avg_quality"] = (
            (pattern["avg_quality"] * (n - 1) + execution_record["result_quality"]) / n
        )
        pattern["avg_time"] = (
            (pattern["avg_time"] * (n - 1) + execution_record["execution_time"]) / n
        )
        pattern["avg_tokens"] = (
            (pattern["avg_tokens"] * (n - 1) + execution_record["tokens_used"]) / n
        )

    async def get_insights(
        self,
        task_description: str,
        language: str
    ) -> Dict[str, Any]:
        """Get insights for a specific task."""

        # Find similar past tasks
        similar_tasks = []
        for record in self.execution_history[-100:]:  # Last 100 executions
            if record["language"] == language:
                # Simple similarity check (in production use embeddings)
                if any(word in record["task_description"]
                      for word in task_description.split()[:5]):
                    similar_tasks.append(record)

        if not similar_tasks:
            return {}

        # Analyze similar tasks
        quality_scores = [t["result_quality"] for t in similar_tasks]
        execution_times = [t["execution_time"] for t in similar_tasks]
        token_usage = [t["tokens_used"] for t in similar_tasks]

        return {
            "similar_tasks_count": len(similar_tasks),
            "avg_quality_score": sum(quality_scores) / len(quality_scores),
            "avg_execution_time": sum(execution_times) / len(execution_times),
            "avg_token_usage": sum(token_usage) / len(token_usage),
            "recommendations": self._generate_recommendations(similar_tasks)
        }

    def _generate_recommendations(
        self,
        similar_tasks: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations based on similar tasks."""

        recommendations = []

        # Analyze common patterns
        if len(similar_tasks) >= 3:
            # Check for performance patterns
            times = [t["execution_time"] for t in similar_tasks]
            avg_time = sum(times) / len(times)

            if avg_time > 300:  # More than 5 minutes
                recommendations.append(
                    "Consider breaking task into smaller subtasks for better performance"
                )

            # Check for quality patterns
            qualities = [t["result_quality"] for t in similar_tasks]
            avg_quality = sum(qualities) / len(qualities)

            if avg_quality < 80:
                recommendations.append(
                    "Consider increasing iteration count for higher quality"
                )

        return recommendations

    async def optimize_prompts(self) -> Dict[str, Any]:
        """Optimize prompts based on history."""

        # Analyze which prompts lead to best results
        # This is a simplified implementation

        language_patterns = {}
        for language, pattern in self.patterns.items():
            if pattern["total_executions"] >= 10:  # Need enough data
                language_patterns[language] = {
                    "efficiency": pattern["avg_quality"] / max(pattern["avg_tokens"], 1),
                    "avg_quality": pattern["avg_quality"]
                }

        return {
            "optimization_type": "prompt_efficiency",
            "language_patterns": language_patterns,
            "recommendations": [
                f"Consider adjusting temperature for {lang} based on efficiency {data['efficiency']:.2f}"
                for lang, data in language_patterns.items()
            ]
        }

    async def optimize_tool_selection(self) -> Dict[str, Any]:
        """Optimize tool selection based on history."""

        # Count tool usage by language
        tool_usage = {}
        for record in self.execution_history:
            language = record["language"]
            tool_calls = record.get("tool_calls", 0)

            if language not in tool_usage:
                tool_usage[language] = {"total": 0, "count": 0}

            tool_usage[language]["total"] += tool_calls
            tool_usage[language]["count"] += 1

        # Calculate averages
        tool_recommendations = {}
        for language, usage in tool_usage.items():
            if usage["count"] > 0:
                avg_tools = usage["total"] / usage["count"]
                tool_recommendations[language] = {
                    "avg_tool_calls": avg_tools,
                    "recommendation": (
                        "Consider reducing tool calls" if avg_tools > 10 else
                        "Tool usage is optimal" if avg_tools > 5 else
                        "Consider increasing tool usage for better results"
                    )
                }

        return {
            "optimization_type": "tool_selection",
            "tool_usage_patterns": tool_recommendations
        }

    async def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""

        return {
            "total_learned_executions": len(self.execution_history),
            "language_patterns": self.patterns,
            "optimizations_stored": len(self.optimizations)
        }


class ErrorRecoverySystem:
    """System for handling and recovering from errors."""

    def __init__(self):
        self.error_history = []
        self.recovery_strategies = {
            "timeout": self._recover_from_timeout,
            "connection_error": self._recover_from_connection_error,
            "validation_error": self._recover_from_validation_error,
            "llm_error": self._recover_from_llm_error
        }
        self.recovery_stats = {strategy: {"attempts": 0, "successes": 0}
                              for strategy in self.recovery_strategies.keys()}

    async def attempt_recovery(
        self,
        task_description: str,
        requirements: List[str],
        language: str,
        error_message: str
    ) -> Dict[str, Any]:
        """Attempt to recover from an error."""

        # Determine error type
        error_type = self._classify_error(error_message)

        # Record error
        self.error_history.append({
            "error_type": error_type,
            "error_message": error_message,
            "task_description": task_description,
            "language": language,
            "timestamp": datetime.now().isoformat()
        })

        # Apply recovery strategy
        if error_type in self.recovery_strategies:
            self.recovery_stats[error_type]["attempts"] += 1

            recovery_result = await self.recovery_strategies[error_type](
                task_description, requirements, language, error_message
            )

            if recovery_result["success"]:
                self.recovery_stats[error_type]["successes"] += 1

            return recovery_result

        # Default recovery
        return {
            "success": False,
            "error_type": error_type,
            "recovery_method": "none",
            "message": "No recovery strategy available"
        }

    def _classify_error(self, error_message: str) -> str:
        """Classify error type from message."""

        error_lower = error_message.lower()

        if any(keyword in error_lower for keyword in ["timeout", "timed out"]):
            return "timeout"
        elif any(keyword in error_lower for keyword in ["connection", "connect", "socket"]):
            return "connection_error"
        elif any(keyword in error_lower for keyword in ["validation", "invalid", "validate"]):
            return "validation_error"
        elif any(keyword in error_lower for keyword in ["openai", "anthropic", "llm", "model"]):
            return "llm_error"
        else:
            return "unknown"

    async def _recover_from_timeout(
        self,
        task_description: str,
        requirements: List[str],
        language: str,
        error_message: str
    ) -> Dict[str, Any]:
        """Recover from timeout error."""

        return {
            "success": True,
            "error_type": "timeout",
            "recovery_method": "adjust_timeouts",
            "adjusted_params": {
                "llm_timeout": 120,  # Increase timeout
                "tool_timeout": 60
            },
            "message": "Increased timeouts for retry"
        }

    async def _recover_from_connection_error(
        self,
        task_description: str,
        requirements: List[str],
        language: str,
        error_message: str
    ) -> Dict[str, Any]:
        """Recover from connection error."""

        return {
            "success": True,
            "error_type": "connection_error",
            "recovery_method": "retry_with_backoff",
            "adjusted_params": {
                "max_retries": 5,
                "retry_delay": 2
            },
            "message": "Will retry with exponential backoff"
        }

    async def _recover_from_validation_error(
        self,
        task_description: str,
        requirements: List[str],
        language: str,
        error_message: str
    ) -> Dict[str, Any]:
        """Recover from validation error."""

        # Simplify requirements if validation fails
        simplified_requirements = []
        for req in requirements:
            # Keep only essential requirements
            if len(req.split()) <= 10:  # Short requirements
                simplified_requirements.append(req)

        if not simplified_requirements and requirements:
            # Keep first requirement if all were too long
            simplified_requirements = [requirements[0]]

        return {
            "success": True,
            "error_type": "validation_error",
            "recovery_method": "simplify_requirements",
            "adjusted_params": {
                "requirements": simplified_requirements
            },
            "message": "Simplified requirements for retry"
        }

    async def _recover_from_llm_error(
        self,
        task_description: str,
        requirements: List[str],
        language: str,
        error_message: str
    ) -> Dict[str, Any]:
        """Recover from LLM error."""

        return {
            "success": True,
            "error_type": "llm_error",
            "recovery_method": "adjust_llm_params",
            "adjusted_params": {
                "temperature": 0.3,  # Lower temperature for more deterministic
                "max_tokens": 2000   # Reduce token limit
            },
            "message": "Adjusted LLM parameters for retry"
        }

    def optimize_recovery_strategies(self) -> Dict[str, Any]:
        """Optimize recovery strategies based on history."""

        strategy_success_rates = {}
        for strategy, stats in self.recovery_stats.items():
            if stats["attempts"] > 0:
                success_rate = stats["successes"] / stats["attempts"]
                strategy_success_rates[strategy] = success_rate

        recommendations = []
        for strategy, success_rate in strategy_success_rates.items():
            if success_rate < 0.5:
                recommendations.append(
                    f"Consider improving {strategy} recovery (success rate: {success_rate:.1%})"
                )

        return {
            "optimization_type": "recovery_strategies",
            "success_rates": strategy_success_rates,
            "recommendations": recommendations
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get recovery statistics."""

        total_attempts = sum(stats["attempts"] for stats in self.recovery_stats.values())
        total_successes = sum(stats["successes"] for stats in self.recovery_stats.values())

        return {
            "total_errors": len(self.error_history),
            "recovery_attempts": total_attempts,
            "recovery_successes": total_successes,
            "overall_success_rate": total_successes / total_attempts if total_attempts > 0 else 0,
            "strategy_stats": self.recovery_stats
        }