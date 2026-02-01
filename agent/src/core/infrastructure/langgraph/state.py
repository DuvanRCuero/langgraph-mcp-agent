"""
State management for LangGraph agent.
Defines the state schema and state management utilities.
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from datetime import datetime
import operator
from enum import Enum


class AgentPhase(str, Enum):
    """Phases of the agent workflow."""
    INITIALIZED = "initialized"
    ANALYZING = "analyzing"
    RESEARCHING = "researching"
    DESIGNING = "designing"
    IMPLEMENTING = "implementing"
    TESTING = "testing"
    REVIEWING = "reviewing"
    OPTIMIZING = "optimizing"
    COMPLETED = "completed"
    FAILED = "failed"
    AWAITING_HUMAN_INPUT = "awaiting_human_input"


class AgentState(TypedDict, total=False):
    """
    State schema for LangGraph agent.
    TypedDict provides type hints for state fields.
    """

    # Task Information
    task_id: str
    task_description: str
    requirements: List[str]
    language: str
    framework: Optional[str]
    quality_level: str

    # Phase Tracking
    current_phase: AgentPhase
    previous_phase: Optional[AgentPhase]
    phase_history: List[Dict[str, Any]]

    # Generated Artifacts
    analysis_result: Optional[Dict[str, Any]]
    researched_docs: List[Dict[str, Any]]
    architecture_design: Optional[Dict[str, Any]]
    generated_code: List[Dict[str, Any]]
    generated_tests: List[Dict[str, Any]]
    quality_metrics: Optional[Dict[str, Any]]

    # Workflow Control
    iteration: int
    max_iterations: int
    needs_human_review: bool
    human_feedback: Optional[str]
    optimization_targets: List[str]

    # Context and Memory
    context: Dict[str, Any]
    memory: Annotated[List[Dict[str, Any]], operator.add]
    execution_history: List[Dict[str, Any]]

    # Performance Metrics
    start_time: float
    execution_time: float
    token_usage: Dict[str, int]
    tool_calls: int

    # Error Handling
    error: Optional[str]
    retry_count: int
    max_retries: int

    # Agent Configuration
    agent_config: Dict[str, Any]
    llm_config: Dict[str, Any]
    tool_config: Dict[str, Any]


class StateManager:
    """
    Manages agent state with validation and helper methods.
    Provides a clean interface for state manipulation.
    """

    def __init__(self, initial_state: Optional[Dict[str, Any]] = None):
        self.state: AgentState = self._initialize_state(initial_state or {})
        self.state_validator = StateValidator()

    def _initialize_state(self, initial_data: Dict[str, Any]) -> AgentState:
        """Initialize state with defaults."""

        defaults: AgentState = {
            "task_id": initial_data.get("task_id", ""),
            "task_description": initial_data.get("task_description", ""),
            "requirements": initial_data.get("requirements", []),
            "language": initial_data.get("language", "python"),
            "framework": initial_data.get("framework"),
            "quality_level": initial_data.get("quality_level", "excellent"),

            "current_phase": AgentPhase.INITIALIZED,
            "previous_phase": None,
            "phase_history": [],

            "analysis_result": None,
            "researched_docs": [],
            "architecture_design": None,
            "generated_code": [],
            "generated_tests": [],
            "quality_metrics": None,

            "iteration": 0,
            "max_iterations": 3,
            "needs_human_review": False,
            "human_feedback": None,
            "optimization_targets": [],

            "context": {},
            "memory": [],
            "execution_history": [],

            "start_time": datetime.now().timestamp(),
            "execution_time": 0.0,
            "token_usage": {"prompt": 0, "completion": 0, "total": 0},
            "tool_calls": 0,

            "error": None,
            "retry_count": 0,
            "max_retries": 3,

            "agent_config": initial_data.get("agent_config", {}),
            "llm_config": initial_data.get("llm_config", {}),
            "tool_config": initial_data.get("tool_config", {})
        }

        # Merge with initial data
        for key, value in initial_data.items():
            if key in defaults:
                defaults[key] = value

        return defaults

    def update_phase(self, new_phase: AgentPhase, metadata: Optional[Dict[str, Any]] = None):
        """Update agent phase with history tracking."""

        phase_entry = {
            "phase": new_phase.value,
            "timestamp": datetime.now().isoformat(),
            "iteration": self.state["iteration"],
            "metadata": metadata or {}
        }

        self.state["previous_phase"] = self.state["current_phase"]
        self.state["current_phase"] = new_phase
        self.state["phase_history"].append(phase_entry)

        # Update execution time
        self.state["execution_time"] = datetime.now().timestamp() - self.state["start_time"]

    def add_memory(self, memory_entry: Dict[str, Any]):
        """Add entry to agent memory."""
        memory_entry["timestamp"] = datetime.now().isoformat()
        self.state["memory"].append(memory_entry)

    def add_execution_record(self, record: Dict[str, Any]):
        """Add execution record."""
        record["timestamp"] = datetime.now().isoformat()
        self.state["execution_history"].append(record)

    def update_token_usage(self, prompt_tokens: int, completion_tokens: int):
        """Update token usage metrics."""
        self.state["token_usage"]["prompt"] += prompt_tokens
        self.state["token_usage"]["completion"] += completion_tokens
        self.state["token_usage"]["total"] += prompt_tokens + completion_tokens

    def increment_tool_calls(self):
        """Increment tool call counter."""
        self.state["tool_calls"] += 1

    def increment_iteration(self):
        """Increment iteration counter."""
        self.state["iteration"] += 1

    def set_error(self, error: str, fatal: bool = False):
        """Set error state."""
        self.state["error"] = error
        if fatal:
            self.update_phase(AgentPhase.FAILED)

    def clear_error(self):
        """Clear error state."""
        self.state["error"] = None

    def should_retry(self) -> bool:
        """Check if agent should retry current operation."""
        return self.state["retry_count"] < self.state["max_retries"]

    def increment_retry(self):
        """Increment retry counter."""
        self.state["retry_count"] += 1

    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of current state."""
        return {
            "task_id": self.state["task_id"],
            "current_phase": self.state["current_phase"].value,
            "iteration": self.state["iteration"],
            "execution_time_seconds": self.state["execution_time"],
            "token_usage": self.state["token_usage"],
            "tool_calls": self.state["tool_calls"],
            "error": self.state["error"],
            "needs_human_review": self.state["needs_human_review"]
        }

    def validate_state(self) -> Tuple[bool, List[str]]:
        """Validate current state."""
        return self.state_validator.validate(self.state)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        result = {}
        for key, value in self.state.items():
            if isinstance(value, Enum):
                result[key] = value.value
            else:
                result[key] = value
        return result

    def from_dict(self, data: Dict[str, Any]):
        """Load state from dictionary."""
        for key, value in data.items():
            if key in self.state:
                # Handle Enum conversion
                if key in ["current_phase", "previous_phase"] and value:
                    self.state[key] = AgentPhase(value)
                else:
                    self.state[key] = value


class StateValidator:
    """Validates agent state integrity."""

    def validate(self, state: AgentState) -> Tuple[bool, List[str]]:
        """Validate state and return issues."""

        issues = []

        # Check required fields
        if not state.get("task_description"):
            issues.append("Missing task_description")

        if not state.get("language"):
            issues.append("Missing language")

        # Check phase consistency
        current_phase = state.get("current_phase")
        if current_phase and not isinstance(current_phase, AgentPhase):
            issues.append(f"Invalid current_phase: {current_phase}")

        # Check iteration bounds
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 3)
        if iteration > max_iterations:
            issues.append(f"Iteration {iteration} exceeds max {max_iterations}")

        # Check token usage
        token_usage = state.get("token_usage", {})
        if token_usage.get("total", 0) > 1000000:  # 1M token limit
            issues.append("Token usage exceeds limit")

        return len(issues) == 0, issues