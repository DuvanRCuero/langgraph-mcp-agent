"""
Core domain entities representing the programming agent's business logic.
Built with Domain-Driven Design principles and elite engineering standards.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
from enum import Enum
import uuid
import hashlib
from pathlib import Path

from pydantic import BaseModel, Field, validator


class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    ANALYZING = "analyzing"
    RESEARCHING = "researching"
    DESIGNING = "designing"
    IMPLEMENTING = "implementing"
    TESTING = "testing"
    REVIEWING = "reviewing"
    OPTIMIZING = "optimizing"
    COMPLETED = "completed"
    FAILED = "failed"
    AWAITING_REVIEW = "awaiting_review"


class CodeQualityLevel(str, Enum):
    """Code quality levels following elite engineering standards."""
    ELITE = "elite"  # Top 0.1% - Production excellence
    EXCELLENT = "excellent"  # Top 5% - Production ready
    GOOD = "good"  # Good enough for production
    FAIR = "fair"  # Needs improvement
    POOR = "poor"  # Not production ready


class ArchitectureStyle(str, Enum):
    """Software architecture styles."""
    CLEAN_ARCHITECTURE = "clean_architecture"
    HEXAGONAL = "hexagonal"
    ONION = "onion"
    MICROSERVICES = "microservices"
    EVENT_DRIVEN = "event_driven"
    CQRS = "cqrs"
    LAYERED = "layered"


class ProgrammingLanguage(str, Enum):
    """Supported programming languages."""
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    KOTLIN = "kotlin"
    SWIFT = "swift"
    CPP = "cpp"
    CSHARP = "csharp"


@dataclass
class CodeSnippet:
    """Value object representing a code snippet with metadata."""
    language: ProgrammingLanguage
    content: str
    framework: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)

    @property
    def lines_of_code(self) -> int:
        """Calculate lines of code (excluding empty lines and comments)."""
        lines = self.content.strip().split('\n')
        code_lines = [
            line for line in lines
            if line.strip() and not line.strip().startswith(('#', '//', '/*', '*'))
        ]
        return len(code_lines)

    @property
    def complexity_score(self) -> float:
        """Calculate simplified complexity score."""
        # In production, integrate with tools like radon or lizard
        loc = self.lines_of_code
        return min(loc / 50, 10.0)  # Normalized score

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate code snippet and return issues."""
        issues = []

        if not self.content.strip():
            issues.append("Code content is empty")

        if self.lines_of_code == 0:
            issues.append("No executable code found")

        # Check for common issues
        if "TODO" in self.content.upper():
            issues.append("Contains TODO comments")

        if "FIXME" in self.content.upper():
            issues.append("Contains FIXME comments")

        return len(issues) == 0, issues

    def compute_hash(self) -> str:
        """Compute hash for code snippet."""
        content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]
        return f"{self.language.value}_{content_hash}"


@dataclass
class TestSuite:
    """Test suite with comprehensive coverage."""
    unit_tests: List[CodeSnippet] = field(default_factory=list)
    integration_tests: List[CodeSnippet] = field(default_factory=list)
    e2e_tests: List[CodeSnippet] = field(default_factory=list)
    coverage_percentage: float = 0.0
    passed_tests: int = 0
    total_tests: int = 0

    @property
    def pass_rate(self) -> float:
        """Calculate test pass rate."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100

    @property
    def total_test_lines(self) -> int:
        """Calculate total lines of test code."""
        return sum(test.lines_of_code for test in self.unit_tests +
                   self.integration_tests + self.e2e_tests)


@dataclass
class DocumentationReference:
    """Reference to documentation from MCP server."""
    title: str
    url: str
    content: str
    source: str
    relevance_score: float
    last_updated: datetime
    doc_type: str
    tags: List[str] = field(default_factory=list)

    def is_recent(self, days_threshold: int = 30) -> bool:
        """Check if documentation is recent."""
        age_days = (datetime.now() - self.last_updated).days
        return age_days <= days_threshold

    def extract_code_examples(self) -> List[str]:
        """Extract code examples from documentation content."""
        # Simple extraction - in production use more sophisticated parsing
        import re

        code_patterns = [
            r'```(?:\w+)?\n(.*?)\n```',  # Markdown code blocks
            r'<code>(.*?)</code>',  # HTML code tags
        ]

        examples = []
        for pattern in code_patterns:
            matches = re.findall(pattern, self.content, re.DOTALL)
            examples.extend(matches)

        return examples


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for elite engineering."""
    complexity_score: float = 0.0  # Cyclomatic complexity
    test_coverage: float = 0.0  # Test coverage percentage
    documentation_score: float = 0.0  # Documentation completeness
    performance_score: float = 0.0  # Performance benchmarks
    security_score: float = 0.0  # Security assessment
    maintainability_score: float = 0.0  # Maintainability index
    reliability_score: float = 0.0  # Reliability metrics
    total_issues: int = 0  # Total issues found

    def calculate_overall_score(self) -> float:
        """Calculate weighted overall quality score."""
        weights = {
            'test_coverage': 0.25,
            'complexity_score': 0.20,
            'maintainability_score': 0.20,
            'security_score': 0.15,
            'documentation_score': 0.10,
            'performance_score': 0.10
        }

        # Normalize scores (higher is better for all)
        normalized_complexity = max(0, 10 - self.complexity_score) / 10

        score = (
                weights['test_coverage'] * (self.test_coverage / 100) +
                weights['complexity_score'] * normalized_complexity +
                weights['maintainability_score'] * (self.maintainability_score / 100) +
                weights['security_score'] * (self.security_score / 100) +
                weights['documentation_score'] * (self.documentation_score / 100) +
                weights['performance_score'] * (self.performance_score / 100)
        )

        return min(score * 100, 100.0)

    def to_elite_level(self) -> CodeQualityLevel:
        """Convert metrics to elite quality level."""
        overall = self.calculate_overall_score()

        if overall >= 95 and self.test_coverage >= 95:
            return CodeQualityLevel.ELITE
        elif overall >= 85 and self.test_coverage >= 85:
            return CodeQualityLevel.EXCELLENT
        elif overall >= 75 and self.test_coverage >= 75:
            return CodeQualityLevel.GOOD
        elif overall >= 60:
            return CodeQualityLevel.FAIR
        else:
            return CodeQualityLevel.POOR


@dataclass
class ArchitectureDesign:
    """Software architecture design."""
    style: ArchitectureStyle
    components: List['ComponentDesign'] = field(default_factory=list)
    data_flow: str = ""
    api_spec: Optional[Dict[str, Any]] = None
    database_schema: Optional[Dict[str, Any]] = None
    deployment_diagram: str = ""

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate architecture design."""
        issues = []

        if not self.components:
            issues.append("No components defined")

        if not self.data_flow:
            issues.append("No data flow description")

        # Check for circular dependencies
        if self._has_circular_dependencies():
            issues.append("Circular dependencies detected")

        return len(issues) == 0, issues

    def _has_circular_dependencies(self) -> bool:
        """Check for circular dependencies between components."""
        # Simplified implementation
        visited = set()

        for component in self.components:
            if component.name in visited:
                continue
            if self._dfs_check(component.name, set(), {}):
                return True

        return False

    def _dfs_check(self, component_name: str, path: Set[str], visited: Dict[str, bool]) -> bool:
        """DFS for cycle detection."""
        if component_name in path:
            return True

        if visited.get(component_name, False):
            return False

        path.add(component_name)
        visited[component_name] = True

        component = next((c for c in self.components if c.name == component_name), None)
        if component:
            for dep in component.dependencies:
                if self._dfs_check(dep, path.copy(), visited):
                    return True

        return False


@dataclass
class ComponentDesign:
    """Design for a single component."""
    name: str
    type: str  # "service", "module", "class", "function"
    responsibilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    interfaces: List[str] = field(default_factory=list)
    persistence: Optional[str] = None
    concurrency_model: str = "synchronous"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type,
            "responsibilities": self.responsibilities,
            "dependencies": self.dependencies,
            "interfaces": self.interfaces,
            "persistence": self.persistence,
            "concurrency_model": self.concurrency_model
        }


@dataclass
class ProgrammingTask:
    """
    Aggregate root for programming tasks.
    Represents a complete programming task with full lifecycle.
    """

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    requirements: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    language: ProgrammingLanguage = ProgrammingLanguage.PYTHON
    framework: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Generated artifacts
    analysis: Optional['TaskAnalysis'] = None
    documentation: List[DocumentationReference] = field(default_factory=list)
    architecture: Optional[ArchitectureDesign] = None
    generated_code: List[CodeSnippet] = field(default_factory=list)
    tests: Optional[TestSuite] = None
    quality_metrics: Optional[QualityMetrics] = None
    quality_level: Optional[CodeQualityLevel] = None

    # Workflow tracking
    iterations: int = 0
    feedback: List[str] = field(default_factory=list)
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)

    def update_status(self, new_status: TaskStatus):
        """Update task status with validation."""
        valid_transitions = {
            TaskStatus.PENDING: [TaskStatus.ANALYZING],
            TaskStatus.ANALYZING: [TaskStatus.RESEARCHING],
            TaskStatus.RESEARCHING: [TaskStatus.DESIGNING],
            TaskStatus.DESIGNING: [TaskStatus.IMPLEMENTING],
            TaskStatus.IMPLEMENTING: [TaskStatus.TESTING],
            TaskStatus.TESTING: [TaskStatus.REVIEWING],
            TaskStatus.REVIEWING: [TaskStatus.OPTIMIZING, TaskStatus.COMPLETED],
            TaskStatus.OPTIMIZING: [TaskStatus.REVIEWING, TaskStatus.COMPLETED],
            TaskStatus.AWAITING_REVIEW: [TaskStatus.OPTIMIZING, TaskStatus.COMPLETED],
        }

        if new_status not in valid_transitions.get(self.status, []):
            raise ValueError(f"Invalid transition from {self.status} to {new_status}")

        self.status = new_status
        self.updated_at = datetime.now()

    def add_code_snippet(self, snippet: CodeSnippet):
        """Add code snippet with validation."""
        is_valid, issues = snippet.validate()
        if not is_valid:
            raise ValueError(f"Invalid code snippet: {issues}")

        self.generated_code.append(snippet)
        self.updated_at = datetime.now()

    def add_documentation(self, doc_ref: DocumentationReference):
        """Add documentation reference."""
        self.documentation.append(doc_ref)
        # Keep only most relevant docs (sorted by relevance)
        self.documentation.sort(key=lambda x: x.relevance_score, reverse=True)
        self.documentation = self.documentation[:20]  # Keep top 20

    def assess_quality(self, metrics: QualityMetrics):
        """Assess and update quality metrics."""
        self.quality_metrics = metrics
        self.quality_level = metrics.to_elite_level()
        self.updated_at = datetime.now()

    def record_iteration(self, iteration_data: Dict[str, Any]):
        """Record an iteration for tracking."""
        self.iterations += 1
        iteration_data.update({
            "iteration": self.iterations,
            "timestamp": datetime.now().isoformat(),
            "status": self.status.value
        })
        self.optimization_history.append(iteration_data)

    def add_feedback(self, feedback_text: str):
        """Add human or automated feedback."""
        self.feedback.append(feedback_text)
        self.updated_at = datetime.now()

    @property
    def is_complete(self) -> bool:
        """Check if task is complete."""
        return self.status in [TaskStatus.COMPLETED]

    @property
    def total_lines_of_code(self) -> int:
        """Calculate total lines of code."""
        return sum(snippet.lines_of_code for snippet in self.generated_code)

    @property
    def test_to_code_ratio(self) -> float:
        """Calculate test to code ratio."""
        if not self.tests or self.total_lines_of_code == 0:
            return 0.0

        test_loc = self.tests.total_test_lines
        return (test_loc / self.total_lines_of_code) * 100

    def to_summary_dict(self) -> Dict[str, Any]:
        """Create summary dictionary."""
        return {
            "task_id": self.task_id,
            "title": self.title,
            "status": self.status.value,
            "language": self.language.value,
            "framework": self.framework,
            "iterations": self.iterations,
            "total_lines": self.total_lines_of_code,
            "test_coverage": self.quality_metrics.test_coverage if self.quality_metrics else 0,
            "quality_level": self.quality_level.value if self.quality_level else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class TaskAnalysis:
    """Analysis of a programming task."""
    complexity: str  # "simple", "medium", "complex"
    estimated_effort_hours: float
    required_skills: List[str]
    dependencies: List[str]
    risk_factors: List[str]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "complexity": self.complexity,
            "estimated_effort_hours": self.estimated_effort_hours,
            "required_skills": self.required_skills,
            "dependencies": self.dependencies,
            "risk_factors": self.risk_factors,
            "recommendations": self.recommendations
        }


@dataclass
class ProjectContext:
    """Context for multi-file projects."""
    project_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    root_path: Path
    tasks: List[ProgrammingTask] = field(default_factory=list)
    dependencies: Dict[str, str] = field(default_factory=dict)
    config_files: List[Path] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def add_task(self, task: ProgrammingTask):
        """Add task to project."""
        self.tasks.append(task)

    def get_task_by_id(self, task_id: str) -> Optional[ProgrammingTask]:
        """Get task by ID."""
        return next((task for task in self.tasks if task.task_id == task_id), None)

    @property
    def total_lines_of_code(self) -> int:
        """Calculate total lines of code across all tasks."""
        return sum(task.total_lines_of_code for task in self.tasks)

    @property
    def overall_quality_score(self) -> float:
        """Calculate overall quality score."""
        if not self.tasks:
            return 0.0

        scores = [
            task.quality_metrics.calculate_overall_score()
            for task in self.tasks
            if task.quality_metrics
        ]

        if not scores:
            return 0.0

        return sum(scores) / len(scores)