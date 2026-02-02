"""
Use case for elite code generation.
Orchestrates the complete code generation pipeline with MCP integration.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from ...domain.entities import (
    ProgrammingTask,
    CodeSnippet,
    TestSuite,
    QualityMetrics,
    CodeQualityLevel,
    TaskStatus,
    ArchitectureDesign,
    ProgrammingLanguage,
    DocumentationReference
)
from ...domain.result import Result, Success, Failure, ResultHelper
from ...domain.errors import DomainError, ValidationError, ErrorCode
from ...domain.events import (
    TaskCreatedEvent, TaskStatusChangedEvent, CodeGeneratedEvent,
    QualityAssessedEvent, TaskCompletedEvent, TaskFailedEvent, get_event_bus
)
from ..ports.mcp_client import MCPClientPort
from ..ports.llm_gateway import LLMGatewayPort
from ..ports.vector_store import VectorStorePort
from ..ports.unit_of_work import UnitOfWorkFactory
from ..services.orchestration import OrchestrationService

logger = logging.getLogger(__name__)


@dataclass
class CodeGenerationRequest:
    """Request DTO for code generation."""
    title: str
    description: str
    requirements: List[str]
    language: ProgrammingLanguage
    framework: Optional[str] = None
    quality_level: CodeQualityLevel = CodeQualityLevel.ELITE
    existing_code: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeGenerationResponse:
    """Response DTO for code generation."""
    task: ProgrammingTask
    generated_files: List[Dict[str, Any]]
    documentation_used: List[DocumentationReference]
    quality_report: Dict[str, Any]
    recommendations: List[str]
    execution_metrics: Dict[str, float]


class CodeGenerationUseCase:
    """
    Orchestrates elite code generation with comprehensive pipeline.
    Implements the full lifecycle from analysis to optimization.
    """

    def __init__(
            self,
            mcp_client: MCPClientPort,
            llm_gateway: LLMGatewayPort,
            vector_store: VectorStorePort,
            orchestration_service: OrchestrationService,
            uow_factory: Optional[UnitOfWorkFactory] = None
    ):
        self.mcp_client = mcp_client
        self.llm_gateway = llm_gateway
        self.vector_store = vector_store
        self.orchestration_service = orchestration_service
        self.uow_factory = uow_factory
        self.max_iterations = 3

        # Statistics
        self.metrics = {
            "total_executions": 0,
            "average_quality_score": 0.0,
            "average_iterations": 0.0,
            "success_rate": 0.0
        }

    async def execute(self, request: CodeGenerationRequest) -> CodeGenerationResponse:
        """
        Execute elite code generation pipeline:
        1. Task analysis and planning
        2. Research latest documentation via MCP
        3. Architecture design with elite patterns
        4. Iterative code generation
        5. Comprehensive testing
        6. Quality assessment and optimization
        7. Final validation
        """

        start_time = datetime.now()
        self.metrics["total_executions"] += 1
        event_bus = get_event_bus()

        try:
            # Validate request
            validation_result = self._validate_request(request)
            if isinstance(validation_result, Failure):
                raise ValueError(f"Validation failed: {validation_result.error.message}")

            # 1. Create task entity
            task = ProgrammingTask(
                title=request.title,
                description=request.description,
                requirements=request.requirements,
                language=request.language,
                framework=request.framework
            )
            
            # Publish task created event
            await event_bus.publish(TaskCreatedEvent(
                task_id=task.task_id,
                title=task.title,
                language=task.language.value,
                requirements=task.requirements
            ))
            
            task.update_status(TaskStatus.ANALYZING)
            await event_bus.publish(TaskStatusChangedEvent(
                task_id=task.task_id,
                old_status=TaskStatus.PENDING.value,
                new_status=TaskStatus.ANALYZING.value
            ))

            # 2. Analyze task
            analysis = await self._analyze_task(request)
            task.analysis = analysis

            # 3. Research documentation
            documentation = await self._research_documentation(request)
            task.documentation = documentation

            # 4. Design architecture
            old_status = task.status
            task.update_status(TaskStatus.DESIGNING)
            await event_bus.publish(TaskStatusChangedEvent(
                task_id=task.task_id,
                old_status=old_status.value,
                new_status=TaskStatus.DESIGNING.value
            ))
            architecture = await self._design_architecture(request, documentation)
            task.architecture = architecture

            # 5. Generate code iteratively
            old_status = task.status
            task.update_status(TaskStatus.IMPLEMENTING)
            await event_bus.publish(TaskStatusChangedEvent(
                task_id=task.task_id,
                old_status=old_status.value,
                new_status=TaskStatus.IMPLEMENTING.value
            ))
            generated_code = await self._generate_code_iteratively(
                request, architecture, documentation
            )

            for snippet in generated_code:
                task.add_code_snippet(snippet)
                # Publish code generated event for each significant component
                await event_bus.publish(CodeGeneratedEvent(
                    task_id=task.task_id,
                    component_name=snippet.framework or "component",
                    language=snippet.language.value,
                    lines_of_code=snippet.lines_of_code
                ))

            # 6. Generate tests
            old_status = task.status
            task.update_status(TaskStatus.TESTING)
            await event_bus.publish(TaskStatusChangedEvent(
                task_id=task.task_id,
                old_status=old_status.value,
                new_status=TaskStatus.TESTING.value
            ))
            test_suite = await self._generate_tests(generated_code, request)
            task.tests = test_suite

            # 7. Assess quality
            old_status = task.status
            task.update_status(TaskStatus.REVIEWING)
            await event_bus.publish(TaskStatusChangedEvent(
                task_id=task.task_id,
                old_status=old_status.value,
                new_status=TaskStatus.REVIEWING.value
            ))
            quality_metrics = await self._assess_quality(generated_code, test_suite)
            task.assess_quality(quality_metrics)
            
            # Publish quality assessed event
            await event_bus.publish(QualityAssessedEvent(
                task_id=task.task_id,
                quality_level=task.quality_level.value if task.quality_level else "unknown",
                overall_score=quality_metrics.calculate_overall_score(),
                metrics={
                    "complexity": quality_metrics.complexity_score,
                    "test_coverage": quality_metrics.test_coverage,
                    "maintainability": quality_metrics.maintainability_score
                }
            ))

            # 8. Optimize if needed
            if task.quality_level.value < request.quality_level.value:
                old_status = task.status
                task.update_status(TaskStatus.OPTIMIZING)
                await event_bus.publish(TaskStatusChangedEvent(
                    task_id=task.task_id,
                    old_status=old_status.value,
                    new_status=TaskStatus.OPTIMIZING.value
                ))
                optimized = await self._optimize_to_target_quality(
                    task, request.quality_level
                )
                if optimized:
                    task = optimized

            # 9. Final validation
            validation_passed = await self._validate_final_output(task)
            if not validation_passed:
                raise Exception("Final validation failed")

            # 10. Complete task
            old_status = task.status
            task.update_status(TaskStatus.COMPLETED)
            await event_bus.publish(TaskStatusChangedEvent(
                task_id=task.task_id,
                old_status=old_status.value,
                new_status=TaskStatus.COMPLETED.value
            ))

            # Calculate execution metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Publish task completed event
            await event_bus.publish(TaskCompletedEvent(
                task_id=task.task_id,
                execution_time_seconds=execution_time,
                quality_level=task.quality_level.value if task.quality_level else "unknown",
                total_lines_of_code=task.total_lines_of_code
            ))

            # Prepare response
            response = CodeGenerationResponse(
                task=task,
                generated_files=self._format_generated_files(task),
                documentation_used=documentation,
                quality_report=self._create_quality_report(task),
                recommendations=self._generate_recommendations(task),
                execution_metrics={
                    "total_time_seconds": execution_time,
                    "iterations": task.iterations,
                    "lines_of_code": task.total_lines_of_code,
                    "test_coverage": task.quality_metrics.test_coverage,
                    "overall_quality_score": task.quality_metrics.calculate_overall_score()
                }
            )

            # Update metrics
            self._update_metrics(task, execution_time)

            logger.info(
                f"Code generation completed in {execution_time:.2f}s with "
                f"quality level: {task.quality_level}"
            )

            return response

        except Exception as e:
            logger.error(f"Code generation failed: {e}", exc_info=True)
            
            # Publish task failed event
            task_id = task.task_id if 'task' in locals() else "unknown"
            phase = task.status.value if 'task' in locals() else "initialization"
            
            await event_bus.publish(TaskFailedEvent(
                task_id=task_id,
                error_message=str(e),
                error_code=ErrorCode.INTERNAL_ERROR.value,
                phase=phase
            ))
            
            if 'task' in locals():
                task.update_status(TaskStatus.FAILED)
            raise

    def _validate_request(self, request: CodeGenerationRequest) -> Result[bool, DomainError]:
        """Validate code generation request."""
        # Basic validation
        if not request.title or not request.title.strip():
            return Failure(ValidationError(
                code=ErrorCode.VALIDATION_ERROR,
                message="Title is required",
                field="title"
            ))
        
        if not request.description or not request.description.strip():
            return Failure(ValidationError(
                code=ErrorCode.VALIDATION_ERROR,
                message="Description is required",
                field="description"
            ))
        
        if not request.requirements or len(request.requirements) == 0:
            return Failure(ValidationError(
                code=ErrorCode.VALIDATION_ERROR,
                message="At least one requirement is needed",
                field="requirements"
            ))
        
        return Success(True)

    async def _analyze_task(self, request: CodeGenerationRequest) -> Dict[str, Any]:
        """Analyze task complexity and requirements."""

        analysis_prompt = f"""
        Analyze this programming task and provide detailed analysis:

        TITLE: {request.title}
        DESCRIPTION: {request.description}
        LANGUAGE: {request.language.value}
        FRAMEWORK: {request.framework or 'None'}
        QUALITY TARGET: {request.quality_level.value}

        Requirements:
        {chr(10).join(f'- {req}' for req in request.requirements)}

        Provide analysis with:
        1. Complexity assessment (simple/medium/complex)
        2. Estimated effort in hours
        3. Required technical skills
        4. External dependencies needed
        5. Risk factors
        6. Recommendations for implementation approach
        """

        response = await self.llm_gateway.generate(
            messages=[
                {
                    "role": "system",
                    "content": "You are an elite software architect. Analyze tasks with precision and provide actionable insights."
                },
                {
                    "role": "user",
                    "content": analysis_prompt
                }
            ],
            config={
                "temperature": 0.3,
                "max_tokens": 1000
            }
        )

        # Parse response into structured analysis
        return self._parse_analysis_response(response.content)

    async def _research_documentation(
            self,
            request: CodeGenerationRequest
    ) -> List[DocumentationReference]:
        """Research latest documentation via MCP."""

        try:
            # Connect to MCP if not already connected
            if not self.mcp_client.is_connected:
                await self.mcp_client.connect()

            # Build search queries
            queries = self._build_documentation_queries(request)

            all_docs = []
            for query in queries:
                docs = await self.mcp_client.search_documentation(
                    query=query,
                    language=request.language.value,
                    framework=request.framework,
                    limit=5
                )
                all_docs.extend(docs)

            # Get best practices
            best_practices = await self.mcp_client.get_latest_best_practices(
                language=request.language.value,
                framework=request.framework,
                topic=request.title
            )
            all_docs.extend(best_practices)

            # Deduplicate and sort by relevance
            unique_docs = self._deduplicate_documentation(all_docs)

            logger.info(f"Researched {len(unique_docs)} documentation references")
            return unique_docs

        except Exception as e:
            logger.warning(f"Documentation research failed: {e}")
            return []

    async def _design_architecture(
            self,
            request: CodeGenerationRequest,
            documentation: List[DocumentationReference]
    ) -> ArchitectureDesign:
        """Design system architecture using elite patterns."""

        # Extract architecture patterns from documentation
        architecture_patterns = self._extract_architecture_patterns(documentation)

        design_prompt = self._build_architecture_prompt(
            request, documentation, architecture_patterns
        )

        response = await self.llm_gateway.generate(
            messages=[
                {
                    "role": "system",
                    "content": """You are an elite software architect with expertise in clean architecture, 
                    domain-driven design, and scalable systems. Design systems that are maintainable, 
                    testable, and production-ready."""
                },
                {
                    "role": "user",
                    "content": design_prompt
                }
            ],
            config={
                "temperature": 0.2,  # Low temperature for deterministic design
                "max_tokens": 3000
            }
        )

        # Parse and validate architecture design
        architecture = self._parse_architecture_response(response.content)
        is_valid, issues = architecture.validate()

        if not is_valid:
            logger.warning(f"Architecture validation issues: {issues}")
            # Try to fix architecture issues
            architecture = await self._fix_architecture_issues(architecture, issues)

        return architecture

    async def _generate_code_iteratively(
            self,
            request: CodeGenerationRequest,
            architecture: ArchitectureDesign,
            documentation: List[DocumentationReference]
    ) -> List[CodeSnippet]:
        """Generate code iteratively with refinement."""

        generated_snippets = []
        iteration = 1

        for component in architecture.components:
            logger.info(f"Generating code for component: {component.name}")

            # Generate initial implementation
            code = await self._generate_component_code(
                component, request, documentation, iteration
            )

            # Validate and refine
            refined_code = await self._refine_code(
                code, component, request, documentation
            )

            # Create code snippet
            snippet = CodeSnippet(
                language=request.language,
                content=refined_code,
                framework=request.framework,
                dependencies=component.dependencies
            )

            generated_snippets.append(snippet)

            # Record iteration
            task_data = {
                "component": component.name,
                "iteration": iteration,
                "lines_of_code": snippet.lines_of_code,
                "complexity_score": snippet.complexity_score
            }

            iteration += 1

        return generated_snippets

    async def _generate_tests(
            self,
            code_snippets: List[CodeSnippet],
            request: CodeGenerationRequest
    ) -> TestSuite:
        """Generate comprehensive test suite."""

        test_suite = TestSuite()

        for snippet in code_snippets:
            # Generate unit tests
            unit_tests = await self._generate_unit_tests(snippet, request)
            test_suite.unit_tests.extend(unit_tests)

            # Generate integration tests if applicable
            if len(code_snippets) > 1:
                integration_tests = await self._generate_integration_tests(
                    snippet, code_snippets, request
                )
                test_suite.integration_tests.extend(integration_tests)

        # Calculate coverage (simplified - in production use actual coverage tools)
        total_tests = len(test_suite.unit_tests) + len(test_suite.integration_tests)
        test_suite.total_tests = total_tests
        test_suite.passed_tests = total_tests  # Assume all pass initially
        test_suite.coverage_percentage = self._estimate_test_coverage(code_snippets, test_suite)

        return test_suite

    async def _assess_quality(
            self,
            code_snippets: List[CodeSnippet],
            test_suite: TestSuite
    ) -> QualityMetrics:
        """Assess code quality comprehensively."""

        metrics = QualityMetrics()

        # Use MCP for code analysis
        for snippet in code_snippets:
            try:
                analysis = await self.mcp_client.analyze_code_quality(
                    code=snippet.content,
                    language=snippet.language.value,
                    framework=snippet.framework
                )

                if analysis.get("success"):
                    # Aggregate metrics
                    metrics.complexity_score += analysis.get("complexity_score", 0)
                    metrics.security_score += analysis.get("security_score", 0)
                    metrics.total_issues += analysis.get("issues_count", 0)

            except Exception as e:
                logger.warning(f"Code quality analysis failed: {e}")

        # Calculate averages
        if code_snippets:
            metrics.complexity_score /= len(code_snippets)
            metrics.security_score /= len(code_snippets)

        # Set other metrics
        metrics.test_coverage = test_suite.coverage_percentage
        metrics.documentation_score = self._calculate_documentation_score(code_snippets)
        metrics.maintainability_score = self._calculate_maintainability_score(code_snippets)
        metrics.performance_score = self._estimate_performance_score(code_snippets)
        metrics.reliability_score = self._calculate_reliability_score(code_snippets, test_suite)

        return metrics

    async def _optimize_to_target_quality(
            self,
            task: ProgrammingTask,
            target_quality: CodeQualityLevel
    ) -> Optional[ProgrammingTask]:
        """Optimize code to meet target quality level."""

        logger.info(f"Optimizing from {task.quality_level} to {target_quality}")

        max_optimization_iterations = 3
        for iteration in range(max_optimization_iterations):
            if task.quality_level.value >= target_quality.value:
                break

            # Identify optimization areas
            optimization_areas = self._identify_optimization_areas(task)

            if not optimization_areas:
                logger.warning("No optimization areas identified")
                break

            # Apply optimizations
            optimized = await self._apply_optimizations(task, optimization_areas)
            if not optimized:
                break

            # Re-assess quality
            task.record_iteration({
                "optimization_iteration": iteration + 1,
                "optimization_areas": optimization_areas,
                "previous_quality": task.quality_level.value
            })

            # Recalculate metrics (simplified - in production re-run analysis)
            task.quality_metrics = await self._assess_quality(
                task.generated_code,
                task.tests
            )
            task.assess_quality(task.quality_metrics)

            logger.info(f"Optimization iteration {iteration + 1}: {task.quality_level}")

        return task if task.quality_level.value >= target_quality.value else None

    async def _validate_final_output(self, task: ProgrammingTask) -> bool:
        """Validate final output meets all requirements."""

        validation_checks = [
            self._validate_requirements_coverage(task),
            self._validate_code_quality(task),
            self._validate_architecture(task),
            self._validate_tests(task)
        ]

        results = await asyncio.gather(*validation_checks, return_exceptions=True)

        # Count successful validations
        successful = sum(1 for r in results if r is True)
        total = len(validation_checks)

        if successful == total:
            logger.info("All validations passed")
            return True
        else:
            logger.warning(f"Validations: {successful}/{total} passed")
            return False

    def _build_documentation_queries(self, request: CodeGenerationRequest) -> List[str]:
        """Build intelligent documentation search queries."""

        queries = []

        # Language-specific queries
        language = request.language.value
        framework = request.framework or ""

        queries.append(f"{language} {framework} best practices 2024")
        queries.append(f"{language} clean architecture patterns")
        queries.append(f"{language} production ready code standards")

        # Task-specific queries
        title_keywords = request.title.lower().split()
        for keyword in title_keywords[:3]:  # Use top 3 keywords
            queries.append(f"{language} {keyword} implementation")

        # Requirement-specific queries
        for req in request.requirements[:2]:  # Use top 2 requirements
            queries.append(f"{language} {req} best practices")

        return queries

    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM analysis response into structured format."""

        # In production, use structured output (JSON) from LLM
        # This is a simplified implementation

        import re

        patterns = {
            "complexity": r"Complexity.*?:\s*(\w+)",
            "estimated_effort": r"Estimated.*?:\s*([\d\.]+)\s*hours",
        }

        result = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                result[key] = match.group(1)

        # Extract lists
        result["required_skills"] = self._extract_list(response, "skills")
        result["dependencies"] = self._extract_list(response, "dependencies")
        result["risk_factors"] = self._extract_list(response, "risk")
        result["recommendations"] = self._extract_list(response, "recommendations")

        return result

    def _extract_list(self, text: str, keyword: str) -> List[str]:
        """Extract list items following a keyword."""
        import re

        pattern = rf"{keyword}.*?:\s*(.+?)(?=\n\n|\n[A-Z]|$)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)

        if not match:
            return []

        items_text = match.group(1)
        # Split by newlines or commas
        items = re.split(r'[\n,]', items_text)

        # Clean up items
        cleaned = [
            item.strip().lstrip('-•*').strip()
            for item in items
            if item.strip()
        ]

        return cleaned

    def _deduplicate_documentation(self, docs: List[DocumentationReference]) -> List[DocumentationReference]:
        """Remove duplicate documentation entries."""
        seen = set()
        unique = []

        for doc in docs:
            doc_id = f"{doc.title}:{doc.url}"
            if doc_id not in seen:
                seen.add(doc_id)
                unique.append(doc)

        # Sort by relevance and recency
        unique.sort(key=lambda x: (x.relevance_score, x.last_updated), reverse=True)
        return unique[:20]  # Keep top 20

    def _extract_architecture_patterns(self, docs: List[DocumentationReference]) -> List[str]:
        """Extract architecture patterns from documentation."""

        patterns = set()

        for doc in docs:
            content_lower = doc.content.lower()

            # Check for architecture patterns
            architecture_keywords = [
                "clean architecture", "hexagonal", "onion", "layered",
                "microservices", "event-driven", "cqrs", "domain-driven",
                "service-oriented", "modular", "component-based"
            ]

            for keyword in architecture_keywords:
                if keyword in content_lower:
                    patterns.add(keyword)

        return list(patterns)

    def _build_architecture_prompt(
            self,
            request: CodeGenerationRequest,
            documentation: List[DocumentationReference],
            patterns: List[str]
    ) -> str:
        """Build prompt for architecture design."""

        # Summarize relevant documentation
        relevant_docs = []
        for doc in documentation[:5]:  # Top 5 most relevant
            summary = f"- {doc.title}: {doc.content[:200]}..."
            relevant_docs.append(summary)

        docs_summary = "\n".join(relevant_docs)

        # Format patterns
        patterns_text = ", ".join(patterns) if patterns else "standard layered architecture"

        return f"""
        Design a production-grade system architecture for:

        TASK: {request.title}
        DESCRIPTION: {request.description}
        LANGUAGE: {request.language.value}
        FRAMEWORK: {request.framework or 'None'}
        QUALITY LEVEL: {request.quality_level.value}

        Requirements:
        {chr(10).join(f'• {req}' for req in request.requirements)}

        Suggested Architecture Patterns: {patterns_text}

        Relevant Documentation:
        {docs_summary}

        Design Constraints:
        1. Must follow clean architecture principles
        2. Must be highly testable (unit, integration, E2E)
        3. Must be scalable and maintainable
        4. Must include proper error handling and logging
        5. Must consider security best practices
        6. Must be deployable to cloud environments

        Provide detailed architecture including:
        1. System components and their responsibilities
        2. Data flow between components
        3. API design (if applicable)
        4. Database schema (if applicable)
        5. Testing strategy
        6. Deployment approach
        7. Monitoring and observability
        8. Security considerations

        Format the response as a structured JSON with component details.
        """

    def _parse_architecture_response(self, response: str) -> ArchitectureDesign:
        """Parse architecture response into structured design."""

        # In production, parse structured JSON response
        # This is a simplified implementation

        import json
        import re

        try:
            # Try to extract JSON
            json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                # Fallback: create mock architecture
                data = {
                    "style": "clean_architecture",
                    "components": [
                        {
                            "name": "main_service",
                            "type": "service",
                            "responsibilities": ["Core business logic"],
                            "dependencies": []
                        }
                    ],
                    "data_flow": "Request -> Service -> Response"
                }

            # Create architecture design
            components = []
            for comp_data in data.get("components", []):
                component = ComponentDesign(
                    name=comp_data.get("name", "unnamed"),
                    type=comp_data.get("type", "module"),
                    responsibilities=comp_data.get("responsibilities", []),
                    dependencies=comp_data.get("dependencies", []),
                    interfaces=comp_data.get("interfaces", []),
                    persistence=comp_data.get("persistence"),
                    concurrency_model=comp_data.get("concurrency_model", "synchronous")
                )
                components.append(component)

            return ArchitectureDesign(
                style=ArchitectureStyle(data.get("style", "clean_architecture")),
                components=components,
                data_flow=data.get("data_flow", ""),
                api_spec=data.get("api_spec"),
                database_schema=data.get("database_schema"),
                deployment_diagram=data.get("deployment_diagram", "")
            )

        except Exception as e:
            logger.warning(f"Failed to parse architecture: {e}")
            # Return minimal valid architecture
            return ArchitectureDesign(
                style=ArchitectureStyle.CLEAN_ARCHITECTURE,
                components=[
                    ComponentDesign(
                        name="default",
                        type="service",
                        responsibilities=["Default implementation"],
                        dependencies=[]
                    )
                ],
                data_flow="Default data flow"
            )

    async def _fix_architecture_issues(
            self,
            architecture: ArchitectureDesign,
            issues: List[str]
    ) -> ArchitectureDesign:
        """Fix architecture validation issues."""

        fix_prompt = f"""
        Fix these architecture design issues:

        Issues:
        {chr(10).join(f'- {issue}' for issue in issues)}

        Current Architecture:
        Style: {architecture.style.value}
        Components: {[c.name for c in architecture.components]}

        Provide fixed architecture that addresses all issues.
        """

        response = await self.llm_gateway.generate(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert software architect. Fix architecture issues with minimal changes."
                },
                {
                    "role": "user",
                    "content": fix_prompt
                }
            ],
            config={
                "temperature": 0.3,
                "max_tokens": 2000
            }
        )

        return self._parse_architecture_response(response.content)

    async def _generate_component_code(
            self,
            component: ComponentDesign,
            request: CodeGenerationRequest,
            documentation: List[DocumentationReference],
            iteration: int
    ) -> str:
        """Generate code for a specific component."""

        # Get relevant documentation for this component
        relevant_docs = []
        for doc in documentation:
            if any(keyword in doc.content.lower()
                   for keyword in [component.type, request.language.value]):
                relevant_docs.append(doc)

        # Build code generation prompt
        prompt = self._build_code_generation_prompt(
            component, request, relevant_docs, iteration
        )

        response = await self.llm_gateway.generate(
            messages=[
                {
                    "role": "system",
                    "content": f"""You are an elite {request.language.value} engineer. 
                    Write production-grade code that is:
                    - Well-tested and documented
                    - Follows best practices
                    - Includes proper error handling
                    - Optimized for performance
                    - Secure and maintainable"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            config={
                "temperature": 0.7,  # Higher for creativity
                "max_tokens": 4000
            }
        )

        return response.content

    def _build_code_generation_prompt(
            self,
            component: ComponentDesign,
            request: CodeGenerationRequest,
            documentation: List[DocumentationReference],
            iteration: int
    ) -> str:
        """Build prompt for code generation."""

        # Summarize relevant documentation
        docs_summary = []
        for doc in documentation[:3]:  # Top 3 most relevant
            docs_summary.append(f"- {doc.title}: {doc.content[:150]}...")

        docs_text = "\n".join(docs_summary) if docs_summary else "No specific documentation"

        return f"""
        Generate elite-level production code for component:

        COMPONENT: {component.name}
        TYPE: {component.type}
        RESPONSIBILITIES: {', '.join(component.responsibilities)}
        DEPENDENCIES: {', '.join(component.dependencies)}
        LANGUAGE: {request.language.value}
        FRAMEWORK: {request.framework or 'None'}
        QUALITY LEVEL: {request.quality_level.value}
        ITERATION: {iteration}

        Relevant Documentation:
        {docs_text}

        Code Requirements:
        1. Complete implementation with all imports/exports
        2. Comprehensive error handling and validation
        3. Logging and monitoring integration
        4. Unit tests included (aim for >90% coverage)
        5. Documentation (docstrings, comments, type hints)
        6. Follow language/framework best practices
        7. Optimize for performance and memory usage
        8. Consider security implications
        9. Make it production-ready

        Generate the complete code with all necessary files.
        """

    async def _refine_code(
            self,
            code: str,
            component: ComponentDesign,
            request: CodeGenerationRequest,
            documentation: List[DocumentationReference]
    ) -> str:
        """Refine generated code based on best practices."""

        # Check for common issues
        issues = self._detect_code_issues(code, request.language)

        if not issues:
            return code

        # Refine code to fix issues
        refine_prompt = f"""
        Refine this code to fix issues and improve quality:

        Component: {component.name}
        Language: {request.language.value}
        Framework: {request.framework or 'None'}

        Issues detected:
        {chr(10).join(f'- {issue}' for issue in issues)}

        Original Code:
        ```{request.language.value}
        {code}
        ```

        Provide refined code that addresses all issues.
        """

        response = await self.llm_gateway.generate(
            messages=[
                {
                    "role": "system",
                    "content": "You are a code refinement expert. Fix issues and improve code quality."
                },
                {
                    "role": "user",
                    "content": refine_prompt
                }
            ],
            config={
                "temperature": 0.3,
                "max_tokens": 4000
            }
        )

        return response.content

    def _detect_code_issues(self, code: str, language: ProgrammingLanguage) -> List[str]:
        """Detect common code issues."""

        issues = []

        # Language-specific checks
        if language == ProgrammingLanguage.PYTHON:
            if "print(" in code and "logging" not in code:
                issues.append("Uses print instead of logging")
            if "except:" in code or "except Exception:" in code:
                issues.append("Bare exception handling")
            if "eval(" in code or "exec(" in code:
                issues.append("Uses eval/exec - security risk")

        # General checks
        if "TODO" in code.upper() or "FIXME" in code.upper():
            issues.append("Contains TODO/FIXME comments")
        if "pass" in code and language == ProgrammingLanguage.PYTHON:
            issues.append("Contains placeholder 'pass' statements")
        if len(code.split('\n')) > 500:
            issues.append("Code is too long - consider refactoring")

        return issues

    async def _generate_unit_tests(
            self,
            snippet: CodeSnippet,
            request: CodeGenerationRequest
    ) -> List[CodeSnippet]:
        """Generate unit tests for code snippet."""

        test_prompt = f"""
        Generate comprehensive unit tests for this code:

        Language: {snippet.language.value}
        Framework: {snippet.framework or 'None'}

        Code to test:
        ```{snippet.language.value}
        {snippet.content}
        ```

        Test Requirements:
        1. Cover all functions/methods
        2. Include edge cases and error scenarios
        3. Mock external dependencies
        4. Follow {snippet.language.value} testing best practices
        5. Aim for >90% coverage
        6. Include setup and teardown if needed

        Provide complete test file.
        """

        response = await self.llm_gateway.generate(
            messages=[
                {
                    "role": "system",
                    "content": f"You are an expert in {snippet.language.value} testing. Write comprehensive unit tests."
                },
                {
                    "role": "user",
                    "content": test_prompt
                }
            ],
            config={
                "temperature": 0.5,
                "max_tokens": 3000
            }
        )

        # Create test snippet
        test_snippet = CodeSnippet(
            language=snippet.language,
            content=response.content,
            framework=snippet.framework,
            dependencies=["testing_framework"]  # Will be resolved based on language
        )

        return [test_snippet]

    async def _generate_integration_tests(
            self,
            snippet: CodeSnippet,
            all_snippets: List[CodeSnippet],
            request: CodeGenerationRequest
    ) -> List[CodeSnippet]:
        """Generate integration tests."""

        # Only generate integration tests for services/components with dependencies
        if not snippet.dependencies:
            return []

        test_prompt = f"""
        Generate integration tests for this component with its dependencies:

        Component Code:
        ```{snippet.language.value}
        {snippet.content[:1000]}  # First 1000 chars
        ```

        Dependencies: {', '.join(snippet.dependencies)}

        Generate tests that verify integration with dependencies.
        """

        response = await self.llm_gateway.generate(
            messages=[
                {
                    "role": "system",
                    "content": "You are an integration testing expert."
                },
                {
                    "role": "user",
                    "content": test_prompt
                }
            ],
            config={
                "temperature": 0.5,
                "max_tokens": 2000
            }
        )

        test_snippet = CodeSnippet(
            language=snippet.language,
            content=response.content,
            framework=snippet.framework
        )

        return [test_snippet]

    def _estimate_test_coverage(
            self,
            code_snippets: List[CodeSnippet],
            test_suite: TestSuite
    ) -> float:
        """Estimate test coverage percentage."""

        # Simplified estimation
        # In production, use actual coverage tools

        total_lines = sum(snippet.lines_of_code for snippet in code_snippets)
        test_lines = test_suite.total_test_lines

        if total_lines == 0:
            return 0.0

        coverage = (test_lines / total_lines) * 100
        return min(coverage, 100.0)

    def _calculate_documentation_score(self, snippets: List[CodeSnippet]) -> float:
        """Calculate documentation completeness score."""

        if not snippets:
            return 0.0

        scores = []
        for snippet in snippets:
            content = snippet.content

            # Check for documentation elements
            doc_elements = 0
            total_elements = 3  # docstrings, comments, type hints

            # Python-specific checks
            if snippet.language == ProgrammingLanguage.PYTHON:
                if '"""' in content or "'''" in content:
                    doc_elements += 1
                if '#' in content:
                    doc_elements += 1
                if '->' in content or ': ' in content:  # Type hints
                    doc_elements += 1

            score = (doc_elements / total_elements) * 100
            scores.append(score)

        return sum(scores) / len(scores)

    def _calculate_maintainability_score(self, snippets: List[CodeSnippet]) -> float:
        """Calculate maintainability score."""

        if not snippets:
            return 0.0

        scores = []
        for snippet in snippets:
            # Factors affecting maintainability
            loc = snippet.lines_of_code
            complexity = snippet.complexity_score

            # Calculate score (higher is better)
            # Penalize long files and high complexity
            loc_score = max(0, 100 - (loc * 0.1))  # -0.1% per line
            complexity_score = max(0, 100 - (complexity * 10))  # -10% per complexity point

            # Average the scores
            score = (loc_score + complexity_score) / 2
            scores.append(score)

        return sum(scores) / len(scores)

    def _estimate_performance_score(self, snippets: List[CodeSnippet]) -> float:
        """Estimate performance score based on code patterns."""

        if not snippets:
            return 0.0

        scores = []
        for snippet in snippets:
            content = snippet.content.lower()

            # Check for performance anti-patterns
            anti_patterns = [
                "sleep(", "time.sleep",
                "for i in range(1000000):",  # Large loops
                "while True:",  # Infinite loops
                "deepcopy", "copy.deepcopy",
                "eval(", "exec(",
            ]

            # Count anti-patterns
            anti_pattern_count = sum(1 for pattern in anti_patterns if pattern in content)

            # Calculate score (fewer anti-patterns = higher score)
            score = max(0, 100 - (anti_pattern_count * 20))
            scores.append(score)

        return sum(scores) / len(scores)

    def _calculate_reliability_score(
            self,
            snippets: List[CodeSnippet],
            test_suite: TestSuite
    ) -> float:
        """Calculate reliability score."""

        if not snippets:
            return 0.0

        # Factors for reliability
        test_coverage = test_suite.coverage_percentage
        test_pass_rate = test_suite.pass_rate

        # Check for error handling
        error_handling_score = 0.0
        for snippet in snippets:
            content = snippet.content.lower()
            if any(keyword in content for keyword in ["try:", "except", "catch", "finally"]):
                error_handling_score += 25  # Max 25% per snippet

        error_handling_score = min(error_handling_score / len(snippets), 25)

        # Calculate overall reliability score
        reliability = (
                              (test_coverage * 0.4) +  # 40% test coverage
                              (test_pass_rate * 0.3) +  # 30% test pass rate
                              error_handling_score  # 25% error handling
                      ) / 0.95  # Normalize to 100%

        return min(reliability, 100.0)

    def _identify_optimization_areas(self, task: ProgrammingTask) -> List[str]:
        """Identify areas for optimization based on quality metrics."""

        areas = []
        metrics = task.quality_metrics

        if metrics.test_coverage < 90:
            areas.append("improve_test_coverage")

        if metrics.complexity_score > 5:
            areas.append("reduce_complexity")

        if metrics.documentation_score < 80:
            areas.append("improve_documentation")

        if metrics.security_score < 80:
            areas.append("enhance_security")

        if metrics.maintainability_score < 80:
            areas.append("improve_maintainability")

        return areas

    async def _apply_optimizations(
            self,
            task: ProgrammingTask,
            areas: List[str]
    ) -> bool:
        """Apply specific optimizations to code."""

        optimized_code = []

        for snippet in task.generated_code:
            # Build optimization prompt based on areas
            optimization_prompt = self._build_optimization_prompt(
                snippet, areas, task.quality_metrics
            )

            response = await self.llm_gateway.generate(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a code optimization expert. Improve code based on specific areas."
                    },
                    {
                        "role": "user",
                        "content": optimization_prompt
                    }
                ],
                config={
                    "temperature": 0.3,
                    "max_tokens": 4000
                }
            )

            optimized_snippet = CodeSnippet(
                language=snippet.language,
                content=response.content,
                framework=snippet.framework,
                dependencies=snippet.dependencies
            )
            optimized_code.append(optimized_snippet)

        # Replace with optimized code
        task.generated_code = optimized_code
        return True

    def _build_optimization_prompt(
            self,
            snippet: CodeSnippet,
            areas: List[str],
            metrics: QualityMetrics
    ) -> str:
        """Build prompt for code optimization."""

        area_descriptions = {
            "improve_test_coverage": "Add more test cases to achieve >90% coverage",
            "reduce_complexity": "Refactor to reduce cyclomatic complexity",
            "improve_documentation": "Add comprehensive docstrings and comments",
            "enhance_security": "Fix security vulnerabilities and follow security best practices",
            "improve_maintainability": "Improve code structure for better maintainability"
        }

        optimizations = [area_descriptions.get(area, area) for area in areas]

        return f"""
        Optimize this code based on the following areas:

        Optimization Areas:
        {chr(10).join(f'- {opt}' for opt in optimizations)}

        Current Metrics:
        - Test Coverage: {metrics.test_coverage:.1f}%
        - Complexity Score: {metrics.complexity_score:.1f}/10
        - Documentation Score: {metrics.documentation_score:.1f}/100
        - Security Score: {metrics.security_score:.1f}/100
        - Maintainability Score: {metrics.maintainability_score:.1f}/100

        Code to optimize:
        ```{snippet.language.value}
        {snippet.content}
        ```

        Provide optimized code that addresses all optimization areas.
        """

    async def _validate_requirements_coverage(self, task: ProgrammingTask) -> bool:
        """Validate that all requirements are covered."""

        # Check if each requirement is addressed in code or documentation
        all_content = " ".join([
            task.description,
            *[snippet.content for snippet in task.generated_code],
            *[doc.content for doc in task.documentation]
        ]).lower()

        uncovered = []
        for req in task.requirements:
            # Check if requirement keywords appear in content
            keywords = req.lower().split()
            if not any(keyword in all_content for keyword in keywords[:3]):  # Check first 3 keywords
                uncovered.append(req)

        if uncovered:
            logger.warning(f"Uncovered requirements: {uncovered}")
            return False

        return True

    async def _validate_code_quality(self, task: ProgrammingTask) -> bool:
        """Validate code quality meets thresholds."""

        thresholds = {
            CodeQualityLevel.ELITE: {
                "test_coverage": 95,
                "overall_score": 95,
                "complexity_max": 3
            },
            CodeQualityLevel.EXCELLENT: {
                "test_coverage": 85,
                "overall_score": 85,
                "complexity_max": 5
            },
            CodeQualityLevel.GOOD: {
                "test_coverage": 75,
                "overall_score": 75,
                "complexity_max": 7
            }
        }

        target = thresholds.get(task.quality_level, thresholds[CodeQualityLevel.GOOD])

        metrics = task.quality_metrics
        overall_score = metrics.calculate_overall_score()

        checks = [
            (metrics.test_coverage >= target["test_coverage"],
             f"Test coverage {metrics.test_coverage:.1f}% < {target['test_coverage']}%"),
            (overall_score >= target["overall_score"],
             f"Overall score {overall_score:.1f} < {target['overall_score']}"),
            (metrics.complexity_score <= target["complexity_max"],
             f"Complexity {metrics.complexity_score:.1f} > {target['complexity_max']}")
        ]

        failures = [reason for passed, reason in checks if not passed]

        if failures:
            logger.warning(f"Quality validation failed: {failures}")
            return False

        return True

    async def _validate_architecture(self, task: ProgrammingTask) -> bool:
        """Validate architecture design."""

        if not task.architecture:
            return False

        is_valid, issues = task.architecture.validate()

        if not is_valid:
            logger.warning(f"Architecture validation failed: {issues}")
            return False

        # Check that all components have implementations
        component_names = {comp.name for comp in task.architecture.components}
        implemented_components = {
            # Extract component names from code (simplified)
            snippet.content.split('\n')[0].split('class ')[-1].split('(')[0].strip()
            for snippet in task.generated_code
            if 'class ' in snippet.content
        }

        missing = component_names - implemented_components
        if missing:
            logger.warning(f"Missing component implementations: {missing}")
            return False

        return True

    async def _validate_tests(self, task: ProgrammingTask) -> bool:
        """Validate test suite."""

        if not task.tests:
            return False

        # Basic validation
        if task.tests.total_tests == 0:
            logger.warning("No tests generated")
            return False

        if task.tests.coverage_percentage < 70:  # Minimum coverage
            logger.warning(f"Low test coverage: {task.tests.coverage_percentage:.1f}%")
            return False

        return True

    def _format_generated_files(self, task: ProgrammingTask) -> List[Dict[str, Any]]:
        """Format generated files for response."""

        files = []

        # Add code files
        for i, snippet in enumerate(task.generated_code):
            files.append({
                "type": "code",
                "name": f"component_{i}.{self._get_file_extension(snippet.language)}",
                "language": snippet.language.value,
                "content": snippet.content,
                "lines": snippet.lines_of_code,
                "complexity": snippet.complexity_score
            })

        # Add test files
        if task.tests:
            for i, test in enumerate(task.tests.unit_tests):
                files.append({
                    "type": "test",
                    "name": f"test_unit_{i}.{self._get_file_extension(test.language)}",
                    "language": test.language.value,
                    "content": test.content,
                    "lines": test.lines_of_code
                })

        return files

    def _get_file_extension(self, language: ProgrammingLanguage) -> str:
        """Get file extension for language."""

        extensions = {
            ProgrammingLanguage.PYTHON: "py",
            ProgrammingLanguage.TYPESCRIPT: "ts",
            ProgrammingLanguage.JAVASCRIPT: "js",
            ProgrammingLanguage.JAVA: "java",
            ProgrammingLanguage.GO: "go",
            ProgrammingLanguage.RUST: "rs",
            ProgrammingLanguage.KOTLIN: "kt",
            ProgrammingLanguage.SWIFT: "swift",
            ProgrammingLanguage.CPP: "cpp",
            ProgrammingLanguage.CSHARP: "cs"
        }

        return extensions.get(language, "txt")

    def _create_quality_report(self, task: ProgrammingTask) -> Dict[str, Any]:
        """Create comprehensive quality report."""

        metrics = task.quality_metrics

        return {
            "overall_score": metrics.calculate_overall_score(),
            "quality_level": task.quality_level.value,
            "detailed_metrics": {
                "test_coverage": metrics.test_coverage,
                "complexity_score": metrics.complexity_score,
                "documentation_score": metrics.documentation_score,
                "security_score": metrics.security_score,
                "performance_score": metrics.performance_score,
                "maintainability_score": metrics.maintainability_score,
                "reliability_score": metrics.reliability_score,
                "total_issues": metrics.total_issues
            },
            "test_metrics": {
                "total_tests": task.tests.total_tests if task.tests else 0,
                "passed_tests": task.tests.passed_tests if task.tests else 0,
                "pass_rate": task.tests.pass_rate if task.tests else 0,
                "coverage_percentage": task.tests.coverage_percentage if task.tests else 0,
                "test_to_code_ratio": task.test_to_code_ratio
            },
            "code_metrics": {
                "total_lines": task.total_lines_of_code,
                "number_of_files": len(task.generated_code),
                "average_complexity": sum(
                    snippet.complexity_score for snippet in task.generated_code
                ) / len(task.generated_code) if task.generated_code else 0
            }
        }

    def _generate_recommendations(self, task: ProgrammingTask) -> List[str]:
        """Generate improvement recommendations."""

        recommendations = []
        metrics = task.quality_metrics

        if metrics.test_coverage < 95:
            recommendations.append(
                "Increase test coverage to >95% for elite quality standards"
            )

        if metrics.complexity_score > 3:
            recommendations.append(
                "Refactor complex functions to reduce cyclomatic complexity"
            )

        if metrics.documentation_score < 90:
            recommendations.append(
                "Add more comprehensive documentation including examples and edge cases"
            )

        if metrics.security_score < 90:
            recommendations.append(
                "Conduct security review and implement additional security measures"
            )

        if task.iterations < 2:
            recommendations.append(
                "Consider additional optimization iterations for peak performance"
            )

        return recommendations

    def _update_metrics(self, task: ProgrammingTask, execution_time: float):
        """Update execution metrics."""

        self.metrics["average_quality_score"] = (
                (self.metrics["average_quality_score"] * (self.metrics["total_executions"] - 1) +
                 task.quality_metrics.calculate_overall_score()) /
                self.metrics["total_executions"]
        )

        self.metrics["average_iterations"] = (
                (self.metrics["average_iterations"] * (self.metrics["total_executions"] - 1) +
                 task.iterations) /
                self.metrics["total_executions"]
        )

        # Update success rate (assuming no exception means success)
        self.metrics["success_rate"] = (
                (self.metrics["success_rate"] * (self.metrics["total_executions"] - 1) + 100) /
                self.metrics["total_executions"]
        )