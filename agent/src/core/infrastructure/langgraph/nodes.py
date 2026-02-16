"""
LangGraph nodes for the programming agent.
Each node implements a specific step in the workflow.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from langgraph.graph import END
from ...application.use_cases.code_generation import CodeGenerationUseCase
from ...application.use_cases.documentation_query import DocumentationQueryUseCase
from ...application.ports.mcp_client import MCPClientPort
from ...application.ports.llm_gateway import LLMGatewayPort
from .state import AgentState, AgentPhase, StateManager
from .tools import AgentTools

logger = logging.getLogger(__name__)


class BaseNode:
    """Base class for all agent nodes with common functionality."""

    def __init__(self, node_name: str):
        self.node_name = node_name
        self.logger = logging.getLogger(f"node.{node_name}")

    async def execute(self, state: AgentState) -> AgentState:
        """Execute node logic. To be overridden by subclasses."""
        try:
            self.logger.info(f"Executing {self.node_name}")
            result = await self._execute_impl(state)
            self.logger.info(f"Completed {self.node_name}")
            return result
        except Exception as e:
            self.logger.error(f"Node {self.node_name} failed: {e}", exc_info=True)
            state["error"] = str(e)
            state["current_phase"] = AgentPhase.FAILED
            return state

    async def _execute_impl(self, state: AgentState) -> AgentState:
        """Node-specific implementation."""
        raise NotImplementedError


class AnalysisNode(BaseNode):
    """Analyzes the programming task."""

    def __init__(self, llm_gateway: LLMGatewayPort):
        super().__init__("analysis")
        self.llm_gateway = llm_gateway

    async def _execute_impl(self, state: AgentState) -> AgentState:
        """Analyze task complexity and requirements."""

        analysis_prompt = f"""
        Analyze this programming task:

        Task: {state['task_description']}
        Language: {state['language']}
        Framework: {state.get('framework', 'None')}
        Quality Level: {state['quality_level']}

        Requirements:
        {chr(10).join(f'- {req}' for req in state['requirements'])}

        Provide analysis including:
        1. Complexity assessment (simple/medium/complex)
        2. Estimated effort (development hours)
        3. Key technical challenges
        4. Required knowledge areas
        5. Recommended architecture patterns
        """

        response = await self.llm_gateway.generate(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert software architect. Analyze tasks precisely."
                },
                {
                    "role": "user",
                    "content": analysis_prompt
                }
            ],
            config={
                "temperature": 0.3,
                "max_tokens": 1500
            }
        )

        # Parse and store analysis
        analysis = self._parse_analysis(response.content)
        state["analysis_result"] = analysis
        state["current_phase"] = AgentPhase.ANALYZING

        # Add to memory
        if "memory" in state:
            state["memory"].append({
                "type": "analysis",
                "content": analysis,
                "timestamp": datetime.now().isoformat()
            })

        # Update token usage
        if "token_usage" in state and hasattr(response, 'usage'):
            state["token_usage"]["prompt"] += response.usage.get("prompt_tokens", 0)
            state["token_usage"]["completion"] += response.usage.get("completion_tokens", 0)
            state["token_usage"]["total"] += response.usage.get("total_tokens", 0)

        return state

    def _parse_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """Parse analysis text into structured format."""

        # Simplified parsing - in production use structured output
        import re

        result = {
            "complexity": "medium",
            "estimated_hours": 8.0,
            "challenges": [],
            "knowledge_areas": [],
            "architecture_patterns": []
        }

        # Extract complexity
        complexity_match = re.search(r'complexity.*?:\s*(\w+)', analysis_text, re.IGNORECASE)
        if complexity_match:
            result["complexity"] = complexity_match.group(1).lower()

        # Extract hours
        hours_match = re.search(r'(\d+\.?\d*)\s*hours', analysis_text, re.IGNORECASE)
        if hours_match:
            result["estimated_hours"] = float(hours_match.group(1))

        # Extract lists (simplified)
        sections = re.split(r'\n\d+\.\s+', analysis_text)
        for section in sections[1:]:  # Skip first (empty or title)
            if "challenge" in section.lower():
                lines = section.split('\n')
                result["challenges"] = [line.strip() for line in lines if line.strip()]
            elif "knowledge" in section.lower():
                lines = section.split('\n')
                result["knowledge_areas"] = [line.strip() for line in lines if line.strip()]
            elif "architecture" in section.lower():
                lines = section.split('\n')
                result["architecture_patterns"] = [line.strip() for line in lines if line.strip()]

        return result


class ResearchNode(BaseNode):
    """Researches documentation using MCP."""

    def __init__(self, mcp_client: MCPClientPort):
        super().__init__("research")
        self.mcp_client = mcp_client

    async def _execute_impl(self, state: AgentState) -> AgentState:
        """Research documentation via MCP."""

        # Ensure connected to MCP
        if not self.mcp_client.is_connected:
            await self.mcp_client.connect()

        # Build search queries from analysis
        queries = self._build_search_queries(state)

        all_docs = []
        for query in queries:
            try:
                docs = await self.mcp_client.search_documentation(
                    query=query,
                    language=state["language"],
                    framework=state.get("framework"),
                    limit=5
                )

                # Convert to dict for state
                doc_dicts = []
                for doc in docs:
                    doc_dicts.append({
                        "title": doc.title,
                        "url": doc.url,
                        "content": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                        "source": doc.source,
                        "relevance_score": doc.relevance_score,
                        "last_updated": doc.last_updated.isoformat(),
                        "doc_type": doc.doc_type,
                        "tags": doc.tags
                    })

                all_docs.extend(doc_dicts)

                # Update tool calls
                if "tool_calls" in state:
                    state["tool_calls"] += 1

            except Exception as e:
                self.logger.warning(f"Documentation search failed for query '{query}': {e}")
                continue

        # Sort by relevance and deduplicate
        unique_docs = self._deduplicate_docs(all_docs)
        state["researched_docs"] = unique_docs[:15]  # Keep top 15

        # Update phase
        state["current_phase"] = AgentPhase.RESEARCHING

        # Add to memory
        if "memory" in state:
            state["memory"].append({
                "type": "research",
                "query_count": len(queries),
                "doc_count": len(state["researched_docs"]),
                "timestamp": datetime.now().isoformat()
            })

        return state

    def _build_search_queries(self, state: AgentState) -> List[str]:
        """Build intelligent search queries."""

        queries = []
        language = state["language"]
        framework = state.get("framework", "")

        # Basic queries
        queries.append(f"{language} {framework} best practices")
        queries.append(f"{language} clean architecture")

        # Task-specific queries
        task_words = state["task_description"].lower().split()[:5]
        for word in task_words:
            if len(word) > 3:  # Skip short words
                queries.append(f"{language} {word} implementation")

        # Requirements-specific queries
        for req in state["requirements"][:3]:  # First 3 requirements
            req_words = req.lower().split()[:3]
            queries.append(f"{language} {' '.join(req_words)}")

        # Analysis-based queries
        if state.get("analysis_result"):
            analysis = state["analysis_result"]
            if "architecture_patterns" in analysis:
                for pattern in analysis["architecture_patterns"][:2]:
                    queries.append(f"{language} {pattern} pattern")

        return list(set(queries))  # Remove duplicates

    def _deduplicate_docs(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate documentation entries."""
        seen = set()
        unique = []

        for doc in docs:
            doc_id = f"{doc.get('title', '')}:{doc.get('url', '')}"
            if doc_id not in seen:
                seen.add(doc_id)
                unique.append(doc)

        # Sort by relevance
        unique.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return unique


class DesignNode(BaseNode):
    """Designs system architecture."""

    def __init__(self, llm_gateway: LLMGatewayPort):
        super().__init__("design")
        self.llm_gateway = llm_gateway

    async def _execute_impl(self, state: AgentState) -> AgentState:
        """Design system architecture."""

        # Build design prompt with research context
        prompt = self._build_design_prompt(state)

        response = await self.llm_gateway.generate(
            messages=[
                {
                    "role": "system",
                    "content": """You are an elite software architect. Design systems that are:
                    - Clean and maintainable
                    - Highly testable
                    - Scalable and performant
                    - Secure and reliable
                    - Production-ready"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            config={
                "temperature": 0.2,
                "max_tokens": 2500
            }
        )

        # Parse design
        design = self._parse_design_response(response.content)
        state["architecture_design"] = design
        state["current_phase"] = AgentPhase.DESIGNING

        # Update token usage
        if "token_usage" in state and hasattr(response, 'usage'):
            state["token_usage"]["prompt"] += response.usage.get("prompt_tokens", 0)
            state["token_usage"]["completion"] += response.usage.get("completion_tokens", 0)
            state["token_usage"]["total"] += response.usage.get("total_tokens", 0)

        # Add to memory
        if "memory" in state:
            state["memory"].append({
                "type": "design",
                "components": len(design.get("components", [])),
                "architecture_style": design.get("style", "unknown"),
                "timestamp": datetime.now().isoformat()
            })

        return state

    def _build_design_prompt(self, state: AgentState) -> str:
        """Build architecture design prompt."""

        # Summarize relevant documentation
        relevant_docs = []
        for doc in state.get("researched_docs", [])[:5]:
            summary = f"- {doc.get('title', 'Untitled')}: {doc.get('content', '')[:200]}..."
            relevant_docs.append(summary)

        docs_summary = "\n".join(relevant_docs) if relevant_docs else "No specific documentation"

        # Include analysis if available
        analysis_text = ""
        if state.get("analysis_result"):
            analysis = state["analysis_result"]
            analysis_text = f"""
            Analysis Results:
            - Complexity: {analysis.get('complexity', 'unknown')}
            - Estimated Hours: {analysis.get('estimated_hours', 0)}
            - Key Challenges: {', '.join(analysis.get('challenges', []))}
            - Recommended Patterns: {', '.join(analysis.get('architecture_patterns', []))}
            """

        return f"""
        Design a production-grade system architecture for:

        TASK: {state['task_description']}
        LANGUAGE: {state['language']}
        FRAMEWORK: {state.get('framework', 'None')}
        QUALITY LEVEL: {state['quality_level'].upper()}

        Requirements:
        {chr(10).join(f'• {req}' for req in state['requirements'])}

        {analysis_text}

        Relevant Documentation:
        {docs_summary}

        Design must include:
        1. System components and responsibilities
        2. Data flow between components
        3. API design (if applicable)
        4. Database schema (if applicable)
        5. Testing strategy
        6. Deployment approach
        7. Security considerations

        Provide the design in structured JSON format.
        """

    def _parse_design_response(self, response_text: str) -> Dict[str, Any]:
        """Parse design response into structured format."""

        import json
        import re

        try:
            # Try to extract JSON
            json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))

            # Fallback: minimal design
            return {
                "style": "clean_architecture",
                "components": [
                    {
                        "name": "main_service",
                        "type": "service",
                        "responsibilities": ["Core business logic"],
                        "dependencies": []
                    }
                ],
                "data_flow": "Request -> Service -> Response",
                "testing_strategy": "Unit + Integration tests",
                "deployment": "Docker container"
            }

        except Exception as e:
            self.logger.warning(f"Failed to parse design: {e}")
            return {
                "style": "minimal",
                "components": [],
                "data_flow": "Unknown",
                "error": str(e)
            }


class CodeGenNode(BaseNode):
    """Generates code implementation."""

    def __init__(self, llm_gateway: LLMGatewayPort, agent_tools: AgentTools):
        super().__init__("code_generation")
        self.llm_gateway = llm_gateway
        self.tools = agent_tools

    async def _execute_impl(self, state: AgentState) -> AgentState:
        """Generate code implementation."""

        # Get design components
        design = state.get("architecture_design", {})
        components = design.get("components", [])

        generated_code = []

        for component in components:
            # Generate code for each component
            code = await self._generate_component_code(component, state)

            # Validate code
            is_valid, issues = self._validate_code(code, state["language"])

            if not is_valid:
                self.logger.warning(f"Code validation issues for {component.get('name')}: {issues}")
                # Try to fix code
                code = await self._fix_code_issues(code, issues, state)

            # Store generated code
            generated_code.append({
                "component": component.get("name"),
                "language": state["language"],
                "framework": state.get("framework"),
                "code": code,
                "lines": len(code.split('\n')),
                "valid": is_valid
            })

        state["generated_code"] = generated_code
        state["current_phase"] = AgentPhase.IMPLEMENTING

        # Update token usage (approximate)
        total_lines = sum(item["lines"] for item in generated_code)
        estimated_tokens = total_lines * 10  # Approximate tokens per line
        if "token_usage" in state:
            state["token_usage"]["completion"] += estimated_tokens
            state["token_usage"]["total"] += estimated_tokens

        # Add to memory
        if "memory" in state:
            state["memory"].append({
                "type": "code_generation",
                "components": len(generated_code),
                "total_lines": total_lines,
                "timestamp": datetime.now().isoformat()
            })

        return state

    async def _generate_component_code(
            self,
            component: Dict[str, Any],
            state: AgentState
    ) -> str:
        """Generate code for a component."""

        prompt = self._build_code_prompt(component, state)

        response = await self.llm_gateway.generate(
            messages=[
                {
                    "role": "system",
                    "content": f"""You are an elite {state['language']} developer. 
                    Write production-grade code that is:
                    - Well-documented and tested
                    - Follows best practices
                    - Includes error handling
                    - Secure and performant"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            config={
                "temperature": 0.7,
                "max_tokens": 3000
            }
        )

        return response.content

    def _build_code_prompt(self, component: Dict[str, Any], state: AgentState) -> str:
        """Build code generation prompt."""

        # Get relevant documentation for this component
        relevant_docs = []
        for doc in state.get("researched_docs", []):
            if component.get("type") in doc.get("content", "").lower():
                relevant_docs.append(f"- {doc.get('title')}: {doc.get('content', '')[:150]}...")

        docs_text = "\n".join(relevant_docs[:3]) if relevant_docs else "No specific documentation"

        return f"""
        Generate elite-level production code for:

        COMPONENT: {component.get('name', 'unnamed')}
        TYPE: {component.get('type', 'service')}
        RESPONSIBILITIES: {', '.join(component.get('responsibilities', []))}
        DEPENDENCIES: {', '.join(component.get('dependencies', []))}
        LANGUAGE: {state['language']}
        FRAMEWORK: {state.get('framework', 'None')}
        QUALITY LEVEL: {state['quality_level'].upper()}

        Relevant Documentation:
        {docs_text}

        Code Requirements:
        1. Complete implementation with imports
        2. Comprehensive error handling
        3. Logging and monitoring
        4. Unit tests included
        5. Documentation (docstrings, comments)
        6. Follow {state['language']} best practices
        7. Optimized for performance

        Provide the complete code file.
        """

    def _validate_code(self, code: str, language: str) -> Tuple[bool, List[str]]:
        """Validate generated code."""

        issues = []

        # Basic validation
        if not code.strip():
            issues.append("Empty code")

        # Language-specific validation
        if language == "python":
            if "print(" in code and "logging" not in code:
                issues.append("Uses print instead of logging")
            if "except:" in code:
                issues.append("Bare except clause")

        # Check for TODOs
        if "TODO" in code.upper():
            issues.append("Contains TODO comments")

        return len(issues) == 0, issues

    async def _fix_code_issues(
            self,
            code: str,
            issues: List[str],
            state: AgentState
    ) -> str:
        """Fix code validation issues."""

        fix_prompt = f"""
        Fix these issues in the code:

        Issues: {', '.join(issues)}

        Original Code:
        ```{state['language']}
        {code}
        ```

        Provide fixed code that addresses all issues.
        """

        response = await self.llm_gateway.generate(
            messages=[
                {
                    "role": "system",
                    "content": f"You are a {state['language']} code fixer. Fix issues while maintaining functionality."
                },
                {
                    "role": "user",
                    "content": fix_prompt
                }
            ],
            config={
                "temperature": 0.3,
                "max_tokens": 3000
            }
        )

        return response.content


class TestGenNode(BaseNode):
    """Generates tests for the code."""

    def __init__(self, llm_gateway: LLMGatewayPort):
        super().__init__("test_generation")
        self.llm_gateway = llm_gateway

    async def _execute_impl(self, state: AgentState) -> AgentState:
        """Generate tests for generated code."""

        generated_tests = []

        for code_item in state.get("generated_code", []):
            # Generate tests for each component
            tests = await self._generate_tests_for_component(code_item, state)

            generated_tests.append({
                "component": code_item["component"],
                "language": code_item["language"],
                "tests": tests,
                "test_lines": len(tests.split('\n'))
            })

        state["generated_tests"] = generated_tests
        state["current_phase"] = AgentPhase.TESTING

        # Update token usage (approximate)
        total_test_lines = sum(item["test_lines"] for item in generated_tests)
        estimated_tokens = total_test_lines * 8  # Approximate tokens per line
        if "token_usage" in state:
            state["token_usage"]["completion"] += estimated_tokens
            state["token_usage"]["total"] += estimated_tokens

        # Add to memory
        if "memory" in state:
            state["memory"].append({
                "type": "test_generation",
                "components": len(generated_tests),
                "total_test_lines": total_test_lines,
                "timestamp": datetime.now().isoformat()
            })

        return state

    async def _generate_tests_for_component(
            self,
            code_item: Dict[str, Any],
            state: AgentState
    ) -> str:
        """Generate tests for a component."""

        prompt = self._build_test_prompt(code_item, state)

        response = await self.llm_gateway.generate(
            messages=[
                {
                    "role": "system",
                    "content": f"You are an expert in {state['language']} testing. Write comprehensive tests."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            config={
                "temperature": 0.5,
                "max_tokens": 2000
            }
        )

        return response.content

    def _build_test_prompt(self, code_item: Dict[str, Any], state: AgentState) -> str:
        """Build test generation prompt."""

        return f"""
        Generate comprehensive tests for this code:

        Component: {code_item['component']}
        Language: {code_item['language']}
        Framework: {code_item.get('framework', 'None')}
        Quality Level: {state['quality_level'].upper()}

        Code to test:
        ```{code_item['language']}
        {code_item['code']}
        ```

        Test Requirements:
        1. Cover all functions/methods
        2. Include edge cases and error scenarios
        3. Mock external dependencies
        4. Follow {code_item['language']} testing best practices
        5. Aim for >90% coverage
        6. Include setup and teardown if needed

        Provide the complete test file.
        """


class ReviewNode(BaseNode):
    """Reviews generated code for quality and correctness."""
    
    def __init__(self, llm_gateway: LLMGatewayPort, agent_tools: AgentTools):
        super().__init__("review")
        self.llm_gateway = llm_gateway
        self.tools = agent_tools
    
    async def _execute_impl(self, state: AgentState) -> AgentState:
        """Review generated code."""
        
        reviews = []
        
        for code_item in state.get("generated_code", []):
            # Validate code quality
            quality_report = self.tools.validate_code_quality(
                code_item["code"],
                state["language"]
            )
            
            # Get LLM review
            review_prompt = self._build_review_prompt(code_item, state)
            
            response = await self.llm_gateway.generate(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior code reviewer. Provide constructive feedback."
                    },
                    {
                        "role": "user",
                        "content": review_prompt
                    }
                ],
                config={
                    "temperature": 0.3,
                    "max_tokens": 1500
                }
            )
            
            reviews.append({
                "component": code_item["component"],
                "quality_report": quality_report,
                "review_comments": response.content,
                "approved": quality_report["valid"] and len(quality_report["issues"]) == 0
            })
        
        state["code_reviews"] = reviews
        state["current_phase"] = AgentPhase.TESTING
        
        # Add to memory
        if "memory" in state:
            state["memory"].append({
                "type": "review",
                "reviews_count": len(reviews),
                "approved_count": sum(1 for r in reviews if r["approved"]),
                "timestamp": datetime.now().isoformat()
            })
        
        return state
    
    def _build_review_prompt(self, code_item: Dict[str, Any], state: AgentState) -> str:
        """Build code review prompt."""
        
        return f"""
        Review this {state['language']} code for quality and correctness:
        
        Component: {code_item['component']}
        
        Code:
        ```{state['language']}
        {code_item['code']}
        ```
        
        Review criteria:
        1. Code correctness and functionality
        2. Best practices adherence
        3. Error handling
        4. Performance considerations
        5. Security concerns
        6. Maintainability
        7. Documentation quality
        
        Provide specific feedback and improvement suggestions.
        """


class OptimizationNode(BaseNode):
    """Optimizes code for performance and quality."""
    
    def __init__(self, llm_gateway: LLMGatewayPort, agent_tools: AgentTools):
        super().__init__("optimization")
        self.llm_gateway = llm_gateway
        self.tools = agent_tools
    
    async def _execute_impl(self, state: AgentState) -> AgentState:
        """Optimize generated code."""
        
        optimized_code = []
        
        for code_item in state.get("generated_code", []):
            # Check if optimization is needed
            complexity = self.tools.calculate_complexity_score(
                code_item["code"],
                state["language"]
            )
            
            if complexity > 50 or state.get("quality_level") == "elite":
                # Optimize code
                optimized = await self._optimize_code(code_item, state)
                optimized_code.append(optimized)
            else:
                # No optimization needed
                optimized_code.append(code_item)
        
        state["generated_code"] = optimized_code
        state["current_phase"] = AgentPhase.COMPLETION
        
        # Add to memory
        if "memory" in state:
            state["memory"].append({
                "type": "optimization",
                "optimized_count": len(optimized_code),
                "timestamp": datetime.now().isoformat()
            })
        
        return state
    
    async def _optimize_code(
        self,
        code_item: Dict[str, Any],
        state: AgentState
    ) -> Dict[str, Any]:
        """Optimize a code component."""
        
        optimization_prompt = f"""
        Optimize this {state['language']} code for:
        - Performance
        - Memory efficiency
        - Readability
        - Best practices
        
        Original code:
        ```{state['language']}
        {code_item['code']}
        ```
        
        Provide the optimized version maintaining the same functionality.
        """
        
        response = await self.llm_gateway.generate(
            messages=[
                {
                    "role": "system",
                    "content": "You are a performance optimization expert."
                },
                {
                    "role": "user",
                    "content": optimization_prompt
                }
            ],
            config={
                "temperature": 0.3,
                "max_tokens": 3000
            }
        )
        
        # Update code item
        optimized_item = code_item.copy()
        optimized_item["code"] = response.content
        optimized_item["optimized"] = True
        
        return optimized_item


class HumanReviewNode(BaseNode):
    """Placeholder for human review integration."""
    
    def __init__(self):
        super().__init__("human_review")
    
    async def _execute_impl(self, state: AgentState) -> AgentState:
        """
        Placeholder for human review.
        
        In a production system, this would:
        - Pause execution and wait for human input
        - Present code for review via UI
        - Accept feedback and suggestions
        - Resume with modifications
        """
        
        self.logger.info("Human review node - auto-approved (not implemented)")
        
        # For now, just mark as reviewed
        state["human_reviewed"] = True
        state["human_approved"] = True
        
        # Add to memory
        if "memory" in state:
            state["memory"].append({
                "type": "human_review",
                "status": "auto_approved",
                "timestamp": datetime.now().isoformat()
            })
        
        return state
