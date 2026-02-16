"""
Agent tools for LangGraph nodes.
Provides utility tools and functions for agent operations.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import re
import ast


logger = logging.getLogger(__name__)


class AgentTools:
    """
    Collection of tools and utilities for the LangGraph agent.
    
    Provides:
    - Code validation tools
    - Documentation search helpers
    - Quality assessment utilities
    - Parsing and formatting tools
    """
    
    def __init__(self):
        """Initialize agent tools."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate_code_syntax(self, code: str, language: str) -> Tuple[bool, List[str]]:
        """
        Validate code syntax for a given language.
        
        Args:
            code: Source code to validate
            language: Programming language
        
        Returns:
            Tuple of (is_valid, list of errors)
        """
        
        errors = []
        
        try:
            if language.lower() == "python":
                # Try to parse as Python AST
                try:
                    ast.parse(code)
                except SyntaxError as e:
                    errors.append(f"Syntax error: {str(e)}")
            
            elif language.lower() in ["javascript", "typescript"]:
                # Basic validation for JS/TS
                # Check for common syntax issues
                if code.count('{') != code.count('}'):
                    errors.append("Mismatched braces")
                if code.count('(') != code.count(')'):
                    errors.append("Mismatched parentheses")
            
            elif language.lower() in ["java", "c", "c++", "csharp"]:
                # Basic validation for C-family languages
                if code.count('{') != code.count('}'):
                    errors.append("Mismatched braces")
                if code.count('(') != code.count(')'):
                    errors.append("Mismatched parentheses")
            
            # Common checks for all languages
            if not code.strip():
                errors.append("Empty code")
            
            return len(errors) == 0, errors
        
        except Exception as e:
            self.logger.warning(f"Syntax validation failed: {e}")
            return False, [str(e)]
    
    def validate_code_quality(self, code: str, language: str) -> Dict[str, Any]:
        """
        Validate code quality and best practices.
        
        Args:
            code: Source code to validate
            language: Programming language
        
        Returns:
            Dictionary with quality metrics and issues
        """
        
        issues = []
        warnings = []
        
        # Language-specific quality checks
        if language.lower() == "python":
            # Check for common Python issues
            if "print(" in code and "logging" not in code.lower():
                warnings.append("Uses print() instead of logging")
            
            if "except:" in code:
                warnings.append("Bare except clause detected")
            
            if re.search(r'^\s*import \*', code, re.MULTILINE):
                warnings.append("Uses wildcard imports")
            
            # Check for TODO/FIXME
            if re.search(r'(TODO|FIXME)', code, re.IGNORECASE):
                warnings.append("Contains TODO/FIXME comments")
        
        # Common checks
        lines = code.split('\n')
        
        # Check line length
        long_lines = [i+1 for i, line in enumerate(lines) if len(line) > 120]
        if long_lines:
            warnings.append(f"Lines exceed 120 characters: {long_lines[:5]}")
        
        # Check for hardcoded secrets (basic patterns)
        secret_patterns = [
            r'password\s*=\s*["\']',
            r'api_key\s*=\s*["\']',
            r'secret\s*=\s*["\']',
            r'token\s*=\s*["\']'
        ]
        
        for pattern in secret_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append("Possible hardcoded secret detected")
                break
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "lines_of_code": len([l for l in lines if l.strip()]),
            "total_lines": len(lines)
        }
    
    def extract_code_blocks(self, text: str, language: Optional[str] = None) -> List[str]:
        """
        Extract code blocks from markdown text.
        
        Args:
            text: Text containing code blocks
            language: Optional language filter
        
        Returns:
            List of extracted code blocks
        """
        
        # Pattern to match markdown code blocks
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        code_blocks = []
        for lang, code in matches:
            if language is None or lang.lower() == language.lower():
                code_blocks.append(code.strip())
        
        return code_blocks
    
    def format_documentation_snippet(
        self,
        doc: Dict[str, Any],
        max_length: int = 500
    ) -> str:
        """
        Format a documentation snippet for display.
        
        Args:
            doc: Documentation dictionary
            max_length: Maximum length of content
        
        Returns:
            Formatted documentation string
        """
        
        title = doc.get("title", "Untitled")
        content = doc.get("content", "")
        source = doc.get("source", "Unknown")
        url = doc.get("url", "")
        
        # Truncate content
        if len(content) > max_length:
            content = content[:max_length] + "..."
        
        snippet = f"**{title}**\n"
        snippet += f"Source: {source}\n"
        if url:
            snippet += f"URL: {url}\n"
        snippet += f"\n{content}\n"
        
        return snippet
    
    def parse_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract and parse JSON from text.
        
        Args:
            text: Text potentially containing JSON
        
        Returns:
            Parsed JSON dictionary or None
        """
        
        import json
        
        try:
            # Try to parse entire text as JSON
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON in markdown code blocks
        json_pattern = r'```json\n(.*?)\n```'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # Try to find JSON object in text
        brace_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(brace_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        return None
    
    def calculate_complexity_score(self, code: str, language: str) -> float:
        """
        Calculate a simple complexity score for code.
        
        Args:
            code: Source code
            language: Programming language
        
        Returns:
            Complexity score (0-100, higher is more complex)
        """
        
        score = 0
        lines = code.split('\n')
        
        # Line count contribution
        loc = len([l for l in lines if l.strip()])
        score += min(loc / 10, 20)  # Max 20 points for line count
        
        # Nesting depth
        max_depth = 0
        current_depth = 0
        
        for line in lines:
            current_depth += line.count('{') - line.count('}')
            max_depth = max(max_depth, current_depth)
        
        score += min(max_depth * 5, 30)  # Max 30 points for nesting
        
        # Control flow statements
        control_keywords = ['if', 'else', 'for', 'while', 'switch', 'case', 'try', 'catch']
        control_count = sum(len(re.findall(rf'\b{kw}\b', code)) for kw in control_keywords)
        score += min(control_count * 2, 30)  # Max 30 points for control flow
        
        # Function/method count
        if language.lower() == "python":
            func_count = len(re.findall(r'\bdef\s+\w+', code))
        elif language.lower() in ["javascript", "typescript"]:
            func_count = len(re.findall(r'\bfunction\s+\w+', code))
        else:
            func_count = 0
        
        score += min(func_count * 2, 20)  # Max 20 points for functions
        
        return min(score, 100)
    
    def generate_test_cases(
        self,
        function_signature: str,
        language: str
    ) -> List[Dict[str, Any]]:
        """
        Generate basic test case templates for a function.
        
        Args:
            function_signature: Function signature
            language: Programming language
        
        Returns:
            List of test case templates
        """
        
        test_cases = [
            {
                "name": "test_happy_path",
                "description": "Test normal/expected input",
                "type": "positive"
            },
            {
                "name": "test_edge_cases",
                "description": "Test boundary conditions",
                "type": "edge"
            },
            {
                "name": "test_error_handling",
                "description": "Test error conditions",
                "type": "negative"
            },
            {
                "name": "test_invalid_input",
                "description": "Test invalid input handling",
                "type": "negative"
            }
        ]
        
        return test_cases
    
    def sanitize_code_output(self, text: str) -> str:
        """
        Sanitize and clean code output from LLM.
        
        Args:
            text: Raw text from LLM
        
        Returns:
            Cleaned code
        """
        
        # Remove markdown code block markers
        text = re.sub(r'```\w*\n', '', text)
        text = re.sub(r'```$', '', text)
        
        # Remove common LLM artifacts
        text = re.sub(r'^(Here\'s|Here is).*?:\s*\n', '', text, flags=re.IGNORECASE)
        
        return text.strip()
