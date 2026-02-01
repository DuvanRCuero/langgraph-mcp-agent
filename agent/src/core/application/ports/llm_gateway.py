"""
Port definition for LLM gateway.
Abstracts different LLM providers (OpenAI, Anthropic, etc.)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import dataclass
from enum import Enum


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    GROQ = "groq"
    TOGETHER = "together"
    LOCAL = "local"


class LLMModel(str, Enum):
    """Supported LLM models."""
    # OpenAI
    GPT4_TURBO = "gpt-4-turbo-preview"
    GPT4 = "gpt-4"
    GPT35_TURBO = "gpt-3.5-turbo"

    # Anthropic
    CLAUDE3_OPUS = "claude-3-opus"
    CLAUDE3_SONNET = "claude-3-sonnet"
    CLAUDE3_HAIKU = "claude-3-haiku"

    # Open Source
    LLAMA2_70B = "llama-2-70b"
    MISTRAL_LARGE = "mistral-large"
    CODEGEN = "codegen"


@dataclass
class LLMMessage:
    """LLM message with role and content."""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


@dataclass
class LLMGenerationConfig:
    """Configuration for LLM generation."""
    temperature: float = 0.7
    max_tokens: int = 4000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    seed: Optional[int] = None
    stream: bool = False


@dataclass
class LLMResponse:
    """LLM response with metadata."""
    content: str
    model: str
    usage: Dict[str, int]  # prompt_tokens, completion_tokens, total_tokens
    finish_reason: str
    response_time: float
    metadata: Dict[str, Any] = None


class LLMGatewayPort(ABC):
    """
    Interface for LLM gateway.
    Abstracts different LLM providers and models.
    """

    @abstractmethod
    async def generate(
            self,
            messages: List[LLMMessage],
            config: LLMGenerationConfig,
            model: Optional[LLMModel] = None
    ) -> LLMResponse:
        """Generate text completion."""
        pass

    @abstractmethod
    async def generate_stream(
            self,
            messages: List[LLMMessage],
            config: LLMGenerationConfig,
            model: Optional[LLMModel] = None
    ) -> AsyncGenerator[str, None]:
        """Stream text completion."""
        pass

    @abstractmethod
    async def chat_with_tools(
            self,
            messages: List[LLMMessage],
            tools: List[Dict[str, Any]],
            config: LLMGenerationConfig,
            model: Optional[LLMModel] = None
    ) -> LLMResponse:
        """Chat with tool calling support."""
        pass

    @abstractmethod
    async def embed_text(
            self,
            text: str,
            model: str = "text-embedding-ada-002"
    ) -> List[float]:
        """Generate embeddings for text."""
        pass

    @abstractmethod
    async def embed_batch(
            self,
            texts: List[str],
            model: str = "text-embedding-ada-002"
    ) -> List[List[float]]:
        """Generate embeddings for batch of texts."""
        pass

    @abstractmethod
    async def analyze_code_complexity(
            self,
            code: str,
            language: str
    ) -> Dict[str, Any]:
        """Analyze code complexity using LLM."""
        pass

    @abstractmethod
    async def generate_tests(
            self,
            code: str,
            language: str,
            framework: Optional[str] = None
    ) -> str:
        """Generate tests for code."""
        pass

    @abstractmethod
    async def refactor_code(
            self,
            code: str,
            language: str,
            improvements: List[str]
    ) -> str:
        """Refactor code with specific improvements."""
        pass

    @property
    @abstractmethod
    def available_models(self) -> List[str]:
        """Get list of available models."""
        pass

    @property
    @abstractmethod
    def provider(self) -> LLMProvider:
        """Get LLM provider."""
        pass