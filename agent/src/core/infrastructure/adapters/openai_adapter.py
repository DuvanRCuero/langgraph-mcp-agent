"""
OpenAI adapter implementing the LLM Gateway Port.
Provides integration with OpenAI's GPT models.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, AsyncGenerator

import openai
from openai import AsyncOpenAI

from ...application.ports.llm_gateway import (
    LLMGatewayPort,
    LLMProvider,
    LLMModel,
    LLMMessage,
    LLMGenerationConfig,
    LLMResponse
)

logger = logging.getLogger(__name__)


class OpenAIAdapter(LLMGatewayPort):
    """
    OpenAI adapter for LLM operations.
    Implements the LLMGatewayPort interface using OpenAI's API.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        organization: Optional[str] = None
    ):
        """
        Initialize OpenAI adapter.
        
        Args:
            api_key: OpenAI API key
            model: Default model to use
            temperature: Default temperature
            max_tokens: Default max tokens
            organization: Optional organization ID
        """
        self.api_key = api_key
        self.default_model = model
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self.organization = organization
        
        # Initialize async client
        self.client = AsyncOpenAI(
            api_key=api_key,
            organization=organization
        )
        
        logger.info(f"OpenAI adapter initialized with model: {model}")
    
    async def generate(
        self,
        messages: List[LLMMessage],
        config: LLMGenerationConfig,
        model: Optional[LLMModel] = None
    ) -> LLMResponse:
        """Generate text completion using OpenAI."""
        
        start_time = time.time()
        
        try:
            # Convert messages to OpenAI format
            openai_messages = self._convert_messages(messages)
            
            # Determine model to use
            model_name = model.value if model else self.default_model
            
            # Make API call
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=openai_messages,
                temperature=config.temperature if hasattr(config, 'temperature') else self.default_temperature,
                max_tokens=config.max_tokens if hasattr(config, 'max_tokens') else self.default_max_tokens,
                top_p=config.top_p if hasattr(config, 'top_p') else 1.0,
                frequency_penalty=config.frequency_penalty if hasattr(config, 'frequency_penalty') else 0.0,
                presence_penalty=config.presence_penalty if hasattr(config, 'presence_penalty') else 0.0,
                stop=config.stop if hasattr(config, 'stop') else None,
                seed=config.seed if hasattr(config, 'seed') else None
            )
            
            # Extract response
            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Build usage stats
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            return LLMResponse(
                content=content,
                model=model_name,
                usage=usage,
                finish_reason=finish_reason,
                response_time=response_time,
                metadata={
                    "id": response.id,
                    "created": response.created,
                    "system_fingerprint": getattr(response, 'system_fingerprint', None)
                }
            )
        
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
    
    async def generate_stream(
        self,
        messages: List[LLMMessage],
        config: LLMGenerationConfig,
        model: Optional[LLMModel] = None
    ) -> AsyncGenerator[str, None]:
        """Stream text completion from OpenAI."""
        
        try:
            # Convert messages
            openai_messages = self._convert_messages(messages)
            
            # Determine model
            model_name = model.value if model else self.default_model
            
            # Create streaming request
            stream = await self.client.chat.completions.create(
                model=model_name,
                messages=openai_messages,
                temperature=config.temperature if hasattr(config, 'temperature') else self.default_temperature,
                max_tokens=config.max_tokens if hasattr(config, 'max_tokens') else self.default_max_tokens,
                stream=True
            )
            
            # Yield chunks
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}")
            raise
    
    async def chat_with_tools(
        self,
        messages: List[LLMMessage],
        tools: List[Dict[str, Any]],
        config: LLMGenerationConfig,
        model: Optional[LLMModel] = None
    ) -> LLMResponse:
        """Chat with tool calling support."""
        
        start_time = time.time()
        
        try:
            # Convert messages
            openai_messages = self._convert_messages(messages)
            
            # Determine model
            model_name = model.value if model else self.default_model
            
            # Convert tools to OpenAI format
            openai_tools = [
                {
                    "type": "function",
                    "function": tool
                }
                for tool in tools
            ]
            
            # Make API call with tools
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=openai_messages,
                tools=openai_tools,
                temperature=config.temperature if hasattr(config, 'temperature') else self.default_temperature,
                max_tokens=config.max_tokens if hasattr(config, 'max_tokens') else self.default_max_tokens
            )
            
            # Extract response
            choice = response.choices[0]
            content = choice.message.content or ""
            finish_reason = choice.finish_reason
            
            # Handle tool calls
            tool_calls = []
            if choice.message.tool_calls:
                for tool_call in choice.message.tool_calls:
                    tool_calls.append({
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    })
            
            response_time = time.time() - start_time
            
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            return LLMResponse(
                content=content,
                model=model_name,
                usage=usage,
                finish_reason=finish_reason,
                response_time=response_time,
                metadata={
                    "tool_calls": tool_calls,
                    "id": response.id
                }
            )
        
        except Exception as e:
            logger.error(f"OpenAI tool calling failed: {e}")
            raise
    
    async def embed_text(
        self,
        text: str,
        model: str = "text-embedding-ada-002"
    ) -> List[float]:
        """Generate embeddings for text."""
        
        try:
            response = await self.client.embeddings.create(
                model=model,
                input=text
            )
            
            return response.data[0].embedding
        
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise
    
    async def embed_batch(
        self,
        texts: List[str],
        model: str = "text-embedding-ada-002"
    ) -> List[List[float]]:
        """Generate embeddings for batch of texts."""
        
        try:
            response = await self.client.embeddings.create(
                model=model,
                input=texts
            )
            
            return [item.embedding for item in response.data]
        
        except Exception as e:
            logger.error(f"OpenAI batch embedding failed: {e}")
            raise
    
    async def analyze_code_complexity(
        self,
        code: str,
        language: str
    ) -> Dict[str, Any]:
        """Analyze code complexity using LLM."""
        
        prompt = f"""Analyze the complexity of this {language} code:

```{language}
{code}
```

Provide a JSON response with:
- cyclomatic_complexity (integer estimate)
- cognitive_complexity (integer estimate)
- maintainability_score (0-100)
- issues (list of strings)
- suggestions (list of strings)
"""
        
        messages = [
            LLMMessage(role="system", content="You are a code analysis expert. Respond with valid JSON only."),
            LLMMessage(role="user", content=prompt)
        ]
        
        config = LLMGenerationConfig(temperature=0.3, max_tokens=1000)
        response = await self.generate(messages, config)
        
        # Parse JSON response
        import json
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse complexity analysis JSON")
            return {
                "cyclomatic_complexity": 0,
                "cognitive_complexity": 0,
                "maintainability_score": 50,
                "issues": [],
                "suggestions": []
            }
    
    async def generate_tests(
        self,
        code: str,
        language: str,
        framework: Optional[str] = None
    ) -> str:
        """Generate tests for code."""
        
        framework_text = f" using {framework}" if framework else ""
        prompt = f"""Generate comprehensive unit tests for this {language} code{framework_text}:

```{language}
{code}
```

Include:
- Test setup and teardown
- Happy path tests
- Edge cases
- Error handling tests
- Mock external dependencies
"""
        
        messages = [
            LLMMessage(role="system", content=f"You are an expert in {language} testing."),
            LLMMessage(role="user", content=prompt)
        ]
        
        config = LLMGenerationConfig(temperature=0.5, max_tokens=2000)
        response = await self.generate(messages, config)
        
        return response.content
    
    async def refactor_code(
        self,
        code: str,
        language: str,
        improvements: List[str]
    ) -> str:
        """Refactor code with specific improvements."""
        
        improvements_text = "\n".join(f"- {imp}" for imp in improvements)
        prompt = f"""Refactor this {language} code with these improvements:

{improvements_text}

Original code:
```{language}
{code}
```

Provide the refactored code with improvements applied.
"""
        
        messages = [
            LLMMessage(role="system", content=f"You are an expert {language} developer focused on clean code."),
            LLMMessage(role="user", content=prompt)
        ]
        
        config = LLMGenerationConfig(temperature=0.3, max_tokens=3000)
        response = await self.generate(messages, config)
        
        return response.content
    
    @property
    def available_models(self) -> List[str]:
        """Get list of available OpenAI models."""
        return [
            "gpt-4",
            "gpt-4-turbo-preview",
            "gpt-3.5-turbo",
            "gpt-4-32k"
        ]
    
    @property
    def provider(self) -> LLMProvider:
        """Get LLM provider."""
        return LLMProvider.OPENAI
    
    def _convert_messages(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """Convert LLMMessage objects to OpenAI format."""
        
        openai_messages = []
        
        for msg in messages:
            message_dict = {
                "role": msg.role,
                "content": msg.content
            }
            
            if msg.name:
                message_dict["name"] = msg.name
            
            if msg.tool_call_id:
                message_dict["tool_call_id"] = msg.tool_call_id
            
            if msg.tool_calls:
                message_dict["tool_calls"] = msg.tool_calls
            
            openai_messages.append(message_dict)
        
        return openai_messages
