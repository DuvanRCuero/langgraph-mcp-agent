"""
Application settings using Pydantic for configuration management.
Settings are loaded from environment variables and .env files.
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    
    All settings can be overridden via environment variables.
    Example: OPENAI_API_KEY=sk-xxx python main.py
    """
    
    # MCP Server Configuration
    mcp_server_url: str = Field(
        default="ws://localhost:8000/mcp",
        description="URL for MCP server WebSocket connection"
    )
    
    # OpenAI Configuration
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key for LLM access"
    )
    
    # LLM Configuration
    llm_model: str = Field(
        default="gpt-4",
        description="LLM model name to use"
    )
    llm_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature setting for LLM generation"
    )
    llm_max_tokens: int = Field(
        default=4000,
        gt=0,
        description="Maximum tokens for LLM generation"
    )
    
    # Qdrant Configuration
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant vector database URL"
    )
    qdrant_collection: str = Field(
        default="code_documentation",
        description="Qdrant collection name"
    )
    
    # Agent Configuration
    persist_checkpoints: bool = Field(
        default=True,
        description="Whether to persist agent checkpoints"
    )
    checkpoint_db: str = Field(
        default="checkpoints.db",
        description="Checkpoint database path"
    )
    
    # Feature Flags
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    enable_learning: bool = Field(
        default=True,
        description="Enable adaptive learning features"
    )
    enable_monitoring: bool = Field(
        default=True,
        description="Enable performance monitoring"
    )
    
    # Server Configuration
    host: str = Field(
        default="0.0.0.0",
        description="Server host address"
    )
    port: int = Field(
        default=8000,
        gt=0,
        lt=65536,
        description="Server port number"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    workers: int = Field(
        default=4,
        gt=0,
        description="Number of worker processes"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    def validate_required_settings(self) -> bool:
        """
        Validate that required settings are properly configured.
        
        Returns:
            bool: True if all required settings are valid
        
        Raises:
            ValueError: If required settings are missing or invalid
        """
        errors = []
        
        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY is required but not set")
        
        if not self.mcp_server_url:
            errors.append("MCP_SERVER_URL is required but not set")
        
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            errors.append(f"Invalid LOG_LEVEL: {self.log_level}")
        
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
        
        return True
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
