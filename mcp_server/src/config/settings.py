"""
Configuration management for MCP server.
Uses Pydantic settings with environment variable support.
Production-ready with validation and type safety.
"""

import os
import json
from typing import List, Optional, Dict, Any
from pydantic import BaseSettings, Field, validator, PostgresDsn
from pydantic.networks import AnyHttpUrl
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    Uses 12-factor app principles.
    """

    # Application
    app_name: str = "Code Documentation MCP Server"
    app_version: str = "2.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="production", env="ENVIRONMENT")

    # Server
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=4, env="WORKERS")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        env="CORS_ORIGINS"
    )

    # Database
    database_url: Optional[PostgresDsn] = Field(
        default=None,
        env="DATABASE_URL"
    )

    # Redis (for caching)
    redis_url: str = Field(
        default="redis://localhost:6379",
        env="REDIS_URL"
    )
    redis_ttl: int = Field(
        default=3600,
        env="REDIS_TTL"
    )

    # Vector Database
    qdrant_url: str = Field(
        default="http://localhost:6333",
        env="QDRANT_URL"
    )
    qdrant_collection: str = Field(
        default="documentation",
        env="QDRANT_COLLECTION"
    )

    # Embedding Model
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL"
    )
    embedding_dimension: int = Field(
        default=384,
        env="EMBEDDING_DIMENSION"
    )

    # Rate Limiting
    rate_limit_per_minute: int = Field(
        default=100,
        env="RATE_LIMIT_PER_MINUTE"
    )
    rate_limit_burst: int = Field(
        default=20,
        env="RATE_LIMIT_BURST"
    )

    # Documentation Sources
    docs_update_interval_hours: int = Field(
        default=6,
        env="DOCS_UPDATE_INTERVAL_HOURS"
    )
    docs_cache_dir: str = Field(
        default="./data/documentation",
        env="DOCS_CACHE_DIR"
    )

    # Monitoring
    metrics_port: int = Field(
        default=9090,
        env="METRICS_PORT"
    )
    health_check_interval: int = Field(
        default=30,
        env="HEALTH_CHECK_INTERVAL"
    )

    # Security
    api_key: Optional[str] = Field(
        default=None,
        env="API_KEY"
    )
    enable_auth: bool = Field(
        default=False,
        env="ENABLE_AUTH"
    )

    # External APIs
    openai_api_key: Optional[str] = Field(
        default=None,
        env="OPENAI_API_KEY"
    )
    github_token: Optional[str] = Field(
        default=None,
        env="GITHUB_TOKEN"
    )

    # Performance
    max_concurrent_tools: int = Field(
        default=50,
        env="MAX_CONCURRENT_TOOLS"
    )
    tool_timeout_seconds: int = Field(
        default=60,
        env="TOOL_TIMEOUT_SECONDS"
    )

    # Feature Flags
    enable_background_updates: bool = Field(
        default=True,
        env="ENABLE_BACKGROUND_UPDATES"
    )
    enable_semantic_search: bool = Field(
        default=True,
        env="ENABLE_SEMANTIC_SEARCH"
    )
    enable_code_examples: bool = Field(
        default=True,
        env="ENABLE_CODE_EXAMPLES"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @validator("log_level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()

    @validator("environment")
    def validate_environment(cls, v):
        valid_environments = ["development", "staging", "production", "test"]
        if v not in valid_environments:
            raise ValueError(f"Environment must be one of {valid_environments}")
        return v

    @property
    def is_development(self) -> bool:
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_testing(self) -> bool:
        return self.environment == "test"

    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        if self.database_url:
            return {
                "url": str(self.database_url),
                "pool_size": 20,
                "max_overflow": 10,
                "pool_recycle": 300,
                "echo": self.debug
            }
        return {}

    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration."""
        return {
            "url": self.redis_url,
            "ttl": self.redis_ttl,
            "encoding": "utf-8",
            "decode_responses": True
        }

    def get_qdrant_config(self) -> Dict[str, Any]:
        """Get Qdrant configuration."""
        return {
            "url": self.qdrant_url,
            "collection": self.qdrant_collection,
            "timeout": 30,
            "prefer_grpc": False
        }

    def get_tool_config(self) -> Dict[str, Any]:
        """Get tool configuration."""
        return {
            "max_concurrent": self.max_concurrent_tools,
            "timeout_seconds": self.tool_timeout_seconds,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "rate_limit_burst": self.rate_limit_burst
        }

    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration."""
        return {
            "metrics_port": self.metrics_port,
            "health_check_interval": self.health_check_interval,
            "enable_prometheus": self.is_production
        }

    def get_feature_config(self) -> Dict[str, Any]:
        """Get feature configuration."""
        return {
            "background_updates": self.enable_background_updates,
            "semantic_search": self.enable_semantic_search,
            "code_examples": self.enable_code_examples
        }

    def log_configuration(self):
        """Log configuration (excluding secrets)."""
        config_dict = self.dict(exclude={
            "api_key", "openai_api_key", "github_token",
            "database_url"  # Don't log full URL
        })

        logger.info("Application Configuration:")
        for key, value in config_dict.items():
            logger.info(f"  {key}: {value}")

    def validate_configuration(self):
        """Validate configuration and log warnings."""
        warnings = []

        if self.is_production and self.debug:
            warnings.append("Debug mode is enabled in production")

        if not self.api_key and self.enable_auth:
            warnings.append("Authentication enabled but no API key configured")

        if not self.openai_api_key and self.enable_semantic_search:
            warnings.append("Semantic search enabled but no OpenAI API key")

        for warning in warnings:
            logger.warning(f"Configuration warning: {warning}")

        return len(warnings) == 0


# Global settings instance
settings = Settings()

# Validate on import
if __name__ == "__main__":
    settings.log_configuration()
    is_valid = settings.validate_configuration()
    if not is_valid:
        logger.warning("Configuration validation failed")