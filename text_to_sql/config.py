"""
Configuration Module
Centralized configuration management for the application.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    connection_string: str = ""
    pool_size: int = 10
    pool_timeout: int = 30
    
    def __post_init__(self):
        if not self.connection_string:
            self.connection_string = os.environ.get('connection_String', '')


@dataclass
class RedisConfig:
    """Redis configuration settings"""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    ssl: bool = False
    socket_timeout: int = 5
    connection_timeout: int = 5
    
    def __post_init__(self):
        self.host = os.environ.get('REDIS_HOST', self.host)
        self.port = int(os.environ.get('REDIS_PORT', self.port))
        self.password = os.environ.get('REDIS_PASSWORD', self.password)
        self.db = int(os.environ.get('REDIS_DB', self.db))
        self.ssl = os.environ.get('REDIS_SSL', 'false').lower() == 'true'


@dataclass
class AzureOpenAIConfig:
    """Azure OpenAI configuration settings"""
    api_key: str = ""
    endpoint: str = ""
    api_version: str = "2023-12-01-preview"
    deployment_name: str = "gpt-35-turbo"
    temperature: float = 0
    max_tokens: Optional[int] = None
    max_retries: int = 2
    
    def __post_init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY", self.api_key)
        self.endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", self.endpoint)
        self.deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", self.deployment_name)


@dataclass
class ConversationConfig:
    """Conversation management configuration"""
    max_conversations_per_user: int = 50
    cache_ttl: int = 3600  # 1 hour in seconds
    max_history_context: int = 3


@dataclass
class AppConfig:
    """Application-wide configuration"""
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 5000
    
    def __post_init__(self):
        self.debug = os.environ.get('DEBUG', 'false').lower() == 'true'
        self.host = os.environ.get('HOST', self.host)
        self.port = int(os.environ.get('PORT', self.port))


@dataclass
class SecurityConfig:
    """Security configuration settings"""
    enable_cors: bool = True
    cors_origins: str = "*"
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_period: int = 3600  # per hour
    
    def __post_init__(self):
        self.enable_cors = os.environ.get('ENABLE_CORS', 'true').lower() == 'true'
        self.cors_origins = os.environ.get('CORS_ORIGINS', self.cors_origins)
        self.rate_limit_enabled = os.environ.get('RATE_LIMIT_ENABLED', 'true').lower() == 'true'


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    file_path: Optional[str] = None
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5
    
    def __post_init__(self):
        self.level = os.environ.get('LOG_LEVEL', self.level)
        self.file_path = os.environ.get('LOG_FILE_PATH', self.file_path)


@dataclass
class ProductionConfig:
    """Main configuration class containing all config sections"""
    database: DatabaseConfig
    redis: RedisConfig
    azure_openai: AzureOpenAIConfig
    conversation: ConversationConfig
    app: AppConfig
    security: SecurityConfig
    logging: LoggingConfig
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.azure_openai = AzureOpenAIConfig()
        self.conversation = ConversationConfig()
        self.app = AppConfig()
        self.security = SecurityConfig()
        self.logging = LoggingConfig()
    
    def validate(self) -> list:
        """Validate configuration and return list of errors"""
        errors = []
        
        if not self.database.connection_string:
            errors.append("Database connection string is required")
        
        if not self.azure_openai.api_key:
            errors.append("Azure OpenAI API key is required")
        
        if not self.azure_openai.endpoint:
            errors.append("Azure OpenAI endpoint is required")
        
        return errors
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return not self.app.debug and os.environ.get('ENVIRONMENT', '').lower() == 'production'


# Global configuration instance
config = ProductionConfig()

# Validate configuration on import
config_errors = config.validate()
if config_errors:
    import logging
    logger = logging.getLogger(__name__)
    for error in config_errors:
        logger.warning("Configuration warning: %s", error)