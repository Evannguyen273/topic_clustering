"""
Text-to-SQL Chatbot Package
A modular chatbot application for natural language to SQL queries.

This package provides:
- Natural language to SQL conversion using Azure OpenAI
- Multi-tier caching with Redis and memory fallback
- Conversation history management
- RESTful API endpoints
- Production-ready error handling and monitoring
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@company.com"

# Import main components for easy access
try:
    from .config import config
    from .exceptions import *
    from .database_fixed import db_manager
    from .conversation_manager import conversation_manager
    from .ai_services import sql_agent_service
    from .utils import *
    
    # Package-level logging
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Text-to-SQL package loaded successfully (v%s)", __version__)
    
except ImportError as e:
    # Allow package to be imported even if some dependencies are missing
    import logging
    logging.warning("Some package components could not be imported: %s", e)

# Public API
__all__ = [
    # Core modules
    'config',
    'db_manager', 
    'conversation_manager',
    'sql_agent_service',
    
    # Exceptions
    'DatabaseError',
    'CacheError', 
    'ValidationError',
    'RateLimitError',
    'AIServiceError',
    'ConversationError',
    'ConfigurationError',
    'AuthenticationError',
    'AuthorizationError',
    
    # Utilities
    'monitor_performance',
    'CircuitBreaker',
    'RateLimiter',
    'validate_user_input',
    'Timer',
]

def get_version():
    """Get package version"""
    return __version__

def get_package_info():
    """Get package information"""
    return {
        "name": "text-to-sql-chatbot",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "description": "A modular chatbot application for natural language to SQL queries"
    }