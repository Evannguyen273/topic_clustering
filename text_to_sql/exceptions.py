"""
Custom Exceptions Module
Defines application-specific exceptions for better error handling.
"""


class BaseAppException(Exception):
    """Base exception class for the application"""
    
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code or self.__class__.__name__.upper()
        super().__init__(self.message)


class DatabaseError(BaseAppException):
    """Raised when database operations fail"""


class CacheError(BaseAppException):
    """Raised when cache operations fail"""


class ConfigurationError(BaseAppException):
    """Raised when configuration is invalid"""


class ValidationError(BaseAppException):
    """Raised when input validation fails"""


class RateLimitError(BaseAppException):
    """Raised when rate limits are exceeded"""


class AIServiceError(BaseAppException):
    """Raised when AI service operations fail"""


class ConversationError(BaseAppException):
    """Raised when conversation management fails"""


class AuthenticationError(BaseAppException):
    """Raised when authentication fails"""


class AuthorizationError(BaseAppException):
    """Raised when authorization fails"""