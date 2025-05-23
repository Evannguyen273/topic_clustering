"""
Utilities Module
Common utility functions and decorators for monitoring, validation, and helper functions.
"""

import time
import logging
import functools
import threading
from typing import Any, Callable, Dict, Optional
import json
from datetime import datetime

try:
    from exceptions import RateLimitError, ValidationError
except ImportError:
    # Fallback for direct execution
    import os
    import sys
    sys.path.append(os.path.dirname(__file__))
    from exceptions import RateLimitError, ValidationError


def monitor_performance(func: Callable) -> Callable:
    """Decorator to monitor function performance"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            logging.info("Function %s executed successfully in %.2f seconds", 
                        func.__name__, execution_time)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logging.error("Function %s failed after %.2f seconds: %s", 
                         func.__name__, execution_time, e)
            raise
    
    return wrapper


class CircuitBreaker:
    """Circuit breaker pattern implementation for external services"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function with circuit breaker protection"""
        with self._lock:
            if self.state == 'OPEN':
                if self._should_attempt_reset():
                    self.state = 'HALF_OPEN'
                else:
                    raise Exception(f"Circuit breaker is OPEN. Service unavailable.")
            
            try:
                result = func(*args, **kwargs)
                
                if self.state == 'HALF_OPEN':
                    self.state = 'CLOSED'
                    self.failure_count = 0
                    logging.info("Circuit breaker reset to CLOSED state")
                
                return result
                
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = 'OPEN'
                    logging.warning("Circuit breaker opened due to failures")
                
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        return (time.time() - self.last_failure_time) >= self.recovery_timeout


class RateLimiter:
    """Simple rate limiter implementation"""
    
    def __init__(self, max_requests: int = 100, time_window: int = 3600):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = {}
        self._lock = threading.Lock()
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for the given identifier"""
        current_time = time.time()
        
        with self._lock:
            if identifier not in self.requests:
                self.requests[identifier] = []
            
            # Remove old requests outside the time window
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if current_time - req_time < self.time_window
            ]
            
            # Check if under the limit
            if len(self.requests[identifier]) < self.max_requests:
                self.requests[identifier].append(current_time)
                return True
            
            return False
    
    def check_or_raise(self, identifier: str):
        """Check rate limit and raise exception if exceeded"""
        if not self.is_allowed(identifier):
            raise RateLimitError(f"Rate limit exceeded for {identifier}")


def validate_user_input(text: str, max_length: int = 1000, min_length: int = 1) -> str:
    """Validate and sanitize user input"""
    if not isinstance(text, str):
        raise ValidationError("Input must be a string")
    
    # Strip whitespace
    text = text.strip()
    
    if len(text) < min_length:
        raise ValidationError(f"Input must be at least {min_length} characters long")
    
    if len(text) > max_length:
        raise ValidationError(f"Input must not exceed {max_length} characters")
    
    # Basic SQL injection prevention (simple check)
    dangerous_patterns = [
        'DROP TABLE', 'DELETE FROM', 'INSERT INTO', 'UPDATE SET',
        'ALTER TABLE', 'CREATE TABLE', 'EXEC ', 'EXECUTE',
        'UNION SELECT', '--', '/*', '*/', 'xp_', 'sp_'
    ]
    
    text_upper = text.upper()
    for pattern in dangerous_patterns:
        if pattern in text_upper:
            raise ValidationError(f"Input contains potentially dangerous content: {pattern}")
    
    return text


def format_conversation_history(conversations: list, max_history: int = 3) -> str:
    """Format conversation history for display or logging"""
    if not conversations:
        return "No conversation history"
    
    # Get the last few conversations
    recent_conversations = conversations[-max_history:]
    
    formatted_lines = []
    for i, (question, response) in enumerate(recent_conversations, 1):
        formatted_lines.append(f"Q{i}: {question[:100]}{'...' if len(question) > 100 else ''}")
        formatted_lines.append(f"A{i}: {response[:100]}{'...' if len(response) > 100 else ''}")
        formatted_lines.append("-" * 50)
    
    return "\n".join(formatted_lines)


def safe_json_loads(json_str: str, default=None) -> Any:
    """Safely parse JSON string with fallback"""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError) as e:
        logging.warning("Failed to parse JSON: %s", e)
        return default if default is not None else {}


def safe_json_dumps(obj: Any, default=None) -> str:
    """Safely serialize object to JSON string"""
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except (TypeError, ValueError) as e:
        logging.warning("Failed to serialize to JSON: %s", e)
        return json.dumps({"error": "Serialization failed"}) if default is None else default


def get_timestamp() -> str:
    """Get current timestamp in ISO format"""
    return datetime.now().isoformat()


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length with suffix"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations"""
    import re
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip('. ')
    # Limit length
    return sanitized[:255]


def retry_with_exponential_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0
):
    """Decorator for retrying functions with exponential backoff"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logging.error("Function %s failed after %d attempts: %s", 
                                    func.__name__, max_retries, e)
                        raise
                    
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    logging.warning("Function %s failed (attempt %d/%d), retrying in %.2f seconds: %s",
                                  func.__name__, attempt + 1, max_retries, delay, e)
                    time.sleep(delay)
            
            return None  # Should never reach here
        
        return wrapper
    return decorator


class Timer:
    """Context manager for timing operations"""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        logging.debug("Starting: %s", self.description)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if exc_type is None:
            logging.debug("Completed: %s in %.2f seconds", self.description, duration)
        else:
            logging.error("Failed: %s after %.2f seconds", self.description, duration)
    
    @property
    def duration(self) -> float:
        """Get the duration of the timed operation"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0


def health_check() -> Dict[str, Any]:
    """Basic health check function"""
    return {
        "status": "healthy",
        "timestamp": get_timestamp(),
        "uptime": time.time()  # This should be calculated from app start time in real implementation
    }


def memory_usage_mb() -> float:
    """Get current memory usage in MB"""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0