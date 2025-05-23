"""
Conversation Manager Module
Handles conversation storage, caching, and management with Redis support.
"""

import json
import threading
import logging
from datetime import datetime
from typing import List, Optional, Tuple

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available, using memory cache only")

try:
    from config import config
    from exceptions import CacheError
    from database import db_manager
except ImportError:
    # Fallback for direct execution
    import os
    import sys
    sys.path.append(os.path.dirname(__file__))
    from config import config
    from exceptions import CacheError
    from database import db_manager


class ConversationManager:
    """Enhanced conversation manager with multi-tier caching (Redis + Memory)"""
    
    def __init__(self, database_manager=None):
        self.db_manager = database_manager or db_manager
        self._lock = threading.Lock()
        self._memory_cache = {}  # Always available as fallback
        
        # Initialize Redis if available and configured
        self.redis_client = None
        self.use_redis = False
        
        if REDIS_AVAILABLE and config.redis.password:
            self._initialize_redis()
        
        # Configuration
        self.max_conversations_per_user = config.conversation.max_conversations_per_user
        self.cache_ttl = config.conversation.cache_ttl
        
        logging.info("ConversationManager initialized with Redis: %s", self.use_redis)
    
    def _initialize_redis(self):
        """Initialize Redis connection with error handling"""
        try:
            redis_config = {
                'host': config.redis.host,
                'port': config.redis.port,
                'decode_responses': True,
                'socket_timeout': config.redis.socket_timeout,
                'socket_connect_timeout': config.redis.connection_timeout,
            }
            
            if config.redis.password:
                redis_config['password'] = config.redis.password
            
            if config.redis.ssl:
                redis_config['ssl'] = True
                redis_config['ssl_cert_reqs'] = None
            
            if config.redis.db:
                redis_config['db'] = config.redis.db
            
            self.redis_client = redis.Redis(**redis_config)
            
            # Test connection
            self.redis_client.ping()
            self.use_redis = True
            logging.info("Redis connection established successfully")
            
        except Exception as e:
            logging.warning("Failed to connect to Redis: %s. Using memory cache only.", e)
            self.redis_client = None
            self.use_redis = False
    
    def _get_cache_key(self, user_id: str) -> str:
        """Generate cache key for user conversations"""
        return f"conversations:{user_id}"
    
    def _serialize_conversations(self, conversations: List[Tuple[str, str]]) -> str:
        """Serialize conversations to JSON string"""
        conversations_data = []
        for question, response in conversations:
            conversations_data.append({
                "user_query": question,
                "response": response,
                "timestamp": datetime.now().isoformat()
            })
        return json.dumps(conversations_data)
    
    def _deserialize_conversations(self, data: str) -> List[Tuple[str, str]]:
        """Deserialize conversations from JSON string"""
        try:
            conversations_data = json.loads(data)
            conversations = []
            
            for conv in conversations_data:
                if isinstance(conv, dict):
                    question = conv.get('user_query', '')
                    response = conv.get('response', '')
                    conversations.append((question, response))
            
            return conversations
            
        except json.JSONDecodeError as e:
            logging.error("Failed to deserialize conversations: %s", e)
            return []
    
    def _get_from_redis(self, cache_key: str) -> Optional[List[Tuple[str, str]]]:
        """Get conversations from Redis cache"""
        if not self.use_redis:
            return None
        
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return self._deserialize_conversations(cached_data)
            
        except redis.RedisError as e:
            logging.error("Redis get error: %s", e)
            # Don't raise, fall back to other methods
        
        return None
    
    def _set_to_redis(self, cache_key: str, conversations: List[Tuple[str, str]]):
        """Set conversations to Redis cache"""
        if not self.use_redis:
            return
        
        try:
            serialized_data = self._serialize_conversations(conversations)
            self.redis_client.setex(cache_key, self.cache_ttl, serialized_data)
            
        except redis.RedisError as e:
            logging.error("Redis set error: %s", e)
            # Don't raise, continue with memory cache
    
    def _get_from_memory(self, cache_key: str) -> Optional[List[Tuple[str, str]]]:
        """Get conversations from memory cache"""
        return self._memory_cache.get(cache_key)
    
    def _set_to_memory(self, cache_key: str, conversations: List[Tuple[str, str]]):
        """Set conversations to memory cache"""
        self._memory_cache[cache_key] = conversations
    
    def _get_from_database(self, user_id: str) -> List[Tuple[str, str]]:
        """Load conversations from database"""
        try:
            conversation_data = self.db_manager.get_conversation_history(user_id)
            if conversation_data:
                return self._deserialize_conversations(conversation_data)
            
        except Exception as e:
            logging.error("Failed to load conversations from database for user %s: %s", user_id, e)
        
        return []
    
    def get_user_conversations(self, user_id: str) -> List[Tuple[str, str]]:
        """Get user conversations with multi-tier caching"""
        cache_key = self._get_cache_key(user_id)
        
        with self._lock:
            # Try Redis cache first
            conversations = self._get_from_redis(cache_key)
            if conversations is not None:
                logging.debug("Conversations loaded from Redis for user %s", user_id)
                return conversations
            
            # Try memory cache
            conversations = self._get_from_memory(cache_key)
            if conversations is not None:
                logging.debug("Conversations loaded from memory for user %s", user_id)
                # Backfill Redis if available
                self._set_to_redis(cache_key, conversations)
                return conversations
            
            # Load from database
            conversations = self._get_from_database(user_id)
            logging.debug("Conversations loaded from database for user %s", user_id)
            
            # Cache in both Redis and memory
            self._set_to_redis(cache_key, conversations)
            self._set_to_memory(cache_key, conversations)
            
            return conversations
    
    def add_conversation(self, user_id: str, question: str, response: str) -> List[Tuple[str, str]]:
        """Add new conversation with automatic cleanup and persistence"""
        with self._lock:
            conversations = self.get_user_conversations(user_id)
            conversations.append((question, response))
            
            # Trim conversations if exceeded limit
            if len(conversations) > self.max_conversations_per_user:
                conversations = conversations[-self.max_conversations_per_user:]
                logging.debug("Trimmed conversations for user %s to %d items", 
                            user_id, self.max_conversations_per_user)
            
            cache_key = self._get_cache_key(user_id)
            
            # Update caches
            self._set_to_redis(cache_key, conversations)
            self._set_to_memory(cache_key, conversations)
            
            # Save to database asynchronously
            self._save_conversations_async(user_id, conversations)
            
            return conversations
    
    def _save_conversations_async(self, user_id: str, conversations: List[Tuple[str, str]]):
        """Save conversations to database in background thread"""
        def save_to_db():
            try:
                serialized_data = self._serialize_conversations(conversations)
                success = self.db_manager.save_conversation_history(user_id, serialized_data)
                
                if success:
                    logging.debug("Conversations saved to database for user %s", user_id)
                else:
                    logging.error("Failed to save conversations to database for user %s", user_id)
                    
            except Exception as e:
                logging.error("Error in async save for user %s: %s", user_id, e)
        
        thread = threading.Thread(target=save_to_db, daemon=True)
        thread.start()
    
    def clear_user_conversations(self, user_id: str) -> bool:
        """Clear all conversations for a user"""
        cache_key = self._get_cache_key(user_id)
        
        with self._lock:
            # Clear from caches
            if self.use_redis:
                try:
                    self.redis_client.delete(cache_key)
                except redis.RedisError as e:
                    logging.error("Failed to clear Redis cache for user %s: %s", user_id, e)
            
            self._memory_cache.pop(cache_key, None)
            
            # Clear from database
            try:
                return self.db_manager.save_conversation_history(user_id, "[]")
            except Exception as e:
                logging.error("Failed to clear database conversations for user %s: %s", user_id, e)
                return False
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        stats = {
            "memory_cache_size": len(self._memory_cache),
            "redis_available": self.use_redis,
            "max_conversations_per_user": self.max_conversations_per_user,
            "cache_ttl": self.cache_ttl
        }
        
        if self.use_redis:
            try:
                redis_info = self.redis_client.info('memory')
                stats["redis_memory_usage"] = redis_info.get('used_memory_human', 'Unknown')
                stats["redis_connected_clients"] = self.redis_client.info().get('connected_clients', 0)
            except redis.RedisError as e:
                logging.error("Failed to get Redis stats: %s", e)
                stats["redis_error"] = str(e)
        
        return stats
    
    def cleanup_cache(self, max_age_hours: int = 24):
        """Cleanup old entries from memory cache"""
        # This is a simple implementation - in production you might want
        # to track timestamps for memory cache entries too
        if len(self._memory_cache) > 1000:  # Arbitrary limit
            # Keep only the most recently accessed half
            cache_items = list(self._memory_cache.items())
            self._memory_cache = dict(cache_items[-500:])
            logging.info("Cleaned up memory cache, kept 500 most recent entries")


# Global conversation manager instance
conversation_manager = ConversationManager()