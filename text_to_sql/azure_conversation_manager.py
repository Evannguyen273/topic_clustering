"""
Azure Redis Conversation Manager Implementation
This module provides Azure-optimized conversation management with Redis caching.
"""

import json
import logging
import threading
from datetime import datetime
from typing import List, Optional, Tuple
import os


class AzureConversationManager:
    """Enhanced ConversationManager with Azure Cache for Redis support"""
    
    def __init__(self, db_manager, config):
        self.db_manager = db_manager
        self.config = config
        self._conversation_cache = {}
        self._feedback_cache = {}
        self._lock = threading.Lock()
        self.max_conversations_per_user = config.MAX_CONVERSATIONS_PER_USER
        
        # Initialize Azure Redis Cache connection
        self._setup_azure_redis()
    
    def _setup_azure_redis(self):
        """Setup Azure Cache for Redis connection"""
        try:
            import redis
            
            # Azure Redis configuration
            redis_config = {
                'host': self.config.AZURE_REDIS_HOST,
                'port': self.config.AZURE_REDIS_PORT,
                'decode_responses': True,
                'socket_connect_timeout': 30,
                'socket_timeout': 30,
                'retry_on_timeout': True,
                'health_check_interval': 30
            }
            
            # Add SSL configuration for Azure
            if self.config.AZURE_REDIS_SSL:
                redis_config.update({
                    'ssl': True,
                    'ssl_cert_reqs': None,
                    'ssl_check_hostname': False
                })
            
            # Add password if provided
            if self.config.AZURE_REDIS_PASSWORD:
                redis_config['password'] = self.config.AZURE_REDIS_PASSWORD
            
            # Add database if specified
            if self.config.AZURE_REDIS_DB:
                redis_config['db'] = self.config.AZURE_REDIS_DB
            
            self.redis_client = redis.Redis(**redis_config)
            
            # Test connection
            self.redis_client.ping()
            self.use_redis = True
            logging.info(f"Azure Redis Cache connected successfully to {self.config.AZURE_REDIS_HOST}")
            
        except ImportError:
            logging.warning("Redis library not installed. Install with: pip install redis")
            self._fallback_to_memory()
        except redis.ConnectionError as e:
            logging.warning(f"Azure Redis connection failed: {e}")
            self._fallback_to_memory()
        except Exception as e:
            logging.warning(f"Unexpected Redis error: {e}")
            self._fallback_to_memory()
    
    def _fallback_to_memory(self):
        """Fallback to memory-based caching"""
        self.redis_client = None
        self.use_redis = False
        self._memory_cache = {}
        logging.info("Using memory cache as fallback")
    
    def get_user_conversations(self, user_id: str) -> List[Tuple[str, str]]:
        """Get user conversations from cache or database"""
        cache_key = f"conversations:{user_id}"
        
        with self._lock:
            # Try Redis cache first
            if self.use_redis:
                try:
                    cached_data = self.redis_client.get(cache_key)
                    if cached_data:
                        conversations_data = json.loads(cached_data)
                        return [(conv.get('user_query', ''), conv.get('response', '')) 
                               for conv in conversations_data if isinstance(conv, dict)]
                except Exception as e:
                    logging.error(f"Redis error while fetching conversations: {e}")
            
            # Try memory cache
            if cache_key in self._conversation_cache:
                return self._conversation_cache[cache_key]
                
            # Load from database
            try:
                with self.db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT conversation_history FROM stage.UserConversations WHERE user_id = ?",
                        (user_id,)
                    )
                    result = cursor.fetchone()
                    
                    if result and result[0]:
                        try:
                            conversations_data = json.loads(result[0])
                            conversations = []
                            for conv in conversations_data:
                                if isinstance(conv, dict):
                                    conversations.append((
                                        conv.get('user_query', ''), 
                                        conv.get('response', '')
                                    ))
                            
                            # Cache the conversations
                            self._cache_conversations(cache_key, conversations)
                            return conversations
                        except json.JSONDecodeError:
                            logging.error(f"Failed to parse conversation history for user {user_id}")
                            
            except Exception as e:
                logging.error(f"Error loading conversations for user {user_id}: {e}")
                
            # Return empty list if no conversations found
            empty_conversations = []
            self._conversation_cache[cache_key] = empty_conversations
            return empty_conversations
    
    def _cache_conversations(self, cache_key: str, conversations: List[Tuple[str, str]]):
        """Cache conversations in Azure Redis and memory"""
        # Convert tuples to dict format for JSON serialization
        conversations_data = []
        for q, r in conversations:
            conversations_data.append({
                "user_query": q,
                "response": r,
                "timestamp": datetime.now().isoformat()
            })
        
        # Cache in Azure Redis if available
        if self.use_redis:
            try:
                self.redis_client.setex(
                    cache_key, 
                    self.config.CONVERSATION_CACHE_TTL, 
                    json.dumps(conversations_data)
                )
                logging.debug(f"Cached conversations for {cache_key} in Azure Redis")
            except Exception as e:
                logging.error(f"Azure Redis caching error: {e}")
        
        # Always cache in memory as backup
        self._conversation_cache[cache_key] = conversations
    
    def add_conversation(self, user_id: str, question: str, response: str):
        """Add conversation with automatic cleanup"""
        with self._lock:
            conversations = self.get_user_conversations(user_id)
            conversations.append((question, response))
            
            # Keep only last N conversations to prevent memory issues
            if len(conversations) > self.max_conversations_per_user:
                conversations = conversations[-self.max_conversations_per_user:]
                
            cache_key = f"conversations:{user_id}"
            self._cache_conversations(cache_key, conversations)
            
            # Save to database asynchronously
            self._save_conversations_async(user_id, conversations)
            
            return conversations
    
    def _save_conversations_async(self, user_id: str, conversations: List[Tuple[str, str]]):
        """Save conversations to database in background thread"""
        def save_to_db():
            try:
                # Convert tuples to dict format for JSON storage
                conversations_data = []
                for q, r in conversations:
                    conversations_data.append({
                        "user_query": q,
                        "response": r,
                        "timestamp": datetime.now().isoformat()
                    })
                
                with self.db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1 FROM stage.UserConversations WHERE user_id = ?", (user_id,))
                    exists = cursor.fetchone()
                    
                    if exists:
                        cursor.execute(
                            "UPDATE stage.UserConversations SET conversation_history = ?, last_updated_time = ? WHERE user_id = ?",
                            (json.dumps(conversations_data), datetime.now(), user_id)
                        )
                    else:
                        cursor.execute(
                            "INSERT INTO stage.UserConversations (user_id, conversation_history, last_updated_time) VALUES (?, ?, ?)",
                            (user_id, json.dumps(conversations_data), datetime.now())
                        )
                    conn.commit()
                    logging.debug(f"Saved conversations for user {user_id} to database")
                    
            except Exception as e:
                logging.error(f"Failed to save conversations for user {user_id}: {e}")
        
        thread = threading.Thread(target=save_to_db)
        thread.daemon = True
        thread.start()
    
    def clear_user_cache(self, user_id: str):
        """Clear cache for a specific user"""
        cache_key = f"conversations:{user_id}"
        
        # Clear from Redis
        if self.use_redis:
            try:
                self.redis_client.delete(cache_key)
            except Exception as e:
                logging.error(f"Error clearing Redis cache for {user_id}: {e}")
        
        # Clear from memory
        if cache_key in self._conversation_cache:
            del self._conversation_cache[cache_key]
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        stats = {
            "redis_connected": self.use_redis,
            "memory_cache_size": len(self._conversation_cache)
        }
        
        if self.use_redis:
            try:
                info = self.redis_client.info()
                stats.update({
                    "redis_memory_usage": info.get('used_memory_human', 'N/A'),
                    "redis_connected_clients": info.get('connected_clients', 'N/A'),
                    "redis_total_commands_processed": info.get('total_commands_processed', 'N/A')
                })
            except Exception as e:
                stats["redis_error"] = str(e)
        
        return stats