"""
Database Manager Module
Handles all database operations including connection management, initialization, and queries.
"""

import pyodbc
import threading
import logging
from contextlib import contextmanager
from typing import Optional

try:
    from config import config
    from exceptions import DatabaseError
except ImportError:
    # Fallback for direct execution
    import os
    import sys
    sys.path.append(os.path.dirname(__file__))
    from config import config
    from exceptions import DatabaseError


class DatabaseManager:
    """Thread-safe database connection manager with connection pooling"""
    
    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string or config.database.connection_string
        self._lock = threading.Lock()
        self._validate_connection_string()
        
    def _validate_connection_string(self):
        """Validate the connection string format"""
        if not self.connection_string:
            raise DatabaseError("Database connection string is required")
        
        # Basic validation for SQL Server connection string
        required_parts = ['Server=', 'Database=']
        for part in required_parts:
            if part not in self.connection_string:
                logging.warning("Connection string may be missing %s", part)
    
    @contextmanager
    def get_connection(self):
        """Thread-safe database connection context manager with retry logic"""
        conn = None
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                conn = pyodbc.connect(
                    self.connection_string,
                    timeout=config.database.pool_timeout
                )
                logging.debug("Database connection established successfully")
                yield conn
                break
                
            except pyodbc.Error as e:
                retry_count += 1
                
                if retry_count < max_retries:
                    logging.warning("Database connection failed (attempt %d/%d): %s. Retrying...", 
                                   retry_count, max_retries, e)
                    continue
                else:
                    logging.error("Database connection failed (attempt %d/%d): %s", 
                                 retry_count, max_retries, e)
                    raise DatabaseError(f"Failed to connect to database after {max_retries} attempts: {e}") from e
                    
            except Exception as e:
                logging.error("Unexpected database error: %s", e)
                raise DatabaseError(f"Unexpected database connection error: {e}") from e
                
            finally:
                if conn:
                    try:
                        conn.close()
                        logging.debug("Database connection closed successfully")
                    except pyodbc.Error as e:
                        logging.warning("Error closing database connection: %s", e)
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                return result[0] == 1
        except DatabaseError:
            raise
        except Exception as e:
            logging.error("Database connection test failed: %s", e)
            return False
    
    def initialize_tables(self):
        """Initialize required database tables"""
        create_user_conversations_table = """
        IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES 
                      WHERE TABLE_SCHEMA = 'stage' 
                      AND TABLE_NAME = 'UserConversations')
        BEGIN
            CREATE TABLE stage.UserConversations (
                user_id NVARCHAR(50) PRIMARY KEY,
                conversation_history NVARCHAR(MAX),
                last_updated_time DATETIME DEFAULT GETDATE(),
                feedback NVARCHAR(MAX),
                created_time DATETIME DEFAULT GETDATE()
            )
            
            -- Create indexes for better performance
            CREATE INDEX IX_UserConversations_LastUpdated 
            ON stage.UserConversations(last_updated_time);
        END
        """
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(create_user_conversations_table)
                conn.commit()
                logging.info("Database tables initialized successfully")
                
        except Exception as e:
            logging.error("Failed to initialize database tables: %s", e)
            raise DatabaseError(f"Database initialization failed: {e}") from e
    
    def execute_query(self, query: str, params: tuple = None, fetch_one: bool = False):
        """Execute a query with parameters"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                if query.strip().upper().startswith('SELECT'):
                    if fetch_one:
                        return cursor.fetchone()
                    else:
                        return cursor.fetchall()
                else:
                    conn.commit()
                    return cursor.rowcount
                    
        except Exception as e:
            logging.error("Query execution failed: %s", e)
            raise DatabaseError(f"Query execution error: {e}") from e
    
    def get_conversation_history(self, user_id: str) -> Optional[str]:
        """Get conversation history for a specific user"""
        query = "SELECT conversation_history FROM stage.UserConversations WHERE user_id = ?"
        
        try:
            result = self.execute_query(query, (user_id,), fetch_one=True)
            return result[0] if result else None
            
        except DatabaseError:
            raise
        except Exception as e:
            logging.error("Failed to get conversation history for user %s: %s", user_id, e)
            return None
    
    def save_conversation_history(self, user_id: str, conversation_data: str) -> bool:
        """Save or update conversation history for a user"""
        # Check if user exists
        check_query = "SELECT 1 FROM stage.UserConversations WHERE user_id = ?"
        
        try:
            exists = self.execute_query(check_query, (user_id,), fetch_one=True)
            
            if exists:
                # Update existing record
                update_query = """
                UPDATE stage.UserConversations 
                SET conversation_history = ?, last_updated_time = GETDATE() 
                WHERE user_id = ?
                """
                self.execute_query(update_query, (conversation_data, user_id))
            else:
                # Insert new record
                insert_query = """
                INSERT INTO stage.UserConversations 
                (user_id, conversation_history, last_updated_time, created_time) 
                VALUES (?, ?, GETDATE(), GETDATE())
                """
                self.execute_query(insert_query, (user_id, conversation_data))
            
            logging.debug("Conversation history saved for user %s", user_id)
            return True
            
        except Exception as e:
            logging.error("Failed to save conversation history for user %s: %s", user_id, e)
            return False
    
    def cleanup_old_conversations(self, days_old: int = 30) -> int:
        """Clean up old conversation records"""
        cleanup_query = """
        DELETE FROM stage.UserConversations 
        WHERE last_updated_time < DATEADD(day, -?, GETDATE())
        """
        
        try:
            rows_affected = self.execute_query(cleanup_query, (days_old,))
            logging.info("Cleaned up %d old conversation records", rows_affected)
            return rows_affected
            
        except Exception as e:
            logging.error("Failed to cleanup old conversations: %s", e)
            return 0
    
    def get_database_stats(self) -> dict:
        """Get database statistics"""
        stats_query = """
        SELECT 
            COUNT(*) as total_users,
            MAX(last_updated_time) as last_activity,
            MIN(created_time) as first_activity
        FROM stage.UserConversations
        """
        
        try:
            result = self.execute_query(stats_query, fetch_one=True)
            if result:
                return {
                    "total_users": result[0],
                    "last_activity": result[1],
                    "first_activity": result[2]
                }
            return {}
            
        except Exception as e:
            logging.error("Failed to get database stats: %s", e)
            return {}


# Global database manager instance
db_manager = DatabaseManager()