"""
Integration script to replace the existing ConversationManager with Azure Redis support
"""

import os
import logging
from azure_conversation_manager import AzureConversationManager

# Import your existing config and db_manager
from app import config, db_manager

def upgrade_to_azure_redis():
    """Replace the existing conversation manager with Azure Redis support"""
    
    # Initialize the new Azure conversation manager
    azure_conversation_manager = AzureConversationManager(db_manager, config)
    
    # Test the connection
    try:
        stats = azure_conversation_manager.get_cache_stats()
        logging.info(f"Azure Redis Cache Stats: {stats}")
        
        if stats['redis_connected']:
            logging.info("✅ Azure Redis Cache is working properly!")
        else:
            logging.warning("⚠️ Azure Redis not available, using memory fallback")
            
        return azure_conversation_manager
        
    except Exception as e:
        logging.error(f"❌ Error initializing Azure Redis: {e}")
        return None

def test_conversation_operations(conversation_manager):
    """Test basic conversation operations"""
    test_user_id = "test_user_123"
    
    try:
        # Test adding conversations
        conversation_manager.add_conversation(
            test_user_id, 
            "What is the total number of problems?", 
            "There are 1,234 problems in the database."
        )
        
        # Test retrieving conversations
        conversations = conversation_manager.get_user_conversations(test_user_id)
        logging.info(f"Retrieved {len(conversations)} conversations for test user")
        
        # Test cache clearing
        conversation_manager.clear_user_cache(test_user_id)
        logging.info("Cache cleared successfully")
        
        return True
        
    except Exception as e:
        logging.error(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Upgrade to Azure Redis
    azure_manager = upgrade_to_azure_redis()
    
    if azure_manager:
        # Test the functionality
        if test_conversation_operations(azure_manager):
            print("🎉 Azure Redis integration successful!")
            print("\nTo use in your main app.py, replace:")
            print("  conversation_manager = ConversationManager(db_manager)")
            print("With:")
            print("  from azure_conversation_manager import AzureConversationManager")
            print("  conversation_manager = AzureConversationManager(db_manager, config)")
        else:
            print("❌ Integration test failed")
    else:
        print("❌ Failed to initialize Azure Redis")