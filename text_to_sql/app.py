"""
Legacy app.py - Updated to use modular structure
This file maintains backward compatibility while using the new modular components.
"""

# Import from modular structure
from api_routes import app
from conversation_manager import conversation_manager
from database_fixed import db_manager
from ai_services import sql_agent_service
from config import config

# For backward compatibility, expose some components
ConversationManager = conversation_manager.__class__
DatabaseManager = db_manager.__class__

if __name__ == '__main__':
    print("⚠️  Using legacy app.py entry point")
    print("💡 Consider using 'python main.py' for the full modular experience")
    
    # Run Flask app directly
    app.run(
        host=config.app.host,
        port=config.app.port,
        debug=config.app.debug
    )