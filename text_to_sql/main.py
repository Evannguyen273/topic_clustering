"""
Text-to-SQL Chatbot Application
Main entry point for the modularized application.
"""

import os
import sys
import logging
from pathlib import Path

# Add current directory to Python path for module imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def setup_logging():
    """Configure application logging"""
    from config import config
    
    logging.basicConfig(
        level=getattr(logging, config.logging.level),
        format=config.logging.format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.logging.file_path) if config.logging.file_path else logging.NullHandler()
        ]
    )
    
    # Set specific logger levels
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('azure').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info("Logging configured successfully")
    return logger


def initialize_services():
    """Initialize all application services"""
    logger = logging.getLogger(__name__)
    
    try:
        # Import and validate configuration
        from config import config
        config_errors = config.validate()
        if config_errors:
            for error in config_errors:
                logger.warning("Configuration issue: %s", error)
        
        # Initialize database
        logger.info("Initializing database service...")
        from database_fixed import db_manager
        if db_manager.test_connection():
            logger.info("Database connection successful")
            # Initialize tables if needed
            try:
                db_manager.initialize_tables()
            except Exception as e:
                logger.warning("Table initialization skipped: %s", e)
        else:
            logger.warning("Database connection failed")
        
        # Initialize AI services
        logger.info("Initializing AI services...")
        from ai_services import sql_agent_service, database_service, llm_service
        logger.info("AI services initialized successfully")
        
        # Initialize conversation manager
        logger.info("Initializing conversation manager...")
        from conversation_manager import conversation_manager
        cache_stats = conversation_manager.get_cache_stats()
        logger.info("Conversation manager initialized - Redis: %s, Memory cache: %d entries", 
                   cache_stats['redis_available'], cache_stats['memory_cache_size'])
        
        return True
        
    except Exception as e:
        logger.error("Service initialization failed: %s", e)
        return False


def create_app():
    """Create and configure the Flask application"""
    logger = logging.getLogger(__name__)
    
    try:
        # Import Flask app from api_routes
        from api_routes import app
        
        # Add custom configuration
        from config import config
        app.config['SECRET_KEY'] = config.security.cors_origins
        app.config['DEBUG'] = config.app.debug
        
        logger.info("Flask application created successfully")
        return app
        
    except Exception as e:
        logger.error("Failed to create Flask app: %s", e)
        raise


def main():
    """Main application entry point"""
    print("🚀 Starting Text-to-SQL Chatbot Application...")
    
    # Setup logging
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("Text-to-SQL Chatbot Application Starting")
    logger.info("=" * 60)
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        env_file = current_dir / '.env'
        if env_file.exists():
            load_dotenv(env_file)
            logger.info("Environment variables loaded from .env file")
        else:
            logger.info("No .env file found, using system environment variables")
    except ImportError:
        logger.warning("python-dotenv not installed, using system environment variables only")
    
    # Initialize services
    logger.info("Initializing application services...")
    if not initialize_services():
        logger.error("❌ Service initialization failed. Exiting.")
        sys.exit(1)
    
    # Create Flask app
    logger.info("Creating Flask application...")
    app = create_app()
    
    # Get configuration
    from config import config
    
    # Print startup information
    print(f"✅ Application initialized successfully!")
    print(f"🔧 Configuration Summary:")
    print(f"   - Debug Mode: {config.app.debug}")
    print(f"   - Host: {config.app.host}")
    print(f"   - Port: {config.app.port}")
    print(f"   - Database: {'✅ Connected' if config.database.connection_string else '❌ Not configured'}")
    print(f"   - Azure OpenAI: {'✅ Configured' if config.azure_openai.api_key else '❌ Not configured'}")
    print(f"   - Redis Cache: {'✅ Available' if config.redis.password else '💾 Memory only'}")
    print(f"   - CORS: {'✅ Enabled' if config.security.enable_cors else '❌ Disabled'}")
    print(f"   - Rate Limiting: {'✅ Enabled' if config.security.rate_limit_enabled else '❌ Disabled'}")
    
    print(f"\n🌐 Available Endpoints:")
    print(f"   - Health Check: http://{config.app.host}:{config.app.port}/health")
    print(f"   - Query API: http://{config.app.host}:{config.app.port}/api/query")
    print(f"   - Conversations: http://{config.app.host}:{config.app.port}/api/conversations/<user_id>")
    print(f"   - Statistics: http://{config.app.host}:{config.app.port}/api/stats")
    
    # Run the application
    logger.info("Starting Flask development server...")
    print(f"\n🎯 Server starting on http://{config.app.host}:{config.app.port}")
    print("Press Ctrl+C to stop the server")
    
    try:
        app.run(
            host=config.app.host,
            port=config.app.port,
            debug=config.app.debug,
            threaded=True
        )
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
        print("\n👋 Application stopped gracefully")
    except Exception as e:
        logger.error("Application crashed: %s", e)
        print(f"\n❌ Application error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()