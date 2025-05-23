"""
API Routes Module
Flask/FastAPI route definitions and request handlers.
"""

import logging
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

try:
    from config import config
    from exceptions import ValidationError, RateLimitError, AIServiceError
    from conversation_manager import conversation_manager
    from ai_services import sql_agent_service
    from utils import validate_user_input, RateLimiter, Timer, monitor_performance
except ImportError:
    # Fallback for direct execution
    import os
    import sys
    sys.path.append(os.path.dirname(__file__))
    from config import config
    from exceptions import ValidationError, RateLimitError, AIServiceError
    from conversation_manager import conversation_manager
    from ai_services import sql_agent_service
    from utils import validate_user_input, RateLimiter, Timer, monitor_performance


# Initialize Flask app
app = Flask(__name__)

# Configure CORS
if config.security.enable_cors:
    CORS(app, origins=config.security.cors_origins)

# Initialize rate limiter
rate_limiter = RateLimiter(
    max_requests=config.security.rate_limit_requests,
    time_window=config.security.rate_limit_period
) if config.security.rate_limit_enabled else None


def get_client_ip():
    """Get client IP address for rate limiting"""
    return request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)


@app.errorhandler(ValidationError)
def handle_validation_error(e):
    """Handle validation errors"""
    logging.warning("Validation error: %s", e.message)
    return jsonify({
        "success": False,
        "error": e.message,
        "error_code": e.error_code
    }), 400


@app.errorhandler(RateLimitError)
def handle_rate_limit_error(e):
    """Handle rate limit errors"""
    logging.warning("Rate limit error: %s", e.message)
    return jsonify({
        "success": False,
        "error": "Too many requests. Please try again later.",
        "error_code": e.error_code
    }), 429


@app.errorhandler(AIServiceError)
def handle_ai_service_error(e):
    """Handle AI service errors"""
    logging.error("AI service error: %s", e.message)
    return jsonify({
        "success": False,
        "error": "AI service temporarily unavailable. Please try again.",
        "error_code": e.error_code
    }), 503


@app.errorhandler(Exception)
def handle_generic_error(e):
    """Handle unexpected errors"""
    logging.error("Unexpected error: %s", e)
    return jsonify({
        "success": False,
        "error": "An unexpected error occurred. Please try again.",
        "error_code": "INTERNAL_ERROR"
    }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Basic health check
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "services": {
                "database": "checking...",
                "ai_service": "checking...",
                "cache": "checking..."
            }
        }
        
        # Check database connection
        try:
            from ai_services import database_service
            if database_service.test_connection():
                health_status["services"]["database"] = "healthy"
            else:
                health_status["services"]["database"] = "unhealthy"
                health_status["status"] = "degraded"
        except Exception as e:
            health_status["services"]["database"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
        
        # Check AI service
        try:
            # Simple test - this could be improved with actual service ping
            health_status["services"]["ai_service"] = "healthy"
        except Exception as e:
            health_status["services"]["ai_service"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
        
        # Check cache
        try:
            cache_stats = conversation_manager.get_cache_stats()
            health_status["services"]["cache"] = "healthy"
            health_status["cache_stats"] = cache_stats
        except Exception as e:
            health_status["services"]["cache"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
        
        status_code = 200 if health_status["status"] == "healthy" else 503
        return jsonify(health_status), status_code
        
    except Exception as e:
        logging.error("Health check failed: %s", e)
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route('/api/query', methods=['POST'])
@monitor_performance
def query_database():
    """Main query endpoint"""
    try:
        # Rate limiting
        if rate_limiter:
            client_ip = get_client_ip()
            rate_limiter.check_or_raise(client_ip)
        
        # Get request data
        data = request.get_json()
        if not data:
            raise ValidationError("Request body is required")
        
        user_id = data.get('user_id')
        question = data.get('question')
        
        if not user_id:
            raise ValidationError("user_id is required")
        
        if not question:
            raise ValidationError("question is required")
        
        # Validate input
        user_id = validate_user_input(user_id, max_length=50)
        question = validate_user_input(question, max_length=1000)
        
        with Timer(f"Query processing for user {user_id}"):
            # Get conversation history for context
            conversation_history = conversation_manager.get_user_conversations(user_id)
            
            # Update agent context if there's history
            if conversation_history:
                sql_agent_service.update_conversation_context(
                    conversation_history, 
                    max_history=config.conversation.max_history_context
                )
            
            # Process the query
            result = sql_agent_service.query(question)
            
            # Save conversation
            if result["success"]:
                conversation_manager.add_conversation(user_id, question, result["response"])
            
            # Prepare response
            response_data = {
                "success": result["success"],
                "user_id": user_id,
                "question": question,
                "response": result["response"],
                "timestamp": datetime.now().isoformat(),
                "conversation_count": len(conversation_history) + 1
            }
            
            # Include SQL queries in debug mode
            if config.app.debug and result.get("sql_queries"):
                response_data["debug"] = {
                    "sql_queries": result["sql_queries"]
                }
            
            # Include error details if failed
            if not result["success"] and result.get("error"):
                response_data["error_details"] = result["error"]
            
            logging.info("Query processed successfully for user %s", user_id)
            return jsonify(response_data)
    
    except (ValidationError, RateLimitError, AIServiceError):
        # These are handled by error handlers
        raise
    except Exception as e:
        logging.error("Unexpected error in query endpoint: %s", e)
        raise


@app.route('/api/conversations/<user_id>', methods=['GET'])
def get_conversations(user_id):
    """Get conversation history for a user"""
    try:
        # Rate limiting
        if rate_limiter:
            client_ip = get_client_ip()
            rate_limiter.check_or_raise(client_ip)
        
        # Validate user_id
        user_id = validate_user_input(user_id, max_length=50)
        
        # Get conversations
        conversations = conversation_manager.get_user_conversations(user_id)
        
        # Format for response
        formatted_conversations = []
        for i, (question, response) in enumerate(conversations, 1):
            formatted_conversations.append({
                "id": i,
                "question": question,
                "response": response,
                "timestamp": datetime.now().isoformat()  # In real app, store actual timestamps
            })
        
        return jsonify({
            "success": True,
            "user_id": user_id,
            "conversations": formatted_conversations,
            "total_count": len(formatted_conversations)
        })
    
    except (ValidationError, RateLimitError):
        raise
    except Exception as e:
        logging.error("Error getting conversations for user %s: %s", user_id, e)
        raise


@app.route('/api/conversations/<user_id>', methods=['DELETE'])
def clear_conversations(user_id):
    """Clear conversation history for a user"""
    try:
        # Rate limiting
        if rate_limiter:
            client_ip = get_client_ip()
            rate_limiter.check_or_raise(client_ip)
        
        # Validate user_id
        user_id = validate_user_input(user_id, max_length=50)
        
        # Clear conversations
        success = conversation_manager.clear_user_conversations(user_id)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Conversations cleared for user {user_id}"
            })
        else:
            return jsonify({
                "success": False,
                "error": "Failed to clear conversations"
            }), 500
    
    except (ValidationError, RateLimitError):
        raise
    except Exception as e:
        logging.error("Error clearing conversations for user %s: %s", user_id, e)
        raise


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get application statistics"""
    try:
        # Rate limiting
        if rate_limiter:
            client_ip = get_client_ip()
            rate_limiter.check_or_raise(client_ip)
        
        # Get various stats
        cache_stats = conversation_manager.get_cache_stats()
        
        # Get database stats if available
        db_stats = {}
        try:
            from database import db_manager
            db_stats = db_manager.get_database_stats()
        except Exception as e:
            logging.warning("Could not get database stats: %s", e)
        
        stats = {
            "cache": cache_stats,
            "database": db_stats,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify({
            "success": True,
            "stats": stats
        })
    
    except (RateLimitError):
        raise
    except Exception as e:
        logging.error("Error getting stats: %s", e)
        raise


@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback"""
    try:
        # Rate limiting
        if rate_limiter:
            client_ip = get_client_ip()
            rate_limiter.check_or_raise(client_ip)
        
        data = request.get_json()
        if not data:
            raise ValidationError("Request body is required")
        
        user_id = data.get('user_id')
        rating = data.get('rating')
        feedback = data.get('feedback', '')
        
        if not user_id:
            raise ValidationError("user_id is required")
        
        if rating is None or not isinstance(rating, int) or rating < 1 or rating > 5:
            raise ValidationError("rating must be an integer between 1 and 5")
        
        # Validate inputs
        user_id = validate_user_input(user_id, max_length=50)
        feedback = validate_user_input(feedback, max_length=1000, min_length=0)
        
        # Log feedback (in a real app, you'd save this to database)
        feedback_data = {
            "user_id": user_id,
            "rating": rating,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat(),
            "ip_address": get_client_ip()
        }
        
        logging.info("Feedback received: %s", feedback_data)
        
        return jsonify({
            "success": True,
            "message": "Thank you for your feedback!"
        })
    
    except (ValidationError, RateLimitError):
        raise
    except Exception as e:
        logging.error("Error submitting feedback: %s", e)
        raise


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.logging.level),
        format=config.logging.format
    )
    
    # Run the app
    app.run(
        host=config.app.host,
        port=config.app.port,
        debug=config.app.debug
    )