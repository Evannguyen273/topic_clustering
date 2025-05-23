# Text-to-SQL Chatbot

A modular, production-ready chatbot application that converts natural language questions into SQL queries using Azure OpenAI and LangChain.

## 🏗️ Architecture

The application is organized into the following modules:

```
text_to_sql/
├── __init__.py              # Package initialization
├── main.py                  # Application entry point
├── config.py                # Configuration management
├── exceptions.py            # Custom exceptions
├── database_fixed.py        # Database connection management
├── conversation_manager.py  # Conversation & caching logic
├── ai_services.py          # LangChain SQL agent & Azure OpenAI
├── api_routes.py           # Flask API endpoints
├── utils.py                # Utilities and monitoring
├── requirements.txt        # Python dependencies
└── .env.example           # Environment variables template
```

## ✨ Features

### Core Functionality
- **Natural Language to SQL**: Convert questions to SQL using Azure OpenAI
- **Multi-tier Caching**: Redis primary cache with memory fallback
- **Conversation History**: Persistent conversation management
- **Context Awareness**: Use conversation history for better responses

### Production Features
- **Error Handling**: Comprehensive exception handling with custom error types
- **Rate Limiting**: Configurable rate limiting per IP/user
- **Monitoring**: Performance monitoring and health checks
- **Security**: Input validation and SQL injection prevention
- **Circuit Breaker**: Fault tolerance for external service calls

### API Endpoints
- `GET /health` - Health check and service status
- `POST /api/query` - Main query endpoint
- `GET /api/conversations/<user_id>` - Get conversation history
- `DELETE /api/conversations/<user_id>` - Clear conversation history
- `GET /api/stats` - Application statistics
- `POST /api/feedback` - Submit user feedback

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd text_to_sql
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
# Minimum required:
# - connection_String (SQL Server)
# - OPENAI_API_KEY (Azure OpenAI)
# - AZURE_OPENAI_ENDPOINT
# - AZURE_OPENAI_DEPLOYMENT_NAME
```

### 3. Run Application

```bash
# Run the application
python main.py
```

The application will start on `http://localhost:5000`

## 📋 Configuration

### Required Configuration
- **Database**: SQL Server connection string
- **Azure OpenAI**: API key, endpoint, and deployment name

### Optional Configuration  
- **Redis**: For production caching (falls back to memory)
- **Security**: CORS origins, rate limiting
- **Logging**: Log level and file output

### Environment Variables

See `.env.example` for all available configuration options.

## 🔧 Usage Examples

### Query the Database

```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123", 
    "question": "How many critical problems were created last month?"
  }'
```

### Get Conversation History

```bash
curl http://localhost:5000/api/conversations/user123
```

### Health Check

```bash
curl http://localhost:5000/health
```

## 📊 Monitoring

### Health Check Response
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "services": {
    "database": "healthy",
    "ai_service": "healthy", 
    "cache": "healthy"
  },
  "cache_stats": {
    "redis_available": true,
    "memory_cache_size": 150
  }
}
```

### Application Statistics
```bash
curl http://localhost:5000/api/stats
```

## 🏢 Production Deployment

### Using Gunicorn (Linux/Mac)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 main:app
```

### Using Waitress (Windows)
```bash
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 main:app
```

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "main.py"]
```

## 🔒 Security Features

### Input Validation
- SQL injection prevention
- Input length limits
- Content filtering

### Rate Limiting
- Per-IP rate limiting
- Configurable limits
- Graceful error responses

### Error Handling
- No sensitive data in error responses
- Structured error codes
- Comprehensive logging

## 🧪 Testing

```bash
# Install test dependencies
pip install pytest pytest-flask

# Run tests (when test files are created)
pytest tests/
```

## 📈 Performance

### Caching Strategy
1. **Redis Cache**: Primary cache with TTL
2. **Memory Cache**: Fallback for Redis failures
3. **Database**: Persistent storage

### Connection Pooling
- Database connection pooling
- Retry logic with exponential backoff
- Circuit breaker for fault tolerance

## 🔧 Customization

### Adding New Endpoints
Add routes in `api_routes.py`:

```python
@app.route('/api/custom', methods=['POST'])
@monitor_performance
def custom_endpoint():
    # Your logic here
    pass
```

### Custom Prompts
Modify the prompt in `ai_services.py` `SQLAgentService._create_prompt()` method.

### Additional Caching
Extend `conversation_manager.py` for custom caching strategies.

## 🐛 Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Check connection string format
   - Verify firewall settings
   - Ensure SQL Server allows connections

2. **Azure OpenAI Errors**
   - Verify API key and endpoint
   - Check deployment name
   - Ensure sufficient quota

3. **Redis Connection Issues**
   - Check Redis host/port/password
   - Application will fall back to memory cache

### Logging

Set `LOG_LEVEL=DEBUG` in `.env` for detailed logging.

## 🤝 Contributing

1. Follow the existing module structure
2. Add comprehensive error handling
3. Include logging for debugging
4. Update documentation for new features

## 📝 License

This project is proprietary software for internal use.

---

**Need Help?** Check the logs in debug mode or contact the development team.