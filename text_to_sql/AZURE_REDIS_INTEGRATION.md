# Azure Redis Integration Guide

## Overview
This guide shows how to integrate Azure Cache for Redis with your text-to-SQL chatbot application for improved performance and scalability.

## Benefits of Azure Cache for Redis

- **High Performance**: Sub-millisecond latency for conversation retrieval
- **Scalability**: Handle thousands of concurrent users
- **Reliability**: 99.9% SLA with built-in failover
- **Security**: SSL/TLS encryption and VNet integration
- **Cost-Effective**: Pay only for what you use

## Setup Instructions

### 1. Create Azure Cache for Redis

```bash
# Using Azure CLI
az redis create \
    --name your-redis-cache \
    --resource-group your-resource-group \
    --location eastus \
    --sku Basic \
    --vm-size c0
```

### 2. Install Dependencies

```bash
pip install redis==5.0.1
```

### 3. Environment Variables

Add to your `.env` file:

```env
# Azure Cache for Redis Configuration
AZURE_REDIS_HOST=your-cache.redis.cache.windows.net
AZURE_REDIS_PORT=6380
AZURE_REDIS_PASSWORD=your-redis-access-key
AZURE_REDIS_SSL=True
AZURE_REDIS_DB=0
```

### 4. Update Configuration

Your `ProductionConfig` class already includes Azure Redis settings:

```python
AZURE_REDIS_HOST: str = os.getenv('AZURE_REDIS_HOST', 'localhost')
AZURE_REDIS_PORT: int = int(os.getenv('AZURE_REDIS_PORT', '6380'))
AZURE_REDIS_PASSWORD: Optional[str] = os.getenv('AZURE_REDIS_PASSWORD')
AZURE_REDIS_SSL: bool = os.getenv('AZURE_REDIS_SSL', 'True').lower() == 'true'
```

### 5. Replace ConversationManager

In your `app.py`, replace the existing ConversationManager initialization:

```python
# OLD:
# conversation_manager = ConversationManager(db_manager)

# NEW:
from azure_conversation_manager import AzureConversationManager
conversation_manager = AzureConversationManager(db_manager, config)
```

## Performance Comparison

| Operation | Memory Cache | Azure Redis |
|-----------|-------------|-------------|
| Single User Lookup | ~1ms | ~2ms |
| 100 Concurrent Users | Degraded | Consistent |
| Cross-Instance Sharing | No | Yes |
| Persistence | No | Yes |
| Memory Usage | High | Low |

## Code Examples

### Basic Usage

```python
# Add conversation
conversation_manager.add_conversation(
    user_id="user123",
    question="How many critical problems are open?",
    response="There are 42 critical problems currently open."
)

# Get conversation history
history = conversation_manager.get_user_conversations("user123")

# Clear user cache
conversation_manager.clear_user_cache("user123")

# Get cache statistics
stats = conversation_manager.get_cache_stats()
```

### Error Handling

The Azure Redis implementation includes automatic fallback:

```python
# If Redis fails, automatically falls back to memory cache
# No code changes needed - transparent failover
```

## Monitoring

### Cache Statistics

```python
stats = conversation_manager.get_cache_stats()
print(f"Redis Connected: {stats['redis_connected']}")
print(f"Memory Usage: {stats['redis_memory_usage']}")
print(f"Connected Clients: {stats['redis_connected_clients']}")
```

### Logging

The implementation provides detailed logging:

```
INFO: Azure Redis Cache connected successfully to your-cache.redis.cache.windows.net
WARNING: Azure Redis not available: Connection timeout, using memory cache
DEBUG: Cached conversations for conversations:user123 in Azure Redis
```

## Production Considerations

### Security
- Use SSL/TLS encryption (enabled by default)
- Implement VNet integration for enhanced security
- Rotate access keys regularly

### Performance
- Use Premium tier for production workloads
- Enable clustering for high availability
- Configure appropriate TTL values

### Cost Optimization
- Use Basic tier for development/testing
- Standard tier for production
- Premium tier for enterprise workloads

### Monitoring
- Set up Azure Monitor alerts
- Monitor memory usage and connection counts
- Track cache hit ratios

## Troubleshooting

### Common Issues

1. **Connection Timeout**
   - Check firewall rules
   - Verify SSL settings
   - Ensure correct port (6380 for SSL, 6379 for non-SSL)

2. **Authentication Failed**
   - Verify access key
   - Check if access keys were rotated

3. **Memory Issues**
   - Monitor Redis memory usage
   - Implement appropriate TTL values
   - Consider data eviction policies

### Health Check

```python
def health_check():
    try:
        conversation_manager.redis_client.ping()
        return "Redis: Healthy"
    except:
        return "Redis: Unhealthy - Using memory fallback"
```

## Migration Strategy

1. **Phase 1**: Deploy with fallback enabled
2. **Phase 2**: Monitor performance and errors
3. **Phase 3**: Optimize cache TTL and memory usage
4. **Phase 4**: Remove memory fallback if desired

This ensures zero-downtime migration and provides a safety net during the transition.