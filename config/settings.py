"""
Configuration settings for the topic clustering package.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

try:
    from pydantic import BaseSettings, Field, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    logging.warning("Pydantic not installed. Using basic config class instead.")
    
if PYDANTIC_AVAILABLE:
    class AzureOpenAISettings(BaseSettings):
        """Azure OpenAI configuration settings"""
        api_version: str = Field("2023-05-15", description="Azure OpenAI API version")
        chat_deployment_name: str = Field("gpt-4o", description="Deployment name for chat completions")
        api_key: Optional[str] = Field(None, description="OpenAI API key")
        endpoint: Optional[str] = Field(None, description="Azure OpenAI endpoint")
        embedding_key: Optional[str] = Field(None, description="Azure OpenAI embedding key")
        embedding_api_version: Optional[str] = Field(None, description="Azure OpenAI embedding API version")
        embedding_endpoint: Optional[str] = Field(None, description="Azure OpenAI embedding endpoint")
        embedding_model: str = Field("text-embedding-3-large", description="Embedding model to use")
        
        class Config:
            env_prefix = "AZURE_OPENAI_"
            
    class ClusteringSettings(BaseSettings):
        """Clustering algorithm settings"""
        min_cluster_size: int = Field(25, ge=1, description="Minimum cluster size for HDBSCAN")
        min_samples: int = Field(5, ge=1, description="Minimum samples for HDBSCAN")
        umap_n_components: int = Field(50, ge=2, description="Number of dimensions for UMAP")
        umap_n_neighbors: int = Field(100, ge=5, description="Number of neighbors for UMAP")
        umap_min_dist: float = Field(0.1, ge=0.0, lt=1.0, description="Minimum distance for UMAP")
        metric: str = Field("cosine", description="Distance metric for UMAP")
        random_state: int = Field(42, description="Random seed for reproducibility")
        
        class Config:
            env_prefix = ""
    
    class PydanticConfig(BaseSettings):
        """Pydantic configuration for topic clustering package"""
        # Azure OpenAI Configuration
        azure_openai_config: AzureOpenAISettings = Field(default_factory=AzureOpenAISettings)
        
        # Azure Blob Storage Configuration
        blob_connection_string: Optional[str] = Field(None, description="Azure Blob Storage connection string")
        container_name: str = Field("prediction-artifact", description="Azure Blob Storage container name")
        
        # BigQuery Configuration
        service_account_key_path: Optional[str] = Field(None, description="BigQuery service account key path")
        
        # Pipeline Defaults
        result_path: str = Field("results", description="Local path to store results")
        batch_size: int = Field(25, ge=1, description="Batch size for API calls")
        write_disposition: str = Field("WRITE_APPEND", description="BigQuery write disposition")
        
        # Clustering Parameters
        clustering: ClusteringSettings = Field(default_factory=ClusteringSettings)
        
        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
        
        @validator("blob_connection_string", "service_account_key_path", "azure_openai_config")
        def warn_missing_configs(cls, v, values, field):
            if v is None:
                logging.warning(f"Missing configuration: {field.name}")
            return v

# Initialize configuration - use Pydantic if available, otherwise use the original Config class
if PYDANTIC_AVAILABLE:
    try:
        config = PydanticConfig()
    except Exception as e:
        logging.error(f"Error initializing Pydantic config: {e}")
        # Fall back to original Config class
        from .original_settings import Config
        config = Config()
else:
    # Load environment variables from multiple possible locations
    for path in [".env", "../.env", "../../.env", "../../../.env"]:
        if os.path.exists(path):
            load_dotenv(path)
            logging.info(f"Loaded environment from {path}")
            break
    
    # Use the traditional Config class
    from .original_settings import Config
    config = Config()
