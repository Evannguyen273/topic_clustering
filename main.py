"""
Main entry point for the topic clustering package, providing easy access to key functionality.
"""
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables early
if os.path.exists(".env"):
    load_dotenv(".env")
elif os.path.exists("../.env"):
    load_dotenv("../.env")

# Import and re-export core functionality
from topic_clustering.pipeline.pipeline import run_full_pipeline, run_modular_pipeline
from topic_clustering.models.embeddings import EmbeddingClient
from topic_clustering.models.clustering import TopicClusterer
from topic_clustering.config.settings import config

# Provide convenient access to utilities
from topic_clustering.utils.bigquery_utils import run_query, save_results_to_bigquery
from topic_clustering.utils.azure_utils import blob_storage

# Re-export both pipeline functions
__all__ = ['analyze_topics', 'run_full_pipeline', 'run_modular_pipeline']

def analyze_topics(input_query: str, dataset_name: str, log_to_blob: bool = True, 
                  env_file: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """
    Simplified interface to run the complete topic clustering pipeline.
    
    Args:
        input_query: BigQuery query to load data
        dataset_name: Name for this analysis run
        log_to_blob: Whether to save logs to Azure Blob Storage
        env_file: Optional path to specific .env file to load
        **kwargs: Additional arguments for the pipeline
        
    Returns:
        Dictionary of results from all pipeline stages
    """
    # Load specific environment file if provided
    if env_file and os.path.exists(env_file):
        load_dotenv(env_file, override=True)
        
    return run_full_pipeline(
        dataset_name=dataset_name,
        input_query=input_query,
        log_to_blob=log_to_blob,
        **kwargs
    )

# Direct execution example
if __name__ == "__main__":
    import logging
    from topic_clustering.utils.logging_utils import setup_logging
    
    # Setup logging
    setup_logging(log_level=logging.INFO)
    
    # Example query
    example_query = """
    SELECT * FROM `your_project.your_dataset.your_table` 
    LIMIT 1000
    """
    
    # Run pipeline
    analyze_topics(
        input_query=example_query,
        dataset_name="test_run",
        embeddings_table_id="your_project.your_dataset.embeddings",
        results_table_id="your_project.your_dataset.results",
        log_to_blob=True
    )
