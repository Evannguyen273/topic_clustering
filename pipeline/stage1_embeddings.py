"""
First pipeline stage: Generate embeddings from text data.
"""

import os
import time
import logging
import pandas as pd
from datetime import datetime

from ..config.settings import config
from ..utils.bigquery_utils import run_query, save_embeddings_to_bigquery
from ..utils.text_utils import get_safe_text, clean_text_for_summary, normalize_business_service
from ..models.embeddings import HybridEmbeddingGenerator, EmbeddingClient
from ..utils.json_utils import save_json, NumpyEncoder
from ..utils.azure_utils import blob_storage

def process_incident_summaries(df, batch_size=10):
    """
    Process incident text for embedding with LLM summarization in batches
    
    Args:
        df: DataFrame with incidents
        batch_size: Number of incidents to process in each batch
        
    Returns:
        Tuple of (result_series, fallback_stats): Series with summarized texts and dictionary of fallback statistics
    """
    import uuid
    from tqdm import tqdm
    import re
    
    result_series = pd.Series(index=df.index)
    num_batches = (len(df) + batch_size - 1) // batch_size
    
    # Initialize fallback tracking statistics
    fallback_stats = {
        "total_incidents": len(df),
        "short_desc_fallbacks": 0,
        "api_failure_fallbacks": 0,
        "final_sweep_fallbacks": 0,
        "llm_processed": 0
    }
    
    # Initialize progress bar
    pbar = tqdm(total=len(df), desc="Summarizing incidents")
    
    # Process in batches
    for batch_idx in range(num_batches):
        # ... existing batch processing code ...
        # This would include:
        # 1. Extracting each incident's text fields
        # 2. Cleaning and normalizing text
        # 3. Using LLM to summarize longer descriptions
        # 4. Handling fallbacks for API failures
        
        # Update progress
        pbar.update(batch_size)
    
    pbar.close()
    
    # Final sweep - fill any missing values with their short description
    final_sweep_count = 0
    for idx in df.index:
        if idx not in result_series or pd.isna(result_series[idx]):
            short_desc = get_safe_text(df.loc[idx], 'short_description')
            business_svc = get_safe_text(df.loc[idx], 'business_service')
            business_svc = normalize_business_service(business_svc)
            result_series[idx] = f"{short_desc} - {business_svc}".strip()
            final_sweep_count += 1
    
    fallback_stats["final_sweep_fallbacks"] = final_sweep_count
    
    # Calculate success rate
    fallback_stats["total_fallbacks"] = (
        fallback_stats["short_desc_fallbacks"] +
        fallback_stats["api_failure_fallbacks"] +
        fallback_stats["final_sweep_fallbacks"]
    )
    fallback_stats["llm_success_count"] = fallback_stats["llm_processed"] - fallback_stats["api_failure_fallbacks"] - fallback_stats["final_sweep_fallbacks"]
    fallback_stats["llm_success_rate"] = fallback_stats["llm_success_count"] / max(1, fallback_stats["llm_processed"]) * 100
    fallback_stats["overall_llm_rate"] = fallback_stats["llm_success_count"] / len(df) * 100
    
    logging.info(f"Summarization fallback stats: {fallback_stats}")
    
    return result_series, fallback_stats

def generate_embeddings_pipeline(input_query, dataset_name, embeddings_table_id=None, 
                              batch_size=None, save_to_bq=True, summary_path=None,
                              write_disposition=None, result_path=None):
    """
    Generate hybrid embeddings from text data with performance optimizations
    
    Args:
        input_query: BigQuery query to load data
        dataset_name: Name for the dataset/run
        embeddings_table_id: BigQuery table to save embeddings
        batch_size: Size of batches for embedding API calls
        save_to_bq: Whether to save embeddings to BigQuery
        summary_path: Path to precomputed summaries to skip LLM processing
        write_disposition: BigQuery write disposition
        result_path: Base path to store results (used only as a fallback if blob storage is unavailable)
        
    Returns:
        Tuple of (DataFrame with embeddings, classification result, fallback statistics)
    """
    # Use default configuration if not provided
    batch_size = batch_size or config.batch_size
    write_disposition = write_disposition or config.write_disposition
    
    start_time = time.time()
    
    # Create versioned training cycle path
    training_cycle = datetime.now().strftime("%Y%m%d")
    base_blob_path = f"{dataset_name}/training_cycle_{training_cycle}"
    embeddings_blob_path = f"{base_blob_path}/embeddings"
    intermediate_blob_path = f"{base_blob_path}/intermediate"
    
    # Create training cycle info but DO NOT update the global pointer yet
    # Instead, save cycle-specific info
    training_cycle_info = {
        "dataset_name": dataset_name,
        "training_cycle": training_cycle,
        "timestamp": datetime.now().isoformat(),
        "base_path": base_blob_path,
        "status": "embedding_stage_started"  # Track current status
    }
    
    # Save training cycle info to the cycle-specific path, not the global pointer
    if blob_storage and blob_storage._container_client:
        blob_storage.upload_json(training_cycle_info, f"{base_blob_path}/training_info.json")
        
        # Optionally, create a "current" pointer that doesn't replace the "latest" successful one
        blob_storage.upload_json(training_cycle_info, f"{dataset_name}/current_training_cycle.json")
    
    # Check if blob storage is available, create local dirs as fallback if not
    if not blob_storage or not blob_storage._container_client:
        logging.warning("Azure Blob Storage not available. Using local storage as fallback.")
        result_path = result_path or config.result_path
        output_dir = f"{result_path}/{dataset_name}/training_cycle_{training_cycle}/embeddings"
        intermediate_dir = f"{result_path}/{dataset_name}/training_cycle_{training_cycle}/intermediate"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(intermediate_dir, exist_ok=True)
    
    logging.info(f"Generating embeddings for dataset '{dataset_name}'")
    
    # Check if we should use previously computed summaries
    default_summary_blob_path = f"{intermediate_blob_path}/df_with_summaries.parquet"
    
    # If no summary path provided, try to find one in blob storage
    if not summary_path and blob_storage and blob_storage._container_client:
        if blob_storage.blob_exists(default_summary_blob_path):
            logging.info(f"Found precomputed summaries in blob storage at {default_summary_blob_path}")
            summary_path = default_summary_blob_path
    # Otherwise check local fallback
    elif not summary_path and result_path:
        default_summary_path = f"{result_path}/{dataset_name}/training_cycle_{training_cycle}/intermediate/df_with_summaries.parquet"
        if os.path.exists(default_summary_path):
            logging.info(f"Found precomputed summaries at {default_summary_path}")
            summary_path = default_summary_path
    
    # 1. Load data from BigQuery
    logging.info(f"Loading data with query: {input_query}")
    df = run_query(input_query)
    logging.info(f"Loaded {len(df)} records")
    
    # Save raw data
    if blob_storage and blob_storage._container_client:
        blob_storage.upload_dataframe(df, f"{embeddings_blob_path}/raw_data.parquet")
    elif result_path:
        df.to_parquet(f"{result_path}/{dataset_name}/training_cycle_{training_cycle}/embeddings/raw_data.parquet", index=False)
    
    # Log telemetry 
    perf_metrics = {
        "start_time": datetime.now().isoformat(),
        "record_count": len(df),
        "batch_size": batch_size,
        "dataset_name": dataset_name,
        "using_precomputed_summaries": summary_path is not None
    }
    
    # 2. Process text into summaries
    if summary_path:
        # Load precomputed summaries - handle both blob and local paths
        logging.info(f"Loading precomputed summaries from {summary_path}")
        
        # Try loading from blob storage first
        if blob_storage and blob_storage._container_client and not os.path.exists(summary_path):
            summary_df = blob_storage.download_dataframe(summary_path, format="parquet")
        else:
            summary_df = pd.read_parquet(summary_path)
            
        df['combined_incidents_summary'] = summary_df['combined_incidents_summary']
        # We don't have fallback stats when loading precomputed summaries
        fallback_stats = {"precomputed": True, "loaded_from": summary_path}
    else:
        # Process text for embedding using batch processing
        logging.info("Processing text for embedding...")
        combined_summaries, fallback_stats = process_incident_summaries(
            df, 
            batch_size=min(10, len(df))  # Use smaller batch for LLM summarization
        )
        df['combined_incidents_summary'] = combined_summaries
        
        # Save intermediate dataframe with summaries
        summary_df = df[['number', 'combined_incidents_summary']].copy()
        if blob_storage and blob_storage._container_client:
            logging.info(f"Saving intermediate dataframe with summaries to blob storage")
            blob_storage.upload_dataframe(summary_df, f"{intermediate_blob_path}/df_with_summaries.parquet")
        elif result_path:
            summary_save_path = f"{result_path}/{dataset_name}/training_cycle_{training_cycle}/intermediate/df_with_summaries.parquet"
            logging.info(f"Saving intermediate dataframe with summaries to {summary_save_path}")
            summary_df.to_parquet(summary_save_path, index=False)
    
    # 3. Generate hybrid embeddings
    logging.info("Generating hybrid embeddings...")
    embedding_client = EmbeddingClient()
    hybrid_generator = HybridEmbeddingGenerator(embedding_client)
    
    df_with_embeddings, classification_result = hybrid_generator.generate_hybrid_embeddings(
        df, 
        text_column='combined_incidents_summary',
        batch_size=batch_size
    )
    
    # Save embeddings to file
    if blob_storage and blob_storage._container_client:
        blob_storage.upload_dataframe(df_with_embeddings, f"{embeddings_blob_path}/df_with_embeddings.parquet")
        blob_storage.upload_json(classification_result, f"{embeddings_blob_path}/classification_result.json")
        blob_storage.upload_json(fallback_stats, f"{embeddings_blob_path}/fallback_stats.json")
    elif result_path:
        df_with_embeddings.to_parquet(f"{result_path}/{dataset_name}/training_cycle_{training_cycle}/embeddings/df_with_embeddings.parquet", index=False)
        save_json(classification_result, f"{result_path}/{dataset_name}/training_cycle_{training_cycle}/embeddings/classification_result.json")
        save_json(fallback_stats, f"{result_path}/{dataset_name}/training_cycle_{training_cycle}/embeddings/fallback_stats.json")
    
    # Add fallback statistics to performance metrics if available
    if "precomputed" not in fallback_stats:
        perf_metrics["text_summarization"] = {
            "llm_processed": fallback_stats["llm_processed"],
            "llm_success_count": fallback_stats["llm_success_count"],
            "short_desc_fallbacks": fallback_stats["short_desc_fallbacks"],
            "api_failure_fallbacks": fallback_stats["api_failure_fallbacks"],
            "final_sweep_fallbacks": fallback_stats["final_sweep_fallbacks"],
            "llm_success_rate": fallback_stats["llm_success_rate"],
            "overall_llm_rate": fallback_stats["overall_llm_rate"]
        }
    
    # 4. Optionally save embeddings to BigQuery
    if save_to_bq and embeddings_table_id:
        save_embeddings_to_bigquery(df_with_embeddings, embeddings_table_id, write_disposition)
    
    # Log runtime
    total_time = time.time() - start_time
    logging.info(f"Embeddings generation completed in {total_time:.2f} seconds")
    
    # Update and save performance metrics
    perf_metrics.update({
        "end_time": datetime.now().isoformat(),
        "runtime_seconds": total_time,
        "avg_time_per_record": total_time / max(1, len(df)),
        "records_per_second": len(df) / max(1, total_time)
    })
    
    # Save metrics
    if blob_storage and blob_storage._container_client:
        blob_storage.upload_json(perf_metrics, f"{embeddings_blob_path}/embedding_metadata.json")
    elif result_path:
        save_json(perf_metrics, f"{result_path}/{dataset_name}/training_cycle_{training_cycle}/embeddings/embedding_metadata.json")
    
    return df_with_embeddings, classification_result, fallback_stats
