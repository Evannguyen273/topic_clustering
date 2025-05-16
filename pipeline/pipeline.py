"""
Main pipeline integration for the topic clustering package.
"""

import os
import time
import logging
from datetime import datetime

from ..config.settings import config
from ..utils.logging_utils import setup_logging, log_section
from ..utils.json_utils import save_json

from .stage1_embeddings import generate_embeddings_pipeline
from .stage2_clustering import train_hdbscan_pipeline
from .stage3_analysis import analyze_clusters_pipeline
from .stage4_save import save_results_pipeline

def run_modular_pipeline(stage=None, dataset_name=None, input_query=None,
                       embeddings_table_id=None, results_table_id=None,
                       input_data=None, input_path=None, result_path=None,
                       **kwargs):
    """
    Run a specific stage of the pipeline with maximum flexibility
    
    Args:
        stage: Pipeline stage to run (1, 2, 3, 4 or stage name)
        dataset_name: Name for the dataset/run
        input_query: BigQuery query for loading data (stage 1)
        embeddings_table_id: BigQuery table to save embeddings (stage 1)
        results_table_id: BigQuery table to save final results (stage 4)
        input_data: Direct input data objects for the stage
        input_path: Path to input data files for the stage
        result_path: Base path to store results
        **kwargs: Additional arguments for the specific stage
        
    Returns:
        Result of the specified pipeline stage
    """
    # Use default configuration if not provided
    result_path = result_path or config.result_path
    
    # Setup logging
    log_dir = f"{result_path}/{dataset_name}/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/pipeline_stage{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(
        log_file=log_file,
        log_to_console=True,
        dataset_name=dataset_name,
        log_to_blob=True  # Enable blob storage logging
    )
    
    log_section(f"Starting Pipeline Stage {stage} for dataset '{dataset_name}'")
    start_time = time.time()
    
    # Map stage names to numbers if provided as strings
    stage_map = {
        "embeddings": 1, "embedding": 1,
        "clustering": 2, "cluster": 2,
        "analysis": 3, "analyze": 3,
        "save": 4, "export": 4, "visualization": 4
    }
    
    if isinstance(stage, str) and stage.lower() in stage_map:
        stage = stage_map[stage.lower()]
    
    # Log parameters
    logger.info(f"Parameters:")
    logger.info(f"  - Dataset: {dataset_name}")
    logger.info(f"  - Result path: {result_path}")
    
    # Execute the specified stage
    try:
        if stage == 1:
            logger.info("Running Stage 1: Embeddings Generation")
            if input_query is None and input_data is None:
                raise ValueError("Stage 1 requires either input_query or input_data")
                
            return generate_embeddings_pipeline(
                input_query=input_query,
                dataset_name=dataset_name,
                embeddings_table_id=embeddings_table_id,
                result_path=result_path,
                **kwargs
            )
            
        elif stage == 2:
            logger.info("Running Stage 2: HDBSCAN Clustering")
            return train_hdbscan_pipeline(
                df_with_embeddings=input_data,
                dataset_name=dataset_name,
                embedding_path=input_path,
                result_path=result_path,
                **kwargs
            )
            
        elif stage == 3:
            logger.info("Running Stage 3: Cluster Analysis")
            return analyze_clusters_pipeline(
                clustered_df=input_data,
                dataset_name=dataset_name,
                result_path=result_path,
                **kwargs
            )
            
        elif stage == 4:
            logger.info("Running Stage 4: Save Results")
            return save_results_pipeline(
                final_df=input_data,
                dataset_name=dataset_name,
                table_id=results_table_id,
                result_path=result_path,
                **kwargs
            )
            
        else:
            raise ValueError(f"Unknown pipeline stage: {stage}")
            
    except Exception as e:
        logger.error(f"Error in pipeline stage {stage}: {e}", exc_info=True)
        raise
    finally:
        total_time = time.time() - start_time
        log_section(f"Pipeline Stage {stage} completed in {total_time:.2f} seconds")
        
        # Flush any pending logs to blob storage
        for handler in logger.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()

def run_full_pipeline(dataset_name, input_query, embeddings_table_id=None, 
                     results_table_id=None, result_path=None, 
                     cluster_params=None, **kwargs):
    """
    Run the full topic clustering pipeline end-to-end
    
    Args:
        dataset_name: Name for the dataset/run
        input_query: BigQuery query for loading data
        embeddings_table_id: BigQuery table to save embeddings
        results_table_id: BigQuery table to save final results
        result_path: Base path to store results
        cluster_params: Dictionary with clustering parameters
        **kwargs: Additional arguments for specific stages
        
    Returns:
        Dictionary with results of all pipeline stages
    """
    # Use default configuration if not provided
    result_path = result_path or config.result_path
    
    # Setup logging
    log_dir = f"{result_path}/{dataset_name}/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/full_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(
        log_file=log_file,
        log_to_console=True,
        dataset_name=dataset_name,
        log_to_blob=True  # Enable blob storage logging
    )
    
    log_section(f"Starting Full Pipeline for dataset '{dataset_name}'")
    start_time = time.time()
    
    # Create run metadata
    metadata = {
        "dataset": dataset_name,
        "start_time": datetime.now().isoformat(),
        "query": input_query,
        "embeddings_table": embeddings_table_id,
        "results_table": results_table_id,
        "cluster_params": cluster_params or {}
    }
    
    # Output directory for the run
    output_dir = f"{result_path}/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results
    results = {}
    
    try:
        # Stage 1: Generate embeddings
        log_section("Stage 1: Embeddings Generation")
        stage1_start = time.time()
        
        df_with_embeddings, classification_result, fallback_stats = generate_embeddings_pipeline(
            input_query=input_query,
            dataset_name=dataset_name,
            embeddings_table_id=embeddings_table_id,
            result_path=result_path,
            **kwargs
        )
        
        stage1_time = time.time() - stage1_start
        logger.info(f"Stage 1 completed in {stage1_time:.2f} seconds")
        results["stage1"] = {
            "df_shape": df_with_embeddings.shape,
            "entity_terms": len(classification_result.get("ENTITY", {})),
            "action_terms": len(classification_result.get("ACTION", {})),
            "fallback_stats": fallback_stats,
            "runtime_seconds": stage1_time
        }
        
        # Stage 2: HDBSCAN Clustering
        log_section("Stage 2: HDBSCAN Clustering")
        stage2_start = time.time()
        
        # Extract clustering parameters
        cluster_kwargs = {}
        if cluster_params:
            cluster_kwargs.update(cluster_params)
        
        clustered_df, umap_embeddings, clusterer, reducer = train_hdbscan_pipeline(
            df_with_embeddings=df_with_embeddings,
            dataset_name=dataset_name,
            result_path=result_path,
            **cluster_kwargs
        )
        
        stage2_time = time.time() - stage2_start
        logger.info(f"Stage 2 completed in {stage2_time:.2f} seconds")
        results["stage2"] = {
            "n_clusters": len(set(clustered_df["cluster"])) - (1 if -1 in set(clustered_df["cluster"]) else 0),
            "n_noise": list(clustered_df["cluster"]).count(-1),
            "noise_percentage": 100 * list(clustered_df["cluster"]).count(-1) / len(clustered_df),
            "runtime_seconds": stage2_time
        }
        
        # Stage 3: Cluster Analysis
        log_section("Stage 3: Cluster Analysis")
        stage3_start = time.time()
        
        final_df, clusters_info, labeled_clusters, domains = analyze_clusters_pipeline(
            clustered_df=clustered_df,
            dataset_name=dataset_name,
            result_path=result_path,
            **kwargs
        )
        
        stage3_time = time.time() - stage3_start
        logger.info(f"Stage 3 completed in {stage3_time:.2f} seconds")
        results["stage3"] = {
            "num_clusters": len(clusters_info) - (1 if "-1" in clusters_info else 0),
            "num_domains": len(domains["domains"]) - 1,  # Exclude noise domain
            "runtime_seconds": stage3_time
        }
        
        # Stage 4: Save Results
        log_section("Stage 4: Save Results")
        stage4_start = time.time()
        
        saved_files = save_results_pipeline(
            final_df=final_df,
            clusters_info=clusters_info,
            labeled_clusters=labeled_clusters,
            domains=domains,
            dataset_name=dataset_name,
            table_id=results_table_id,
            result_path=result_path,
            metadata={
                "dataset": dataset_name,
                "stage1_time": stage1_time,
                "stage2_time": stage2_time,
                "stage3_time": stage3_time,
                "processing_time": time.time() - start_time
            },
            **kwargs
        )
        
        stage4_time = time.time() - stage4_start
        logger.info(f"Stage 4 completed in {stage4_time:.2f} seconds")
        results["stage4"] = {
            "saved_files": saved_files,
            "runtime_seconds": stage4_time
        }
        
        # Update and save metadata
        total_time = time.time() - start_time
        metadata.update({
            "end_time": datetime.now().isoformat(),
            "total_runtime_seconds": total_time,
            "stage_runtimes": {
                "stage1": stage1_time,
                "stage2": stage2_time,
                "stage3": stage3_time,
                "stage4": stage4_time
            },
            "results": results
        })
        
        metadata_path = f"{output_dir}/pipeline_metadata.json"
        save_json(metadata, metadata_path)
        
        log_section(f"Full Pipeline completed in {total_time:.2f} seconds")
        logger.info(f"Final results saved to {output_dir}")
        logger.info(f"Metadata saved to {metadata_path}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error in full pipeline: {e}", exc_info=True)
        
        # Save partial metadata even if pipeline fails
        metadata.update({
            "error": str(e),
            "end_time": datetime.now().isoformat(),
            "total_runtime_seconds": time.time() - start_time,
            "partial_results": results
        })
        
        failure_metadata_path = f"{output_dir}/pipeline_failure_metadata.json"
        save_json(metadata, failure_metadata_path)
        
        logger.info(f"Failure metadata saved to {failure_metadata_path}")
        raise
    
    finally:
        # Flush any pending logs to blob storage
        for handler in logger.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()
