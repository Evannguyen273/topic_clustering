"""
Second pipeline stage: Train HDBSCAN clustering model on embeddings.
"""

import os
import time
import logging
import uuid
import pandas as pd
from datetime import datetime

from ..config.settings import config
from ..models.clustering import TopicClusterer, process_data_for_clustering
from ..utils.json_utils import save_json
from ..utils.azure_utils import blob_storage

def train_hdbscan_pipeline(df_with_embeddings=None, dataset_name=None, embedding_path=None, 
                         min_cluster_size=None, min_samples=None, umap_n_components=None, 
                         umap_n_neighbors=None, umap_min_dist=None, metric=None, 
                         random_state=None, result_path=None):
    """
    Train HDBSCAN clustering model on embeddings
    
    Args:
        df_with_embeddings: DataFrame with 'embedding' column
        dataset_name: Name for the dataset/run
        embedding_path: Optional custom path to load embeddings from
        min_cluster_size: Minimum cluster size for HDBSCAN
        min_samples: Minimum samples for HDBSCAN
        umap_n_components: Number of dimensions for UMAP reduction
        umap_n_neighbors: Number of neighbors for UMAP
        umap_min_dist: Minimum distance for UMAP
        metric: Distance metric for UMAP
        random_state: Random seed for reproducibility
        result_path: Base path to store results locally (used only as fallback if blob storage is unavailable)
        
    Returns:
        Tuple of (clustered_df, umap_embeddings, clusterer, reducer)
    """
    # Use default configuration if not provided
    min_cluster_size = min_cluster_size or config.clustering["min_cluster_size"]
    min_samples = min_samples or config.clustering["min_samples"]
    umap_n_components = umap_n_components or config.clustering["umap_n_components"]
    umap_n_neighbors = umap_n_neighbors or config.clustering["umap_n_neighbors"]
    umap_min_dist = umap_min_dist or config.clustering["umap_min_dist"]
    metric = metric or config.clustering["metric"]
    random_state = random_state or config.clustering["random_state"]
    
    start_time = time.time()
    correlation_id = str(uuid.uuid4())
    
    # Get the training cycle (either from passed data or by loading latest)
    training_cycle = datetime.now().strftime("%Y%m%d")
    if blob_storage and blob_storage._container_client:
        try:
            cycle_info = blob_storage.download_json(f"{dataset_name}/latest_training_cycle.json")
            if cycle_info and "training_cycle" in cycle_info:
                training_cycle = cycle_info["training_cycle"]
                logging.info(f"Using training cycle {training_cycle} from previous stage")
        except Exception as e:
            logging.warning(f"Could not load training cycle info, using current date: {e}")
    
    # Define Azure paths with training cycle
    base_blob_path = f"{dataset_name}/training_cycle_{training_cycle}"
    clustering_blob_path = f"{base_blob_path}/clustering"
    
    # Check if blob storage is available, create local dirs as fallback if not
    if not blob_storage or not blob_storage._container_client:
        logging.warning("Azure Blob Storage not available. Using local storage as fallback.")
        result_path = result_path or config.result_path
        output_dir = f"{result_path}/{dataset_name}/clustering" if dataset_name else f"{result_path}/clustering"
        os.makedirs(output_dir, exist_ok=True)
    else:
        # Use None to skip local disk operations
        output_dir = None
    
    # Check if we need to load data from file
    if df_with_embeddings is None:
        if embedding_path:
            # Try to load from blob storage first
            if blob_storage and blob_storage._container_client and not os.path.exists(embedding_path):
                logging.info(f"Loading embeddings from blob storage: {embedding_path}")
                df_with_embeddings = blob_storage.download_dataframe(embedding_path, format="parquet")
                if df_with_embeddings is None:
                    raise FileNotFoundError(f"Embeddings file not found in blob storage: {embedding_path}")
            # Fall back to local file
            elif os.path.exists(embedding_path):
                logging.info(f"Loading embeddings from local path: {embedding_path}")
                df_with_embeddings = pd.read_parquet(embedding_path)
            else:
                raise FileNotFoundError(f"Embeddings file not found: {embedding_path}")
                
        elif dataset_name:
            # Try blob storage first
            if blob_storage and blob_storage._container_client:
                blob_path = f"{dataset_name}/embeddings/df_with_embeddings.parquet"
                logging.info(f"Loading embeddings from blob storage: {blob_path}")
                df_with_embeddings = blob_storage.download_dataframe(blob_path, format="parquet")
            
            # Fall back to local file if needed
            if df_with_embeddings is None and result_path:
                default_path = f"{result_path}/{dataset_name}/embeddings/df_with_embeddings.parquet"
                if os.path.exists(default_path):
                    logging.info(f"Loading embeddings from local path: {default_path}")
                    df_with_embeddings = pd.read_parquet(default_path)
                else:
                    raise FileNotFoundError(f"Embeddings file not found at default path: {default_path}")
            
            if df_with_embeddings is None:
                raise FileNotFoundError(f"Embeddings not found for dataset: {dataset_name}")
        else:
            raise ValueError("Either df_with_embeddings, dataset_name, or embedding_path must be provided")
    
    logging.info(f"Training HDBSCAN for dataset '{dataset_name or 'custom'}'")
    
    # Process data for clustering
    embeddings = process_data_for_clustering(df_with_embeddings, embedding_column='embedding')
    
    # Initialize and fit the clusterer
    clusterer = TopicClusterer(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        umap_n_components=umap_n_components,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        metric=metric,
        random_state=random_state
    )
    
    # Fit the model
    clusterer, cluster_labels = clusterer.fit(embeddings)
    
    # Add cluster labels to dataframe
    result_df = df_with_embeddings.copy()
    result_df['cluster'] = cluster_labels
    
    # Add cluster probabilities
    if hasattr(clusterer.clusterer, 'probabilities_'):
        result_df['cluster_probability'] = clusterer.clusterer.probabilities_
    
    # Save clustered data
    if blob_storage and blob_storage._container_client:
        blob_storage.upload_dataframe(result_df, f"{clustering_blob_path}/clustered_df.parquet")
    elif output_dir:
        result_df.to_parquet(f"{output_dir}/clustered_df.parquet", index=False)
    
    # Save model
    saved_files = clusterer.save(dataset_name=dataset_name, output_dir=output_dir)
    
    # Log runtime
    total_time = time.time() - start_time
    logging.info(f"HDBSCAN training completed in {total_time:.2f} seconds")
    
    # Add correlation ID to metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "umap_n_components": umap_n_components,
            "umap_n_neighbors": umap_n_neighbors,
            "umap_min_dist": umap_min_dist,
            "metric": metric,
            "random_state": random_state
        },
        "results": clusterer.clustering_stats,
        "runtime_seconds": total_time,
        "dataset_name": dataset_name,
        "correlation_id": correlation_id,
        "saved_files": saved_files
    }
    
    # Save metadata
    if blob_storage and blob_storage._container_client:
        blob_storage.upload_json(metadata, f"{clustering_blob_path}/clustering_pipeline_metadata.json")
    elif output_dir:
        save_json(metadata, f"{output_dir}/clustering_pipeline_metadata.json")
    
    # Update the training cycle info with clustering details
    if blob_storage and blob_storage._container_client:
        try:
            cycle_info = blob_storage.download_json(f"{dataset_name}/latest_training_cycle.json")
            if cycle_info:
                cycle_info["clustering_completed"] = True
                cycle_info["clustering_timestamp"] = datetime.now().isoformat()
                cycle_info["clustering_model_path"] = f"{clustering_blob_path}/hdbscan_clusterer.pkl"
                blob_storage.upload_json(cycle_info, f"{dataset_name}/latest_training_cycle.json")
        except Exception as e:
            logging.warning(f"Could not update training cycle info: {e}")
    
    return result_df, clusterer.umap_embeddings, clusterer.clusterer, clusterer.reducer
