"""
Clustering models for the topic clustering package.
"""

import time
import logging
import os
import pickle
import numpy as np
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan
from datetime import datetime

from ..config.settings import config
from ..utils.json_utils import NumpyEncoder, save_json
from ..utils.azure_utils import blob_storage

class TopicClusterer:
    """Class for clustering topics using UMAP and HDBSCAN"""
    
    def __init__(self, min_cluster_size=None, min_samples=None, 
                 umap_n_components=None, umap_n_neighbors=None, 
                 umap_min_dist=None, metric=None, random_state=None):
        
        # Use provided values or defaults from config
        self.min_cluster_size = min_cluster_size or config.clustering["min_cluster_size"]
        self.min_samples = min_samples or config.clustering["min_samples"]
        self.umap_n_components = umap_n_components or config.clustering["umap_n_components"]
        self.umap_n_neighbors = umap_n_neighbors or config.clustering["umap_n_neighbors"]
        self.umap_min_dist = umap_min_dist or config.clustering["umap_min_dist"]
        self.metric = metric or config.clustering["metric"]
        self.random_state = random_state or config.clustering["random_state"]
        
        # Initialize models to None (will be created during fitting)
        self.reducer = None
        self.clusterer = None
        self.scaler = None
        
        # Results
        self.umap_embeddings = None
        self.cluster_labels = None
        self.probabilities = None
        self.clustering_stats = {}
    
    def fit(self, embeddings):
        """
        Fit the UMAP reducer and HDBSCAN clusterer to the data
        
        Args:
            embeddings: Numpy array of embeddings (can be loaded from JSON strings)
            
        Returns:
            self, cluster_labels
        """
        start_time = time.time()
        logging.info(f"Starting HDBSCAN clustering for {len(embeddings)} documents")
        
        # Standardize embeddings
        logging.info("Standardizing embeddings...")
        self.scaler = StandardScaler()
        scaled_embeddings = self.scaler.fit_transform(embeddings)
        
        # Apply UMAP for dimensionality reduction
        logging.info(f"Applying UMAP to reduce dimensions from {scaled_embeddings.shape[1]} to {self.umap_n_components}...")
        self.reducer = umap.UMAP(
            n_components=self.umap_n_components,
            n_neighbors=self.umap_n_neighbors,
            min_dist=self.umap_min_dist,
            metric=self.metric,
            random_state=self.random_state,
            low_memory=True,
            spread=1.0,
            local_connectivity=2,
            verbose=True
        )
        
        self.umap_embeddings = self.reducer.fit_transform(scaled_embeddings)
        
        # Apply HDBSCAN for clustering
        logging.info(f"Applying HDBSCAN with min_cluster_size={self.min_cluster_size}, min_samples={self.min_samples}...")
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean',
            cluster_selection_method='leaf',
            cluster_selection_epsilon=0.2,
            prediction_data=True,
            core_dist_n_jobs=-1
        )
        
        self.cluster_labels = self.clusterer.fit_predict(self.umap_embeddings)
        
        # Store cluster probabilities if available
        if hasattr(self.clusterer, 'probabilities_'):
            self.probabilities = self.clusterer.probabilities_
        
        # Count clusters and noise points
        clusters = set(self.cluster_labels)
        n_clusters = len(clusters) - (1 if -1 in clusters else 0)
        n_noise = list(self.cluster_labels).count(-1)
        noise_percentage = 100 * n_noise / len(self.cluster_labels) if len(self.cluster_labels) > 0 else 0
        
        logging.info(f"HDBSCAN found {n_clusters} clusters")
        logging.info(f"Noise points: {n_noise} ({noise_percentage:.2f}% of data)")
        
        total_time = time.time() - start_time
        logging.info(f"Total HDBSCAN processing time: {total_time:.2f} seconds")
        
        self.clustering_stats = {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_percentage": noise_percentage,
            "processing_time": total_time
        }
        
        return self, self.cluster_labels
    
    def transform(self, embeddings):
        """
        Transform new data using the fitted model
        
        Args:
            embeddings: New embeddings to transform
            
        Returns:
            Cluster labels for the new data
        """
        if self.reducer is None or self.clusterer is None:
            raise ValueError("Models must be fitted before transform can be called")
        
        # Scale the embeddings
        scaled_embeddings = self.scaler.transform(embeddings)
        
        # Reduce dimensionality with UMAP
        reduced_embeddings = self.reducer.transform(scaled_embeddings)
        
        # Get approximate cluster labels
        cluster_labels, strengths = hdbscan.approximate_predict(self.clusterer, reduced_embeddings)
        
        return cluster_labels, strengths
    
    def save(self, dataset_name=None, output_dir=None):
        """
        Save the model to Azure Blob Storage (and optionally to disk)
        
        Args:
            dataset_name: Dataset name for Azure Blob Storage paths
            output_dir: Optional local directory to save the model (if None, only saves to blob storage)
            
        Returns:
            Dictionary with saved file paths/blob paths
        """
        saved_files = {
            "local": {},
            "blob": {}
        }
        
        # Check if blob storage is available
        if not blob_storage or not blob_storage._container_client:
            if not output_dir:
                raise ValueError("Either blob_storage must be configured or output_dir must be provided")
            logging.warning("Blob storage not available. Only saving to local disk.")
        
        # Prepare metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "min_cluster_size": self.min_cluster_size,
                "min_samples": self.min_samples,
                "umap_n_components": self.umap_n_components,
                "umap_n_neighbors": self.umap_n_neighbors,
                "umap_min_dist": self.umap_min_dist,
                "metric": self.metric,
                "random_state": self.random_state
            },
            "results": self.clustering_stats,
            "dataset_name": dataset_name
        }
        
        # Define blob paths
        if dataset_name and blob_storage and blob_storage._container_client is not None:
            base_blob_path = f"{dataset_name}/clustering"
            umap_blob_path = f"{base_blob_path}/umap_reducer.pkl"
            hdbscan_blob_path = f"{base_blob_path}/hdbscan_clusterer.pkl"
            scaler_blob_path = f"{base_blob_path}/scaler.pkl"
            metadata_blob_path = f"{base_blob_path}/clustering_metadata.json"
            umap_embeddings_blob_path = f"{base_blob_path}/umap_embeddings.npy"
            
            # Upload models to blob storage
            if blob_storage.upload_pickle(self.reducer, umap_blob_path):
                saved_files["blob"]["umap_reducer"] = umap_blob_path
                
            if blob_storage.upload_pickle(self.clusterer, hdbscan_blob_path):
                saved_files["blob"]["hdbscan_clusterer"] = hdbscan_blob_path
                
            if blob_storage.upload_pickle(self.scaler, scaler_blob_path):
                saved_files["blob"]["scaler"] = scaler_blob_path
                
            if blob_storage.upload_json(metadata, metadata_blob_path):
                saved_files["blob"]["metadata"] = metadata_blob_path
                
            # Try to save UMAP embeddings
            if self.umap_embeddings is not None:
                if blob_storage.upload_numpy(self.umap_embeddings, umap_embeddings_blob_path):
                    saved_files["blob"]["umap_embeddings"] = umap_embeddings_blob_path
            
            # Create a manifest file
            manifest = {
                "dataset_name": dataset_name,
                "timestamp": datetime.now().isoformat(),
                "artifacts": saved_files["blob"]
            }
            
            manifest_blob_path = f"{base_blob_path}/manifest.json"
            if blob_storage.upload_json(manifest, manifest_blob_path):
                saved_files["blob"]["manifest"] = manifest_blob_path
                logging.info(f"Successfully created model manifest in Azure: {manifest_blob_path}")
        
        # Optionally save to local disk as well
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save UMAP reducer
            umap_path = os.path.join(output_dir, "umap_reducer.pkl")
            with open(umap_path, "wb") as f:
                pickle.dump(self.reducer, f)
            saved_files["local"]["umap_reducer"] = umap_path
            
            # Save HDBSCAN clusterer
            hdbscan_path = os.path.join(output_dir, "hdbscan_clusterer.pkl")
            with open(hdbscan_path, "wb") as f:
                pickle.dump(self.clusterer, f)
            saved_files["local"]["hdbscan_clusterer"] = hdbscan_path
            
            # Save scaler
            scaler_path = os.path.join(output_dir, "scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)
            saved_files["local"]["scaler"] = scaler_path
            
            # Save metadata
            metadata_path = os.path.join(output_dir, "clustering_metadata.json")
            save_json(metadata, metadata_path)
            saved_files["local"]["metadata"] = metadata_path
            
            # Try to save UMAP embeddings as NPY file
            try:
                umap_embeddings_path = os.path.join(output_dir, "umap_embeddings.npy")
                np.save(umap_embeddings_path, self.umap_embeddings)
                saved_files["local"]["umap_embeddings"] = umap_embeddings_path
            except Exception as e:
                logging.warning(f"Could not save UMAP embeddings as numpy array: {e}")
        
        # Log success summary
        blob_count = len(saved_files["blob"])
        local_count = len(saved_files["local"])
        if blob_count > 0:
            logging.info(f"Saved {blob_count} model artifacts to Azure Blob Storage")
        if local_count > 0:
            logging.info(f"Saved {local_count} model artifacts to local disk")
        
        return saved_files
    
    @classmethod
    def load(cls, dataset_name=None, model_dir=None, training_cycle=None):
        """
        Load the model from Azure Blob Storage or local disk
        
        Args:
            dataset_name: Dataset name for Azure Blob Storage paths (preferred source)
            model_dir: Local directory containing saved model files (fallback)
            training_cycle: Specific training cycle to load (if None, uses latest)
            
        Returns:
            Loaded TopicClusterer instance
        """
        instance = cls()
        
        # Try loading from Azure Blob Storage first
        if dataset_name and blob_storage and blob_storage._container_client is not None:
            # First, try to determine which training cycle to use
            if not training_cycle:
                # Try to load latest training cycle info
                try:
                    cycle_info = blob_storage.download_json(f"{dataset_name}/latest_training_cycle.json")
                    if cycle_info and "training_cycle" in cycle_info:
                        training_cycle = cycle_info["training_cycle"]
                        logging.info(f"Loading model from training cycle {training_cycle}")
                    else:
                        # No training cycle info, look for any model_manifest
                        manifest = blob_storage.download_json(f"{dataset_name}/latest_model_manifest.json")
                        if manifest and "training_cycle" in manifest:
                            training_cycle = manifest["training_cycle"]
                            logging.info(f"Loading model from manifest training cycle {training_cycle}")
                except Exception as e:
                    logging.warning(f"Could not determine latest training cycle: {e}")
            
            # Define paths based on training cycle
            if training_cycle:
                base_blob_path = f"{dataset_name}/training_cycle_{training_cycle}/clustering"
            else:
                # Use old path format as fallback (for backward compatibility)
                base_blob_path = f"{dataset_name}/clustering"
                
            umap_blob_path = f"{base_blob_path}/umap_reducer.pkl"
            hdbscan_blob_path = f"{base_blob_path}/hdbscan_clusterer.pkl"
            scaler_blob_path = f"{base_blob_path}/scaler.pkl"
            
            # Check if required blobs exist
            if blob_storage.blob_exists(umap_blob_path) and blob_storage.blob_exists(hdbscan_blob_path):
                logging.info(f"Loading model from Azure Blob Storage: {base_blob_path}")
                
                # Load the models
                instance.reducer = blob_storage.download_pickle(umap_blob_path)
                instance.clusterer = blob_storage.download_pickle(hdbscan_blob_path)
                
                # Load scaler if available, otherwise create new one
                instance.scaler = blob_storage.download_pickle(scaler_blob_path)
                if instance.scaler is None:
                    logging.warning(f"Could not load scaler from {scaler_blob_path}, creating a new one")
                    instance.scaler = StandardScaler()
                
                # Extract parameters
                if instance.reducer and instance.clusterer:
                    instance.min_cluster_size = instance.clusterer.min_cluster_size
                    instance.min_samples = instance.clusterer.min_samples
                    instance.umap_n_components = instance.reducer.n_components
                    instance.umap_n_neighbors = instance.reducer.n_neighbors
                    instance.umap_min_dist = instance.reducer.min_dist
                    instance.metric = instance.reducer.metric
                    instance.random_state = instance.reducer.random_state
                    
                    logging.info("Successfully loaded model from Azure Blob Storage")
                    return instance
                else:
                    logging.error("Failed to load required model components from Azure Blob Storage")
            else:
                # If not found in training cycle path, try legacy path
                if training_cycle:
                    logging.warning(f"Required model files not found in training cycle path, trying legacy path")
                    return cls.load(dataset_name=dataset_name, model_dir=model_dir, training_cycle=None)
                else:
                    logging.warning(f"Required model files not found in Azure Blob Storage")
        
        # Fall back to loading from local directory
        if model_dir:
            logging.info(f"Loading model from local disk: {model_dir}")
            
            umap_path = os.path.join(model_dir, "umap_reducer.pkl")
            hdbscan_path = os.path.join(model_dir, "hdbscan_clusterer.pkl")
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            
            if not os.path.exists(umap_path):
                raise FileNotFoundError(f"UMAP reducer not found at {umap_path}")
            
            if not os.path.exists(hdbscan_path):
                raise FileNotFoundError(f"HDBSCAN clusterer not found at {hdbscan_path}")
            
            # Load the models
            with open(umap_path, "rb") as f:
                instance.reducer = pickle.load(f)
            
            with open(hdbscan_path, "rb") as f:
                instance.clusterer = pickle.load(f)
            
            if os.path.exists(scaler_path):
                with open(scaler_path, "rb") as f:
                    instance.scaler = pickle.load(f)
            else:
                # Create a new scaler if not found
                instance.scaler = StandardScaler()
                
            # Extract parameters
            instance.min_cluster_size = instance.clusterer.min_cluster_size
            instance.min_samples = instance.clusterer.min_samples
            instance.umap_n_components = instance.reducer.n_components
            instance.umap_n_neighbors = instance.reducer.n_neighbors
            instance.umap_min_dist = instance.reducer.min_dist
            instance.metric = instance.reducer.metric
            instance.random_state = instance.reducer.random_state
            
            return instance
                
        raise ValueError("Either dataset_name (with configured blob_storage) or model_dir must be provided")

def process_data_for_clustering(df, embedding_column='embedding'):
    """
    Process DataFrame with embedding column for clustering
    
    Args:
        df: DataFrame with embeddings stored as JSON strings
        embedding_column: Name of column containing embeddings
        
    Returns:
        Numpy array of embeddings
    """
    import json
    
    # Extract embeddings from the dataframe
    logging.info(f"Extracting embeddings from column '{embedding_column}'...")
    embeddings = np.array([json.loads(emb) for emb in df[embedding_column].tolist()])
    
    return embeddings

def generate_cluster_info(df, text_column='short_description', cluster_column='cluster', sample_size=5, output_dir=None):
    """
    Generate information about each cluster in the dataframe
    
    Args:
        df: DataFrame with clustering results
        text_column: Column containing text to sample for cluster descriptions
        cluster_column: Column containing cluster labels
        sample_size: Number of samples to include for each cluster
        output_dir: Directory to save cluster info
        
    Returns:
        Dictionary with cluster information
    """
    from ..utils.json_utils import save_json, NumpyEncoder
    
    # Dictionary to store cluster information
    clusters = {}
    
    # Get unique clusters
    unique_clusters = sorted(df[cluster_column].unique())
    
    for cluster_id in unique_clusters:
        # Get rows for this cluster
        cluster_df = df[df[cluster_column] == cluster_id]
        count = len(cluster_df)
        
        # Get sample texts
        samples = cluster_df[text_column].sample(min(sample_size, count)).tolist()
        
        # Store the cluster information with explicit type conversion
        clusters[str(cluster_id)] = {
            "id": int(cluster_id),
            "size": int(count),
            "percentage": float(round((count / len(df)) * 100, 2)),
            "is_noise": bool(cluster_id == -1),
            "samples": samples
        }
    
    # Save cluster info if output_dir provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "cluster_details.json"), "w") as f:
            json.dump(clusters, f, indent=2, cls=NumpyEncoder)
    
    return clusters
