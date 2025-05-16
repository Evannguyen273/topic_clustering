"""
Prediction functionality for topic clustering.
"""

import os
import logging
import json
import time
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union

from ..config.settings import config
from ..models.embeddings import HybridEmbeddingGenerator, EmbeddingClient
from ..models.clustering import TopicClusterer, process_data_for_clustering
from ..utils.bigquery_utils import run_query, save_results_to_bigquery
from ..utils.azure_utils import blob_storage, RetryHandler
from ..utils.text_utils import get_safe_text, clean_text_for_summary
from ..utils.json_utils import save_json

class PredictionRegistry:
    """Manages the prediction state registry for all tech centers"""
    
    def __init__(self, registry_path: str = "prediction/registry.json"):
        """
        Initialize the prediction registry
        
        Args:
            registry_path: Path to the registry file in blob storage
        """
        self.registry_path = registry_path
        self._registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load registry from blob storage or create a new one if it doesn't exist"""
        if blob_storage and blob_storage._container_client:
            if blob_storage.blob_exists(self.registry_path):
                registry = blob_storage.download_json(self.registry_path)
                if registry:
                    return registry
        
        # If registry doesn't exist or couldn't be loaded, create a new one
        return {
            "tech_centers": {},
            "last_updated": datetime.now().isoformat()
        }
    
    def save_registry(self) -> bool:
        """Save registry to blob storage"""
        self._registry["last_updated"] = datetime.now().isoformat()
        
        if blob_storage and blob_storage._container_client:
            return blob_storage.upload_json(self._registry, self.registry_path)
        return False
    
    def get_tech_center_info(self, tech_center: str) -> Dict[str, Any]:
        """Get information for a specific tech center"""
        if tech_center not in self._registry["tech_centers"]:
            # Initialize with default values if tech center not in registry
            self._registry["tech_centers"][tech_center] = {
                "last_training_run": None,
                "last_prediction_run": None,
                "last_processed_timestamp": "2024-01-01T00:00:00",  # Start from beginning of 2024
                "model_artifacts": {},
                "prediction_stats": {}
            }
            self.save_registry()
            
        return self._registry["tech_centers"][tech_center]
    
    def update_tech_center(self, tech_center: str, info: Dict[str, Any]) -> bool:
        """Update information for a specific tech center"""
        self._registry["tech_centers"][tech_center] = info
        return self.save_registry()
    
    def update_last_processed(self, tech_center: str, timestamp: str) -> bool:
        """Update the last processed timestamp for a tech center"""
        if tech_center in self._registry["tech_centers"]:
            self._registry["tech_centers"][tech_center]["last_processed_timestamp"] = timestamp
            self._registry["tech_centers"][tech_center]["last_prediction_run"] = datetime.now().isoformat()
            return self.save_registry()
        return False
    
    def add_prediction_stats(self, tech_center: str, run_date: str, stats: Dict[str, Any]) -> bool:
        """Add prediction statistics for a tech center"""
        if tech_center in self._registry["tech_centers"]:
            if "prediction_stats" not in self._registry["tech_centers"][tech_center]:
                self._registry["tech_centers"][tech_center]["prediction_stats"] = {}
                
            self._registry["tech_centers"][tech_center]["prediction_stats"][run_date] = stats
            return self.save_registry()
        return False
    
    def list_tech_centers(self) -> List[str]:
        """List all tech centers in the registry"""
        return list(self._registry["tech_centers"].keys())

def generate_tech_center_query(tech_center: str, since_timestamp: str, until_timestamp: Optional[str] = None) -> str:
    """
    Generate a query for fetching incidents for a specific tech center
    
    Args:
        tech_center: Name of the tech center
        since_timestamp: Only fetch incidents created after this timestamp
        until_timestamp: Only fetch incidents created before this timestamp (default: current time)
        
    Returns:
        BigQuery query string
    """
    if not until_timestamp:
        until_timestamp = datetime.now().isoformat()
        
    query = f"""
    SELECT t3.number, t3.priority, t3.sys_created_on, t1.TeamDepartment, t1.Area, t1.Unit, t1.TechCenter,
           t3.short_description, t3.close_code, t3.vendor, t3.assignment_group, t3.state, t3.business_service,
           t3.close_notes, t3.work_notes, t3.contact_type as chanel, t3.description
    FROM `enterprise-dashboardnp-cd35.bigquery_datasets_hone_srv_dev.oa_snow_incident_mgmt_srv_dev` t3
    JOIN `enterprise-dashboardnp-cd35.bigquery_datasets_spoke_oa_dev.Team_services` t1
    ON t3.business_service = t1.Services
    WHERE t1.TechCenter = '{tech_center}' 
    AND t3.sys_created_on > '{since_timestamp}'
    AND t3.sys_created_on <= '{until_timestamp}'
    ORDER BY t3.sys_created_on
    """
    
    return query

def load_models(tech_center: str, training_cycle: Optional[str] = None) -> Tuple[Optional[TopicClusterer], Optional[Dict[str, Dict[str, str]]]]:
    """
    Load prediction models for a tech center
    
    Args:
        tech_center: Name of the tech center
        training_cycle: Specific training cycle to load (if None, uses latest)
        
    Returns:
        Tuple of (TopicClusterer model, labeled_clusters dict)
    """
    # Sanitize tech center name for use in paths
    tech_center_path = tech_center.replace(" ", "_").replace("-", "_").replace("/", "_")
    
    # Try to determine which training cycle to use
    if not training_cycle and blob_storage and blob_storage._container_client:
        try:
            # First check if we have tracking info
            cycle_info = blob_storage.download_json(f"{tech_center_path}/latest_training_cycle.json")
            if cycle_info and "training_cycle" in cycle_info:
                training_cycle = cycle_info["training_cycle"]
                logging.info(f"Using training cycle {training_cycle} from tracking info")
            else:
                # Check if we have a model manifest
                manifest = blob_storage.download_json(f"{tech_center_path}/latest_model_manifest.json")
                if manifest and "training_cycle" in manifest:
                    training_cycle = manifest["training_cycle"]
                    logging.info(f"Using training cycle {training_cycle} from model manifest")
        except Exception as e:
            logging.warning(f"Could not determine latest training cycle: {e}")
    
    # Load the model
    try:
        # Pass training cycle to the load method
        clusterer = TopicClusterer.load(dataset_name=tech_center_path, training_cycle=training_cycle)
        logging.info(f"Loaded clustering model for {tech_center} from {'training cycle ' + training_cycle if training_cycle else 'default location'}")
    except Exception as e:
        logging.error(f"Error loading model for {tech_center}: {e}")
        return None, None
    
    # Define paths based on training cycle
    if training_cycle:
        base_path = f"{tech_center_path}/training_cycle_{training_cycle}"
    else:
        # Legacy path format
        base_path = tech_center_path
    
    # Load labeled clusters
    labeled_clusters = None
    if blob_storage and blob_storage._container_client:
        # Try versioned paths first
        labels_path = f"{base_path}/analysis/labeled_clusters.json"
        fallback_path = f"{base_path}/analysis/labeled_clusters_fallback.json"
        
        if blob_storage.blob_exists(labels_path):
            labeled_clusters = blob_storage.download_json(labels_path)
        elif blob_storage.blob_exists(fallback_path):
            labeled_clusters = blob_storage.download_json(fallback_path)
        else:
            # Try legacy paths (root level)
            legacy_labels_path = f"{tech_center_path}/analysis/labeled_clusters.json"
            legacy_fallback_path = f"{tech_center_path}/analysis/labeled_clusters_fallback.json"
            
            if blob_storage.blob_exists(legacy_labels_path):
                labeled_clusters = blob_storage.download_json(legacy_labels_path)
                logging.info(f"Loaded labeled clusters from legacy path: {legacy_labels_path}")
            elif blob_storage.blob_exists(legacy_fallback_path):
                labeled_clusters = blob_storage.download_json(legacy_fallback_path)
                logging.info(f"Loaded labeled clusters from legacy fallback path: {legacy_fallback_path}")
    
    if not labeled_clusters:
        logging.error(f"Could not load labeled clusters for {tech_center}")
        return clusterer, None
    
    return clusterer, labeled_clusters

def process_incident_texts(df: pd.DataFrame) -> pd.Series:
    """
    Process incident texts for embedding
    
    Args:
        df: DataFrame with incident data
        
    Returns:
        Series with processed texts
    """
    result = pd.Series(index=df.index)
    
    for idx, row in df.iterrows():
        # Combine short description and business service
        short_desc = get_safe_text(row, 'short_description')
        business_svc = get_safe_text(row, 'business_service')
        
        # Get description if available
        desc = get_safe_text(row, 'description')
        if len(desc) > 20:  # Only use description if it has meaningful content
            # Clean and trim description
            desc = clean_text_for_summary(desc)
            desc = desc[:1000]  # Limit to 1000 chars
            combined = f"{short_desc} - {business_svc}. {desc}"
        else:
            combined = f"{short_desc} - {business_svc}"
        
        result[idx] = combined
    
    return result

def predict_incidents(tech_center: str, registry: Optional[PredictionRegistry] = None,
                    since_timestamp: Optional[str] = None, 
                    until_timestamp: Optional[str] = None) -> Dict[str, Any]:
    """
    Run prediction on new incidents for a tech center
    
    Args:
        tech_center: Name of the tech center
        registry: Prediction registry (created if not provided)
        since_timestamp: Override for start timestamp
        until_timestamp: Override for end timestamp
        
    Returns:
        Dictionary with prediction results and statistics
    """
    start_time = time.time()
    run_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    tech_center_path = tech_center.replace(" ", "_").replace("-", "_").replace("/", "_")
    
    # Initialize or use provided registry
    if registry is None:
        registry = PredictionRegistry()
    
    # Get tech center info from registry
    tech_center_info = registry.get_tech_center_info(tech_center)
    
    # Determine time range to process
    if not since_timestamp:
        since_timestamp = tech_center_info["last_processed_timestamp"]
    
    if not until_timestamp:
        until_timestamp = datetime.now().isoformat()
    
    logging.info(f"Running prediction for {tech_center} from {since_timestamp} to {until_timestamp}")
    
    # Create blob path for this prediction run
    prediction_path = f"{tech_center_path}/prediction/{datetime.now().strftime('%Y%m%d')}"
    log_path = f"{prediction_path}/prediction_{run_date}.log"
    
    # Initialize statistics
    stats = {
        "tech_center": tech_center,
        "start_time": datetime.now().isoformat(),
        "since_timestamp": since_timestamp,
        "until_timestamp": until_timestamp,
        "run_date": run_date,
        "prediction_path": prediction_path,
        "incidents_processed": 0,
        "processing_time": 0,
        "success": False,
        "error": None,
        "max_timestamp": since_timestamp  # Will be updated with actual max timestamp
    }
    
    try:
        # 1. Build query for new data
        query = generate_tech_center_query(tech_center, since_timestamp, until_timestamp)
        
        # 2. Fetch incidents from BigQuery
        incidents_df = run_query(query)
        
        if incidents_df is None or len(incidents_df) == 0:
            logging.info(f"No new incidents for {tech_center} since {since_timestamp}")
            stats["incidents_processed"] = 0
            stats["success"] = True
            stats["processing_time"] = time.time() - start_time
            
            # Save prediction log
            if blob_storage and blob_storage._container_client:
                blob_storage.upload_json(stats, f"{prediction_path}/stats_{run_date}.json")
            
            return stats
        
        logging.info(f"Found {len(incidents_df)} new incidents for {tech_center}")
        stats["incidents_processed"] = len(incidents_df)
        
        # 3. Load models
        clusterer, labeled_clusters = load_models(tech_center)
        if clusterer is None or labeled_clusters is None:
            stats["error"] = "Failed to load prediction models"
            stats["success"] = False
            stats["processing_time"] = time.time() - start_time
            
            # Save prediction log
            if blob_storage and blob_storage._container_client:
                blob_storage.upload_json(stats, f"{prediction_path}/stats_{run_date}.json")
                
            raise ValueError(f"Failed to load prediction models for {tech_center}")
        
        # 4. Process texts for embedding
        incidents_df['combined_incidents_summary'] = process_incident_texts(incidents_df)
        
        # 5. Generate embeddings
        embedding_client = EmbeddingClient()
        hybrid_generator = HybridEmbeddingGenerator(embedding_client)
        
        df_with_embeddings, _ = hybrid_generator.generate_hybrid_embeddings(
            incidents_df, 
            text_column='combined_incidents_summary',
            batch_size=config.batch_size
        )
        
        # 6. Process embeddings for clustering
        embeddings = process_data_for_clustering(df_with_embeddings)
        
        # 7. Apply clustering model
        cluster_labels, strengths = clusterer.transform(embeddings)
        
        # 8. Add cluster labels to dataframe
        result_df = df_with_embeddings.copy()
        result_df['cluster'] = cluster_labels
        result_df['cluster_probability'] = strengths
        
        # 9. Apply domain and topic labels
        domain_mapping = {}
        topic_mapping = {}
        
        # Load domains
        domains = None
        if blob_storage and blob_storage._container_client:
            domains_path = f"{tech_center_path}/analysis/domains.json"
            fallback_path = f"{tech_center_path}/analysis/domains_fallback.json"
            
            if blob_storage.blob_exists(domains_path):
                domains = blob_storage.download_json(domains_path)
            elif blob_storage.blob_exists(fallback_path):
                domains = blob_storage.download_json(fallback_path)
        
        # Create mappings
        if domains:
            for domain in domains.get("domains", []):
                domain_name = domain["domain_name"]
                for cluster_id in domain.get("clusters", []):
                    domain_mapping[int(cluster_id)] = domain_name
        
        # Create topic mapping
        for cluster_id, label_data in labeled_clusters.items():
            if cluster_id != "-1":
                topic_mapping[int(cluster_id)] = label_data.get("topic", f"Cluster {cluster_id}")
        
        # Add noise
        domain_mapping[-1] = "Noise"
        topic_mapping[-1] = "Noise"
        
        # Apply mappings
        result_df["subcategory"] = result_df["cluster"].map(topic_mapping).fillna("Unknown")
        result_df["category"] = result_df["cluster"].map(domain_mapping).fillna("Other")
        
        # 10. Save results to BigQuery
        table_id = f"your_project.your_dataset.{tech_center_path}_predictions"
        save_success = save_results_to_bigquery(result_df, table_id)
        
        if save_success:
            logging.info(f"Successfully saved {len(result_df)} predictions to BigQuery table {table_id}")
        else:
            logging.warning(f"Failed to save predictions to BigQuery table {table_id}")
        
        # 11. Save prediction data to blob storage
        if blob_storage and blob_storage._container_client:
            # Save full results
            blob_storage.upload_dataframe(result_df, f"{prediction_path}/predictions_{run_date}.parquet")
            
            # Save a CSV sample for easy viewing
            sample_size = min(100, len(result_df))
            sample_df = result_df.sample(sample_size) if sample_size < len(result_df) else result_df
            blob_storage.upload_dataframe(sample_df, f"{prediction_path}/predictions_sample_{run_date}.csv", format="csv")
        
        # 12. Update statistics
        stats["success"] = True
        stats["processing_time"] = time.time() - start_time
        stats["bigquery_table"] = table_id
        stats["cluster_counts"] = result_df["cluster"].value_counts().to_dict()
        stats["category_counts"] = result_df["category"].value_counts().to_dict()
        
        # 13. Update registry with new last processed timestamp
        # Find the latest timestamp in the data
        if "sys_created_on" in result_df.columns and len(result_df) > 0:
            max_timestamp = result_df["sys_created_on"].max()
            stats["max_timestamp"] = max_timestamp
            registry.update_last_processed(tech_center, max_timestamp)
        
        # 14. Add prediction stats to registry
        registry.add_prediction_stats(tech_center, run_date, {
            "incidents_processed": stats["incidents_processed"],
            "success": stats["success"],
            "processing_time": stats["processing_time"],
            "since_timestamp": since_timestamp,
            "max_timestamp": stats["max_timestamp"]
        })
        
        # 15. Save prediction log
        if blob_storage and blob_storage._container_client:
            blob_storage.upload_json(stats, f"{prediction_path}/stats_{run_date}.json")
        
        logging.info(f"Prediction completed for {tech_center} in {stats['processing_time']:.2f} seconds")
        return stats
        
    except Exception as e:
        logging.error(f"Error during prediction for {tech_center}: {e}", exc_info=True)
        
        # Update stats with error
        stats["error"] = str(e)
        stats["success"] = False
        stats["processing_time"] = time.time() - start_time
        
        # Save prediction log even when failed
        if blob_storage and blob_storage._container_client:
            blob_storage.upload_json(stats, f"{prediction_path}/stats_{run_date}.json")
        
        # Re-raise the exception for higher-level handling
        raise
