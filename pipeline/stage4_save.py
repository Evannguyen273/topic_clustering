"""
Fourth pipeline stage: Save results to BigQuery and blob storage.
"""

import os
import time
import logging
import pandas as pd
import json
from datetime import datetime

from ..config.settings import config
from ..utils.bigquery_utils import save_results_to_bigquery
from ..utils.json_utils import save_json
from ..utils.azure_utils import blob_storage

def generate_visualization_data(clusters_info, labeled_clusters, domains, dataset_name):
    """
    Generate visualization-friendly data for dashboard or UI
    
    Args:
        clusters_info: Cluster information dictionary
        labeled_clusters: Dictionary of labeled clusters
        domains: Dictionary of domains
        dataset_name: Dataset name for blob storage path
        
    Returns:
        Dictionary with visualization data
    """
    # Combine cluster info with labels
    viz_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_clusters": len(clusters_info) - (1 if "-1" in clusters_info else 0),
            "num_domains": len(domains["domains"]) - 1,  # Exclude noise domain
            "noise_percentage": clusters_info.get("-1", {}).get("percentage", 0) if "-1" in clusters_info else 0
        },
        "domains": domains["domains"],
        "clusters": []
    }
    
    # Create detailed cluster information for visualization
    for cluster_id, info in clusters_info.items():
        if cluster_id == "-1":  # Skip noise in detailed view
            continue
            
        cluster_label = labeled_clusters.get(cluster_id, {})
        
        # Find which domain this cluster belongs to
        domain_name = "Other"
        for domain in domains["domains"]:
            if int(cluster_id) in domain.get("clusters", []):
                domain_name = domain["domain_name"]
                break
                
        # Create cluster entry with all needed information for visualization
        cluster_entry = {
            "id": int(cluster_id),
            "size": info["size"],
            "percentage": info["percentage"],
            "topic": cluster_label.get("topic", f"Cluster {cluster_id}"),
            "description": cluster_label.get("description", "No description available"),
            "domain": domain_name,
            "samples": info.get("samples", [])[:3]  # Include up to 3 samples
        }
        
        viz_data["clusters"].append(cluster_entry)
    
    # Add a summary of noise
    if "-1" in clusters_info:
        viz_data["noise"] = {
            "size": clusters_info["-1"]["size"],
            "percentage": clusters_info["-1"]["percentage"],
            "samples": clusters_info["-1"].get("samples", [])[:3]
        }
    
    # Save visualization data to blob storage
    if blob_storage and blob_storage._container_client:
        blob_path = f"{dataset_name}/results/visualization_data.json"
        blob_storage.upload_json(viz_data, blob_path)
    
    return viz_data

def generate_summary_report(clusters_info, labeled_clusters, domains, metadata, dataset_name):
    """
    Generate a textual summary report of the clustering results
    
    Args:
        clusters_info: Cluster information dictionary
        labeled_clusters: Dictionary of labeled clusters
        domains: Dictionary of domains
        metadata: Metadata about the clustering run
        dataset_name: Dataset name for blob storage path
        
    Returns:
        Generated report content
    """
    # Start building the report
    report = []
    report.append("# Cluster Analysis Summary Report")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Dataset: {metadata.get('dataset', 'Unknown')}")
    report.append("")
    
    # Overview section
    report.append("## Overview")
    report.append(f"- Total clusters: {len(clusters_info) - (1 if '-1' in clusters_info else 0)}")
    report.append(f"- Total domains: {len(domains['domains']) - 1}")  # Exclude noise domain
    
    if "-1" in clusters_info:
        noise_pct = clusters_info["-1"]["percentage"]
        report.append(f"- Noise percentage: {noise_pct:.2f}%")
        report.append(f"- Clustering rate: {100 - noise_pct:.2f}%")
    
    # Add processing time if available
    if "processing_time" in metadata:
        report.append(f"- Processing time: {metadata['processing_time']:.2f} seconds")
    
    report.append("")
    
    # Domains section
    report.append("## Domains")
    for i, domain in enumerate([d for d in domains["domains"] if d["domain_name"] != "Noise"]):
        num_clusters = len(domain["clusters"])
        report.append(f"### {i+1}. {domain['domain_name']} ({num_clusters} clusters)")
        report.append(f"*{domain['description']}*")
        report.append("")
        
        # List clusters in this domain
        for cluster_id in domain["clusters"]:
            if str(cluster_id) in labeled_clusters:
                topic = labeled_clusters[str(cluster_id)]["topic"]
                size = clusters_info[str(cluster_id)]["size"]
                pct = clusters_info[str(cluster_id)]["percentage"]
                report.append(f"- Cluster {cluster_id}: **{topic}** ({size} incidents, {pct:.2f}%)")
        
        report.append("")
    
    # Noise section if present
    if "-1" in clusters_info:
        noise = clusters_info["-1"]
        report.append("## Noise (Unclustered Data)")
        report.append(f"- {noise['size']} incidents ({noise['percentage']:.2f}% of total)")
        report.append("- These incidents couldn't be grouped into meaningful clusters")
        report.append("")
    
    # Convert the report to a single string
    report_content = "\n".join(report)
    
    # Save to blob storage
    if blob_storage and blob_storage._container_client:
        blob_path = f"{dataset_name}/results/cluster_report.md"
        blob_storage.upload_blob(report_content, blob_path, 
                               content_type="text/markdown", is_file_path=False)
    
    return report_content

def save_results_pipeline(final_df=None, clusters_info=None, labeled_clusters=None, domains=None,
                        dataset_name=None, table_id=None, result_path=None, save_to_bq=True,
                        metadata=None):
    """
    Save analysis results to BigQuery and Azure Blob Storage
    """
    # Get the training cycle from latest_training_cycle.json
    training_cycle = datetime.now().strftime("%Y%m%d")
    if blob_storage and blob_storage._container_client:
        try:
            cycle_info = blob_storage.download_json(f"{dataset_name}/latest_training_cycle.json")
            if cycle_info and "training_cycle" in cycle_info:
                training_cycle = cycle_info["training_cycle"]
                logging.info(f"Using training cycle {training_cycle} from previous stages")
        except Exception as e:
            logging.warning(f"Could not load training cycle info, using current date: {e}")
    
    # Define Azure paths with training cycle
    base_blob_path = f"{dataset_name}/training_cycle_{training_cycle}"
    results_blob_path = f"{base_blob_path}/results"
    analysis_blob_path = f"{base_blob_path}/analysis"
    
    start_time = time.time()
    
    # Fallback to local storage if blob storage is not available
    if not blob_storage or not blob_storage._container_client:
        logging.warning("Azure Blob Storage not available. Using local storage as fallback.")
        if not result_path:
            result_path = config.result_path
        output_dir = f"{result_path}/{dataset_name}/results"
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = None
    
    logging.info(f"Saving results for dataset '{dataset_name}'")
    
    # Load data from previous stage if not provided
    if any(x is None for x in [final_df, clusters_info, labeled_clusters, domains]):
        # Try to load from Azure Blob Storage first
        if blob_storage and blob_storage._container_client:
            if final_df is None:
                final_df_blob_path = f"{dataset_name}/analysis/final_df.parquet"
                if blob_storage.blob_exists(final_df_blob_path):
                    logging.info(f"Loading final dataframe from blob storage: {final_df_blob_path}")
                    final_df = blob_storage.download_dataframe(final_df_blob_path)
                else:
                    # Check for chunked files with manifest
                    manifest_blob_path = f"{dataset_name}/analysis/final_df_manifest.json"
                    if blob_storage.blob_exists(manifest_blob_path):
                        manifest = blob_storage.download_json(manifest_blob_path)
                        if manifest and "file_paths" in manifest:
                            chunks = []
                            for chunk_path in manifest["file_paths"]:
                                chunk_df = blob_storage.download_dataframe(chunk_path)
                                if chunk_df is not None:
                                    chunks.append(chunk_df)
                            
                            if chunks:
                                final_df = pd.concat(chunks, ignore_index=True)
                                logging.info(f"Loaded chunked dataframe with {len(final_df)} rows from blob storage")
            
            if clusters_info is None:
                clusters_blob_path = f"{dataset_name}/analysis/cluster_details.json"
                if blob_storage.blob_exists(clusters_blob_path):
                    clusters_info = blob_storage.download_json(clusters_blob_path)
            
            if labeled_clusters is None:
                # Try regular and fallback paths
                labels_blob_path = f"{dataset_name}/analysis/labeled_clusters.json"
                fallback_labels_blob_path = f"{dataset_name}/analysis/labeled_clusters_fallback.json"
                
                if blob_storage.blob_exists(labels_blob_path):
                    labeled_clusters = blob_storage.download_json(labels_blob_path)
                elif blob_storage.blob_exists(fallback_labels_blob_path):
                    labeled_clusters = blob_storage.download_json(fallback_labels_blob_path)
            
            if domains is None:
                # Try regular and fallback paths
                domains_blob_path = f"{dataset_name}/analysis/domains.json"
                fallback_domains_blob_path = f"{dataset_name}/analysis/domains_fallback.json"
                
                if blob_storage.blob_exists(domains_blob_path):
                    domains = blob_storage.download_json(domains_blob_path)
                elif blob_storage.blob_exists(fallback_domains_blob_path):
                    domains = blob_storage.download_json(fallback_domains_blob_path)
        
        # Fall back to local files if needed
        if result_path and any(x is None for x in [final_df, clusters_info, labeled_clusters, domains]):
            analysis_dir = f"{result_path}/{dataset_name}/analysis"
            
            if final_df is None and os.path.exists(f"{analysis_dir}/final_df.parquet"):
                final_df = pd.read_parquet(f"{analysis_dir}/final_df.parquet")
            
            if clusters_info is None and os.path.exists(f"{analysis_dir}/cluster_details.json"):
                with open(f"{analysis_dir}/cluster_details.json", "r") as f:
                    clusters_info = json.load(f)
            
            if labeled_clusters is None:
                if os.path.exists(f"{analysis_dir}/labeled_clusters.json"):
                    with open(f"{analysis_dir}/labeled_clusters.json", "r") as f:
                        labeled_clusters = json.load(f)
                elif os.path.exists(f"{analysis_dir}/labeled_clusters_fallback.json"):
                    with open(f"{analysis_dir}/labeled_clusters_fallback.json", "r") as f:
                        labeled_clusters = json.load(f)
            
            if domains is None:
                if os.path.exists(f"{analysis_dir}/domains.json"):
                    with open(f"{analysis_dir}/domains.json", "r") as f:
                        domains = json.load(f)
                elif os.path.exists(f"{analysis_dir}/domains_fallback.json"):
                    with open(f"{analysis_dir}/domains_fallback.json", "r") as f:
                        domains = json.load(f)
    
    # Use provided or loaded metadata, or create new minimal metadata
    if metadata is None:
        metadata = {
            "dataset": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "processing_time": 0
        }
    
    # Track saved resources
    saved_resources = {
        "bigquery": None,
        "blob_storage": {},
        "local": {}
    }
    
    # 1. Save to BigQuery if requested
    if save_to_bq and final_df is not None and table_id:
        logging.info(f"Saving {len(final_df)} rows to BigQuery table {table_id}")
        try:
            success = save_results_to_bigquery(final_df, table_id)
            saved_resources["bigquery"] = table_id if success else None
        except Exception as e:
            logging.error(f"Error saving to BigQuery: {e}")
    
    # 2. Generate and save visualization data
    if all(x is not None for x in [clusters_info, labeled_clusters, domains]):
        logging.info("Generating visualization data")
        viz_data = generate_visualization_data(
            clusters_info,
            labeled_clusters,
            domains,
            dataset_name
        )
        
        # Save a local copy if needed
        if output_dir:
            viz_local_path = f"{output_dir}/visualization_data.json"
            save_json(viz_data, viz_local_path)
            saved_resources["local"]["visualization"] = viz_local_path
        
        # Blob storage path already saved during generation
        saved_resources["blob_storage"]["visualization"] = f"{dataset_name}/results/visualization_data.json"
        
        # Generate summary report
        report_content = generate_summary_report(
            clusters_info,
            labeled_clusters,
            domains,
            metadata,
            dataset_name
        )
        
        # Save local copy if needed
        if output_dir:
            report_local_path = f"{output_dir}/cluster_report.md"
            with open(report_local_path, "w") as f:
                f.write(report_content)
            saved_resources["local"]["report"] = report_local_path
        
        # Blob storage path already saved during generation
        saved_resources["blob_storage"]["report"] = f"{dataset_name}/results/cluster_report.md"
    
    # 3. Save final dataframe in various useful formats
    if final_df is not None:
        # Sample CSV for quick viewing
        sample_size = min(1000, len(final_df))
        sample_df = final_df.sample(sample_size) if sample_size < len(final_df) else final_df
        
        if blob_storage and blob_storage._container_client:
            # Save sample to blob storage
            sample_blob_path = f"{dataset_name}/results/final_df_sample.csv"
            blob_storage.upload_dataframe(sample_df, sample_blob_path, format="csv")
            saved_resources["blob_storage"]["sample_csv"] = sample_blob_path
            
            # Save complete compressed data
            compressed_blob_path = f"{dataset_name}/results/final_df_compressed.parquet"
            blob_storage.upload_dataframe(final_df, compressed_blob_path, format="parquet", compression="gzip")
            saved_resources["blob_storage"]["compressed_parquet"] = compressed_blob_path
        
        # Save local copies if needed
        if output_dir:
            sample_local_path = f"{output_dir}/final_df_sample.csv"
            sample_df.to_csv(sample_local_path, index=False)
            saved_resources["local"]["sample_csv"] = sample_local_path
            
            compressed_local_path = f"{output_dir}/final_df_compressed.parquet"
            final_df.to_parquet(compressed_local_path, compression="gzip", index=False)
            saved_resources["local"]["compressed_parquet"] = compressed_local_path
    
    # 4. Save run metadata
    metadata.update({
        "saved_resources": saved_resources,
        "runtime_seconds": time.time() - start_time
    })
    
    if blob_storage and blob_storage._container_client:
        metadata_blob_path = f"{dataset_name}/results/save_results_metadata.json"
        blob_storage.upload_json(metadata, metadata_blob_path)
        saved_resources["blob_storage"]["metadata"] = metadata_blob_path
        
        # Create a manifest file
        manifest = {
            "dataset": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "files": saved_resources["blob_storage"]
        }
        manifest_blob_path = f"{dataset_name}/results/manifest.json"
        blob_storage.upload_json(manifest, manifest_blob_path)
        saved_resources["blob_storage"]["manifest"] = manifest_blob_path
    
    # Save local metadata if needed
    if output_dir:
        metadata_local_path = f"{output_dir}/save_results_metadata.json"
        save_json(metadata, metadata_local_path)
        saved_resources["local"]["metadata"] = metadata_local_path
    
    # Create a model manifest file that includes all important artifacts
    model_manifest = {
        "dataset_name": dataset_name,
        "training_cycle": training_cycle,
        "timestamp": datetime.now().isoformat(),
        "artifacts": {
            "embedding_model": f"{base_blob_path}/embeddings",
            "clustering_model": f"{base_blob_path}/clustering/hdbscan_clusterer.pkl",
            "umap_reducer": f"{base_blob_path}/clustering/umap_reducer.pkl",
            "scaler": f"{base_blob_path}/clustering/scaler.pkl",
            "labeled_clusters": f"{base_blob_path}/analysis/labeled_clusters.json",
            "domains": f"{base_blob_path}/analysis/domains.json",
            "visualization_data": f"{base_blob_path}/results/visualization_data.json",
            "final_dataframe": f"{base_blob_path}/results/final_df_compressed.parquet"
        },
        "bigquery_table": table_id,
        "cluster_stats": {
            "num_clusters": labeled_clusters and len(labeled_clusters) - (1 if "-1" in labeled_clusters else 0),
            "num_domains": domains and len(domains.get("domains", [])) - 1
        }
    }
    
    # Save to both the training cycle folder and the dataset root for discovery
    if blob_storage and blob_storage._container_client:
        blob_storage.upload_json(model_manifest, f"{base_blob_path}/model_manifest.json")
        blob_storage.upload_json(model_manifest, f"{dataset_name}/latest_model_manifest.json")
        
        # Update training cycle info
        try:
            cycle_info = blob_storage.download_json(f"{base_blob_path}/training_info.json")
            if cycle_info:
                cycle_info["status"] = "completed" 
                cycle_info["completed_timestamp"] = datetime.now().isoformat()
                cycle_info["model_manifest_path"] = f"{base_blob_path}/model_manifest.json"
                cycle_info["bigquery_table"] = table_id
                
                blob_storage.upload_json(cycle_info, f"{base_blob_path}/training_info.json")
                blob_storage.upload_json(cycle_info, f"{dataset_name}/latest_training_cycle.json")
                logging.info(f"Updated latest_training_cycle.json to point to {training_cycle}")
        except Exception as e:
            logging.warning(f"Could not update training cycle info: {e}")
    
    logging.info(f"Results saving completed in {time.time() - start_time:.2f} seconds")
    
    return saved_resources
