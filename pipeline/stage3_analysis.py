"""
Third pipeline stage: Generate cluster info, label clusters and group into domains.
"""

import os
import time
import logging
import json
import uuid
import pandas as pd
import numpy as np
import io
from datetime import datetime

from ..config.settings import config
from ..models.clustering import generate_cluster_info
from ..utils.json_utils import save_json, robust_json_parser
from ..utils.azure_utils import RetryHandler, blob_storage

def read_parquet_optimized(file_path, columns=None, chunksize=None):
    """
    Read a Parquet file with memory optimization options
    
    Args:
        file_path: Path to the Parquet file
        columns: List of columns to load (None loads all)
        chunksize: Number of rows to read at once (None reads all at once)
        
    Returns:
        DataFrame with the loaded data
    """
    import pyarrow.parquet as pq
    import pyarrow as pa
    
    try:
        # First approach: Try with memory_map=True which may reduce memory usage
        try:
            if columns is not None:
                return pd.read_parquet(file_path, columns=columns, memory_map=True)
            else:
                return pd.read_parquet(file_path, memory_map=True)
        except Exception as e:
            logging.warning(f"Memory-mapped read failed: {e}, trying with chunking")
        
        # Second approach: Read in chunks if chunksize provided
        if chunksize is not None:
            table = pq.read_table(file_path, columns=columns)
            total_rows = table.num_rows
            chunks = []
            for i in range(0, total_rows, chunksize):
                end = min(i + chunksize, total_rows)
                chunk = table.slice(i, end - i).to_pandas()
                chunks.append(chunk)
            return pd.concat(chunks, ignore_index=True)
        
        # Third approach: Try with default options but limit columns
        if columns is not None:
            return pd.read_parquet(file_path, columns=columns)
        else:
            # Last resort - try to read with default settings
            return pd.read_parquet(file_path)
    
    except Exception as e:
        logging.error(f"Error reading Parquet file: {e}")
        # If we still have issues, try an even more memory efficient approach
        try:
            # Read only essential columns for clustering
            essential_cols = ['number', 'short_description', 'cluster', 'cluster_probability']
            if columns is not None:
                essential_cols = [col for col in essential_cols if col in columns]
            
            logging.info(f"Attempting to read only essential columns: {essential_cols}")
            return pd.read_parquet(file_path, columns=essential_cols)
        except Exception as e2:
            logging.error(f"Failed to read even with essential columns: {e2}")
            raise

def label_clusters_with_llm(clusters, max_samples=300, chunk_size=25, output_dir=None):
    """
    Use LLM to label HDBSCAN clusters with topics and descriptions
    
    Args:
        clusters: Dictionary of cluster information
        max_samples: Maximum number of samples to include for all clusters
        chunk_size: Number of clusters to process in each chunk
        output_dir: Directory to save labeled clusters
        
    Returns:
        Dictionary of labeled clusters
    """
    import openai
    from ..config.settings import config
    
    # Initialize dictionary for labeled clusters
    labeled_clusters = {}
    
    # Skip the noise cluster (-1) for labeling
    clusters_to_label = {k: v for k, v in clusters.items() if k != "-1"}
    
    # If no clusters to label, return early
    if not clusters_to_label:
        logging.warning("No clusters to label")
        labeled_clusters["-1"] = {"topic": "Noise", "description": "Unclustered data points"}
        return labeled_clusters
    
    # Initialize Azure OpenAI client
    openai_client = openai.AzureOpenAI(
        azure_endpoint=config.azure_openai_config["endpoint"],
        api_key=config.azure_openai_config["api_key"],
        api_version=config.azure_openai_config["api_version"]
    )
    
    # Process clusters in chunks to avoid token limits
    cluster_ids = list(clusters_to_label.keys())
    all_chunk_results = {}
    
    # First pass: Label each chunk independently
    for i in range(0, len(cluster_ids), chunk_size):
        chunk = cluster_ids[i:i+chunk_size]
        logging.info(f"Labeling clusters {i} to {i+len(chunk)-1}")
        correlation_id = str(uuid.uuid4())
        
        # Create prompt for this chunk with consistency guidance
        prompt_text = "You are an IT topic labeling expert. Please analyze these incident clusters and provide a concise topic name and description for each.\n\n"
        prompt_text += "IMPORTANT GUIDELINES for topic naming:\n"
        prompt_text += "1. Be consistent in terminology - use the same term for the same concept\n"
        prompt_text += "2. Use specific system names (like 'SAP', 'Synapps', 'Azure') when clearly identified\n"
        prompt_text += "3. Use format [System/Area] + [Issue Type] - e.g. 'SAP Login Issues', 'VPN Connection Failures'\n"
        prompt_text += "4. Be concise but descriptive\n\n"
        
        for cluster_id in chunk:
            cluster = clusters[cluster_id]
            samples = cluster["samples"][:5]  # Limit to 5 samples per cluster
            sample_text = "\n".join([f"- {s}" for s in samples])
            
            prompt_text += f"\nCLUSTER {cluster_id} (Size: {cluster['size']} incidents, {cluster['percentage']}%):\n{sample_text}\n"
        
        # Enhanced JSON structure specification
        prompt_text += "\nYOU MUST RESPOND WITH VALID JSON using this format:\n"
        prompt_text += "{\n"
        prompt_text += '  "cluster_id": {"topic": "Brief topic name", "description": "Concise description of what unifies these incidents"},\n'
        prompt_text += '  "another_cluster_id": {"topic": "Another topic name", "description": "Another description"}\n'
        prompt_text += "}\n\n"
        prompt_text += "Do not include any text outside the JSON structure."
        
        def _label_cluster_chunk():
            # Make API call to Azure OpenAI with detailed settings for reliable JSON
            response = openai_client.chat.completions.create(
                model=config.azure_openai_config["chat_deployment_name"],
                messages=[
                    {"role": "system", "content": "You are a JSON-generating IT incident classifier. Always return syntactically valid JSON with no explanations outside the JSON."},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.1,  # Reduced temperature for more reliable structured output
                response_format={"type": "json_object"},
                timeout=60,  # Add timeout for Azure OpenAI API call
                user=correlation_id  # For Azure request tracking
            )
            
            # Handle the response with our robust parser for maximum reliability
            response_text = response.choices[0].message.content
            
            # First check if response appears to be JSON (starts with { and ends with })
            if not (response_text.strip().startswith('{') and response_text.strip().endswith('}')):
                raise ValueError("Response is not in JSON format")
            
            result = robust_json_parser(response_text)
            
            if not result:
                raise ValueError("Empty result after parsing")
                
            # Validate result has expected structure for at least one cluster
            valid_response = False
            for k, v in result.items():
                if isinstance(v, dict) and "topic" in v and "description" in v:
                    valid_response = True
                    break
            
            if not valid_response:
                raise ValueError("Response doesn't match expected cluster label format")
            
            return result
        
        try:
            chunk_result = RetryHandler.with_retry(
                _label_cluster_chunk,
                max_retries=3,
                logging_prefix=f"Cluster labeling (chunks {i}-{i+len(chunk)-1})"
            )
            all_chunk_results.update(chunk_result)
        except Exception as e:
            logging.error(f"Error labeling clusters {chunk}: {e}")
            # Add fallback labels for this chunk
            for cluster_id in chunk:
                all_chunk_results[cluster_id] = {"topic": f"Cluster {cluster_id}", "description": "Auto-generated label"}
    
    # Second pass: Standardize labels across all chunks
    if all_chunk_results:
        try:
            logging.info("Standardizing labels across all chunks")
            
            # Create a standardization prompt
            standardize_prompt = "Review and standardize these IT incident cluster labels to ensure consistency across the entire dataset.\n\n"
            standardize_prompt += "RULES FOR STANDARDIZATION:\n"
            standardize_prompt += "1. If multiple clusters clearly refer to the same issue type, use identical terminology\n"
            standardize_prompt += "2. Use format [System/Area] + [Issue Type] consistently\n"
            standardize_prompt += "3. Maintain specificity - don't overgeneralize\n\n"
            standardize_prompt += "CURRENT LABELS:\n"
            
            # Add all current labels to the prompt
            for cluster_id, info in all_chunk_results.items():
                description_excerpt = info['description'][:100] + "..." if len(info['description']) > 100 else info['description']
                standardize_prompt += f"Cluster {cluster_id}: \"{info['topic']}\" - {description_excerpt}\n"
                
            standardize_prompt += "\nRESPOND WITH STANDARDIZED LABELS IN JSON FORMAT:\n"
            standardize_prompt += "{\n"
            standardize_prompt += '  "cluster_id": {"topic": "Standardized topic name", "description": "Original or improved description"}'
            standardize_prompt += "\n  ...\n}"
            
            def _standardize_labels():
                response = openai_client.chat.completions.create(
                    model=config.azure_openai_config["chat_deployment_name"],
                    messages=[
                        {"role": "system", "content": "You are a standardization expert for IT incident classifications. Your responses must be valid JSON."},
                        {"role": "user", "content": standardize_prompt}
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"},
                    timeout=120  # Increase timeout for large responses
                )
                
                # Parse standardized labels with robust parser
                response_text = response.choices[0].message.content
                standardized_result = robust_json_parser(response_text)
                
                if not standardized_result:
                    raise ValueError("Could not parse standardized labels")
                
                return standardized_result
            
            try:
                labeled_clusters = RetryHandler.with_retry(
                    _standardize_labels,
                    max_retries=3,
                    logging_prefix="Label standardization"
                )
            except Exception as e:
                logging.error(f"Error standardizing labels: {e}. Using original labels.")
                labeled_clusters = all_chunk_results
        
        except Exception as e:
            logging.error(f"Error in standardization process: {e}. Using original labels.")
            labeled_clusters = all_chunk_results
    else:
        labeled_clusters = all_chunk_results
    
    # Add noise cluster label
    labeled_clusters["-1"] = {"topic": "Noise", "description": "Unclustered data points"}

    # Save labeled clusters
    if output_dir:
        with open(f"{output_dir}/labeled_clusters.json", "w") as f:
            json.dump(labeled_clusters, f, indent=2)
        
    return labeled_clusters

def group_clusters_into_domains(labeled_clusters, clusters, num_domains=15, output_dir=None, standardize_labels=True):
    """
    Group labeled clusters into higher-level domains
    
    Args:
        labeled_clusters: Dictionary of labeled clusters
        clusters: Dictionary of cluster information
        num_domains: Target number of domains
        output_dir: Directory to save domains information
        standardize_labels: Whether to standardize labels during domain creation
        
    Returns:
        Dictionary with domains
    """
    import openai
    from ..config.settings import config
    
    # Skip the noise cluster for domain grouping
    clusters_to_group = {k: v for k, v in labeled_clusters.items() if k != "-1"}
    
    # If very few clusters, don't bother with domains
    if len(clusters_to_group) <= 3:
        domains = {
            "domains": [
                {
                    "domain_name": "All Incidents", 
                    "description": "All incident clusters", 
                    "clusters": [int(k) for k in clusters_to_group.keys()]
                }
            ]
        }
        domains["domains"].append({"domain_name": "Noise", "description": "Uncategorized incidents", "clusters": [-1]})
        
        if output_dir:
            with open(f"{output_dir}/domains.json", "w") as f:
                json.dump(domains, f, indent=2)
                
        return domains
    
    # Initialize OpenAI client
    openai_client = openai.AzureOpenAI(
        azure_endpoint=config.azure_openai_config["endpoint"],
        api_key=config.azure_openai_config["api_key"],
        api_version=config.azure_openai_config["api_version"]
    )
    
    # Create prompt for GPT to group clusters
    prompt_text = f"""Group the following IT incident clusters into {num_domains} meaningful domains or categories:

CLUSTERS:
"""
    
    # Add each cluster's info to the prompt
    for cluster_id, info in clusters_to_group.items():
        size = clusters[cluster_id]["size"]
        percentage = clusters[cluster_id]["percentage"]
        prompt_text += f"\nCluster {cluster_id} (Size: {size}, {percentage}%):\n"
        prompt_text += f"Topic: {info['topic']}\n"
        prompt_text += f"Description: {info['description']}\n"
    
    correlation_id = str(uuid.uuid4())
    
    # Specify exact JSON schema to enforce structure
    if standardize_labels:
        expected_keys = ["domains", "standardized_labels"]
        prompt_text += f"""
Group these clusters into at most {num_domains} logical domains. Ignore the noise cluster (-1).
Each domain should have a meaningful name and brief description.

IMPORTANT INSTRUCTIONS:
1. Find clusters with very similar topics and standardize their labels
2. For example, if multiple clusters are about "password resets for Synapps" but have slightly different labels,
   they should all be grouped in the same domain AND their labels should be standardized.
3. You must maintain the original domain structure with domain_name, description and clusters.
4. Standardized labels should be applied consistently within each domain.

YOU MUST RETURN A VALID JSON OBJECT with this exact structure:
{{
  "domains": [
    {{
      "domain_name": "Domain 1 Name",
      "description": "Brief description of domain",
      "clusters": [cluster_id_1, cluster_id_2, ...]
    }},
    ...
  ],
  "standardized_labels": {{
    "cluster_id": "Standardized topic label",
    ...
  }}
}}

Do not include any explanation or text outside of the JSON structure.
"""
    else:
        expected_keys = ["domains"]
        # Original prompt without standardization
        prompt_text += f"""
Group these clusters into at most {num_domains} logical domains. Ignore the noise cluster (-1).
Each domain should have a meaningful name and brief description.

YOU MUST RETURN A VALID JSON OBJECT with this exact structure:
{{
  "domains": [
    {{
      "domain_name": "Domain 1 Name",
      "description": "Brief description of domain",
      "clusters": [cluster_id_1, cluster_id_2, ...]
    }},
    ...
  ]
}}

Do not include any explanation or text outside of the JSON structure.
"""
    
    def _create_domains():
        response = openai_client.chat.completions.create(
            model=config.azure_openai_config["chat_deployment_name"],
            messages=[
                {"role": "system", "content": "You are a JSON-generating IT domain classifier. Always respond with properly structured JSON and no additional text."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.1,  # Lower temperature for more consistent JSON
            response_format={"type": "json_object"},  # Enforce JSON format
            timeout=60,  # Add timeout for API call
            user=correlation_id  # For Azure telemetry
        )
        
        # Parse response with validation
        response_text = response.choices[0].message.content
        
        # Parse and validate with our robust parser
        result = robust_json_parser(response_text)
        
        # Verify expected structure exists
        if not all(key in result for key in expected_keys):
            missing_keys = [key for key in expected_keys if key not in result]
            raise ValueError(f"Missing required keys in response: {missing_keys}")
        
        return result
    
    try:
        result = RetryHandler.with_retry(
            _create_domains,
            max_retries=3,
            logging_prefix="Domain grouping"
        )
        
        # Extract domains from the response
        domains = {"domains": result["domains"]}
        
        # Apply standardized labels to the labeled_clusters if provided
        if standardize_labels and "standardized_labels" in result:
            standardized_labels = result["standardized_labels"]
            
            # Update the topics in labeled_clusters with standardized versions
            for cluster_id, new_label in standardized_labels.items():
                if cluster_id in labeled_clusters:
                    labeled_clusters[cluster_id]["topic"] = new_label
                    
            # Save the updated labeled clusters
            if output_dir:
                with open(f"{output_dir}/labeled_clusters_standardized.json", "w") as f:
                    json.dump(labeled_clusters, f, indent=2)
        
        # Convert cluster IDs from strings to integers
        for domain in domains["domains"]:
            fixed_clusters = []
            for cluster_id in domain["clusters"]:
                try:
                    # Try direct conversion first
                    fixed_clusters.append(int(cluster_id))
                except (ValueError, TypeError):
                    # If that fails, try to extract the numeric part
                    if isinstance(cluster_id, str) and "cluster_" in cluster_id.lower():
                        try:
                            numeric_part = cluster_id.lower().replace("cluster_", "")
                            fixed_clusters.append(int(numeric_part))
                        except ValueError:
                            logging.warning(f"Could not convert {cluster_id} to integer, skipping")
                    else:
                        logging.warning(f"Unrecognized cluster ID format: {cluster_id}, skipping")
        
            domain["clusters"] = fixed_clusters
        
        # Add noise domain
        noise_domain = {
            "domain_name": "Noise", 
            "description": "Uncategorized incidents", 
            "clusters": [-1]
        }
        domains["domains"].append(noise_domain)
        
    except Exception as e:
        logging.error(f"Error grouping clusters into domains: {e}")
        # Fallback domains
        domains = {
            "domains": [
                {
                    "domain_name": "All Incidents", 
                    "description": "All incident clusters", 
                    "clusters": [int(k) for k in clusters_to_group.keys()]
                },
                {
                    "domain_name": "Noise", 
                    "description": "Uncategorized incidents", 
                    "clusters": [-1]
                }
            ]
        }
    
    # Save domains
    if output_dir:
        with open(f"{output_dir}/domains.json", "w") as f:
            json.dump(domains, f, indent=2)
            
    return domains

def apply_labels_to_data(df, labeled_clusters, domains):
    """
    Apply cluster labels and domains to the data
    
    Args:
        df: DataFrame with cluster assignments
        labeled_clusters: Dictionary of cluster labels
        domains: Dictionary of domains
        
    Returns:
        DataFrame with labels and domains applied
    """
    result_df = df.copy()
    
    # Create mapping dictionaries for efficient lookups
    topic_mapping = {int(k): v["topic"] for k, v in labeled_clusters.items() if k != "-1"}
    topic_mapping[-1] = "Noise"  # Add mapping for noise cluster
    
    # Create domain mapping
    domain_mapping = {}
    for domain in domains.get("domains", []):
        domain_name = domain["domain_name"]
        for cluster_id in domain.get("clusters", []):
            domain_mapping[int(cluster_id)] = domain_name
    
    # Apply mappings
    result_df["subcategory"] = result_df["cluster"].map(topic_mapping).fillna("Unknown")
    result_df["category"] = result_df["cluster"].map(domain_mapping).fillna("Other")
    
    return result_df

def analyze_clusters_pipeline(clustered_df=None, dataset_name=None, num_domains=15, result_path=None):
    """
    Generate cluster info, label clusters and group into domains
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
    analysis_blob_path = f"{base_blob_path}/analysis"
    clustering_blob_path = f"{base_blob_path}/clustering"
    
    # Check if blob storage is available, create local dirs as fallback if not
    if not blob_storage or not blob_storage._container_client:
        logging.warning("Azure Blob Storage not available. Using local storage as fallback.")
        result_path = result_path or config.result_path
        output_dir = f"{result_path}/{dataset_name}/analysis"
        os.makedirs(output_dir, exist_ok=True)
    else:
        # Use None to skip local disk operations
        output_dir = None
    
    start_time = time.time()
    
    # Check if we need to load data from file
    if clustered_df is None:
        if dataset_name is None:
            raise ValueError("Either clustered_df or dataset_name must be provided")
        
        # Try to load from blob storage first with updated path
        cluster_blob_path = f"{base_blob_path}/clustering/clustered_df.parquet"
        if blob_storage and blob_storage._container_client:
            logging.info(f"Loading clustered data from blob storage: {cluster_blob_path}")
            try:
                clustered_df = blob_storage.download_dataframe(cluster_blob_path, format="parquet")
            except Exception as e:
                logging.error(f"Error loading clustered data from blob storage: {e}")
                clustered_df = None
        
        # Fall back to local file if needed
        if clustered_df is None and result_path:
            cluster_path = f"{result_path}/{dataset_name}/clustering/clustered_df.parquet"
            if os.path.exists(cluster_path):
                logging.info(f"Loading clustered data from local path: {cluster_path}")
                try:
                    essential_cols = ['number', 'short_description', 'cluster', 'cluster_probability', 
                                    'combined_incidents_summary', 'business_service']
                    clustered_df = read_parquet_optimized(
                        cluster_path, 
                        columns=essential_cols,
                        chunksize=10000
                    )
                except Exception as e:
                    logging.error(f"Error loading clustered data from local file: {e}")
                    raise
            else:
                raise FileNotFoundError(f"Clustered data not found at {cluster_path}")
        
        if clustered_df is None:
            raise FileNotFoundError(f"Clustered data not found for dataset: {dataset_name}")
    
    logging.info(f"Analyzing clusters for dataset '{dataset_name}'")
    
    # Generate cluster information
    clusters_info = generate_cluster_info(
        clustered_df, 
        text_column='short_description', 
        cluster_column='cluster',
        sample_size=5,
        dataset_name=dataset_name,
        output_dir=output_dir
    )
    
    # Label clusters using LLM
    try:
        labeled_clusters = label_clusters_with_llm(
            clusters_info,
            output_dir=output_dir
        )
        
        # Save labeled clusters to blob storage
        if dataset_name and blob_storage and blob_storage._container_client:
            blob_storage.upload_json(labeled_clusters, f"{analysis_blob_path}/labeled_clusters.json")
            
    except Exception as e:
        logging.error(f"Error in cluster labeling: {e}")
        logging.info("Creating basic labels as fallback")
        # Create basic labels if LLM fails
        labeled_clusters = {
            str(cid): {"topic": f"Cluster {cid}", "description": "Auto-generated label"}
            for cid in clustered_df['cluster'].unique() if cid != -1
        }
        labeled_clusters["-1"] = {"topic": "Noise", "description": "Unclustered data points"}
        
        # Save fallback labels
        if dataset_name and blob_storage and blob_storage._container_client:
            blob_storage.upload_json(labeled_clusters, f"{analysis_blob_path}/labeled_clusters_fallback.json")
        elif output_dir:
            save_json(labeled_clusters, f"{output_dir}/labeled_clusters_fallback.json")
    
    # Group clusters into domains
    try:
        domains = group_clusters_into_domains(
            labeled_clusters, 
            clusters_info, 
            num_domains=num_domains,
            output_dir=output_dir
        )
        
        # Save domains to blob storage
        if dataset_name and blob_storage and blob_storage._container_client:
            blob_storage.upload_json(domains, f"{analysis_blob_path}/domains.json")
            
    except Exception as e:
        logging.error(f"Error in domain grouping: {e}")
        logging.info("Creating basic domains as fallback")
        # Create basic domains if grouping fails
        domains = {"domains": [{
            "domain_name": "All Incidents", 
            "description": "All incident clusters",
            "clusters": [int(cid) for cid in clustered_df['cluster'].unique() if cid != -1]
        }]}
        domains["domains"].append({
            "domain_name": "Noise", 
            "description": "Uncategorized incidents", 
            "clusters": [-1]
        })
        
        # Save fallback domains
        if dataset_name and blob_storage and blob_storage._container_client:
            blob_storage.upload_json(domains, f"{analysis_blob_path}/domains_fallback.json")
        elif output_dir:
            save_json(domains, f"{output_dir}/domains_fallback.json")
    
    # Apply labels to data
    final_df = apply_labels_to_data(clustered_df, labeled_clusters, domains)
    
    # Save final labeled dataframe - for blob storage, handle large datasets with chunking in memory
    if blob_storage and blob_storage._container_client:
        if len(final_df) > 100000:  # If the dataframe is very large
            logging.info(f"Large dataset detected ({len(final_df)} rows). Uploading in chunks...")
            chunk_size = 50000
            
            # Save manifest info
            manifest = {
                "num_parts": (len(final_df) + chunk_size - 1) // chunk_size,
                "total_rows": len(final_df),
                "chunk_size": chunk_size
            }
            
            # Create chunk patterns
            chunk_paths = []
            
            # Upload each chunk
            for i in range(0, len(final_df), chunk_size):
                chunk = final_df.iloc[i:i+chunk_size]
                chunk_path = f"{analysis_blob_path}/final_df_part_{i//chunk_size}.parquet"
                if blob_storage.upload_dataframe(chunk, chunk_path, format="parquet"):
                    chunk_paths.append(chunk_path)
            
            # Update manifest with successful uploads
            manifest["file_paths"] = chunk_paths
            blob_storage.upload_json(manifest, f"{analysis_blob_path}/final_df_manifest.json")
            
            # Upload a sample for quick access
            sample_df = final_df.sample(min(1000, len(final_df)))
            blob_storage.upload_dataframe(sample_df, f"{analysis_blob_path}/final_df_sample.parquet")
            
        else:
            # Upload as a single file
            blob_storage.upload_dataframe(final_df, f"{analysis_blob_path}/final_df.parquet")
    
    # Save to local disk as fallback
    elif output_dir:
        if len(final_df) > 100000:  # If the dataframe is very large
            logging.info(f"Large dataset detected ({len(final_df)} rows). Saving in chunks...")
            chunk_size = 50000
            for i in range(0, len(final_df), chunk_size):
                chunk = final_df.iloc[i:i+chunk_size]
                if i == 0:
                    # First chunk - create new file
                    chunk.to_parquet(f"{output_dir}/final_df.parquet", index=False)
                else:
                    # Append subsequent chunks
                    chunk.to_parquet(f"{output_dir}/final_df_part_{i//chunk_size}.parquet", index=False)
            
            # Save a manifest file with all parts
            manifest = {
                "num_parts": (len(final_df) + chunk_size - 1) // chunk_size,
                "total_rows": len(final_df),
                "chunk_size": chunk_size,
                "file_pattern": "final_df_part_{i}.parquet"
            }
            save_json(manifest, f"{output_dir}/final_df_manifest.json")
        else:
            # Save as a single file
            final_df.to_parquet(f"{output_dir}/final_df.parquet", index=False)
    
    # Save analysis metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_name,
        "num_clusters": len(clusters_info) - (1 if "-1" in clusters_info else 0),
        "num_domains": len(domains["domains"]) - (1 if any(d["domain_name"] == "Noise" for d in domains["domains"]) else 0),
        "total_records": len(final_df),
        "noise_percentage": clusters_info.get("-1", {}).get("percentage", 0) if "-1" in clusters_info else 0,
        "processing_time": time.time() - start_time
    }
    
    # Save metadata
    if blob_storage and blob_storage._container_client:
        blob_storage.upload_json(metadata, f"{analysis_blob_path}/analysis_metadata.json")
    elif output_dir:
        save_json(metadata, f"{output_dir}/analysis_metadata.json")
    
    # Update the training cycle info with analysis details
    if blob_storage and blob_storage._container_client:
        try:
            cycle_info = blob_storage.download_json(f"{dataset_name}/latest_training_cycle.json")
            if cycle_info:
                cycle_info["analysis_completed"] = True
                cycle_info["analysis_timestamp"] = datetime.now().isoformat()
                cycle_info["labeled_clusters_path"] = f"{analysis_blob_path}/labeled_clusters.json"
                cycle_info["domains_path"] = f"{analysis_blob_path}/domains.json"
                blob_storage.upload_json(cycle_info, f"{dataset_name}/latest_training_cycle.json")
        except Exception as e:
            logging.warning(f"Could not update training cycle info: {e}")
    
    logging.info(f"Cluster analysis completed in {time.time() - start_time:.2f} seconds")
    logging.info(f"Found {metadata['num_clusters']} clusters grouped into {metadata['num_domains']} domains")
    
    return final_df, clusters_info, labeled_clusters, domains
