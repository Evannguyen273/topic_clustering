"""
Utility script to migrate local results to Azure Blob Storage.
"""

import os
import logging
import argparse
import json
import glob
from tqdm import tqdm
import pandas as pd

from topic_clustering.utils.azure_utils import AzureBlobStorage
from topic_clustering.utils.logging_utils import setup_logging

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Migrate local files to Azure Blob Storage")
    parser.add_argument("--dataset", required=True, help="Dataset name to migrate")
    parser.add_argument("--local-dir", required=True, help="Local directory containing results")
    parser.add_argument("--blob-connection", help="Azure Blob connection string (if not in environment)")
    parser.add_argument("--container-name", default="prediction-artifact", help="Azure Blob container name")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()

def migrate_dataset(dataset_name, local_dir, blob_storage):
    """Migrate a dataset from local storage to Azure Blob Storage"""
    # Map of file extensions to content types
    content_types = {
        ".json": "application/json",
        ".parquet": "application/octet-stream",
        ".csv": "text/csv",
        ".pkl": "application/octet-stream",
        ".md": "text/markdown",
        ".npy": "application/octet-stream",
        ".txt": "text/plain",
        ".log": "text/plain"
    }
    
    # Find all files in the dataset directory
    dataset_dir = os.path.join(local_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        logging.error(f"Dataset directory not found: {dataset_dir}")
        return False
    
    # Get all files to migrate
    files_to_migrate = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            local_path = os.path.join(root, file)
            rel_path = os.path.relpath(local_path, local_dir)
            blob_path = rel_path.replace("\\", "/")
            
            # Determine content type based on file extension
            _, ext = os.path.splitext(file)
            content_type = content_types.get(ext.lower(), "application/octet-stream")
            
            files_to_migrate.append((local_path, blob_path, content_type))
    
    logging.info(f"Found {len(files_to_migrate)} files to migrate for dataset '{dataset_name}'")
    
    # Migrate files
    successful = 0
    failed = 0
    
    with tqdm(total=len(files_to_migrate), desc="Migrating files") as pbar:
        for local_path, blob_path, content_type in files_to_migrate:
            try:
                # Special handling for parquet files
                if local_path.endswith(".parquet"):
                    try:
                        df = pd.read_parquet(local_path)
                        if blob_storage.upload_dataframe(df, blob_path, format="parquet"):
                            successful += 1
                        else:
                            failed += 1
                    except Exception as e:
                        logging.error(f"Error reading or uploading parquet file {local_path}: {e}")
                        if blob_storage.upload_blob(local_path, blob_path, content_type=content_type):
                            successful += 1
                        else:
                            failed += 1
                # Special handling for JSON files
                elif local_path.endswith(".json"):
                    try:
                        with open(local_path, "r") as f:
                            data = json.load(f)
                        if blob_storage.upload_json(data, blob_path):
                            successful += 1
                        else:
                            failed += 1
                    except Exception as e:
                        logging.error(f"Error reading or uploading JSON file {local_path}: {e}")
                        if blob_storage.upload_blob(local_path, blob_path, content_type=content_type):
                            successful += 1
                        else:
                            failed += 1
                # Default file upload
                else:
                    if blob_storage.upload_blob(local_path, blob_path, content_type=content_type):
                        successful += 1
                    else:
                        failed += 1
            except Exception as e:
                logging.error(f"Error migrating file {local_path}: {e}")
                failed += 1
                
            pbar.update(1)
    
    logging.info(f"Migration completed: {successful} successful, {failed} failed")
    return successful > 0 and failed == 0

def main():
    """Main function"""
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(log_level=log_level)
    
    # Initialize Azure Blob Storage
    blob_connection = args.blob_connection
    blob_storage = AzureBlobStorage(
        connection_string=blob_connection,
        container_name=args.container_name
    )
    
    if not blob_storage._container_client:
        logger.error("Failed to initialize Azure Blob Storage. Check your connection string.")
        return 1
    
    # Migrate dataset
    success = migrate_dataset(args.dataset, args.local_dir, blob_storage)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
