"""
BigQuery utilities for the topic clustering package.
"""

import os
import json
import logging
import time
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from google.cloud import bigquery_storage

class BigQueryClient:
    """A class to handle BigQuery operations with proper retry logic"""
    
    def __init__(self, service_account_key_path=None):
        # Try to get from environment variable if not provided
        self.service_account_key_path = service_account_key_path or os.environ.get("SERVICE_ACCOUNT_KEY_PATH")
        
        if not self.service_account_key_path:
            logging.error("Service account key path not provided and not found in environment variables")
            self._client = None
            self._storage_client = None
        else:
            # Initialize client lazily
            self._client = None
            self._storage_client = None
    
    def _get_credentials(self):
        """Get credentials from service account key file or JSON string"""
        if os.path.isfile(self.service_account_key_path):
            return service_account.Credentials.from_service_account_file(self.service_account_key_path)
        else:
            json_key = json.loads(self.service_account_key_path)
            return service_account.Credentials.from_service_account_info(json_key)
    
    @property
    def client(self):
        """Get or create BigQuery client"""
        if self._client is None:
            credentials = self._get_credentials()
            self._client = bigquery.Client(credentials=credentials)
        return self._client
    
    @property
    def storage_client(self):
        """Get or create BigQuery storage client"""
        if self._storage_client is None:
            credentials = self._get_credentials()
            self._storage_client = bigquery_storage.BigQueryReadClient(credentials=credentials)
        return self._storage_client
    
    def run_query(self, query, max_retries=3):
        """Run a BigQuery query with retry logic"""
        retry_count = 0
        backoff_time = 2  # Start with 2 seconds backoff
        
        while retry_count <= max_retries:
            try:
                query_job = self.client.query(query)
                query_result = query_job.result()
                return query_result.to_dataframe(create_bqstorage_client=False)
            except Exception as e:
                retry_count += 1
                if retry_count > max_retries:
                    logging.error(f"Query failed after {max_retries} attempts: {e}")
                    return pd.DataFrame()
                    
                logging.warning(f"Query attempt {retry_count} failed: {e}. Retrying in {backoff_time}s")
                time.sleep(backoff_time)
                backoff_time *= 2  # Exponential backoff
    
    def save_dataframe(self, df, table_id, write_disposition="WRITE_APPEND", required_columns=None):
        """Save DataFrame to BigQuery table"""
        logging.info(f"Saving {len(df)} rows to {table_id}")
        
        # Filter columns if required_columns provided
        if required_columns:
            available_columns = [col for col in required_columns if col in df.columns]
            results_df = df[available_columns].copy()
        else:
            results_df = df
        
        sanitized_table_id = table_id.replace('`', '')
        write_disp = getattr(bigquery.WriteDisposition, write_disposition)
        job_config = bigquery.LoadJobConfig(write_disposition=write_disp)
        
        try:
            job = self.client.load_table_from_dataframe(
                results_df, 
                sanitized_table_id, 
                job_config=job_config,
                timeout=300  # 5 minutes timeout
            )
            job.result()  # Wait for job to complete
            logging.info(f"Successfully saved {len(results_df)} results to {table_id}")
            return True
        except Exception as e:
            logging.error(f"Error saving to BigQuery: {e}")
            return False

def get_bigquery_client():
    """Get a BigQuery client with service account from environment variables"""
    try:
        return BigQueryClient()
    except ValueError as e:
        logging.error(f"BigQuery client initialization failed: {e}")
        return None

def run_query(query, max_retries=3):
    """Convenience function to run a query using the default client"""
    client = get_bigquery_client()
    if client:
        return client.run_query(query, max_retries)
    return pd.DataFrame()

def save_results_to_bigquery(df, table_id, write_disposition="WRITE_APPEND"):
    """Save final results to BigQuery"""
    client = get_bigquery_client()
    if client:
        # Define standard columns to include when available
        required_columns = [
            'number', 'priority', 'sys_created_on', 'TeamDepartment', 'Area',
            'Unit', 'TechCenter', 'short_description', 'close_code', 'vendor',
            'assignment_group', 'state', 'business_service', 'close_notes',
            'work_notes', 'chanel', 'description', 'cluster', 'subcategory',
            'category'
        ]
        return client.save_dataframe(df, table_id, write_disposition, required_columns)
    return False

def save_embeddings_to_bigquery(df, table_id, write_disposition="WRITE_APPEND"):
    """Save embeddings to BigQuery for persistence"""
    client = get_bigquery_client()
    if client:
        # Only save essential columns for embeddings
        embedding_df = df[['number', 'combined_incidents_summary', 'embedding']].copy()
        return client.save_dataframe(embedding_df, table_id, write_disposition)
    return False
