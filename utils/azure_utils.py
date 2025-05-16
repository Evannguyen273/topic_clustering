"""
Azure utilities for the topic clustering package.
"""

import time
import uuid
import logging
import io
import json
import asyncio
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, Awaitable
import pandas as pd
import numpy as np
from azure.storage.blob import BlobServiceClient, ContainerClient, ContentSettings
from azure.storage.blob.aio import BlobServiceClient as AsyncBlobServiceClient

# Import config once module is fully loaded
import sys
if 'topic_clustering.config.settings' not in sys.modules:
    from ..config.settings import config

T = TypeVar('T')

class RetryHandler:
    """Unified class for handling retries with exponential backoff"""
    
    @staticmethod
    def with_retry(func: Callable[[], T], max_retries: int = 3, base_delay: int = 2, 
                  retry_on_exceptions: Tuple[type[Exception], ...] = (Exception,), 
                  logging_prefix: str = "Operation") -> T:
        """
        Execute a function with retry logic and exponential backoff
        
        Args:
            func: Function to execute
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff
            retry_on_exceptions: Tuple of exceptions that should trigger retry
            logging_prefix: Prefix for log messages
            
        Returns:
            Result of the function
        """
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                return func()
                
            except retry_on_exceptions as e:
                retry_count += 1
                
                if retry_count > max_retries:
                    logging.error(f"{logging_prefix} failed after {max_retries} attempts: {e}")
                    raise
                
                wait_time = min(60, base_delay ** retry_count)  # Cap at 60 seconds
                logging.warning(f"{logging_prefix} attempt {retry_count} failed: {e}. Retrying in {wait_time}s")
                time.sleep(wait_time)
    
    @staticmethod
    async def with_async_retry(func: Callable[[], Awaitable[T]], max_retries: int = 3, 
                              base_delay: int = 2, 
                              retry_on_exceptions: Tuple[type[Exception], ...] = (Exception,), 
                              logging_prefix: str = "Operation") -> T:
        """
        Execute an async function with retry logic and exponential backoff
        
        Args:
            func: Async function to execute
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff
            retry_on_exceptions: Tuple of exceptions that should trigger retry
            logging_prefix: Prefix for log messages
            
        Returns:
            Result of the function
        """
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                return await func()
                
            except retry_on_exceptions as e:
                retry_count += 1
                
                if retry_count > max_retries:
                    logging.error(f"{logging_prefix} failed after {max_retries} attempts: {e}")
                    raise
                
                wait_time = min(60, base_delay ** retry_count)  # Cap at 60 seconds
                logging.warning(f"{logging_prefix} attempt {retry_count} failed: {e}. Retrying in {wait_time}s")
                await asyncio.sleep(wait_time)

class AzureBlobStorage:
    """Class to handle Azure Blob Storage operations"""
    
    def __init__(self, connection_string: Optional[str] = None, container_name: Optional[str] = None):
        from ..config.settings import config
        self.connection_string = connection_string or config.blob_connection_string
        self.container_name = container_name or config.container_name
        
        if not self.connection_string:
            logging.error("Blob connection string not provided")
            self._container_client = None
        else:
            try:
                self._container_client = ContainerClient.from_container_url(self.connection_string)
                logging.info(f"Connected to blob storage container")
            except Exception as e:
                logging.error(f"Failed to connect to blob storage: {e}")
                self._container_client = None
    
    @property
    def container_client(self) -> ContainerClient:
        """Get container client, creating it if not already created"""
        if not self._container_client:
            raise ValueError("Container client is not initialized")
        return self._container_client
    
    def upload_blob(self, file_path_or_data, blob_path, content_type="application/octet-stream", 
                    max_retries=3, is_file_path=True):
        """
        Upload a file or data to blob storage with retry logic
        
        Args:
            file_path_or_data: Path to file on disk OR the actual data to upload
            blob_path: Path/name for the blob in storage
            content_type: MIME content type
            max_retries: Number of retry attempts
            is_file_path: If True, first arg is a file path; if False, it's the actual data
        
        Returns:
            Boolean indicating success
        """
        if not self._container_client:
            logging.error("Container client not initialized. Cannot upload blob.")
            return False
        
        def _upload():
            blob_client = self.container_client.get_blob_client(blob_path)
            content_settings = ContentSettings(content_type=content_type)
            
            if is_file_path:
                with open(file_path_or_data, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True, content_settings=content_settings)
            else:
                # Assume file_path_or_data is the actual data
                blob_client.upload_blob(file_path_or_data, overwrite=True, content_settings=content_settings)
            return True
        
        try:
            result = RetryHandler.with_retry(
                _upload,
                max_retries=max_retries,
                logging_prefix=f"Blob upload ({blob_path})"
            )
            if result:
                logging.info(f"Successfully uploaded blob to {blob_path}")
            return result
        except Exception as e:
            logging.error(f"Failed to upload blob {blob_path}: {e}")
            return False
    
    def upload_json(self, data, blob_path, max_retries=3):
        """Upload JSON data to blob storage"""
        if not self._container_client:
            logging.error("Container client not initialized. Cannot upload JSON blob.")
            return False
        
        from ..utils.json_utils import NumpyEncoder
        
        def _upload():
            blob_client = self.container_client.get_blob_client(blob_path)
            content_settings = ContentSettings(content_type="application/json")
            serialized_data = json.dumps(data, cls=NumpyEncoder)
            blob_client.upload_blob(serialized_data, overwrite=True, content_settings=content_settings)
            return True
        
        try:
            result = RetryHandler.with_retry(
                _upload,
                max_retries=max_retries,
                logging_prefix=f"JSON upload ({blob_path})"
            )
            if result:
                logging.info(f"Successfully uploaded JSON to {blob_path}")
            return result
        except Exception as e:
            logging.error(f"Failed to upload JSON {blob_path}: {e}")
            return False
    
    def upload_dataframe(self, df, blob_path, format="parquet", max_retries=3, **kwargs):
        """
        Upload a pandas DataFrame to blob storage
        
        Args:
            df: DataFrame to upload
            blob_path: Path/name for the blob in storage
            format: Format to save ('parquet', 'csv', etc.)
            max_retries: Number of retry attempts
            **kwargs: Additional arguments to pass to DataFrame.to_* method
            
        Returns:
            Boolean indicating success
        """
        if not self._container_client:
            logging.error("Container client not initialized. Cannot upload DataFrame.")
            return False
            
        def _upload():
            # Create in-memory buffer
            buffer = io.BytesIO()
            
            # Save DataFrame to buffer in specified format
            if format.lower() == "parquet":
                df.to_parquet(buffer, **kwargs)
                content_type = "application/octet-stream"
            elif format.lower() == "csv":
                df.to_csv(buffer, index=False, **kwargs)
                content_type = "text/csv"
            elif format.lower() == "excel":
                df.to_excel(buffer, **kwargs)
                content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            # Reset buffer position to beginning
            buffer.seek(0)
            
            # Upload buffer content
            blob_client = self.container_client.get_blob_client(blob_path)
            content_settings = ContentSettings(content_type=content_type)
            blob_client.upload_blob(buffer.getvalue(), overwrite=True, content_settings=content_settings)
            return True
            
        try:
            result = RetryHandler.with_retry(
                _upload, 
                max_retries=max_retries,
                logging_prefix=f"DataFrame upload ({blob_path})"
            )
            if result:
                logging.info(f"Successfully uploaded DataFrame to {blob_path} in {format} format")
            return result
        except Exception as e:
            logging.error(f"Failed to upload DataFrame to {blob_path}: {e}")
            return False
    
    async def upload_dataframe_async(self, df: pd.DataFrame, blob_path: str, 
                                    format: str = "parquet", max_retries: int = 3, 
                                    **kwargs) -> bool:
        """
        Upload a pandas DataFrame to blob storage asynchronously
        
        Args:
            df: DataFrame to upload
            blob_path: Path/name for the blob in storage
            format: Format to save ('parquet', 'csv', etc.)
            max_retries: Number of retry attempts
            **kwargs: Additional arguments to pass to DataFrame.to_* method
            
        Returns:
            Boolean indicating success
        """
        if not self._container_client:
            logging.error("Container client not initialized. Cannot upload DataFrame.")
            return False
            
        async def _upload():
            # Create in-memory buffer
            buffer = io.BytesIO()
            
            # Save DataFrame to buffer in specified format
            if format.lower() == "parquet":
                df.to_parquet(buffer, **kwargs)
                content_type = "application/octet-stream"
            elif format.lower() == "csv":
                df.to_csv(buffer, index=False, **kwargs)
                content_type = "text/csv"
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            # Reset buffer position to beginning
            buffer.seek(0)
            
            # Set up async blob client
            async_service_client = AsyncBlobServiceClient.from_connection_string(self.connection_string)
            async_container_client = async_service_client.get_container_client(self.container_name)
            async_blob_client = async_container_client.get_blob_client(blob_path)
            
            content_settings = ContentSettings(content_type=content_type)
            await async_blob_client.upload_blob(buffer.getvalue(), overwrite=True, content_settings=content_settings)
            
            # Close clients
            await async_blob_client.close()
            await async_container_client.close()
            await async_service_client.close()
            
            return True
            
        try:
            result = await RetryHandler.with_async_retry(
                _upload, 
                max_retries=max_retries,
                logging_prefix=f"DataFrame async upload ({blob_path})"
            )
            if result:
                logging.info(f"Successfully uploaded DataFrame asynchronously to {blob_path} in {format} format")
            return result
        except Exception as e:
            logging.error(f"Failed to upload DataFrame asynchronously to {blob_path}: {e}")
            return False
    
    def upload_pickle(self, obj, blob_path, max_retries=3):
        """Upload a pickled object to blob storage"""
        if not self._container_client:
            logging.error("Container client not initialized. Cannot upload pickle.")
            return False
            
        import pickle
        
        def _upload():
            # Pickle the object to a buffer
            buffer = io.BytesIO()
            pickle.dump(obj, buffer)
            buffer.seek(0)
            
            # Upload the pickled data
            blob_client = self.container_client.get_blob_client(blob_path)
            content_settings = ContentSettings(content_type="application/octet-stream")
            blob_client.upload_blob(buffer.getvalue(), overwrite=True, content_settings=content_settings)
            return True
            
        try:
            result = RetryHandler.with_retry(
                _upload,
                max_retries=max_retries,
                logging_prefix=f"Pickle upload ({blob_path})"
            )
            if result:
                logging.info(f"Successfully uploaded pickled object to {blob_path}")
            return result
        except Exception as e:
            logging.error(f"Failed to upload pickled object to {blob_path}: {e}")
            return False
    
    def upload_numpy(self, array, blob_path, max_retries=3):
        """Upload a numpy array to blob storage"""
        if not self._container_client:
            logging.error("Container client not initialized. Cannot upload numpy array.")
            return False
            
        def _upload():
            # Save numpy array to a buffer
            buffer = io.BytesIO()
            np.save(buffer, array)
            buffer.seek(0)
            
            # Upload the array data
            blob_client = self.container_client.get_blob_client(blob_path)
            content_settings = ContentSettings(content_type="application/octet-stream")
            blob_client.upload_blob(buffer.getvalue(), overwrite=True, content_settings=content_settings)
            return True
            
        try:
            result = RetryHandler.with_retry(
                _upload,
                max_retries=max_retries,
                logging_prefix=f"Numpy array upload ({blob_path})"
            )
            if result:
                logging.info(f"Successfully uploaded numpy array to {blob_path}")
            return result
        except Exception as e:
            logging.error(f"Failed to upload numpy array to {blob_path}: {e}")
            return False
    
    def download_blob(self, blob_path, file_path=None, max_retries=3):
        """
        Download a blob from storage with retry logic
        
        Args:
            blob_path: Path to the blob in storage
            file_path: Path to save file locally (if None, returns data as bytes)
            max_retries: Number of retry attempts
            
        Returns:
            Downloaded data as bytes (if file_path is None) or boolean success indicator
        """
        if not self._container_client:
            logging.error("Container client not initialized. Cannot download blob.")
            return False if file_path else None
        
        def _download():
            blob_client = self.container_client.get_blob_client(blob_path)
            download_stream = blob_client.download_blob()
            
            if file_path:
                with open(file_path, "wb") as data:
                    data.write(download_stream.readall())
                return True
            else:
                return download_stream.readall()
        
        try:
            result = RetryHandler.with_retry(
                _download,
                max_retries=max_retries,
                logging_prefix=f"Blob download ({blob_path})"
            )
            if result and file_path:
                logging.info(f"Successfully downloaded blob from {blob_path} to {file_path}")
            elif result:
                logging.info(f"Successfully downloaded blob from {blob_path}")
            return result
        except Exception as e:
            logging.error(f"Failed to download blob {blob_path}: {e}")
            return False if file_path else None
    
    def download_json(self, blob_path, max_retries=3):
        """Download and parse JSON data from blob storage"""
        data = self.download_blob(blob_path, file_path=None, max_retries=max_retries)
        if data:
            try:
                return json.loads(data)
            except Exception as e:
                logging.error(f"Failed to parse JSON from {blob_path}: {e}")
                return None
        return None
    
    def download_dataframe(self, blob_path, format="parquet", max_retries=3, **kwargs):
        """
        Download and parse a DataFrame from blob storage
        
        Args:
            blob_path: Path to the blob in storage
            format: Format of the stored DataFrame ('parquet', 'csv', etc.)
            max_retries: Number of retry attempts
            **kwargs: Additional arguments to pass to pd.read_* method
            
        Returns:
            DataFrame or None if download/parsing fails
        """
        data = self.download_blob(blob_path, file_path=None, max_retries=max_retries)
        if data:
            try:
                buffer = io.BytesIO(data)
                if format.lower() == "parquet":
                    return pd.read_parquet(buffer, **kwargs)
                elif format.lower() == "csv":
                    return pd.read_csv(buffer, **kwargs)
                elif format.lower() == "excel":
                    return pd.read_excel(buffer, **kwargs)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            except Exception as e:
                logging.error(f"Failed to parse DataFrame from {blob_path}: {e}")
                return None
        return None
    
    def download_pickle(self, blob_path, max_retries=3):
        """Download and unpickle an object from blob storage"""
        import pickle
        
        data = self.download_blob(blob_path, file_path=None, max_retries=max_retries)
        if data:
            try:
                buffer = io.BytesIO(data)
                return pickle.load(buffer)
            except Exception as e:
                logging.error(f"Failed to unpickle object from {blob_path}: {e}")
                return None
        return None
    
    def download_numpy(self, blob_path, max_retries=3):
        """Download and parse a numpy array from blob storage"""
        data = self.download_blob(blob_path, file_path=None, max_retries=max_retries)
        if data:
            try:
                buffer = io.BytesIO(data)
                return np.load(buffer, allow_pickle=True)
            except Exception as e:
                logging.error(f"Failed to parse numpy array from {blob_path}: {e}")
                return None
        return None
    
    def blob_exists(self, blob_path):
        """Check if a blob exists in storage"""
        if not self._container_client:
            return False
        
        try:
            blob_client = self.container_client.get_blob_client(blob_path)
            return blob_client.exists()
        except Exception as e:
            logging.error(f"Error checking if blob {blob_path} exists: {e}")
            return False
    
    def list_blobs(self, prefix=None):
        """List blobs in the container with an optional prefix"""
        if not self._container_client:
            return []
        
        try:
            return [blob.name for blob in self.container_client.list_blobs(name_starts_with=prefix)]
        except Exception as e:
            logging.error(f"Error listing blobs with prefix {prefix}: {e}")
            return []

# Create a global instance for convenience
try:
    from ..config.settings import config
    blob_storage = AzureBlobStorage(config.blob_connection_string, config.container_name)
except (ImportError, AttributeError):
    # Handle circular import or missing config
    blob_storage = None
    logging.warning("Unable to initialize blob_storage at module load time. Use AzureBlobStorage() directly.")
