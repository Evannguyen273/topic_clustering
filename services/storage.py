"""
Storage service with dependency injection support for the topic clustering package.
"""

import os
import logging
from typing import Protocol, Optional, Union, Dict, List, Any, TypeVar, Generic
import pandas as pd
import numpy as np
import json

from ..utils.azure_utils import AzureBlobStorage, blob_storage as default_blob_storage
from ..config.settings import config

# Define protocol for storage backends
class StorageBackend(Protocol):
    """Protocol defining the required methods for storage backends"""
    
    def upload_dataframe(self, df: pd.DataFrame, path: str, **kwargs) -> bool:
        """Upload a dataframe to storage"""
        ...
    
    def download_dataframe(self, path: str, **kwargs) -> Optional[pd.DataFrame]:
        """Download a dataframe from storage"""
        ...
    
    def upload_json(self, data: Any, path: str) -> bool:
        """Upload JSON data to storage"""
        ...
    
    def download_json(self, path: str) -> Optional[Dict[str, Any]]:
        """Download JSON data from storage"""
        ...
    
    def upload_raw(self, data: Union[str, bytes], path: str, **kwargs) -> bool:
        """Upload raw data to storage"""
        ...
    
    def download_raw(self, path: str) -> Optional[bytes]:
        """Download raw data from storage"""
        ...
    
    def path_exists(self, path: str) -> bool:
        """Check if a path exists in storage"""
        ...

class AzureBlobBackend:
    """Azure Blob Storage backend implementation"""
    
    def __init__(self, blob_client: Optional[AzureBlobStorage] = None):
        self.client = blob_client or default_blob_storage
        if not self.client or not self.client._container_client:
            logging.warning("Azure Blob Storage not available. Operations will fail.")
    
    def upload_dataframe(self, df: pd.DataFrame, path: str, **kwargs) -> bool:
        format = kwargs.pop("format", "parquet")
        return self.client.upload_dataframe(df, path, format=format, **kwargs)
    
    def download_dataframe(self, path: str, **kwargs) -> Optional[pd.DataFrame]:
        format = kwargs.pop("format", "parquet")
        return self.client.download_dataframe(path, format=format, **kwargs)
    
    def upload_json(self, data: Any, path: str) -> bool:
        return self.client.upload_json(data, path)
    
    def download_json(self, path: str) -> Optional[Dict[str, Any]]:
        return self.client.download_json(path)
    
    def upload_raw(self, data: Union[str, bytes], path: str, **kwargs) -> bool:
        is_file = kwargs.pop("is_file", isinstance(data, str) and os.path.exists(data))
        content_type = kwargs.pop("content_type", "application/octet-stream")
        return self.client.upload_blob(data, path, content_type=content_type, is_file_path=is_file)
    
    def download_raw(self, path: str) -> Optional[bytes]:
        return self.client.download_blob(path, file_path=None)
    
    def path_exists(self, path: str) -> bool:
        return self.client.blob_exists(path)

class LocalFileBackend:
    """Local file storage backend implementation"""
    
    def __init__(self, base_path: Optional[str] = None):
        self.base_path = base_path or config.result_path
        os.makedirs(self.base_path, exist_ok=True)
    
    def _get_full_path(self, path: str) -> str:
        # Convert blob-style paths to local OS paths
        local_path = path.replace("/", os.sep)
        full_path = os.path.join(self.base_path, local_path)
        # Ensure directory exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        return full_path
    
    def upload_dataframe(self, df: pd.DataFrame, path: str, **kwargs) -> bool:
        format = kwargs.pop("format", "parquet")
        full_path = self._get_full_path(path)
        try:
            if format.lower() == "parquet":
                df.to_parquet(full_path, **kwargs)
            elif format.lower() == "csv":
                df.to_csv(full_path, index=False, **kwargs)
            else:
                raise ValueError(f"Unsupported format: {format}")
            return True
        except Exception as e:
            logging.error(f"Failed to save DataFrame to {full_path}: {e}")
            return False
    
    def download_dataframe(self, path: str, **kwargs) -> Optional[pd.DataFrame]:
        format = kwargs.pop("format", "parquet")
        full_path = self._get_full_path(path)
        try:
            if not os.path.exists(full_path):
                return None
                
            if format.lower() == "parquet":
                return pd.read_parquet(full_path, **kwargs)
            elif format.lower() == "csv":
                return pd.read_csv(full_path, **kwargs)
            else:
                raise ValueError(f"Unsupported format: {format}")
        except Exception as e:
            logging.error(f"Failed to load DataFrame from {full_path}: {e}")
            return None
    
    def upload_json(self, data: Any, path: str) -> bool:
        from ..utils.json_utils import NumpyEncoder
        full_path = self._get_full_path(path)
        try:
            with open(full_path, "w") as f:
                json.dump(data, f, cls=NumpyEncoder, indent=2)
            return True
        except Exception as e:
            logging.error(f"Failed to save JSON to {full_path}: {e}")
            return False
    
    def download_json(self, path: str) -> Optional[Dict[str, Any]]:
        full_path = self._get_full_path(path)
        try:
            if not os.path.exists(full_path):
                return None
                
            with open(full_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load JSON from {full_path}: {e}")
            return None
    
    def upload_raw(self, data: Union[str, bytes], path: str, **kwargs) -> bool:
        full_path = self._get_full_path(path)
        try:
            is_file = kwargs.get("is_file", isinstance(data, str) and os.path.exists(data))
            
            if is_file:
                # Copy file
                import shutil
                shutil.copy2(data, full_path)
            else:
                # Write data directly
                mode = "wb" if isinstance(data, bytes) else "w"
                with open(full_path, mode) as f:
                    f.write(data)
            return True
        except Exception as e:
            logging.error(f"Failed to write data to {full_path}: {e}")
            return False
    
    def download_raw(self, path: str) -> Optional[bytes]:
        full_path = self._get_full_path(path)
        try:
            if not os.path.exists(full_path):
                return None
                
            with open(full_path, "rb") as f:
                return f.read()
        except Exception as e:
            logging.error(f"Failed to read data from {full_path}: {e}")
            return None
    
    def path_exists(self, path: str) -> bool:
        full_path = self._get_full_path(path)
        return os.path.exists(full_path)

class BackupStorage:
    """
    Storage service with multiple backends and fallback support
    """
    
    def __init__(self, 
                primary_backend: Optional[StorageBackend] = None,
                backup_backend: Optional[StorageBackend] = None,
                dataset_name: Optional[str] = None):
        """
        Initialize the storage service
        
        Args:
            primary_backend: Primary storage backend (default: Azure Blob Storage)
            backup_backend: Backup storage backend (default: Local File Storage)
            dataset_name: Dataset name to use for path prefixes
        """
        self.primary = primary_backend or AzureBlobBackend()
        self.backup = backup_backend or LocalFileBackend()
        self.dataset_name = dataset_name
    
    def _get_full_path(self, path: str) -> str:
        """Get full path including dataset name if provided"""
        if not self.dataset_name:
            return path
        return f"{self.dataset_name}/{path}"
    
    def upload(self, data: pd.DataFrame, path: str, **kwargs) -> Dict[str, bool]:
        """
        Upload data to both primary and backup storage
        
        Args:
            data: DataFrame to upload
            path: Path within storage (dataset name will be prepended if provided)
            **kwargs: Additional arguments for upload
            
        Returns:
            Dictionary with success status for each backend
        """
        full_path = self._get_full_path(path)
        result = {
            "primary": self.primary.upload_dataframe(data, full_path, **kwargs),
            "backup": self.backup.upload_dataframe(data, full_path, **kwargs)
        }
        return result
    
    def download(self, path: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        Download data from storage, trying primary first then falling back to backup
        
        Args:
            path: Path within storage (dataset name will be prepended if provided)
            **kwargs: Additional arguments for download
            
        Returns:
            Downloaded data or None if not found in either backend
        """
        full_path = self._get_full_path(path)
        
        # Try primary first
        result = self.primary.download_dataframe(full_path, **kwargs)
        if result is not None:
            return result
            
        # Fall back to backup
        logging.info(f"Data not found in primary storage, trying backup for {full_path}")
        return self.backup.download_dataframe(full_path, **kwargs)
    
    def upload_json(self, data: Any, path: str) -> Dict[str, bool]:
        """Upload JSON data to both primary and backup storage"""
        full_path = self._get_full_path(path)
        result = {
            "primary": self.primary.upload_json(data, full_path),
            "backup": self.backup.upload_json(data, full_path)
        }
        return result
    
    def download_json(self, path: str) -> Optional[Dict[str, Any]]:
        """Download JSON from storage with fallback"""
        full_path = self._get_full_path(path)
        
        # Try primary first
        result = self.primary.download_json(full_path)
        if result is not None:
            return result
            
        # Fall back to backup
        logging.info(f"JSON not found in primary storage, trying backup for {full_path}")
        return self.backup.download_json(full_path)
    
    def exists(self, path: str) -> Dict[str, bool]:
        """Check if path exists in either storage backend"""
        full_path = self._get_full_path(path)
        return {
            "primary": self.primary.path_exists(full_path),
            "backup": self.backup.path_exists(full_path)
        }
