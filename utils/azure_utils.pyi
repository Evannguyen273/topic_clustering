"""
Type stubs for Azure utilities in the topic clustering package.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, Awaitable
import pandas as pd
import numpy as np
from azure.storage.blob import BlobServiceClient, ContainerClient, ContentSettings

T = TypeVar('T')

class RetryHandler:
    @staticmethod
    def with_retry(
        func: Callable[[], T], 
        max_retries: int = ..., 
        base_delay: int = ...,
        retry_on_exceptions: Tuple[type[Exception], ...] = ..., 
        logging_prefix: str = ...
    ) -> T: ...
    
    @staticmethod
    async def with_async_retry(
        func: Callable[[], Awaitable[T]], 
        max_retries: int = ..., 
        base_delay: int = ...,
        retry_on_exceptions: Tuple[type[Exception], ...] = ..., 
        logging_prefix: str = ...
    ) -> T: ...

class AzureBlobStorage:
    connection_string: Optional[str]
    container_name: str
    _container_client: Optional[ContainerClient]
    
    def __init__(self, connection_string: Optional[str] = ..., container_name: Optional[str] = ...) -> None: ...
    
    @property
    def container_client(self) -> ContainerClient: ...
    
    def upload_blob(
        self, 
        file_path_or_data: Union[str, bytes], 
        blob_path: str, 
        content_type: str = ..., 
        max_retries: int = ..., 
        is_file_path: bool = ...
    ) -> bool: ...
    
    def upload_json(self, data: Any, blob_path: str, max_retries: int = ...) -> bool: ...
    
    def upload_dataframe(
        self, 
        df: pd.DataFrame, 
        blob_path: str, 
        format: str = ..., 
        max_retries: int = ..., 
        **kwargs: Any
    ) -> bool: ...
    
    async def upload_dataframe_async(
        self, 
        df: pd.DataFrame, 
        blob_path: str, 
        format: str = ..., a
        max_retries: int = ..., 
        **kwargs: Any
    ) -> bool: ...
    
    def upload_pickle(self, obj: Any, blob_path: str, max_retries: int = ...) -> bool: ...
    
    def upload_numpy(self, array: np.ndarray, blob_path: str, max_retries: int = ...) -> bool: ...
    
    def download_blob(
        self, 
        blob_path: str, 
        file_path: Optional[str] = ..., 
        max_retries: int = ...
    ) -> Union[bytes, bool]: ...
    
    def download_json(self, blob_path: str, max_retries: int = ...) -> Optional[Any]: ...
    
    def download_dataframe(
        self, 
        blob_path: str, 
        format: str = ..., 
        max_retries: int = ..., 
        **kwargs: Any
    ) -> Optional[pd.DataFrame]: ...
    
    def download_pickle(self, blob_path: str, max_retries: int = ...) -> Optional[Any]: ...
    
    def download_numpy(self, blob_path: str, max_retries: int = ...) -> Optional[np.ndarray]: ...
    
    def blob_exists(self, blob_path: str) -> bool: ...
    
    def list_blobs(self, prefix: Optional[str] = ...) -> List[str]: ...

blob_storage: Optional[AzureBlobStorage]
