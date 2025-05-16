"""
Logging utilities for the topic clustering package.
"""

import logging
import sys
import os
import io
from datetime import datetime
from ..utils.azure_utils import blob_storage

class BlobStorageLogHandler(logging.Handler):
    """Handler for writing logs to Azure Blob Storage"""
    
    def __init__(self, dataset_name, log_name=None):
        """
        Initialize handler
        
        Args:
            dataset_name: Name of dataset for blob path
            log_name: Name of log file (if None, uses timestamp)
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.log_name = log_name or f"pipeline_{get_timestamp()}.log"
        self.buffer = io.StringIO()
        self.blob_path = f"{dataset_name}/logs/{self.log_name}"
        self.last_upload_time = datetime.now()
        self.upload_interval = 30  # seconds
    
    def emit(self, record):
        """Process a log record"""
        try:
            msg = self.format(record)
            self.buffer.write(msg + "\n")
            
            # Upload to blob storage periodically
            now = datetime.now()
            if (now - self.last_upload_time).total_seconds() >= self.upload_interval:
                self.flush()
        except Exception:
            self.handleError(record)
    
    def flush(self):
        """Flush the buffer to blob storage"""
        if not blob_storage or not blob_storage._container_client:
            return
            
        try:
            # Get buffer content
            self.buffer.seek(0)
            content = self.buffer.getvalue()
            
            if content:
                # Upload log content
                blob_storage.upload_blob(
                    content, 
                    self.blob_path, 
                    content_type="text/plain",
                    is_file_path=False
                )
                self.last_upload_time = datetime.now()
                
                # Clear buffer but keep the content (for appending)
                new_buffer = io.StringIO()
                new_buffer.write(content)
                self.buffer = new_buffer
        except Exception as e:
            # Don't raise exceptions from logging
            print(f"Error uploading logs to blob storage: {e}")
    
    def close(self):
        """Close the handler"""
        self.flush()
        self.buffer.close()
        super().close()

def setup_logging(log_level=logging.INFO, log_file=None, log_to_console=True, 
                dataset_name=None, log_to_blob=False):
    """
    Set up logging configuration
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Path to log file (if None, logs only to console)
        log_to_console: Whether to log to console (default: True)
        dataset_name: Dataset name for blob storage path
        log_to_blob: Whether to save logs to Azure Blob Storage
        
    Returns:
        Logger instance
    """
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # Add console handler if requested
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Add blob storage handler if requested
    if log_to_blob and dataset_name:
        if blob_storage and blob_storage._container_client:
            log_name = os.path.basename(log_file) if log_file else None
            blob_handler = BlobStorageLogHandler(dataset_name, log_name)
            blob_handler.setFormatter(formatter)
            blob_handler.setLevel(log_level)
            root_logger.addHandler(blob_handler)
            logging.info(f"Logs will be saved to Azure Blob Storage: {blob_handler.blob_path}")
        else:
            logging.warning("Azure Blob Storage not available. Logs won't be saved to blob storage.")
    
    return root_logger

def get_timestamp(format="%Y-%m-%d_%H-%M-%S"):
    """
    Get current timestamp in a standardized format
    
    Args:
        format: Datetime format string (default: "%Y-%m-%d_%H-%M-%S")
        
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime(format)

def log_section(section_name, logger=None, char="=", length=80):
    """
    Log a section header to clearly separate sections in the logs
    
    Args:
        section_name: Name of the section
        logger: Logger to use (if None, uses root logger)
        char: Character to use for separator line (default: "=")
        length: Length of separator line (default: 80)
    """
    if logger is None:
        logger = logging.getLogger()
    
    separator = char * length
    centered_name = f" {section_name} ".center(length, char)
    
    logger.info(separator)
    logger.info(centered_name)
    logger.info(separator)
