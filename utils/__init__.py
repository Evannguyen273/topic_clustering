"""
Utility functions for the topic clustering package.
"""

from .text_utils import (
    get_safe_text, 
    estimate_tokens, 
    clean_text_for_summary, 
    normalize_business_service,
    remove_emails
)

from .json_utils import (
    NumpyEncoder,
    robust_json_parser,
    validate_json_response,
    save_json
)

from .bigquery_utils import (
    run_query,
    save_results_to_bigquery,
    save_embeddings_to_bigquery,
    get_bigquery_client,
    BigQueryClient
)

from .logging_utils import (
    setup_logging,
    get_timestamp,
    log_section
)

from .azure_utils import (
    RetryHandler,
    AzureBlobStorage,
    blob_storage
)

__all__ = [
    # Text utils
    'get_safe_text', 'estimate_tokens', 'clean_text_for_summary', 
    'normalize_business_service', 'remove_emails',
    
    # JSON utils
    'NumpyEncoder', 'robust_json_parser', 'validate_json_response', 'save_json',
    
    # BigQuery utils
    'run_query', 'save_results_to_bigquery', 'save_embeddings_to_bigquery',
    'get_bigquery_client', 'BigQueryClient',
    
    # Logging utils
    'setup_logging', 'get_timestamp', 'log_section',
    
    # Azure utils
    'RetryHandler', 'AzureBlobStorage', 'blob_storage'
]
