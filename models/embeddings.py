"""
Embedding generation for the topic clustering package.
"""

import time
import uuid
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import openai
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from ..config.settings import config
from ..utils.text_utils import estimate_tokens

class EmbeddingClient:
    """Client for generating embeddings from OpenAI API"""
    
    def __init__(self, api_key=None, api_version=None, endpoint=None, model=None):
        # Use provided values or defaults from config
        self.api_key = api_key or config.azure_openai_config["embedding_key"]
        self.api_version = api_version or config.azure_openai_config["embedding_api_version"]
        self.endpoint = endpoint or config.azure_openai_config["embedding_endpoint"]
        self.model = model or config.azure_openai_config["embedding_model"]
        
        # Initialize OpenAI client
        self._client = openai.AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint
        )
        
        # Circuit breaker state
        self.circuit_open = False
        self.circuit_failures = 0
        self.circuit_threshold = 5
        self.circuit_reset_time = None
        
        # Metrics tracking
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "rate_limited_requests": 0,
            "total_time": 0
        }
        
        logging.info(f"Initialized embedding client with model {self.model}")
    
    def get_embedding(self, text):
        """Get embedding for a single text"""
        try:
            response = self._client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            logging.error(f"Error getting embedding: {e}")
            # Return zeros as fallback
            return [0.0] * 3072  # Assuming text-embedding-3-large dimensions
    
    def generate_batch_embeddings(self, text_series, batch_size=25):
        """Generate embeddings for a series of texts in batches"""
        semantic_embeddings = []
        
        # Reduce batch size for very large texts
        max_estimated_tokens = text_series.apply(lambda x: estimate_tokens(str(x))).max()
        if max_estimated_tokens > 4000:
            adjusted_batch_size = max(1, min(batch_size, 5))
            logging.warning(f"Detected large texts (est. {max_estimated_tokens} tokens). Reducing batch size to {adjusted_batch_size}")
            batch_size = adjusted_batch_size
        
        # Process in batches with progress tracking
        with tqdm(total=len(text_series), desc="Embedding texts") as pbar:
            for i in range(0, len(text_series), batch_size):
                # Check circuit breaker
                if self.circuit_open:
                    if time.time() - self.circuit_reset_time > 60:  # 1 minute timeout
                        self.circuit_open = False
                        self.circuit_failures = 0
                        logging.info("Circuit breaker reset, resuming API calls")
                    else:
                        logging.warning(f"Circuit breaker open, skipping batch and using zeros. Resets in {60 - (time.time() - self.circuit_reset_time):.0f}s")
                        semantic_embeddings.extend([[0.0] * 3072 for _ in range(min(batch_size, len(text_series) - i))])
                        pbar.update(min(batch_size, len(text_series) - i))
                        continue
                
                # Create batch with correlation ID for tracking
                batch = text_series.iloc[i:i+batch_size].fillna('').tolist()
                batch_size_actual = len(batch)
                correlation_id = str(uuid.uuid4())
                
                self.metrics["total_requests"] += 1
                start_time = time.time()
                
                try:
                    # Call the embedding API
                    response = self._client.embeddings.create(
                        input=batch,
                        model=self.model,
                        timeout=30,
                        user=correlation_id
                    )
                    
                    # Extract embeddings
                    batch_embeddings = [item.embedding for item in response.data]
                    semantic_embeddings.extend(batch_embeddings)
                    
                    # Update metrics
                    self.metrics["successful_requests"] += 1
                    self.metrics["total_time"] += time.time() - start_time
                    self.circuit_failures = 0
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    # Handle different error types
                    if "rate limit" in error_msg or "too many requests" in error_msg or "throttl" in error_msg:
                        # Rate limiting - back off and retry
                        self._handle_rate_limit(i, batch_size, correlation_id)
                        continue
                    elif "token" in error_msg or "context length" in error_msg:
                        # Token limit - reduce batch size and retry
                        if batch_size > 1:
                            new_batch_size = self._handle_token_limit(batch_size, i)
                            batch_size = new_batch_size
                            i = max(0, i - batch_size * 2)  # Go back to retry
                            continue
                        else:
                            # Single item too long - use zeros
                            logging.error(f"Single item is too long: {len(text_series.iloc[i])}")
                            semantic_embeddings.extend([[0.0] * 3072])
                            self.metrics["failed_requests"] += 1
                    else:
                        # Other errors - implement circuit breaker
                        self._handle_other_error(error_msg, batch_size_actual)
                
                # Update progress
                pbar.update(batch_size_actual)
        
        # Log final metrics
        self._log_metrics()
        
        # Ensure we return the right number of embeddings
        if len(semantic_embeddings) < len(text_series):
            logging.warning(f"Missing embeddings: expected {len(text_series)}, got {len(semantic_embeddings)}. Padding with zeros.")
            semantic_embeddings.extend([[0.0] * 3072 for _ in range(len(text_series) - len(semantic_embeddings))])
        
        return np.array(semantic_embeddings)
    
    def _handle_rate_limit(self, position, batch_size, correlation_id):
        """Handle rate limiting errors"""
        self.metrics["rate_limited_requests"] += 1
        wait_time = min(60, 2 ** (self.metrics["rate_limited_requests"] % 6))
        logging.warning(f"Azure OpenAI rate limit hit (ID: {correlation_id}). Backing off for {wait_time}s")
        time.sleep(wait_time)
        return position - batch_size  # Retry the same batch
    
    def _handle_token_limit(self, batch_size, position):
        """Handle token limit errors by reducing batch size"""
        new_batch_size = max(1, batch_size // 2)
        logging.warning(f"Token limit exceeded. Reducing batch size from {batch_size} to {new_batch_size}")
        return new_batch_size
    
    def _handle_other_error(self, error_msg, batch_size_actual):
        """Handle other errors with circuit breaker pattern"""
        self.metrics["failed_requests"] += 1
        self.circuit_failures += 1
        logging.error(f"Embedding error: {error_msg} (Failures: {self.circuit_failures}/{self.circuit_threshold})")
        
        # If too many failures, open circuit breaker
        if self.circuit_failures >= self.circuit_threshold:
            self.circuit_open = True
            self.circuit_reset_time = time.time()
            logging.error(f"Circuit breaker activated! Too many failures. Pausing API calls for 60s")
        
        # Use zero vectors for errors
        return [[0.0] * 3072 for _ in range(batch_size_actual)]
    
    def _log_metrics(self):
        """Log metrics about embedding API usage"""
        avg_time = self.metrics["total_time"] / max(1, self.metrics["successful_requests"])
        logging.info(f"Azure OpenAI Embedding API metrics: {self.metrics['total_requests']} requests, "
                    f"{self.metrics['successful_requests']} successful, {self.metrics['failed_requests']} failed, "
                    f"{self.metrics['rate_limited_requests']} rate limited, avg time: {avg_time:.2f}s")

class HybridEmbeddingGenerator:
    """Generate hybrid embeddings combining TF-IDF and semantic embeddings"""
    
    def __init__(self, embedding_client=None):
        self.embedding_client = embedding_client or EmbeddingClient()
    
    def generate_hybrid_embeddings(self, df, text_column, batch_size=25):
        """
        Generate hybrid embeddings combining entity, action, and semantic information
        
        Args:
            df: DataFrame with text data
            text_column: Column containing text to embed
            batch_size: Batch size for embedding API calls
            
        Returns:
            DataFrame with embeddings added
        """
        logging.info(f"Generating hybrid embeddings for {len(df)} records from column {text_column}")
        
        # Get text data
        text_data = df[text_column].fillna('')
        
        # 1. Get entity and action classification
        classification_result = self.classify_terms(df, text_column)
        
        # 2. Extract terms and create TF-IDF embeddings
        entity_terms = list(classification_result.get("ENTITY", {}).keys())
        action_terms = list(classification_result.get("ACTION", {}).keys())
        logging.info(f"Using {len(entity_terms)} entities and {len(action_terms)} action terms")
        
        # Create vectorizers
        vectorizers = {
            'entity': TfidfVectorizer(vocabulary=entity_terms, lowercase=True),
            'action': TfidfVectorizer(vocabulary=action_terms, lowercase=True)
        }
        
        # Generate TF-IDF matrices
        matrices = {
            'entity': vectorizers['entity'].fit_transform(text_data),
            'action': vectorizers['action'].fit_transform(text_data)
        }
        
        # Convert sparse matrices to dense
        dense_matrices = {k: m.toarray() for k, m in matrices.items()}
        
        # 3. Generate semantic embeddings
        logging.info("Generating semantic embeddings...")
        semantic_embeddings = self.embedding_client.generate_batch_embeddings(text_data, batch_size)
        
        # 4. Scale and combine all embedding components
        logging.info("Scaling and combining embeddings...")
        scaled_matrices = {}
        for name, matrix in {**dense_matrices, 'semantic': semantic_embeddings}.items():
            scaler = StandardScaler()
            scaled_matrices[name] = scaler.fit_transform(matrix)
        
        # Set weights for each component
        weights = {'entity': 1.0, 'action': 1.0, 'semantic': 2.0}
        
        # Get dimensions for each component
        dims = {k: m.shape[1] for k, m in scaled_matrices.items()}
        
        # Create combined embeddings array
        total_dims = sum(dims.values())
        combined_embeddings = np.zeros((len(df), total_dims))
        
        # Fill combined embeddings with weighted components
        start_idx = 0
        for name, matrix in scaled_matrices.items():
            end_idx = start_idx + dims[name]
            combined_embeddings[:, start_idx:end_idx] = weights[name] * matrix
            start_idx = end_idx
        
        # Convert to json strings for storage
        import json
        df_with_embeddings = df.copy()
        df_with_embeddings['embedding'] = [json.dumps(emb.tolist()) for emb in combined_embeddings]
        
        logging.info("Hybrid embedding generation complete")
        return df_with_embeddings, classification_result
    
    def classify_terms(self, df, text_column):
        """Classify terms into entities and actions using OpenAI API"""
        from ..utils.azure_utils import RetryHandler
        from ..config.settings import config
        from ..utils.json_utils import robust_json_parser
        import openai
        
        # Initialize OpenAI client for chat
        openai_client = openai.AzureOpenAI(
            azure_endpoint=config.azure_openai_config["endpoint"],
            api_key=config.azure_openai_config["api_key"],
            api_version=config.azure_openai_config["api_version"]
        )
        
        # Combine all text for term extraction (limit to avoid token issues)
        sample_df = df.sample(min(1000, len(df))) if len(df) > 1000 else df
        all_text = " ".join(sample_df[text_column].fillna('').astype(str).tolist())
        
        # Use GPT to extract and classify terms
        prompt = f"""
        Analyze the following IT incidents text and identify:
        1. ENTITIES: Technical components, systems, software, or services mentioned (like SAP, Outlook, VPN)
        2. ACTIONS: Verbs describing what happened or needs to happen (like crashed, failed, reset)
        
        Extract up to 50 most common entities and up to 50 most common actions.
        
        YOU MUST RESPOND WITH VALID JSON using this structure:
        {{
            "ENTITY": {{"term1": frequency, "term2": frequency, ...}},
            "ACTION": {{"term1": frequency, "term2": frequency, ...}}
        }}
        
        Incident text sample:
        {all_text[:3000]}
        """
        
        def _classify_terms():
            response = openai_client.chat.completions.create(
                model=config.azure_openai_config["chat_deployment_name"],
                messages=[
                    {"role": "system", "content": "You are an expert IT analyst that extracts and classifies terms from incident descriptions. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
                timeout=30,
                user=str(uuid.uuid4())
            )
            
            # Use robust JSON parser
            result = robust_json_parser(response.choices[0].message.content)
            
            # Validate expected structure
            if "ENTITY" in result and "ACTION" in result:
                return result
            else:
                raise ValueError("Response missing required ENTITY or ACTION keys")
        
        try:
            result = RetryHandler.with_retry(
                _classify_terms,
                max_retries=3,
                logging_prefix="Term classification"
            )
            return result
        except Exception as e:
            logging.error(f"All term classification attempts failed: {e}")
            # Return empty results as fallback
            return {"ENTITY": {}, "ACTION": {}}
