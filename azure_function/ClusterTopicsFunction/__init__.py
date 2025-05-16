import logging
import json
import os
import azure.functions as func
from datetime import datetime

# Import topic clustering functionality
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from topic_clustering.pipeline.pipeline import run_full_pipeline
from topic_clustering.utils.json_utils import NumpyEncoder

def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    HTTP-triggered Azure Function that runs the topic clustering pipeline.
    
    Request body should include:
    - input_query: BigQuery query to load data
    - dataset_name: Name for this analysis run (optional, will generate if not provided)
    - embeddings_table_id: BigQuery table to save embeddings (optional)
    - results_table_id: BigQuery table to save results (optional)
    - cluster_params: Dictionary with clustering parameters (optional)
    """
    logging.info('Topic clustering function processing a request.')
    
    try:
        # Parse request body
        req_body = req.get_json()
        
        # Extract parameters from request
        input_query = req_body.get('input_query')
        if not input_query:
            return func.HttpResponse(
                json.dumps({"error": "No input_query provided in request body"}),
                mimetype="application/json",
                status_code=400
            )
        
        # Get optional parameters or use defaults
        dataset_name = req_body.get('dataset_name', f"azure_function_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        embeddings_table_id = req_body.get('embeddings_table_id')
        results_table_id = req_body.get('results_table_id')
        cluster_params = req_body.get('cluster_params', {})
        other_params = req_body.get('other_params', {})
        
        # Run the pipeline
        result = run_full_pipeline(
            dataset_name=dataset_name,
            input_query=input_query,
            embeddings_table_id=embeddings_table_id,
            results_table_id=results_table_id,
            cluster_params=cluster_params,
            **other_params
        )
        
        # Return results
        return func.HttpResponse(
            json.dumps({
                "status": "success",
                "dataset_name": dataset_name,
                "result_summary": result
            }, cls=NumpyEncoder),
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.exception(f"Error in topic clustering function: {str(e)}")
        return func.HttpResponse(
            json.dumps({
                "status": "error",
                "error": str(e)
            }),
            mimetype="application/json",
            status_code=500
        )
