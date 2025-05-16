"""
Azure Function App for running the topic clustering prediction pipeline.
"""

import datetime
import logging
import os
import json
import azure.functions as func
from dotenv import load_dotenv

def main(timer: func.TimerRequest) -> None:
    """
    Azure Function entry point for the prediction pipeline.
    Runs every 12 hours to predict clusters for new incidents.
    
    Args:
        timer: Timer trigger information
    """
    # Load environment variables
    load_dotenv()
    
    # Import here to allow environment variables to be loaded first
    from topic_clustering.prediction import run_prediction_job
    
    utc_timestamp = datetime.datetime.utcnow().replace(
        tzinfo=datetime.timezone.utc).isoformat()
    
    logging.info(f'Python timer trigger function started at {utc_timestamp}')
    
    # Get function parameters from settings
    tech_center = os.environ.get("TECHCENTER_TO_PROCESS")
    process_all = os.environ.get("PROCESS_ALL_TECHCENTERS", "False").lower() == "true"
    
    try:
        if tech_center:
            logging.info(f"Running prediction for {tech_center}")
            result = run_prediction_job(tech_center=tech_center)
        elif process_all:
            logging.info(f"Running prediction for all tech centers")
            result = run_prediction_job(all_centers=True)
        else:
            logging.error("No tech center specified and PROCESS_ALL_TECHCENTERS not set to true")
            return
            
        logging.info(f"Prediction job completed: {json.dumps(result, default=str)}")
        
    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
