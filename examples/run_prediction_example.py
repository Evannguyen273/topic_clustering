"""
Example script for running the prediction pipeline.
"""

import os
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

from topic_clustering.prediction import run_prediction_job
from topic_clustering.utils.logging_utils import setup_logging

def main():
    # Load environment variables
    load_dotenv(".env")
    
    # Setup logging
    setup_logging(log_level=logging.INFO)
    
    # Define time range for prediction
    # Process yesterday's incidents
    yesterday = datetime.now() - timedelta(days=1)
    since_time = yesterday.replace(hour=0, minute=0, second=0).isoformat()
    until_time = yesterday.replace(hour=23, minute=59, second=59).isoformat()
    
    # Example 1: Run prediction for a specific tech center
    tech_center = "BT-TC-Product Development & Engineering"
    
    print(f"Running prediction for {tech_center} from {since_time} to {until_time}")
    
    result = run_prediction_job(
        tech_center=tech_center,
        since_timestamp=since_time,
        until_timestamp=until_time
    )
    
    print(f"Prediction completed with result: {result}")
    
    # Example 2: Run prediction for all tech centers
    # Uncomment the following lines to run for all tech centers
    # print("Running prediction for all tech centers")
    # result = run_prediction_job(all_centers=True)
    # print(f"Prediction for all tech centers completed")

if __name__ == "__main__":
    main()
