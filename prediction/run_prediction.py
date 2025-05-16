"""
Command-line tool for running the prediction pipeline.
"""

import os
import time
import logging
import argparse
import json
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Optional

from ..config.settings import config
from ..utils.logging_utils import setup_logging
from ..utils.azure_utils import blob_storage
from .predict import predict_incidents, PredictionRegistry

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run prediction for tech centers")
    parser.add_argument("--tech-center", help="Specific tech center to process (optional)")
    parser.add_argument("--all", action="store_true", help="Process all tech centers in registry")
    parser.add_argument("--list", action="store_true", help="List available tech centers")
    parser.add_argument("--since", help="Process incidents since timestamp (overrides registry)")
    parser.add_argument("--until", help="Process incidents until timestamp (default: now)")
    parser.add_argument("--log-dir", default="logs", help="Directory for log files")
    parser.add_argument("--no-blob", action="store_true", help="Don't use blob storage (local only)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()

def get_available_tech_centers() -> List[str]:
    """Get list of available tech centers from BigQuery"""
    from ..utils.bigquery_utils import run_query
    
    query = """
    SELECT DISTINCT TechCenter
    FROM `enterprise-dashboardnp-cd35.bigquery_datasets_spoke_oa_dev.Team_services`
    WHERE TechCenter IS NOT NULL AND TechCenter != ''
    ORDER BY TechCenter
    """
    
    try:
        df = run_query(query)
        if df is not None and len(df) > 0:
            return df["TechCenter"].tolist()
    except Exception as e:
        logging.error(f"Error fetching tech centers: {e}")
    
    return []

def run_prediction_job(tech_center: Optional[str] = None, all_centers: bool = False,
                     since_timestamp: Optional[str] = None, until_timestamp: Optional[str] = None,
                     log_to_blob: bool = True):
    """
    Run prediction job for one or all tech centers
    
    Args:
        tech_center: Specific tech center to process (None for all)
        all_centers: Process all tech centers in registry
        since_timestamp: Override for start timestamp
        until_timestamp: Override for end timestamp
        log_to_blob: Whether to log to blob storage
        
    Returns:
        Dictionary with job results
    """
    job_start_time = time.time()
    job_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup logging
    log_dir = os.path.join(config.result_path, "prediction_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"prediction_job_{job_date}.log")
    logger = setup_logging(log_file=log_file, log_to_console=True)
    
    # Initialize registry
    registry = PredictionRegistry()
    
    # Determine which tech centers to process
    tech_centers_to_process = []
    
    if tech_center:
        # Process a single specified tech center
        tech_centers_to_process.append(tech_center)
    elif all_centers:
        # Process all tech centers in registry
        tech_centers_to_process = registry.list_tech_centers()
        if not tech_centers_to_process:
            # If registry is empty, fetch from BigQuery
            tech_centers_to_process = get_available_tech_centers()
    else:
        logging.error("No tech center specified and --all not used")
        return {"error": "No tech center specified and --all not used"}
    
    if not tech_centers_to_process:
        logging.error("No tech centers found to process")
        return {"error": "No tech centers found to process"}
    
    logging.info(f"Processing {len(tech_centers_to_process)} tech centers: {', '.join(tech_centers_to_process)}")
    
    # Run prediction for each tech center
    results = {}
    success_count = 0
    failure_count = 0
    
    for tc in tech_centers_to_process:
        try:
            result = predict_incidents(
                tech_center=tc,
                registry=registry,
                since_timestamp=since_timestamp,
                until_timestamp=until_timestamp
            )
            
            results[tc] = {
                "success": True,
                "incidents_processed": result["incidents_processed"],
                "processing_time": result["processing_time"]
            }
            
            success_count += 1
            logging.info(f"Successfully processed {tc}: {result['incidents_processed']} incidents in {result['processing_time']:.2f} seconds")
            
        except Exception as e:
            logging.error(f"Error processing {tc}: {e}")
            results[tc] = {
                "success": False,
                "error": str(e)
            }
            failure_count += 1
        
        # Continue to next tech center regardless
        if tc != tech_centers_to_process[-1]:
            time.sleep(2)
    
    # Summarize results
    job_duration = time.time() - job_start_time
    job_summary = {
        "job_date": job_date,
        "tech_centers_processed": len(tech_centers_to_process),
        "success_count": success_count,
        "failure_count": failure_count,
        "job_duration_seconds": job_duration,
        "results": results
    }
    
    logging.info(f"Job completed in {job_duration:.2f} seconds: {success_count} successful, {failure_count} failed")
    
    # Save job summary
    summary_file = os.path.join(log_dir, f"prediction_summary_{job_date}.json")
    with open(summary_file, "w") as f:
        json.dump(job_summary, f, indent=2)
    
    # Save to blob storage
    if log_to_blob and blob_storage and blob_storage._container_client:
        blob_storage.upload_json(job_summary, f"prediction_jobs/summary_{job_date}.json")
    
    return job_summary

def main():
    """Main function for command-line usage"""
    args = parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level=log_level)
    
    # List available tech centers
    if args.list:
        tech_centers = get_available_tech_centers()
        print(f"Available Tech Centers ({len(tech_centers)}):")
        for i, tc in enumerate(tech_centers, 1):
            print(f"{i}. {tc}")
        return 0
    
    # Run prediction job
    try:
        result = run_prediction_job(
            tech_center=args.tech_center,
            all_centers=args.all,
            since_timestamp=args.since,
            until_timestamp=args.until,
            log_to_blob=not args.no_blob
        )
        
        if "error" in result:
            logging.error(f"Prediction job failed: {result['error']}")
            return 1
        
        logging.info(f"Prediction job completed successfully")
        return 0
        
    except Exception as e:
        logging.error(f"Error running prediction job: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())
