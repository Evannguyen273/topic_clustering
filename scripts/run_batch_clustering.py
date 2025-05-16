"""
Script to run clustering for multiple tech centers sequentially.
"""

import os
import time
import logging
import argparse
import json
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

# Adjust path to ensure we can import the package
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from topic_clustering.pipeline.pipeline import run_full_pipeline
from topic_clustering.utils.bigquery_utils import run_query
from topic_clustering.utils.logging_utils import setup_logging
from topic_clustering.utils.azure_utils import blob_storage

# Hardcoded list of all tech centers
all_tech_center_list = [
    "BT-TC-Product Development & Engineering",

]


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run clustering for multiple tech centers")
    parser.add_argument("--tech-centers", nargs="+", help="List of tech centers to process")
    parser.add_argument("--file", help="JSON file containing tech centers list")
    parser.add_argument("--all", action="store_true", help="Process all available tech centers")
    parser.add_argument("--start-date", default="2024-01-01", help="Start date for incidents (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date for incidents (YYYY-MM-DD, default: today)")
    parser.add_argument("--limit", type=int, help="Limit number of incidents per tech center")
    parser.add_argument("--cluster-size", type=int, default=25, help="Minimum cluster size")
    parser.add_argument("--output-dir", default="batch_results", help="Output directory for results")
    parser.add_argument("--skip-existing", action="store_true", help="Skip tech centers with existing models")
    parser.add_argument("--dry-run", action="store_true", help="Print queries without running clustering")
    return parser.parse_args()


def get_all_tech_centers() -> List[str]:
    """Get all available tech centers from BigQuery"""
    # First check if we have a hardcoded list
    if all_tech_center_list:
        logging.info(f"Using hardcoded list of {len(all_tech_center_list)} tech centers")
        return all_tech_center_list
        
    # Fall back to querying BigQuery if no hardcoded list
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
    
    return []


def generate_tech_center_query(tech_center: str, start_date: str, end_date: Optional[str] = None, 
                             limit: Optional[int] = None) -> str:
    """Generate query for fetching incidents for a specific tech center"""
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
        
    query = f"""
    SELECT t3.number, t3.priority, t3.sys_created_on, t1.TeamDepartment, t1.Area, t1.Unit, 
           t1.TechCenter, t3.short_description, t3.close_code, t3.vendor, t3.assignment_group, 
           t3.state, t3.business_service, t3.close_notes, t3.work_notes, 
           t3.contact_type as chanel, t3.description
    FROM `enterprise-dashboardnp-cd35.bigquery_datasets_hone_srv_dev.oa_snow_incident_mgmt_srv_dev` t3
    JOIN `enterprise-dashboardnp-cd35.bigquery_datasets_spoke_oa_dev.Team_services` t1
    ON t3.business_service = t1.Services
    WHERE t1.TechCenter = '{tech_center}' 
    AND t3.sys_created_on >= '{start_date}'
    AND t3.sys_created_on < '{end_date}'
    ORDER BY t3.sys_created_on
    """
    
    if limit:
        query += f"\nLIMIT {limit}"
    
    return query


def tech_center_has_model(tech_center: str) -> bool:
    """Check if model files exist for this tech center"""
    if not blob_storage or not blob_storage._container_client:
        return False
        
    # Sanitize tech center name for use in paths
    tech_center_path = tech_center.replace(" ", "_").replace("-", "_").replace("/", "_")
    
    # Check for key model files - tech_center_path IS the dataset_name
    model_files = [
        f"{tech_center_path}/clustering/hdbscan_clusterer.pkl",
        f"{tech_center_path}/analysis/final_df.parquet"
    ]
    
    # Log paths being checked
    logging.debug(f"Checking for model files: {model_files}")
    
    # Check if any of the files exist (if any exists, we have a model)
    exists = any(blob_storage.blob_exists(path) for path in model_files)
    if exists:
        logging.info(f"Found existing model files for {tech_center}")
    
    return exists


def run_clustering_for_tech_center(tech_center: str, start_date: str, end_date: Optional[str] = None, 
                                 limit: Optional[int] = None, cluster_params: Dict[str, Any] = None,
                                 dry_run: bool = False) -> Dict[str, Any]:
    """Run clustering for a single tech center"""
    tech_center_path = tech_center.replace(" ", "_").replace("-", "_").replace("/", "_")
    
    # Generate query
    query = generate_tech_center_query(tech_center, start_date, end_date, limit)
    
    # Log query
    logging.info(f"Query for {tech_center}:\n{query}")
    
    if dry_run:
        return {
            "tech_center": tech_center, 
            "status": "dry_run",
            "query": query
        }
    
    # Run clustering pipeline
    try:
        start_time = time.time()
        results = run_full_pipeline(
            dataset_name=tech_center_path,
            input_query=query,
            cluster_params=cluster_params
        )
        runtime = time.time() - start_time
        
        return {
            "tech_center": tech_center,
            "status": "success",
            "runtime_seconds": runtime,
            "results": results
        }
    
    except Exception as e:
        logging.error(f"Error clustering {tech_center}: {e}", exc_info=True)
        return {
            "tech_center": tech_center,
            "status": "error",
            "error": str(e)
        }


def main():
    """Main function"""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    args = parse_args()
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(args.output_dir, f"batch_clustering_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = setup_logging(log_file=log_file, log_to_console=True)
    
    # Get tech centers to process
    tech_centers = []
    
    if args.tech_centers:
        tech_centers = args.tech_centers
    elif args.file:
        try:
            with open(args.file, 'r') as f:
                data = json.load(f)
                tech_centers = data.get('tech_centers', [])
        except Exception as e:
            logging.error(f"Error loading tech centers from file: {e}")
            return 1
    elif args.all:
        tech_centers = get_all_tech_centers()
    else:
        logging.error("No tech centers specified. Use --tech-centers, --file or --all")
        return 1
    
    if not tech_centers:
        logging.error("No tech centers found to process")
        return 1
    
    logging.info(f"Processing {len(tech_centers)} tech centers: {', '.join(tech_centers)}")
    
    # Set up clustering parameters
    cluster_params = {
        "min_cluster_size": args.cluster_size,
        "min_samples": max(5, args.cluster_size // 5)
    }
    
    # Ensure end date is in the correct format
    end_date = args.end_date
    if end_date:
        try:
            # Convert to ensure format is valid
            datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            logging.error("Invalid end date format. Use YYYY-MM-DD")
            return 1
    
    # Run clustering for each tech center
    results = []
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    for i, tc in enumerate(tech_centers, 1):
        logging.info(f"[{i}/{len(tech_centers)}] Processing tech center: {tc}")
        
        # Check if model already exists
        if args.skip_existing and tech_center_has_model(tc):
            logging.info(f"Skipping {tc} - model already exists")
            results.append({
                "tech_center": tc,
                "status": "skipped",
                "reason": "model_exists"
            })
            skipped_count += 1
            continue
        
        # Run clustering
        result = run_clustering_for_tech_center(
            tech_center=tc,
            start_date=args.start_date,
            end_date=end_date,
            limit=args.limit,
            cluster_params=cluster_params,
            dry_run=args.dry_run
        )
        
        results.append(result)
        
        if result["status"] == "success":
            success_count += 1
        elif result["status"] == "error":
            error_count += 1
        
        # Save incremental results
        results_file = os.path.join(args.output_dir, "batch_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "tech_centers_processed": i,
                "total_tech_centers": len(tech_centers),
                "success_count": success_count,
                "error_count": error_count,
                "skipped_count": skipped_count,
                "results": results
            }, f, indent=2)
    
    # Final summary
    logging.info("====== BATCH CLUSTERING COMPLETE ======")
    logging.info(f"Total tech centers: {len(tech_centers)}")
    logging.info(f"Successful: {success_count}")
    logging.info(f"Errors: {error_count}")
    logging.info(f"Skipped: {skipped_count}")
    logging.info(f"Results saved to {results_file}")
    
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    exit(main())
