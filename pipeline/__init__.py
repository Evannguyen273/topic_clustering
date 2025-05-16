"""
Pipeline modules for the topic clustering package.
"""

from .stage1_embeddings import generate_embeddings_pipeline
from .stage2_clustering import train_hdbscan_pipeline
from .stage3_analysis import analyze_clusters_pipeline
from .stage4_save import save_results_pipeline

from .pipeline import run_modular_pipeline, run_full_pipeline

__all__ = [
    # Pipeline stages
    'generate_embeddings_pipeline',
    'train_hdbscan_pipeline',
    'analyze_clusters_pipeline',
    'save_results_pipeline',
    
    # Full pipelines
    'run_modular_pipeline',
    'run_full_pipeline'
]
