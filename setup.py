from setuptools import setup, find_packages

setup(
    name="topic_clustering",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "umap-learn",
        "hdbscan",
        "openai",
        "azure-storage-blob",
        "azure-identity",
        "google-cloud-bigquery",
        "tqdm",
        "python-dotenv"
    ]
)
