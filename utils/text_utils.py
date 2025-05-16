"""
Text processing utilities for the topic clustering package.
"""

import re
import pandas as pd

def get_safe_text(row, column):
    """Helper function to safely get text from a DataFrame row column"""
    return str(row[column]) if pd.notna(row[column]) else ""

def estimate_tokens(text):
    """
    Estimate token count for a given text using a simple approximation.
    
    Args:
        text: Input text string
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    # OpenAI models generally use ~4 chars per token for English text
    return len(text) // 4 + 1  # Add 1 as a safety margin

def clean_text_for_summary(text):
    """
    Clean text by normalizing whitespace and removing special characters
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Replace newlines and tabs with spaces
    text = re.sub(r'[\n\r\t]+', ' ', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove any non-printable characters
    text = ''.join(c for c in text if c.isprintable() or c.isspace())
    
    return text.strip()

def normalize_business_service(text):
    """
    Normalize business service names by removing dashes, underscores and standardizing spacing
    
    Args:
        text: Business service name
        
    Returns:
        Normalized business service name
    """
    if not text:
        return ""
    
    # Remove trailing " - PROD", " - DEV", etc.
    text = re.sub(r'\s*[-_]\s*(PROD|DEV|TEST|UAT|QA)$', '', text)
    
    # Replace dashes and underscores with spaces
    text = re.sub(r'[-_]+', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def remove_emails(text):
    """Remove email addresses from text"""
    if not text:
        return ""
    return re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)
