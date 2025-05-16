"""
JSON utilities for the topic clustering package.
"""

import json
import re
import logging
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)

def robust_json_parser(json_string):
    """
    Robust JSON parser optimized for Azure OpenAI responses with enhanced error recovery.
    Handles unterminated strings, delimiter issues, and malformed JSON with multiple fallback strategies.
    
    Args:
        json_string: JSON string that might be malformed
        
    Returns:
        Parsed JSON object or fallback dictionary
    """
    try:
        # First try standard parsing
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        position = e.pos
        error_type = str(e)
        logging.warning(f"JSON parsing error at position {position}: {error_type}")
        
        # Initialize fixed string
        fixed_string = json_string
        
        # PHASE 1: STRING TERMINATION & ESCAPING FIX
        if "Unterminated string" in error_type or "Invalid \\escape" in error_type:
            logging.info(f"Fixing unterminated string or escape sequence")
            # ... existing code for string termination fix ...
            # This is a complex section that handles string termination issues
            pass
        
        # PHASE 2: STRUCTURAL REPAIRS
        # Fix missing quotes around keys
        fixed_string = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', fixed_string)
        
        # Fix missing commas between objects and array elements
        fixed_string = re.sub(r'}\s*{', '},{', fixed_string)
        fixed_string = re.sub(r'"\s*{', '",{', fixed_string)
        fixed_string = re.sub(r'}\s*"', '},"', fixed_string)
        fixed_string = re.sub(r']\s*\[', '],[', fixed_string)
        fixed_string = re.sub(r']\s*{', '],{', fixed_string)
        fixed_string = re.sub(r'}\s*\[', '},[', fixed_string)
        
        # ... existing code for more structural repairs ...
        
        try:
            return json.loads(fixed_string)
        except json.JSONDecodeError:
            # Continue to extraction strategies
            pass
        
        # PHASE 3: EXTRACTION STRATEGIES
        logging.info("Attempting extraction strategies")
        
        # Strategy 1: Extract the outermost JSON object
        try:
            start = fixed_string.find('{')
            end = fixed_string.rfind('}')
            if start != -1 and end != -1 and start < end:
                extracted = fixed_string[start:end+1]
                logging.info(f"Extracting outermost JSON object from position {start} to {end}")
                return json.loads(extracted)
        except Exception:
            pass
        
        # ... other extraction strategies ...
        
        # PHASE 4: LAST RESORT STRATEGIES
        logging.warning("All JSON repair strategies failed, attempting last resort methods")
        
        # Create minimal return object with any data we can extract
        try:
            # Regex for key-value patterns
            kv_pattern = r'"([^"]+)":\s*"([^"]+)"'
            kv_matches = re.findall(kv_pattern, fixed_string)
            
            if kv_matches:
                logging.info(f"Extracted {len(kv_matches)} key-value pairs as fallback")
                result = {k: v for k, v in kv_matches}
                return result
            else:
                logging.error("Could not extract any data from malformed JSON")
                return {}
        except Exception as final_error:
            logging.error(f"All JSON recovery methods failed: {final_error}")
            return {}

def validate_json_response(json_data, required_keys=None, schema=None):
    """
    Validate that a JSON response matches expected structure
    
    Args:
        json_data: Parsed JSON data to validate
        required_keys: List of top-level keys that must exist
        schema: Optional JSON schema for detailed validation
        
    Returns:
        (bool, str): Tuple of (is_valid, error_message)
    """
    # Check for empty response
    if not json_data:
        return False, "Empty JSON response"
        
    # Check required keys
    if required_keys:
        missing_keys = [key for key in required_keys if key not in json_data]
        if missing_keys:
            return False, f"Missing required keys: {missing_keys}"
    
    # Advanced schema validation if jsonschema module is available
    if schema:
        try:
            import jsonschema
            jsonschema.validate(instance=json_data, schema=schema)
        except ImportError:
            logging.warning("jsonschema module not available, skipping schema validation")
        except jsonschema.exceptions.ValidationError as e:
            return False, f"Schema validation error: {e.message}"
    
    return True, "Valid JSON"

def save_json(data, filepath, encoder=NumpyEncoder):
    """Save data to a JSON file with proper encoding"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, cls=encoder)
