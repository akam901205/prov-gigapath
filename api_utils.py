"""
API Utilities - Helper functions for data processing and response formatting
"""

import numpy as np
import json

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)

def convert_numpy_types(obj):
    """Recursively convert numpy types to JSON-serializable types"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif hasattr(obj, 'item'):  # Handle numpy scalars
        return obj.item()
    elif str(type(obj)).startswith('<class \'numpy.'):  # Catch any numpy types
        try:
            return obj.item() if hasattr(obj, 'item') else float(obj)
        except (TypeError, ValueError):
            return str(obj)
    else:
        return obj

def create_fallback_response(predicted_class, confidence, class_names, status="Model not available"):
    """Create a standardized fallback response"""
    if len(class_names) == 2:  # Binary classification
        probabilities = {class_names[0]: 0.5, class_names[1]: 0.5}
    else:  # Multi-class
        prob_each = 1.0 / len(class_names)
        probabilities = {cls: prob_each for cls in class_names}
    
    return {
        "predicted_class": predicted_class,
        "confidence": float(confidence),
        "probabilities": probabilities,
        "status": status
    }

def validate_features(features):
    """Validate and normalize feature input"""
    if features is None:
        raise ValueError("Features cannot be None")
    
    features = np.array(features)
    if len(features.shape) == 1:
        features = features.reshape(1, -1)
    
    if features.shape[1] != 1536:
        raise ValueError(f"Expected 1536 features, got {features.shape[1]}")
    
    return features

def normalize_l2(features):
    """Apply L2 normalization safely"""
    from sklearn.preprocessing import normalize
    return normalize([features], norm='l2')[0]