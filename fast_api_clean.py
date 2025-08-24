"""
GigaPath FastAPI - Clean, Modular Architecture
Single Image Analysis with BACH and BreakHis Classifiers
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import base64
import io
import pickle
import torch
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.decomposition import PCA

# Import our modular components
from classifier_manager import classifier_manager
from api_utils import convert_numpy_types, create_fallback_response, validate_features, normalize_l2
from correlation_utils import calculate_correlation_predictions

app = FastAPI(title="GigaPath API - Clean Architecture", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
TILE_ENCODER = None
EMBEDDINGS_CACHE = None
TRANSFORM = None

class AnalyzeRequest(BaseModel):
    input: dict

def load_model_on_demand():
    """Load GigaPath model on demand"""
    global TILE_ENCODER, TRANSFORM
    if TILE_ENCODER is None:
        print("Loading GigaPath model on-demand...")
        model_name = "hf_hub:prov-gigapath/prov-gigapath"
        TILE_ENCODER = timm.create_model(model_name, pretrained=True)
        TILE_ENCODER.eval()
        
        TRANSFORM = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("Model loaded successfully!")
    return TILE_ENCODER, TRANSFORM

def load_cache_on_demand():
    """Load embeddings cache on demand"""
    global EMBEDDINGS_CACHE
    if EMBEDDINGS_CACHE is None:
        print("Loading cache on-demand...")
        
        # Use BACH 4-CLASS cache (optimized supervision)
        cache_path = "/workspace/embeddings_cache_4_CLUSTERS_FIXED_TSNE.pkl"
        if os.path.exists(cache_path):
            print(f"🎯 Loading BACH 4-CLASS cache")
            with open(cache_path, 'rb') as f:
                EMBEDDINGS_CACHE = pickle.load(f)
            print(f"✅ Cache loaded: {len(EMBEDDINGS_CACHE['combined']['features'])} images")
        else:
            raise FileNotFoundError(f"Cache not found: {cache_path}")
    return EMBEDDINGS_CACHE

def extract_gigapath_features(image_data):
    """Extract features from image using GigaPath"""
    # Load model and process image
    encoder, transform = load_model_on_demand()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = encoder(input_tensor)
    
    return output.cpu().numpy().flatten()

def run_all_predictions(features):
    """Run all classifier predictions with proper error isolation"""
    # Normalize features for classifiers
    l2_features = normalize_l2(features)
    
    # BACH Predictions (4-class)
    bach_results = {
        'logistic_regression': classifier_manager.predict_bach_lr(l2_features),
        'svm_rbf': classifier_manager.predict_bach_svm(l2_features),
        'xgboost': classifier_manager.predict_bach_xgb(l2_features)
    }
    
    # BreakHis Predictions (binary)
    breakhis_results = {
        'logistic_regression': classifier_manager.predict_breakhis_lr(l2_features),
        'svm_rbf': classifier_manager.predict_breakhis_svm(l2_features),
        'xgboost': classifier_manager.predict_breakhis_xgb(l2_features)
    }
    
    return bach_results, breakhis_results

def build_classifier_response(results, fallback_class, fallback_confidence, class_names):
    """Build classifier response with fallbacks"""
    response = {}
    
    for alg_name, result in results.items():
        if result:
            response[alg_name] = result
        else:
            response[alg_name] = create_fallback_response(
                fallback_class, fallback_confidence, class_names, 
                f"{alg_name.replace('_', ' ').title()} not available"
            )
    
    return response

@app.get("/")
async def root():
    return {"status": "online", "service": "GigaPath Clean API", "message": "Ready for requests"}

@app.post("/api/single-image-analysis")
async def single_image_analysis_clean(request: AnalyzeRequest):
    """
    Clean Single Image Analysis
    Robust architecture with isolated classifier predictions
    """
    try:
        input_data = request.input
        
        # Get image data
        if "image_base64" not in input_data or not input_data["image_base64"]:
            raise HTTPException(status_code=400, detail="image_base64 is required")
        
        image_data = base64.b64decode(input_data["image_base64"])
        
        # Extract GigaPath features
        print("🔥 Extracting GigaPath features...")
        features = extract_gigapath_features(image_data)
        print(f"🔥 Features extracted: shape {features.shape}")
        
        # Run all predictions
        print("🔥 Running classifier predictions...")
        bach_results, breakhis_results = run_all_predictions(features)
        
        # Get auxiliary data
        cache = load_cache_on_demand()
        bach_roc_plot = classifier_manager.generate_roc_plot('bach')
        breakhis_roc_plot = classifier_manager.generate_roc_plot('breakhis')
        bach_model_info = classifier_manager.get_model_info('bach')
        breakhis_model_info = classifier_manager.get_model_info('breakhis')
        
        # Determine consensus prediction
        valid_bach_predictions = [r['predicted_class'] for r in bach_results.values() if r]
        final_prediction = valid_bach_predictions[0] if valid_bach_predictions else 'normal'
        
        # Calculate ensemble confidence
        valid_confidences = [r['confidence'] for r in bach_results.values() if r]
        ensemble_confidence = np.mean(valid_confidences) if valid_confidences else 0.5
        
        print(f"🎯 FINAL RESULTS: {len(valid_bach_predictions)} valid BACH predictions, consensus: {final_prediction}")
        
        # Build response
        result = {
            "status": "success",
            "gigapath_verdict": {
                **build_classifier_response(
                    bach_results, final_prediction, ensemble_confidence,
                    ['normal', 'benign', 'insitu', 'invasive']
                ),
                "breakhis_binary": build_classifier_response(
                    breakhis_results, 'benign', 0.5,
                    ['benign', 'malignant']
                ),
                "roc_plot_base64": bach_roc_plot,
                "model_info": bach_model_info
            },
            "verdict": {
                "final_prediction": final_prediction,
                "confidence": float(ensemble_confidence),
                "ensemble_size": len(valid_bach_predictions),
                "recommendation": f"Ensemble of {len(valid_bach_predictions)} classifiers with {ensemble_confidence:.1%} confidence"
            },
            "features": {
                "encoder_type": "tile",
                "features_shape": list(features.shape),
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            }
        }
        
        # Convert numpy types and return
        return convert_numpy_types(result)
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)