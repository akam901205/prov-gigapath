"""
GigaPath Tiered API - Robust & Optimized Architecture
Core Features:
- Tiered prediction system as primary architecture
- Comprehensive error handling and logging
- Optimized model caching and performance
- Clinical workflow-focused design
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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from preprocessing_pipeline import preprocessor, detect_image_scale
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost available")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("❌ XGBoost not available")

app = FastAPI(title="GigaPath Tiered Clinical API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    input: dict

# Global model cache - singleton pattern for efficiency
class ModelCache:
    def __init__(self):
        self.gigapath_encoder = None
        self.transform = None
        self.embeddings_cache = None
        
        # BreakHis binary models
        self.breakhis_lr = None
        self.breakhis_svm = None
        self.breakhis_xgb = None
        self.breakhis_label_encoder = None
        
        # BACH normal vs benign models
        self.normal_benign_lr = None
        self.normal_benign_svm = None
        self.normal_benign_xgb = None
        self.normal_benign_label_encoder = None
        
        # BACH invasive vs insitu models
        self.invasive_insitu_lr = None
        self.invasive_insitu_svm = None
        self.invasive_insitu_xgb = None
        self.invasive_insitu_label_encoder = None
        
        # Model loading states
        self.models_loaded = {
            'gigapath': False,
            'cache': False,
            'breakhis': False,
            'normal_benign': False,
            'invasive_insitu': False
        }

# Global cache instance
model_cache = ModelCache()

def convert_numpy_types(obj):
    """Convert numpy types to JSON-serializable types"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif hasattr(obj, 'item'):
        return obj.item()
    else:
        return obj

def load_gigapath_model():
    """Load GigaPath model with proper device management"""
    if not model_cache.models_loaded['gigapath']:
        print("🔥 Loading GigaPath model...")
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"🔥 Using device: {device}")
            
            # Load model and move to device
            model_cache.gigapath_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
            model_cache.gigapath_encoder = model_cache.gigapath_encoder.to(device)
            model_cache.gigapath_encoder.eval()
            
            model_cache.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            model_cache.models_loaded['gigapath'] = True
            print(f"✅ GigaPath model loaded successfully on {device}")
        except Exception as e:
            print(f"❌ GigaPath model loading failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    return model_cache.gigapath_encoder, model_cache.transform

def load_embeddings_cache():
    """Load embeddings cache with caching"""
    if not model_cache.models_loaded['cache']:
        print("🔥 Loading embeddings cache...")
        try:
            cache_path = "/workspace/embeddings_cache_L2_REPROCESSED.pkl"
            with open(cache_path, 'rb') as f:
                model_cache.embeddings_cache = pickle.load(f)
            
            model_cache.models_loaded['cache'] = True
            total_samples = len(model_cache.embeddings_cache['combined']['features'])
            print(f"✅ Embeddings cache loaded: {total_samples} samples")
        except Exception as e:
            print(f"❌ Embeddings cache loading failed: {e}")
            raise
    
    return model_cache.embeddings_cache

def load_breakhis_models():
    """Load BreakHis binary classification models"""
    if not model_cache.models_loaded['breakhis']:
        print("🔥 Loading BreakHis binary models (L2 retrained)...")
        try:
            with open('/workspace/breakhis_binary_model_L2_RETRAINED.pkl', 'rb') as f:
                data = pickle.load(f)
            
            model_cache.breakhis_lr = data['lr_model']
            model_cache.breakhis_svm = data['svm_model']
            model_cache.breakhis_xgb = data.get('xgb_model')
            model_cache.breakhis_label_encoder = data['label_encoder']
            
            model_cache.models_loaded['breakhis'] = True
            print(f"✅ BreakHis models loaded: LR={model_cache.breakhis_lr is not None}, SVM={model_cache.breakhis_svm is not None}, XGB={model_cache.breakhis_xgb is not None}")
        except Exception as e:
            print(f"❌ BreakHis models loading failed: {e}")
            # Continue without BreakHis models
    
    return model_cache.models_loaded['breakhis']

def load_normal_benign_models():
    """Load BACH normal vs benign models"""
    if not model_cache.models_loaded['normal_benign']:
        print("🟦 Loading BACH Normal vs Benign models (L2 retrained)...")
        try:
            with open('/workspace/bach_normal_benign_model_L2_RETRAINED.pkl', 'rb') as f:
                data = pickle.load(f)
            
            model_cache.normal_benign_lr = data['lr_model']
            model_cache.normal_benign_svm = data['svm_model']
            model_cache.normal_benign_xgb = data.get('xgb_model')
            model_cache.normal_benign_label_encoder = data['label_encoder']
            
            model_cache.models_loaded['normal_benign'] = True
            print(f"✅ Normal vs Benign models loaded: LR={model_cache.normal_benign_lr is not None}, SVM={model_cache.normal_benign_svm is not None}, XGB={model_cache.normal_benign_xgb is not None}")
        except Exception as e:
            print(f"❌ Normal vs Benign models loading failed: {e}")
            # Continue without these models
    
    return model_cache.models_loaded['normal_benign']

def load_invasive_insitu_models():
    """Load BACH invasive vs insitu models"""
    if not model_cache.models_loaded['invasive_insitu']:
        print("🟪 Loading BACH Invasive vs InSitu models (L2 retrained)...")
        try:
            with open('/workspace/bach_invasive_insitu_model_L2_RETRAINED.pkl', 'rb') as f:
                data = pickle.load(f)
            
            model_cache.invasive_insitu_lr = data['lr_model']
            model_cache.invasive_insitu_svm = data['svm_model']
            model_cache.invasive_insitu_xgb = data.get('xgb_model')
            model_cache.invasive_insitu_label_encoder = data['label_encoder']
            
            model_cache.models_loaded['invasive_insitu'] = True
            print(f"✅ Invasive vs InSitu models loaded: LR={model_cache.invasive_insitu_lr is not None}, SVM={model_cache.invasive_insitu_svm is not None}, XGB={model_cache.invasive_insitu_xgb is not None}")
        except Exception as e:
            print(f"❌ Invasive vs InSitu models loading failed: {e}")
            # Continue without these models
    
    return model_cache.models_loaded['invasive_insitu']

def predict_with_model(model, label_encoder, features, algorithm_name):
    """Generic prediction function for any model"""
    if not model or not label_encoder:
        return None
    
    try:
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        probabilities = model.predict_proba(features)[0]
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = label_encoder.classes_[predicted_class_idx]
        confidence = probabilities[predicted_class_idx]
        
        return {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(label_encoder.classes_, probabilities)
            },
            'algorithm': algorithm_name
        }
    except Exception as e:
        print(f"❌ Prediction failed for {algorithm_name}: {e}")
        return None

def run_breakhis_classification(features):
    """Run BreakHis binary classification"""
    print("🔬 Stage 1: BreakHis Binary Classification")
    
    if not load_breakhis_models():
        print("❌ BreakHis models not available")
        return None
    
    # Run all three algorithms
    lr_result = predict_with_model(model_cache.breakhis_lr, model_cache.breakhis_label_encoder, features, "Logistic Regression")
    svm_result = predict_with_model(model_cache.breakhis_svm, model_cache.breakhis_label_encoder, features, "SVM RBF")
    xgb_result = predict_with_model(model_cache.breakhis_xgb, model_cache.breakhis_label_encoder, features, "XGBoost") if model_cache.breakhis_xgb else None
    
    # Calculate consensus
    predictions = [r['predicted_class'] for r in [lr_result, svm_result, xgb_result] if r]
    malignant_votes = sum(1 for pred in predictions if pred == 'malignant')
    consensus = 'malignant' if malignant_votes >= len(predictions) / 2 else 'benign'
    
    print(f"🔬 BreakHis Results: {len(predictions)} classifiers, {malignant_votes} malignant votes → {consensus}")
    
    return {
        'consensus': consensus,
        'vote_breakdown': {'malignant': malignant_votes, 'benign': len(predictions) - malignant_votes},
        'total_classifiers': len(predictions),
        'classifiers': {
            'logistic_regression': lr_result,
            'svm_rbf': svm_result,
            'xgboost': xgb_result
        }
    }

def run_normal_benign_classification(features):
    """Run BACH normal vs benign classification"""
    print("🟦 Stage 2a: BACH Normal vs Benign Classification")
    
    if not load_normal_benign_models():
        print("❌ Normal vs Benign models not available")
        return None
    
    # Run all three algorithms
    lr_result = predict_with_model(model_cache.normal_benign_lr, model_cache.normal_benign_label_encoder, features, "LR Normal/Benign")
    svm_result = predict_with_model(model_cache.normal_benign_svm, model_cache.normal_benign_label_encoder, features, "SVM Normal/Benign")
    xgb_result = predict_with_model(model_cache.normal_benign_xgb, model_cache.normal_benign_label_encoder, features, "XGB Normal/Benign") if model_cache.normal_benign_xgb else None
    
    # Calculate consensus
    predictions = [r['predicted_class'] for r in [lr_result, svm_result, xgb_result] if r]
    normal_votes = sum(1 for pred in predictions if pred == 'normal')
    consensus = 'normal' if normal_votes >= len(predictions) / 2 else 'benign'
    
    print(f"🟦 Normal vs Benign Results: {len(predictions)} classifiers, {normal_votes} normal votes → {consensus}")
    
    return {
        'task': 'Normal vs Benign',
        'consensus': consensus,
        'vote_breakdown': {'normal': normal_votes, 'benign': len(predictions) - normal_votes},
        'total_classifiers': len(predictions),
        'classifiers': {
            'logistic_regression': lr_result,
            'svm_rbf': svm_result,
            'xgboost': xgb_result
        }
    }

def run_invasive_insitu_classification(features):
    """Run BACH invasive vs insitu classification"""
    print("🟪 Stage 2b: BACH Invasive vs InSitu Classification")
    
    if not load_invasive_insitu_models():
        print("❌ Invasive vs InSitu models not available")
        return None
    
    # Run all three algorithms
    lr_result = predict_with_model(model_cache.invasive_insitu_lr, model_cache.invasive_insitu_label_encoder, features, "LR Invasive/InSitu")
    svm_result = predict_with_model(model_cache.invasive_insitu_svm, model_cache.invasive_insitu_label_encoder, features, "SVM Invasive/InSitu")
    xgb_result = predict_with_model(model_cache.invasive_insitu_xgb, model_cache.invasive_insitu_label_encoder, features, "XGB Invasive/InSitu") if model_cache.invasive_insitu_xgb else None
    
    # Calculate consensus
    predictions = [r['predicted_class'] for r in [lr_result, svm_result, xgb_result] if r]
    invasive_votes = sum(1 for pred in predictions if pred == 'invasive')
    consensus = 'invasive' if invasive_votes >= len(predictions) / 2 else 'insitu'
    
    print(f"🟪 Invasive vs InSitu Results: {len(predictions)} classifiers, {invasive_votes} invasive votes → {consensus}")
    
    return {
        'task': 'Invasive vs InSitu', 
        'consensus': consensus,
        'vote_breakdown': {'invasive': invasive_votes, 'insitu': len(predictions) - invasive_votes},
        'total_classifiers': len(predictions),
        'classifiers': {
            'logistic_regression': lr_result,
            'svm_rbf': svm_result,
            'xgboost': xgb_result
        }
    }

def run_tiered_prediction(features):
    """
    Core Tiered Prediction System
    
    Stage 1: BreakHis Binary (malignant vs benign)
    Stage 2a: If benign → Normal vs Benign  
    Stage 2b: If malignant → Invasive vs InSitu
    """
    print("\n" + "="*60)
    print("🏥 TIERED CLINICAL PREDICTION SYSTEM")
    print("="*60)
    
    # Normalize features for all classifiers
    l2_features = normalize([features], norm='l2')[0]
    
    # Stage 1: BreakHis Binary Classification
    stage1_results = run_breakhis_classification(l2_features)
    
    if not stage1_results:
        print("❌ Stage 1 failed - cannot proceed with tiered prediction")
        return None
    
    breakhis_consensus = stage1_results['consensus']
    print(f"\n🎯 Stage 1 Complete: {breakhis_consensus.upper()}")
    
    # Stage 2: Deploy appropriate specialized classifier
    stage2_results = None
    if breakhis_consensus == 'benign':
        print("🟦 Deploying Normal vs Benign pathway...")
        stage2_results = run_normal_benign_classification(l2_features)
    else:
        print("🟪 Deploying Invasive vs InSitu pathway...")
        stage2_results = run_invasive_insitu_classification(l2_features)
    
    # Final prediction
    final_prediction = stage2_results['consensus'] if stage2_results else breakhis_consensus
    clinical_pathway = f"BreakHis → {'Normal/Benign' if breakhis_consensus == 'benign' else 'Invasive/InSitu'}"
    
    print(f"\n🎯 TIERED PREDICTION COMPLETE")
    print(f"   Clinical Pathway: {clinical_pathway}")
    print(f"   Final Prediction: {final_prediction.upper()}")
    print("="*60 + "\n")
    
    return {
        'stage_1_breakhis': stage1_results,
        'stage_2_bach_specialized': stage2_results,
        'tiered_final_prediction': final_prediction,
        'clinical_pathway': clinical_pathway,
        'system_status': 'fully_operational'
    }

@app.get("/")
async def root():
    return {
        "status": "online", 
        "service": "GigaPath Tiered Clinical API", 
        "version": "3.0.0",
        "message": "Tiered prediction system ready"
    }

@app.post("/api/single-image-analysis")
async def tiered_single_image_analysis(request: AnalyzeRequest):
    """
    Tiered Single Image Analysis
    
    Primary: Tiered clinical prediction system
    Secondary: Additional analysis data
    """
    try:
        print(f"\n{'='*80}")
        print(f"🚀 NEW TIERED ANALYSIS REQUEST")
        print(f"{'='*80}")
        
        input_data = request.input
        
        # Validate input
        if "image_base64" not in input_data or not input_data["image_base64"]:
            raise HTTPException(status_code=400, detail="image_base64 is required")
        
        # Complete 7-Step Preprocessing Pipeline
        print("🔬 Starting complete preprocessing pipeline...")
        image_data = base64.b64decode(input_data["image_base64"])
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Get filename if provided (for scale detection)
        filename = input_data.get("filename", "uploaded_image.jpg")
        
        # Load GigaPath model
        encoder, _ = load_gigapath_model()
        
        # Run complete preprocessing pipeline
        l2_features, preprocessing_metadata = preprocessor.complete_pipeline(
            image=image,
            gigapath_model=encoder,
            source_um_per_pixel=detect_image_scale(image, filename),
            apply_tissue_mask=True,
            apply_stain_norm=True
        )
        
        print(f"✅ Complete preprocessing pipeline executed: {len(preprocessing_metadata['pipeline_steps'])} steps")
        
        # Core Tiered Prediction System  
        tiered_results = run_tiered_prediction(l2_features)
        
        if not tiered_results:
            raise ValueError("Tiered prediction system failed")
        
        # Build optimized response focused on tiered results
        response = {
            "status": "success",
            "tiered_prediction": tiered_results,
            "features": {
                "encoder_type": "tile",
                "features_shape": list(l2_features.shape),
                "feature_dimension": l2_features.shape[0],
                "normalization": "l2"
            },
            "preprocessing": preprocessing_metadata,
            "system_info": {
                "api_version": "3.0.0",
                "prediction_method": "tiered_clinical_system_with_preprocessing",
                "models_loaded": model_cache.models_loaded,
                "pipeline_steps": len(preprocessing_metadata['pipeline_steps'])
            }
        }
        
        print(f"✅ Response built successfully")
        print(f"🎯 Final Tiered Prediction: {tiered_results['tiered_final_prediction'].upper()}")
        print(f"🏥 Clinical Pathway: {tiered_results['clinical_pathway']}")
        
        return convert_numpy_types(response)
        
    except Exception as e:
        print(f"❌ Tiered analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e), "system": "tiered_prediction"}

if __name__ == "__main__":
    print("🚀 Starting GigaPath Tiered Clinical API...")
    uvicorn.run(app, host="0.0.0.0", port=8008)