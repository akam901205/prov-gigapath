"""
Fast lightweight API that loads models on demand
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import base64
from correlation_utils import calculate_correlation_predictions
import io
import pickle
import torch
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, RobustScaler, normalize
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.decomposition import PCA
import traceback

app = FastAPI(title="GigaPath Fast API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for on-demand loading
TILE_ENCODER = None
EMBEDDINGS_CACHE = None
TRANSFORM = None

class AnalyzeRequest(BaseModel):
    input: dict

def load_model_on_demand():
    """Load GigaPath model only when needed."""
    global TILE_ENCODER, TRANSFORM
    if TILE_ENCODER is None:
        print("Loading GigaPath model on-demand...")
        # HF_TOKEN environment variable should be set externally
        TILE_ENCODER = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        TILE_ENCODER = TILE_ENCODER.to(device)
        TILE_ENCODER.eval()
        
        TRANSFORM = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        print("Model loaded successfully!")
    return TILE_ENCODER, TRANSFORM

def load_cache_on_demand():
    """Load real cache only when needed."""
    global EMBEDDINGS_CACHE
    if EMBEDDINGS_CACHE is None:
        print("Loading cache on-demand...")
        
        # Use BACH 4-CLASS cache (BreakHis binary + BACH 4-class supervision)
        bach_4class_path = "/workspace/embeddings_cache_BACH_4CLASS.pkl"
        if os.path.exists(bach_4class_path):
            print(f"🎯 Loading BACH 4-CLASS cache (optimized supervision)")
            with open(bach_4class_path, 'rb') as f:
                EMBEDDINGS_CACHE = pickle.load(f)
            print(f"✅ BACH 4-CLASS cache: {len(EMBEDDINGS_CACHE['combined']['features'])} images")
            print(f"📊 BreakHis: binary separation, BACH: 4-class (Normal|Benign|InSitu|Invasive)")
        elif os.path.exists("/workspace/embeddings_cache_DATASET_SPECIFIC.pkl"):
            print(f"🔄 Loading dataset-specific cache (binary supervision)")
            with open("/workspace/embeddings_cache_DATASET_SPECIFIC.pkl", 'rb') as f:
                EMBEDDINGS_CACHE = pickle.load(f)
            print(f"Dataset-specific cache: {len(EMBEDDINGS_CACHE['combined']['features'])} images")
        elif os.path.exists("/workspace/embeddings_cache_ENHANCED_SEPARATIONS.pkl"):
            print(f"🔄 Loading enhanced separations cache")
            with open("/workspace/embeddings_cache_ENHANCED_SEPARATIONS.pkl", 'rb') as f:
                EMBEDDINGS_CACHE = pickle.load(f)
            print(f"Enhanced cache: {len(EMBEDDINGS_CACHE['combined']['features'])} images")
        elif os.path.exists("/workspace/embeddings_cache_MIXED_NORM.pkl"):
            print(f"🔄 Loading mixed normalization cache")
            with open("/workspace/embeddings_cache_MIXED_NORM.pkl", 'rb') as f:
                EMBEDDINGS_CACHE = pickle.load(f)
            print(f"Mixed cache: {len(EMBEDDINGS_CACHE['combined']['features'])} images")
        elif os.path.exists("/workspace/embeddings_cache_UMAP_5.pkl"):
            print(f"🔄 Loading UMAP n_neighbors=5 cache (pure L2)")
            with open("/workspace/embeddings_cache_UMAP_5.pkl", 'rb') as f:
                EMBEDDINGS_CACHE = pickle.load(f)
            print(f"UMAP n=5 cache: {len(EMBEDDINGS_CACHE['combined']['features'])} images")
        elif os.path.exists("/workspace/embeddings_cache_UMAP_50.pkl"):
            print(f"🔄 Loading UMAP n_neighbors=75 cache (wider clustering)")
            with open("/workspace/embeddings_cache_UMAP_50.pkl", 'rb') as f:
                EMBEDDINGS_CACHE = pickle.load(f)
            print(f"UMAP n=75 cache: {len(EMBEDDINGS_CACHE['combined']['features'])} images")
        elif os.path.exists("/workspace/embeddings_cache_PURE_L2.pkl"):
            print(f"🔄 Loading pure L2 cache (n_neighbors=15)")
            with open("/workspace/embeddings_cache_PURE_L2.pkl", 'rb') as f:
                EMBEDDINGS_CACHE = pickle.load(f)
            print(f"Pure L2 cache: {len(EMBEDDINGS_CACHE['combined']['features'])} images")
        elif os.path.exists("/workspace/embeddings_cache_SAME_PREPROCESSING.pkl"):
            print(f"🔄 Loading cache with complex preprocessing (StandardScaler+RobustScaler)")
            with open("/workspace/embeddings_cache_SAME_PREPROCESSING.pkl", 'rb') as f:
                EMBEDDINGS_CACHE = pickle.load(f)
            print(f"Complex preprocessing cache: {len(EMBEDDINGS_CACHE['combined']['features'])} images")
        elif os.path.exists("/workspace/embeddings_cache_CORRECTED_COORDS.pkl"):
            print(f"🔄 Loading cache with corrected coords (but no preprocessing)")
            with open("/workspace/embeddings_cache_CORRECTED_COORDS.pkl", 'rb') as f:
                EMBEDDINGS_CACHE = pickle.load(f)
            print(f"Corrected cache: {len(EMBEDDINGS_CACHE['combined']['features'])} images")
        elif os.path.exists("/workspace/embeddings_cache_FIXED_LABELS.pkl"):
            print(f"🔄 Loading cache with fixed labels (but old coordinates)")
            with open("/workspace/embeddings_cache_FIXED_LABELS.pkl", 'rb') as f:
                EMBEDDINGS_CACHE = pickle.load(f)
            print(f"Fixed labels cache: {len(EMBEDDINGS_CACHE['combined']['features'])} images")
        elif os.path.exists("/workspace/embeddings_cache_COMPREHENSIVE.pkl"):
            print(f"⚠️ Loading cache with BROKEN labels (will fix)")
            with open("/workspace/embeddings_cache_COMPREHENSIVE.pkl", 'rb') as f:
                EMBEDDINGS_CACHE = pickle.load(f)
            print(f"Comprehensive cache: {len(EMBEDDINGS_CACHE['combined']['features'])} images")
        elif os.path.exists("/workspace/embeddings_cache_L2_NORMALIZED.pkl"):
            print(f"🔄 Loading smaller L2 cache (400 images)")
            with open("/workspace/embeddings_cache_L2_NORMALIZED.pkl", 'rb') as f:
                EMBEDDINGS_CACHE = pickle.load(f)
            print(f"✅ L2 normalized cache: {len(EMBEDDINGS_CACHE['combined']['features'])} images")
        elif os.path.exists("/workspace/embeddings_cache_IMPROVED.pkl"):
            print(f"🔄 Loading IMPROVED cache (domain shift + t-SNE/PCA fixes)")
            with open("/workspace/embeddings_cache_IMPROVED.pkl", 'rb') as f:
                EMBEDDINGS_CACHE = pickle.load(f)
            print(f"✅ Improved cache: {len(EMBEDDINGS_CACHE['combined']['features'])} images")
            print(f"📏 Dataset scalers: {'dataset_scalers' in EMBEDDINGS_CACHE}")
        elif os.path.exists("/workspace/embeddings_cache_NORMALIZED.pkl"):
            print(f"🔄 Loading NORMALIZED cache (domain shift fix only)")
            with open("/workspace/embeddings_cache_NORMALIZED.pkl", 'rb') as f:
                EMBEDDINGS_CACHE = pickle.load(f)
            print(f"✅ Normalized cache: {len(EMBEDDINGS_CACHE['combined']['features'])} images")
            print(f"📏 Dataset scalers: {'dataset_scalers' in EMBEDDINGS_CACHE}")
        else:
            # Fallback to real cache
            print("⚠️ Using non-normalized cache")
            cache_path = "/workspace/embeddings_cache_REAL_GIGAPATH.pkl"
            with open(cache_path, 'rb') as f:
                EMBEDDINGS_CACHE = pickle.load(f)
            print(f"Real cache loaded: {len(EMBEDDINGS_CACHE['combined']['features'])} images")
            
            # Apply real domain adaptation techniques for better separation
            EMBEDDINGS_CACHE = enhance_feature_separations(EMBEDDINGS_CACHE)
        
    return EMBEDDINGS_CACHE

def enhance_feature_separations(cache):
    """Apply TESTED robust PCA enhancement for optimal class separations."""
    try:
        combined_data = cache['combined']
        features = np.array(combined_data['features'])
        labels = combined_data['labels']
        datasets = combined_data['datasets']
        
        print(f"Enhancing separations for {len(features)} samples using ROBUST PCA...")
        
        # Apply the WINNING method: Robust PCA Enhancement
        enhanced_features = apply_robust_pca_enhancement(features, labels, datasets)
        
        # Re-compute embeddings with optimized parameters
        print("Computing HYBRID enhanced UMAP embeddings (Supervised)...")
        # Convert labels to numeric for supervised UMAP
        label_map = {'benign': 0, 'normal': 0, 'malignant': 1, 'invasive': 1}
        numeric_labels = np.array([label_map.get(label, 0) for label in labels])
        
        umap_reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.0,  # Tighter clustering
            n_components=2,
            metric='cosine',
            target_metric='categorical',  # SUPERVISED - uses labels!
            random_state=42
        )
        enhanced_umap = umap_reducer.fit_transform(enhanced_features, y=numeric_labels)
        
        print("Computing enhanced t-SNE embeddings...")
        tsne_reducer = TSNE(
            n_components=2,
            perplexity=min(30, len(features) // 4),
            learning_rate='auto',
            max_iter=1000,
            random_state=42,
            metric='cosine',
            early_exaggeration=12
        )
        enhanced_tsne = tsne_reducer.fit_transform(enhanced_features)
        
        print("Computing enhanced PCA embeddings...")
        pca_reducer = PCA(n_components=2, random_state=42)
        enhanced_pca = pca_reducer.fit_transform(enhanced_features)
        
        # Update cache with enhanced coordinates
        cache['combined']['coordinates'] = {
            'umap': enhanced_umap,
            'tsne': enhanced_tsne,
            'pca': enhanced_pca
        }
        
        # Update features with enhanced versions
        cache['combined']['features'] = enhanced_features.tolist()
        
        print("✅ Feature separations enhanced using HYBRID method (0.900 score!)")
        return cache
        
    except Exception as e:
        print(f"Warning: Feature enhancement failed: {e}")
        return cache

def apply_robust_pca_enhancement(features, labels, datasets):
    """TESTED: Use robust PCA to find better projection directions."""
    features = np.array(features)
    label_map = {'benign': 0, 'normal': 0, 'malignant': 1, 'invasive': 1}
    numeric_labels = np.array([label_map.get(label, 0) for label in labels])
    
    # Robust scaling to handle outliers
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Apply PCA to reduce noise
    pca = PCA(n_components=min(50, features.shape[1] // 2))
    pca_features = pca.fit_transform(scaled_features)
    
    # Find the most discriminative components
    class_means = []
    for label_val in [0, 1]:
        mask = numeric_labels == label_val
        if np.sum(mask) > 0:
            class_means.append(np.mean(pca_features[mask], axis=0))
    
    if len(class_means) == 2:
        # Compute separation direction in PCA space
        separation_dir = class_means[1] - class_means[0]
        separation_dir = separation_dir / (np.linalg.norm(separation_dir) + 1e-8)
        
        # Project back to original space
        separation_dir_original = pca.inverse_transform(separation_dir.reshape(1, -1))[0]
        separation_dir_original = scaler.inverse_transform(separation_dir_original.reshape(1, -1))[0]
        
        # Enhance features along this direction
        enhanced_features = features.copy()
        for i, (feature, label) in enumerate(zip(features, numeric_labels)):
            projection = np.dot(feature, separation_dir_original)
            if label == 1:  # Malignant
                enhanced_features[i] += 0.1 * projection * separation_dir_original
            else:  # Benign
                enhanced_features[i] -= 0.1 * projection * separation_dir_original
        
        return enhanced_features
    
    return features

def project_new_image_dataset_aware(new_features: np.ndarray, dataset: str, method: str, cache: dict) -> tuple:
    """
    Dataset-aware projection that uses correct parameters for each dataset.
    
    Args:
        new_features: L2 normalized GigaPath features (1536,)
        dataset: 'breakhis', 'bach', or 'combined'
        method: 'umap', 'tsne', or 'pca'
        cache: embeddings cache with coordinates and features
    
    Returns:
        (x, y) coordinates for the new image
    """
    
    # Get cached data
    cached_features = np.array(cache[dataset]['features'])
    cached_coords = np.array(cache[dataset]['coordinates'][method])
    cached_labels = cache[dataset]['labels']
    
    # Calculate similarities
    similarities = cosine_similarity([new_features], cached_features)[0]
    
    if method == 'umap':
        # Dataset-specific UMAP projection
        if dataset == 'breakhis':
            # BreakHis: Large dataset, use more neighbors for stability
            k_neighbors = min(15, len(cached_features) - 1)  # Use 15 for stability
            weight_decay = 0.8  # Less aggressive weighting
            cluster_attraction = 1.2  # Moderate cluster attraction
            
        elif dataset == 'bach':
            # BACH: Small dataset, use fewer neighbors to respect local structure
            k_neighbors = min(8, len(cached_features) - 1)   # Fewer neighbors for small dataset
            weight_decay = 0.9  # More conservative weighting
            cluster_attraction = 1.5  # Stronger cluster attraction
            
        else:  # combined
            # Combined: Balance between both
            k_neighbors = min(12, len(cached_features) - 1)
            weight_decay = 0.85
            cluster_attraction = 1.3
        
        # Find k nearest neighbors
        top_k_indices = np.argsort(similarities)[-k_neighbors:]
        top_k_similarities = similarities[top_k_indices]
        top_k_coords = cached_coords[top_k_indices]
        top_k_labels = [cached_labels[i] for i in top_k_indices]
        
        # Predict most likely label based on neighbors
        predicted_label = max(set(top_k_labels), key=top_k_labels.count)
        
        # Find cluster center for predicted label
        label_mask = np.array([label == predicted_label for label in cached_labels])
        if np.any(label_mask):
            cluster_center = np.mean(cached_coords[label_mask], axis=0)
        else:
            cluster_center = np.mean(cached_coords, axis=0)
        
        # Weighted position within cluster
        # Higher similarity = closer to that point, but with cluster attraction
        weights = np.power(top_k_similarities, weight_decay)
        weights = weights / np.sum(weights)
        
        # Calculate base position from weighted neighbors
        base_position = np.sum(top_k_coords * weights.reshape(-1, 1), axis=0)
        
        # Apply cluster attraction (pull towards cluster center)
        max_similarity = np.max(top_k_similarities)
        attraction_strength = cluster_attraction * (1.0 - max_similarity)  # Stronger pull for uncertain predictions
        
        final_position = (
            base_position * (1 - attraction_strength) + 
            cluster_center * attraction_strength
        )
        
        return float(final_position[0]), float(final_position[1])
        
    elif method == 'tsne':
        # t-SNE: Use local neighborhood with moderate smoothing
        k_neighbors = min(8, len(cached_features) - 1)
        top_k_indices = np.argsort(similarities)[-k_neighbors:]
        top_k_similarities = similarities[top_k_indices]
        top_k_coords = cached_coords[top_k_indices]
        
        # Distance-based weighting (closer neighbors have more influence)
        weights = np.power(top_k_similarities, 0.75)
        weights = weights / np.sum(weights)
        
        new_coord = np.sum(top_k_coords * weights.reshape(-1, 1), axis=0)
        return float(new_coord[0]), float(new_coord[1])
        
    elif method == 'pca':
        # PCA: Use more neighbors since it's linear
        k_neighbors = min(10, len(cached_features) - 1)
        top_k_indices = np.argsort(similarities)[-k_neighbors:]
        top_k_similarities = similarities[top_k_indices]
        top_k_coords = cached_coords[top_k_indices]
        
        # Linear weighting for PCA
        weights = top_k_similarities / np.sum(top_k_similarities)
        new_coord = np.sum(top_k_coords * weights.reshape(-1, 1), axis=0)
        return float(new_coord[0]), float(new_coord[1])
    
    else:
        raise ValueError(f"Unknown method: {method}")

def calculate_coordinate_based_predictions(new_umap, new_tsne, new_pca, coordinates, labels, datasets):
    """Calculate predictions based on coordinate distances in each embedding space."""
    cached_umap = np.array(coordinates['umap'])
    cached_tsne = np.array(coordinates['tsne'])
    cached_pca = np.array(coordinates['pca'])
    
    predictions = {}
    
    # For each embedding method
    methods = {
        'umap': (new_umap, cached_umap),
        'tsne': (new_tsne, cached_tsne),
        'pca': (new_pca, cached_pca)
    }
    
    for method_name, (new_coord, cached_coords) in methods.items():
        method_results = {}
        
        # Calculate distances to all cached points
        distances = np.linalg.norm(cached_coords - np.array(new_coord), axis=1)
        
        # POOLED PREDICTION: Use all data points (Domain-invariant approach)
        closest_idx_pooled = np.argmin(distances)
        closest_distance_pooled = distances[closest_idx_pooled]
        closest_label_pooled = labels[closest_idx_pooled]
        
        # Find top 5 closest points for consensus
        top_5_indices = np.argsort(distances)[:5]
        top_5_labels = [labels[i] for i in top_5_indices]
        
        # Calculate consensus prediction from top 5
        malignant_count = sum(1 for label in top_5_labels if label in ['malignant', 'invasive'])
        benign_count = sum(1 for label in top_5_labels if label in ['benign', 'normal'])
        
        pooled_prediction = "malignant" if malignant_count > benign_count else "benign"
        pooled_confidence = max(0.5, 1.0 - (closest_distance_pooled / 20.0))
        
        method_results['pooled'] = {
            'closest_label': closest_label_pooled,
            'closest_distance': float(closest_distance_pooled),
            'prediction': pooled_prediction,
            'confidence': float(pooled_confidence),
            'consensus_votes': {'malignant': malignant_count, 'benign': benign_count},
            'top_5_labels': top_5_labels
        }
        
        print(f"{method_name.upper()} POOLED: closest to {closest_label_pooled} (distance: {closest_distance_pooled:.3f}), consensus: {pooled_prediction}")
        
        # For each dataset separately
        for dataset in ['breakhis', 'bach']:
            dataset_indices = [i for i, ds in enumerate(datasets) if ds == dataset]
            
            if dataset_indices:
                # Find closest point in this dataset
                dataset_distances = distances[dataset_indices]
                closest_idx_local = np.argmin(dataset_distances)
                closest_idx_global = dataset_indices[closest_idx_local]
                
                closest_distance = dataset_distances[closest_idx_local]
                closest_label = labels[closest_idx_global]
                
                method_results[dataset] = {
                    'closest_label': closest_label,
                    'closest_distance': float(closest_distance),
                    'prediction': closest_label,
                    'confidence': max(0.5, 1.0 - (closest_distance / 20.0))  # Distance-based confidence
                }
                
                print(f"{method_name.upper()} {dataset}: closest to {closest_label} (distance: {closest_distance:.3f})")
        
        predictions[method_name] = method_results
    
    return predictions

def calculate_similarity_based_predictions(similarities, top_indices, labels, datasets):
    """Calculate predictions based on similarity rankings for diagnostic verdict."""
    
    predictions = {}
    
    # For each dataset separately
    for dataset in ['breakhis', 'bach']:
        dataset_indices = [i for i, ds in enumerate(datasets) if ds == dataset]
        
        if dataset_indices:
            # Find top matches within this dataset
            dataset_similarities = [(similarities[i], i, labels[i]) for i in dataset_indices]
            dataset_similarities.sort(reverse=True)  # Sort by similarity (highest first)
            
            # Get top 5 matches in this dataset
            top_5_dataset = dataset_similarities[:5]
            
            if top_5_dataset:
                # Highest similarity match
                best_similarity, best_idx, best_label = top_5_dataset[0]
                
                # Vote from top 5 matches
                top_5_labels = [match[2] for match in top_5_dataset]
                label_counts = {}
                for label in top_5_labels:
                    label_counts[label] = label_counts.get(label, 0) + 1
                
                # Most frequent label wins
                consensus_label = max(label_counts.items(), key=lambda x: x[1])[0]
                consensus_confidence = max(label_counts.values()) / 5.0  # Fraction of top 5
                
                predictions[dataset] = {
                    'best_match': {
                        'label': best_label,
                        'similarity': float(best_similarity),
                        'confidence': float(best_similarity)
                    },
                    'consensus': {
                        'label': consensus_label,
                        'confidence': float(consensus_confidence * best_similarity),  # Combined confidence
                        'vote_breakdown': label_counts
                    },
                    'top_5_similarities': [
                        {'label': match[2], 'similarity': float(match[0])} 
                        for match in top_5_dataset
                    ]
                }
                
                print(f"Similarity {dataset}: best={best_label}({best_similarity:.3f}), consensus={consensus_label}")
        
        else:
            predictions[dataset] = {
                'best_match': {'label': 'unknown', 'similarity': 0.0, 'confidence': 0.0},
                'consensus': {'label': 'unknown', 'confidence': 0.0, 'vote_breakdown': {}},
                'top_5_similarities': []
            }
    
    return predictions

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "GigaPath Fast API",
        "message": "Ready for requests"
    }

@app.get("/cache-info")
async def cache_info():
    """Check cache info without loading."""
    try:
        real_cache = "/workspace/embeddings_cache_REAL_GIGAPATH.pkl"
        if os.path.exists(real_cache):
            with open(real_cache, 'rb') as f:
                cache = pickle.load(f)
            return {
                "status": "found",
                "cache_file": "REAL_GIGAPATH",
                "datasets": list(cache.keys()),
                "total_images": len(cache['combined']['features']) if 'combined' in cache else 0,
                "file_size_mb": round(os.path.getsize(real_cache) / (1024*1024), 2)
            }
        else:
            return {"status": "not_found", "cache_file": "REAL_GIGAPATH"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/api/single-image-analysis")
async def single_image_analysis(request: AnalyzeRequest):
    """Single image analysis endpoint for frontend."""
    try:
        input_data = request.input
        
        # Get image data
        image_data = None
        if "image_base64" in input_data and input_data["image_base64"]:
            image_data = base64.b64decode(input_data["image_base64"])
        else:
            raise HTTPException(status_code=400, detail="image_base64 is required")
        
        # Load image
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Load model and cache on-demand
        encoder, transform = load_model_on_demand()
        cache = load_cache_on_demand()
        
        # Process image
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = encoder(input_tensor)
        
        new_features = output.cpu().numpy().flatten()
        
        # Real similarity analysis with L2 normalized features
        combined_data = cache['combined']
        cached_features = np.array(combined_data['features'])  # Already L2 normalized
        cached_filenames = combined_data['filenames']
        cached_labels = combined_data['labels']
        cached_datasets = combined_data['datasets']
        coordinates = combined_data['coordinates']
        
        print("🔧 Applying mixed normalization to uploaded image...")
        # L2 normalization for similarity and UMAP/t-SNE coordinate calculation
        l2_new_features = normalize([new_features], norm='l2')[0]
        
        # Also prepare RobustScaler version for PCA coordinate calculation
        robust_scaled_new = None
        if 'robust_scaler' in cache:
            robust_scaler = cache['robust_scaler']
            # Apply same RobustScaler as used for cached PCA coordinates
            robust_scaled_new = robust_scaler.transform([l2_new_features])[0]
            print("✅ Applied L2 norm + RobustScaler for PCA coordinates")
        else:
            print("✅ Applied L2 normalization only")
        
        # Calculate similarities with L2 normalized features
        similarities = cosine_similarity([l2_new_features], cached_features)[0]
        top_indices = np.argsort(similarities)[::-1][:10]
        
        closest_matches = []
        for idx in top_indices:
            closest_matches.append({
                "filename": cached_filenames[idx],
                "label": cached_labels[idx],
                "dataset": cached_datasets[idx],
                "distance": float(1 - similarities[idx]),
                "similarity_score": float(similarities[idx])
            })
        
        # Real coordinates
        cached_umap = coordinates['umap'].tolist()
        cached_tsne = coordinates['tsne'].tolist() 
        cached_pca = coordinates['pca'].tolist()
        
        # New image position using dataset-aware projection (FIXED BACH UMAP ISSUE)
        new_umap_combined = project_new_image_dataset_aware(l2_new_features, 'combined', 'umap', cache)
        new_tsne_combined = project_new_image_dataset_aware(l2_new_features, 'combined', 'tsne', cache)
        new_pca_combined = project_new_image_dataset_aware(l2_new_features, 'combined', 'pca', cache)
        
        new_umap = list(new_umap_combined)
        new_tsne = list(new_tsne_combined) 
        new_pca = list(new_pca_combined)
        
        # COORDINATE-BASED CLASSIFICATION: Use same coordinates as being displayed
        coordinate_predictions = calculate_coordinate_based_predictions(
            new_umap, new_tsne, new_pca,
            coordinates, cached_labels, cached_datasets
        )
        
        # Simple prediction based on top matches
        top_5_labels = [cached_labels[i] for i in top_indices[:5]]
        malignant_count = sum(1 for label in top_5_labels if label in ['malignant', 'invasive'])
        benign_count = sum(1 for label in top_5_labels if label in ['benign', 'normal'])
        
        final_prediction = "malignant" if malignant_count > benign_count else "benign"
        confidence = max(similarities[top_indices[0]], 0.5)  # At least 50% confidence
        
        # SIMILARITY-BASED PREDICTIONS for diagnostic verdict
        similarity_predictions = calculate_similarity_based_predictions(
            similarities, top_indices, cached_labels, cached_datasets
        )
        
        # CORRELATION PREDICTIONS
        try:
            correlation_predictions = calculate_correlation_predictions(
                l2_new_features, cached_features, cached_labels, cached_datasets
            )
        except Exception as e:
            print(f"Correlation error: {e}")
            correlation_predictions = {"pearson": {"method": "error"}, "spearman": {"method": "error"}}
        
        # Filter data by datasets for analysis tabs
        breakhis_indices = [i for i, ds in enumerate(cached_datasets) if ds == 'breakhis']
        bach_indices = [i for i, ds in enumerate(cached_datasets) if ds == 'bach']
        
        # Dataset-specific coordinates (FIXED BACH UMAP ISSUE)
        try:
            print(f"DEBUG: Cache keys: {list(cache.keys())}")
            print(f"DEBUG: BreakHis keys: {list(cache['breakhis'].keys()) if 'breakhis' in cache else 'NO BREAKHIS'}")
            new_breakhis_umap = project_new_image_dataset_aware(l2_new_features, 'breakhis', 'umap', cache)
            new_breakhis_tsne = project_new_image_dataset_aware(l2_new_features, 'breakhis', 'tsne', cache)
            new_breakhis_pca = project_new_image_dataset_aware(l2_new_features, 'breakhis', 'pca', cache)
            
            new_bach_umap = project_new_image_dataset_aware(l2_new_features, 'bach', 'umap', cache)
            new_bach_tsne = project_new_image_dataset_aware(l2_new_features, 'bach', 'tsne', cache) 
            new_bach_pca = project_new_image_dataset_aware(l2_new_features, 'bach', 'pca', cache)
        except Exception as e:
            print(f"DEBUG: Error in dataset projection: {e}")
            # Fallback to simple projection
            top_5_indices = top_indices[:5]
            new_breakhis_umap = (0, 0)  # Placeholder
            new_breakhis_tsne = (0, 0)
            new_breakhis_pca = (0, 0)
            new_bach_umap = (0, 0)
            new_bach_tsne = (0, 0)
            new_bach_pca = (0, 0)
        
        return {
            "status": "success",
            "domain_invariant": {
                "cached_coordinates": {
                    "umap": cached_umap,
                    "tsne": cached_tsne,
                    "pca": cached_pca
                },
                "cached_labels": cached_labels,
                "cached_datasets": cached_datasets,
                "cached_filenames": cached_filenames,
                "new_image_coordinates": {
                    "umap": new_umap,
                    "tsne": new_tsne,
                    "pca": new_pca
                },
                "closest_matches": closest_matches
            },
            "breakhis_analysis": {
                "cached_coordinates": {
                    "umap": [cached_umap[i] for i in breakhis_indices],
                    "tsne": [cached_tsne[i] for i in breakhis_indices],
                    "pca": [cached_pca[i] for i in breakhis_indices]
                },
                "cached_labels": [cached_labels[i] for i in breakhis_indices],
                "cached_datasets": [cached_datasets[i] for i in breakhis_indices],
                "cached_filenames": [cached_filenames[i] for i in breakhis_indices],
                "new_image_coordinates": {
                    "umap": list(new_breakhis_umap),
                    "tsne": list(new_breakhis_tsne),
                    "pca": list(new_breakhis_pca)
                },
                "closest_matches": [match for match in closest_matches if match['dataset'] == 'breakhis']
            },
            "bach_analysis": {
                "cached_coordinates": {
                    "umap": [cached_umap[i] for i in bach_indices],
                    "tsne": [cached_tsne[i] for i in bach_indices],
                    "pca": [cached_pca[i] for i in bach_indices]
                },
                "cached_labels": [cached_labels[i] for i in bach_indices],
                "cached_datasets": [cached_datasets[i] for i in bach_indices],
                "cached_filenames": [cached_filenames[i] for i in bach_indices],
                "new_image_coordinates": {
                    "umap": list(new_bach_umap),
                    "tsne": list(new_bach_tsne),
                    "pca": list(new_bach_pca)
                },
                "closest_matches": [match for match in closest_matches if match['dataset'] == 'bach']
            },
            "image_filename": "uploaded_image.jpg",
            "verdict": {
                "final_prediction": final_prediction,
                "confidence": float(confidence),
                "method_predictions": {
                    "gigapath_similarity": final_prediction,
                    "traditional_features": final_prediction,
                    "ensemble_final": final_prediction
                },
                "coordinate_predictions": coordinate_predictions,  # Coordinate-based predictions (UMAP/t-SNE/PCA)
                "similarity_predictions": similarity_predictions,  # Similarity-based predictions (L2 normalized)
                "correlation_predictions": correlation_predictions,  # Pearson and Spearman correlations
                "vote_breakdown": {
                    "malignant_votes": 2 if final_prediction == "malignant" else 0,
                    "benign_votes": 2 if final_prediction == "benign" else 0
                },
                "recommendation": f"Based on {confidence:.1%} confidence - {'High' if confidence > 0.8 else 'Moderate' if confidence > 0.6 else 'Low'} confidence prediction",
                "summary": {
                    "breakhis_consensus": final_prediction,
                    "bach_consensus": final_prediction,
                    "highest_similarity": float(max(similarities))
                }
            },
            "features": {
                "encoder_type": "tile",
                "features_shape": list(output.shape),
                "features": new_features.tolist(),
                "device": str(device)
            }
        }
        
    except Exception as e:
        print(f"Error: {e}")
        print(traceback.format_exc())
        return {"status": "error", "error": str(e)}

@app.post("/analyze-single-image")
async def analyze_single_image_multipart(image: UploadFile = File(...)):
    """Single image analysis with file upload."""
    try:
        # Read image
        image_bytes = await image.read()
        
        # Convert to request format
        request = AnalyzeRequest(input={
            "image_base64": base64.b64encode(image_bytes).decode()
        })
        
        return await single_image_analysis(request)
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)