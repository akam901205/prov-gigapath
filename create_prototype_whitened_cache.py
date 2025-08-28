#!/usr/bin/env python3
"""
Create Prototype Whitened Cache
Pre-process all embeddings through the whitening pipeline and save
"""
import numpy as np
import pickle
from sklearn.preprocessing import normalize
from sklearn.covariance import LedoitWolf

def create_prototype_whitened_cache():
    """Create cache with all embeddings pre-processed through whitening pipeline"""
    
    print("🚀 CREATING PROTOTYPE WHITENED CACHE")
    print("=" * 70)
    print("Pipeline: Raw GigaPath → Source Whitener → Whiten + L2 → Cache")
    print("=" * 70)
    
    # Load raw embeddings
    print("📂 Loading raw embeddings...")
    with open("/workspace/embeddings_cache_FRESH_SIMPLE.pkl", 'rb') as f:
        cache = pickle.load(f)
    
    features = np.array(cache['combined']['features'])
    labels = cache['combined']['labels']
    datasets = cache['combined']['datasets']
    filenames = cache['combined']['filenames']
    
    print(f"✅ Loaded {len(features)} raw embeddings")
    print(f"   Feature shape: {features.shape}")
    
    # Check if already L2 normalized
    sample_norms = [np.linalg.norm(features[i]) for i in range(5)]
    avg_norm = np.mean(sample_norms)
    print(f"   Average norm: {avg_norm:.6f}")
    
    # Step 1: Fit source whitener on ALL embeddings
    print("\n🔧 Step 1: Fitting source whitener...")
    
    # Compute source statistics
    source_mean = np.mean(features, axis=0, keepdims=True)
    centered = features - source_mean
    
    # Ledoit-Wolf shrinkage covariance
    print("   Computing Ledoit-Wolf shrinkage covariance...")
    lw = LedoitWolf()
    source_cov = lw.fit(centered).covariance_
    shrinkage = lw.shrinkage_
    
    print(f"   ✅ Auto shrinkage: {shrinkage:.4f}")
    print(f"   Condition number: {np.linalg.cond(source_cov):.2e}")
    
    # Compute whitening matrix (inverse square root)
    print("   Computing whitening matrix...")
    eigenvals, eigenvecs = np.linalg.eigh(source_cov)
    eps = 1e-6
    eigenvals = np.maximum(eigenvals, eps)
    inv_sqrt_eigenvals = 1.0 / np.sqrt(eigenvals)
    whitening_matrix = eigenvecs @ np.diag(inv_sqrt_eigenvals) @ eigenvecs.T
    
    print(f"   ✅ Whitening matrix: {whitening_matrix.shape}")
    print(f"   Min eigenvalue: {np.min(eigenvals):.2e}")
    
    # Step 2: Apply whitening + L2 to ALL embeddings
    print("\n🔄 Step 2: Applying whitening + L2 to all embeddings...")
    
    whitened_embeddings = []
    batch_size = 500
    
    for i in range(0, len(features), batch_size):
        batch_end = min(i + batch_size, len(features))
        batch_features = features[i:batch_end]
        
        # Center batch
        batch_centered = batch_features - source_mean
        
        # Apply whitening
        batch_whitened = batch_centered @ whitening_matrix.T
        
        # L2 normalize each embedding
        batch_l2 = normalize(batch_whitened, norm='l2')
        
        whitened_embeddings.extend(batch_l2)
        
        print(f"   Processed {batch_end}/{len(features)} embeddings...")
    
    whitened_embeddings = np.array(whitened_embeddings)
    
    # Verify L2 normalization
    sample_norms_whitened = [np.linalg.norm(whitened_embeddings[i]) for i in range(5)]
    print(f"   ✅ Whitened L2 norms: {[f'{norm:.6f}' for norm in sample_norms_whitened]}")
    
    # Step 3: Compute class prototypes in whitened space
    print("\n🎯 Step 3: Computing class prototypes...")
    
    # Binary classification mapping
    binary_labels = []
    for label in labels:
        if label in ['benign', 'normal']:
            binary_labels.append('benign')
        elif label in ['malignant', 'invasive', 'insitu']:
            binary_labels.append('malignant')
        else:
            binary_labels.append('unknown')
    
    # Compute centroids
    benign_mask = np.array([bl == 'benign' for bl in binary_labels])
    malignant_mask = np.array([bl == 'malignant' for bl in binary_labels])
    
    benign_prototype = np.mean(whitened_embeddings[benign_mask], axis=0)
    malignant_prototype = np.mean(whitened_embeddings[malignant_mask], axis=0)
    
    print(f"   Benign prototype: {np.sum(benign_mask)} samples, norm = {np.linalg.norm(benign_prototype):.6f}")
    print(f"   Malignant prototype: {np.sum(malignant_mask)} samples, norm = {np.linalg.norm(malignant_prototype):.6f}")
    
    # Step 4: Create comprehensive cache
    print("\n💾 Step 4: Creating comprehensive cache...")
    
    prototype_cache = {
        'combined': {
            'features': whitened_embeddings.tolist(),  # Pre-whitened L2 embeddings
            'labels': labels,
            'datasets': datasets,
            'filenames': filenames,
            'binary_labels': binary_labels
        },
        'whitening_transform': {
            'source_mean': source_mean,
            'whitening_matrix': whitening_matrix,
            'shrinkage_value': shrinkage,
            'eigenvalues': eigenvals,
            'condition_number': np.linalg.cond(source_cov)
        },
        'class_prototypes': {
            'benign': benign_prototype,
            'malignant': malignant_prototype
        },
        'metadata': {
            'pipeline': 'Raw GigaPath → Ledoit-Wolf Whitening → L2 Normalization → Prototypes',
            'total_samples': len(features),
            'feature_dimension': features.shape[1],
            'whitened_dimension': whitened_embeddings.shape[1],
            'shrinkage_method': 'Ledoit-Wolf automatic',
            'shrinkage_value': float(shrinkage),
            'datasets_included': list(set(datasets)),
            'label_distribution': {label: labels.count(label) for label in set(labels)},
            'class_prototype_norms': {
                'benign': float(np.linalg.norm(benign_prototype)),
                'malignant': float(np.linalg.norm(malignant_prototype))
            }
        }
    }
    
    # Save the comprehensive cache
    output_path = "/workspace/embeddings_cache_PROTOTYPE_WHITENED.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(prototype_cache, f)
    
    file_size = os.path.getsize(output_path) / (1024*1024)
    
    print(f"\n✅ PROTOTYPE WHITENED CACHE CREATED!")
    print(f"   📁 Path: {output_path}")
    print(f"   📊 Samples: {len(whitened_embeddings)}")
    print(f"   🧬 Features: {whitened_embeddings.shape[1]}-dim whitened L2 normalized")
    print(f"   🎯 Prototypes: benign + malignant centroids included")
    print(f"   💾 Size: {file_size:.1f} MB")
    print(f"   🔧 Shrinkage: {shrinkage:.4f} (Ledoit-Wolf auto)")
    
    # Quick verification test
    print(f"\n🧪 Quick verification test...")
    
    # Test prediction function
    def quick_predict(embedding_idx):
        """Quick prediction using cached prototypes"""
        whitened_embedding = whitened_embeddings[embedding_idx]
        
        cos_benign = np.dot(whitened_embedding, benign_prototype)
        cos_malignant = np.dot(whitened_embedding, malignant_prototype)
        
        prediction = 'malignant' if cos_malignant > cos_benign else 'benign'
        return prediction, cos_benign, cos_malignant
    
    # Test a few samples
    test_indices = [200, 201, 0, 1]  # b001, b002, iv001, iv002
    for idx in test_indices:
        if idx < len(filenames):
            pred, cos_b, cos_m = quick_predict(idx)
            true_label = labels[idx]
            filename = filenames[idx]
            
            print(f"   {filename}: {true_label} → {pred} (cos_b={cos_b:+.6f}, cos_m={cos_m:+.6f})")
    
    print(f"\n🎉 PROTOTYPE WHITENED CACHE READY FOR PRODUCTION!")
    
    return output_path

if __name__ == "__main__":
    import os
    create_prototype_whitened_cache()