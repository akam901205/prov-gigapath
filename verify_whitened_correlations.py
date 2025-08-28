#!/usr/bin/env python3
"""
Verify Individual Correlations in Whitened Space
Check if every BACH benign truly correlates highest with BreakHis benign
"""
import numpy as np
import pickle
from scipy.stats import spearmanr
from sklearn.preprocessing import normalize
from sklearn.covariance import LedoitWolf

def load_and_whiten_embeddings():
    """Load embeddings and apply whitening transform"""
    
    # Load cache
    with open("/workspace/embeddings_cache_FRESH_SIMPLE.pkl", 'rb') as f:
        cache = pickle.load(f)
    
    features = np.array(cache['combined']['features'])
    labels = cache['combined']['labels']
    datasets = cache['combined']['datasets']
    filenames = cache['combined']['filenames']
    
    print(f"✅ Loaded: {len(features)} embeddings")
    
    # Apply same whitening as prototype classifier
    print("🔧 Applying whitening transform...")
    
    # Compute source statistics
    source_mean = np.mean(features, axis=0, keepdims=True)
    centered = features - source_mean
    
    # Auto shrinkage covariance
    lw = LedoitWolf()
    source_cov = lw.fit(centered).covariance_
    shrinkage = lw.shrinkage_
    
    print(f"   Auto shrinkage: {shrinkage:.4f}")
    
    # Whitening matrix
    eigenvals, eigenvecs = np.linalg.eigh(source_cov)
    eps = 1e-6
    eigenvals = np.maximum(eigenvals, eps)
    inv_sqrt_eigenvals = 1.0 / np.sqrt(eigenvals)
    whitening_matrix = eigenvecs @ np.diag(inv_sqrt_eigenvals) @ eigenvecs.T
    
    # Apply whitening + L2 to all embeddings
    whitened = centered @ whitening_matrix.T
    whitened_l2 = normalize(whitened, norm='l2')
    
    print(f"   ✅ All embeddings whitened and L2 normalized")
    
    return whitened_l2, labels, datasets, filenames

def test_individual_correlations_whitened():
    """Test individual BACH correlations in whitened space"""
    
    print("🔬 TESTING INDIVIDUAL CORRELATIONS IN WHITENED SPACE")
    print("=" * 70)
    
    # Load whitened embeddings
    features, labels, datasets, filenames = load_and_whiten_embeddings()
    
    # Extract indices
    bach_benign_indices = []
    bach_invasive_indices = []
    breakhis_benign_indices = []
    breakhis_malignant_indices = []
    
    for i, (dataset, label) in enumerate(zip(datasets, labels)):
        if dataset == 'bach':
            if label == 'benign':
                bach_benign_indices.append(i)
            elif label == 'invasive':
                bach_invasive_indices.append(i)
        elif dataset == 'breakhis':
            if label == 'benign':
                breakhis_benign_indices.append(i)
            elif label == 'malignant':
                breakhis_malignant_indices.append(i)
    
    bach_benign_indices = bach_benign_indices[:100]
    bach_invasive_indices = bach_invasive_indices[:100]
    
    print(f"Test samples: {len(bach_benign_indices)} benign, {len(bach_invasive_indices)} invasive BACH")
    print(f"Reference: {len(breakhis_benign_indices)} benign, {len(breakhis_malignant_indices)} malignant BreakHis")
    
    # Test BACH benign samples
    print(f"\n=== TESTING BACH BENIGN INDIVIDUAL CORRELATIONS ===")
    benign_correct = 0
    benign_correlations = []
    benign_details = []
    
    for i, bach_idx in enumerate(bach_benign_indices[:10]):  # Test first 10 for detailed analysis
        bach_feature = features[bach_idx]
        
        # Find best correlation with BreakHis benign
        best_benign_corr = -1
        best_benign_file = None
        for breakhis_idx in breakhis_benign_indices[:50]:  # Sample for speed
            try:
                corr, _ = spearmanr(bach_feature, features[breakhis_idx])
                if not np.isnan(corr) and corr > best_benign_corr:
                    best_benign_corr = corr
                    best_benign_file = filenames[breakhis_idx]
            except:
                continue
        
        # Find best correlation with BreakHis malignant
        best_malignant_corr = -1
        best_malignant_file = None
        for breakhis_idx in breakhis_malignant_indices[:50]:
            try:
                corr, _ = spearmanr(bach_feature, features[breakhis_idx])
                if not np.isnan(corr) and corr > best_malignant_corr:
                    best_malignant_corr = corr
                    best_malignant_file = filenames[breakhis_idx]
            except:
                continue
        
        # Check alignment
        correct = best_benign_corr > best_malignant_corr
        if correct:
            benign_correct += 1
        
        benign_details.append({
            'bach_file': filenames[bach_idx],
            'best_benign_corr': best_benign_corr,
            'best_malignant_corr': best_malignant_corr,
            'correct': correct,
            'best_benign_file': best_benign_file,
            'best_malignant_file': best_malignant_file
        })
        
        status = "✅" if correct else "❌"
        print(f"  {status} BACH benign #{i+1}: benign={best_benign_corr:.4f} vs malignant={best_malignant_corr:.4f}")
    
    # Test all 100 quickly
    print(f"\nTesting all 100 BACH benign samples...")
    total_benign_correct = 0
    for bach_idx in bach_benign_indices:
        bach_feature = features[bach_idx]
        
        # Quick test with smaller samples
        benign_corrs = []
        for breakhis_idx in breakhis_benign_indices[:20]:
            try:
                corr, _ = spearmanr(bach_feature, features[breakhis_idx])
                if not np.isnan(corr):
                    benign_corrs.append(corr)
            except:
                continue
        
        malignant_corrs = []
        for breakhis_idx in breakhis_malignant_indices[:20]:
            try:
                corr, _ = spearmanr(bach_feature, features[breakhis_idx])
                if not np.isnan(corr):
                    malignant_corrs.append(corr)
            except:
                continue
        
        if benign_corrs and malignant_corrs:
            best_benign = max(benign_corrs)
            best_malignant = max(malignant_corrs)
            
            if best_benign > best_malignant:
                total_benign_correct += 1
    
    benign_accuracy = total_benign_correct / len(bach_benign_indices)
    print(f"📊 BACH benign accuracy: {total_benign_correct}/{len(bach_benign_indices)} ({benign_accuracy:.1%})")
    
    # Show detailed examples
    print(f"\n📝 DETAILED EXAMPLES:")
    for i, detail in enumerate(benign_details[:5]):
        print(f"  {i+1}. {detail['bach_file']}")
        print(f"     Best benign match: {detail['best_benign_file']} (corr={detail['best_benign_corr']:.4f})")
        print(f"     Best malignant match: {detail['best_malignant_file']} (corr={detail['best_malignant_corr']:.4f})")
        print(f"     Result: {'CORRECT' if detail['correct'] else 'WRONG'}")
    
    return benign_accuracy, benign_details

if __name__ == "__main__":
    accuracy, details = test_individual_correlations_whitened()