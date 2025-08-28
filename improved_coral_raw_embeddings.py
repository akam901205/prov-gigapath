#!/usr/bin/env python3
"""
Improved CORAL with Shrinkage on Raw GigaPath Embeddings (before L2)
Apply CORAL to raw embeddings, then L2 normalize for better domain alignment
"""
import numpy as np
import pickle
import torch
from scipy.stats import spearmanr
from sklearn.preprocessing import normalize

def load_raw_embeddings_cache():
    """Load cache with raw GigaPath embeddings (before L2 normalization)"""
    # Try to find cache with raw embeddings
    cache_paths = [
        "/workspace/embeddings_cache_FRESH_SIMPLE.pkl",
        "/workspace/embeddings_cache_COMPLETE.pkl"
    ]
    
    for cache_path in cache_paths:
        try:
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
            
            features = np.array(cache['combined']['features'])
            labels = cache['combined']['labels']
            datasets = cache['combined']['datasets']
            filenames = cache['combined']['filenames']
            
            print(f"✅ Loaded raw embeddings from: {cache_path}")
            print(f"   Features shape: {features.shape}")
            
            # Check if features are L2 normalized (norm ≈ 1.0)
            sample_norms = [np.linalg.norm(features[i]) for i in range(5)]
            avg_norm = np.mean(sample_norms)
            print(f"   Average feature norm: {avg_norm:.4f}")
            
            if avg_norm > 0.5 and avg_norm < 1.5:
                print("   ⚠️ Features appear to be L2 normalized already")
            else:
                print("   ✅ Features appear to be raw GigaPath embeddings")
            
            return features, labels, datasets, filenames
            
        except FileNotFoundError:
            continue
    
    raise FileNotFoundError("No suitable raw embeddings cache found")

def coral_with_shrinkage(source_features, target_features, shrinkage=0.1):
    """
    Improved CORAL with Ledoit-Wolf shrinkage regularization
    Applies stronger regularization for better covariance estimation
    """
    print(f"🔧 Applying CORAL with shrinkage regularization: {shrinkage}")
    
    # Convert to torch tensors
    source = torch.tensor(source_features, dtype=torch.float32)
    target = torch.tensor(target_features, dtype=torch.float32)
    
    d = source.size(1)  # Feature dimension
    n_source = source.size(0)
    n_target = target.size(0)
    
    print(f"   Source samples: {n_source}, Target samples: {n_target}, Dimensions: {d}")
    
    # Center the features (zero mean)
    source_centered = source - torch.mean(source, dim=0, keepdim=True)
    target_centered = target - torch.mean(target, dim=0, keepdim=True)
    
    # Compute sample covariance matrices
    source_cov_sample = torch.mm(source_centered.t(), source_centered) / (n_source - 1)
    target_cov_sample = torch.mm(target_centered.t(), target_centered) / (n_target - 1)
    
    # Apply Ledoit-Wolf shrinkage: shrunk_cov = (1-α)*sample_cov + α*identity
    identity = torch.eye(d)
    
    # Shrinkage for source
    source_trace = torch.trace(source_cov_sample) / d
    source_cov_shrunk = (1 - shrinkage) * source_cov_sample + shrinkage * source_trace * identity
    
    # Shrinkage for target  
    target_trace = torch.trace(target_cov_sample) / d
    target_cov_shrunk = (1 - shrinkage) * target_cov_sample + shrinkage * target_trace * identity
    
    print(f"   Source trace: {source_trace.item():.6f}, Target trace: {target_trace.item():.6f}")
    
    # Compute CORAL loss (Frobenius norm)
    coral_loss = torch.norm(source_cov_shrunk - target_cov_shrunk, p='fro') ** 2
    coral_loss = coral_loss / (4 * d ** 2)
    print(f"   CORAL loss: {coral_loss.item():.6f}")
    
    # Apply CORAL transformation
    try:
        # Eigendecomposition for more stable matrix square root
        source_eigenvals, source_eigenvecs = torch.linalg.eigh(source_cov_shrunk)
        target_eigenvals, target_eigenvecs = torch.linalg.eigh(target_cov_shrunk)
        
        # Ensure positive eigenvalues
        eps = 1e-5
        source_eigenvals = torch.clamp(source_eigenvals, min=eps)
        target_eigenvals = torch.clamp(target_eigenvals, min=eps)
        
        # Compute matrix square roots via eigendecomposition
        source_sqrt = source_eigenvecs @ torch.diag(torch.sqrt(source_eigenvals)) @ source_eigenvecs.t()
        source_inv_sqrt = source_eigenvecs @ torch.diag(1.0 / torch.sqrt(source_eigenvals)) @ source_eigenvecs.t()
        target_sqrt = target_eigenvecs @ torch.diag(torch.sqrt(target_eigenvals)) @ target_eigenvecs.t()
        
        # CORAL transformation: target_sqrt @ source_inv_sqrt @ source_centered
        transformation_matrix = target_sqrt @ source_inv_sqrt
        coral_aligned = torch.mm(source_centered, transformation_matrix.t())
        
        # Add target mean
        coral_aligned = coral_aligned + torch.mean(target, dim=0, keepdim=True)
        
        print(f"   ✅ CORAL transformation successful")
        return coral_aligned.numpy()
        
    except Exception as e:
        print(f"   ❌ CORAL transformation failed: {e}")
        print(f"   Using identity transformation")
        return source_features

def apply_improved_coral_pipeline(features, labels, datasets, shrinkage=0.1):
    """Apply improved CORAL to raw embeddings, then L2 normalize"""
    
    print("🚀 IMPROVED CORAL PIPELINE:")
    print("   1. Extract raw GigaPath embeddings")
    print("   2. Apply CORAL with shrinkage regularization") 
    print("   3. L2 normalize aligned embeddings")
    print("=" * 60)
    
    # Extract domain-specific features
    bach_mask = np.array([d == 'bach' for d in datasets])
    breakhis_mask = np.array([d == 'breakhis' for d in datasets])
    
    bach_features = features[bach_mask]
    breakhis_features = features[breakhis_mask]
    
    print(f"📊 Domain statistics:")
    print(f"   BACH samples: {len(bach_features)}")
    print(f"   BreakHis samples: {len(breakhis_features)}")
    
    # Apply CORAL with shrinkage: align BACH to BreakHis distribution
    coral_bach_features = coral_with_shrinkage(bach_features, breakhis_features, shrinkage)
    
    # Reconstruct full feature matrix
    coral_features = features.copy()
    coral_features[bach_mask] = coral_bach_features
    
    print(f"\n🔄 Applying L2 normalization to CORAL-aligned embeddings...")
    l2_coral_features = normalize(coral_features, norm='l2')
    
    # Verify L2 normalization
    sample_norms = [np.linalg.norm(l2_coral_features[i]) for i in range(5)]
    print(f"   L2 normalized norms: {[f'{norm:.6f}' for norm in sample_norms]}")
    
    return l2_coral_features

def test_coral_improvement(original_features, coral_features, labels, datasets):
    """Test CORAL improvement using individual BACH correlation test"""
    
    print("\n🧪 Testing CORAL improvement...")
    
    # Extract test indices
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
    
    # Limit to 100 BACH samples each
    bach_benign_indices = bach_benign_indices[:100]
    bach_invasive_indices = bach_invasive_indices[:100]
    
    def test_alignment(features, bach_indices, expected_match, label_type):
        """Test individual BACH sample alignment"""
        correct = 0
        total = len(bach_indices)
        
        for bach_idx in bach_indices:
            bach_feature = features[bach_idx]
            
            # Test against BreakHis benign
            best_benign_corr = -1
            for breakhis_idx in breakhis_benign_indices[:50]:  # Sample for speed
                try:
                    corr, _ = spearmanr(bach_feature, features[breakhis_idx])
                    if not np.isnan(corr) and corr > best_benign_corr:
                        best_benign_corr = corr
                except:
                    continue
            
            # Test against BreakHis malignant
            best_malignant_corr = -1  
            for breakhis_idx in breakhis_malignant_indices[:50]:
                try:
                    corr, _ = spearmanr(bach_feature, features[breakhis_idx])
                    if not np.isnan(corr) and corr > best_malignant_corr:
                        best_malignant_corr = corr
                except:
                    continue
            
            # Check alignment
            if expected_match == 'benign':
                if best_benign_corr > best_malignant_corr:
                    correct += 1
            else:  # malignant
                if best_malignant_corr > best_benign_corr:
                    correct += 1
        
        accuracy = correct / total if total > 0 else 0
        print(f"  {label_type}: {correct}/{total} ({accuracy:.1%})")
        return accuracy
    
    # Test ORIGINAL features (with standard L2)
    print("\n--- BEFORE IMPROVED CORAL ---")
    original_l2 = normalize(original_features, norm='l2')
    orig_benign_acc = test_alignment(original_l2, bach_benign_indices, 'benign', 'BACH benign')
    orig_invasive_acc = test_alignment(original_l2, bach_invasive_indices, 'malignant', 'BACH invasive')
    orig_overall = (orig_benign_acc + orig_invasive_acc) / 2
    
    # Test IMPROVED CORAL features
    print("\n--- AFTER IMPROVED CORAL ---")
    coral_benign_acc = test_alignment(coral_features, bach_benign_indices, 'benign', 'BACH benign')
    coral_invasive_acc = test_alignment(coral_features, bach_invasive_indices, 'malignant', 'BACH invasive')
    coral_overall = (coral_benign_acc + coral_invasive_acc) / 2
    
    # Summary
    improvement = coral_overall - orig_overall
    print(f"\n🎯 IMPROVED CORAL RESULTS:")
    print(f"  Before CORAL: {orig_overall:.1%}")
    print(f"  After CORAL:  {coral_overall:.1%}")
    print(f"  Improvement:  {improvement:+.1%}")
    
    if improvement > 0.10:
        print(f"  ✅ EXCELLENT improvement!")
        status = "EXCELLENT"
    elif improvement > 0.05:
        print(f"  ✅ GOOD improvement!")
        status = "GOOD"
    elif improvement > -0.05:
        print(f"  ⚖️ Minimal change")
        status = "NEUTRAL"
    else:
        print(f"  ❌ Performance decreased")
        status = "DECREASED"
    
    return {
        'before': {'benign': orig_benign_acc, 'invasive': orig_invasive_acc, 'overall': orig_overall},
        'after': {'benign': coral_benign_acc, 'invasive': coral_invasive_acc, 'overall': coral_overall},
        'improvement': improvement,
        'status': status
    }

def main():
    """Main improved CORAL test"""
    print("🚀 IMPROVED CORAL WITH SHRINKAGE ON RAW EMBEDDINGS")
    print("=" * 70)
    
    # Load raw embeddings
    features, labels, datasets, filenames = load_raw_embeddings_cache()
    
    # Test different shrinkage values
    shrinkage_values = [0.01, 0.05, 0.1, 0.2, 0.5]
    
    results = {}
    
    for shrinkage in shrinkage_values:
        print(f"\n{'='*20} TESTING SHRINKAGE = {shrinkage} {'='*20}")
        
        # Apply improved CORAL pipeline
        coral_features = apply_improved_coral_pipeline(features, labels, datasets, shrinkage)
        
        # Test improvement
        result = test_coral_improvement(features, coral_features, labels, datasets)
        results[shrinkage] = result
        
        # Save this version if it's good
        if result['improvement'] > 0.05:
            cache_name = f"IMPROVED_CORAL_SHRINK_{shrinkage:g}"
            output_path = f"/workspace/embeddings_cache_{cache_name}.pkl"
            
            coral_cache = {
                'combined': {
                    'features': coral_features.tolist(),
                    'labels': labels,
                    'datasets': datasets,  
                    'filenames': filenames
                },
                'metadata': {
                    'pipeline': f'Raw GigaPath → CORAL (shrinkage={shrinkage}) → L2',
                    'shrinkage_value': shrinkage,
                    'improvement': result['improvement'],
                    'alignment_accuracy': result['after']['overall']
                }
            }
            
            with open(output_path, 'wb') as f:
                pickle.dump(coral_cache, f)
            
            print(f"  💾 Saved: {output_path}")
    
    # Find best shrinkage
    best_shrinkage = max(results.keys(), key=lambda k: results[k]['improvement'])
    best_result = results[best_shrinkage]
    
    print(f"\n🏆 BEST SHRINKAGE VALUE: {best_shrinkage}")
    print(f"   Improvement: {best_result['improvement']:+.1%}")
    print(f"   Final accuracy: {best_result['after']['overall']:.1%}")
    
    # Save results summary
    with open('/workspace/improved_coral_shrinkage_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results

if __name__ == "__main__":
    main()