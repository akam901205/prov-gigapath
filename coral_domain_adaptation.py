#!/usr/bin/env python3
"""
CORAL Domain Adaptation After GigaPath
Applies CORelation ALignment to reduce domain shift between BACH and BreakHis
"""
import numpy as np
import pickle
import torch
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

def load_fresh_cache():
    """Load fresh simple pipeline cache"""
    with open("/workspace/embeddings_cache_FRESH_SIMPLE.pkl", 'rb') as f:
        cache = pickle.load(f)
    
    features = np.array(cache['combined']['features'])
    labels = cache['combined']['labels']
    datasets = cache['combined']['datasets']
    filenames = cache['combined']['filenames']
    
    print(f"✅ Loaded cache: {len(features)} samples")
    return features, labels, datasets, filenames

def coral_alignment(source_features, target_features):
    """
    CORAL: CORelation ALignment for domain adaptation
    Aligns second-order statistics (covariance) between domains
    """
    # Convert to torch tensors
    source = torch.tensor(source_features, dtype=torch.float32)
    target = torch.tensor(target_features, dtype=torch.float32)
    
    # Center the features (zero mean)
    source_centered = source - torch.mean(source, dim=0, keepdim=True)
    target_centered = target - torch.mean(target, dim=0, keepdim=True)
    
    # Compute covariance matrices
    source_cov = torch.mm(source_centered.t(), source_centered) / (source.size(0) - 1)
    target_cov = torch.mm(target_centered.t(), target_centered) / (target.size(0) - 1)
    
    # CORAL loss: Frobenius norm of covariance difference
    coral_loss = torch.norm(source_cov - target_cov, p='fro') ** 2
    coral_loss = coral_loss / (4 * source.size(1) ** 2)
    
    print(f"  CORAL loss: {coral_loss.item():.6f}")
    
    # Apply CORAL alignment
    # Whiten source features and color with target statistics
    try:
        source_cov_sqrt = torch.linalg.cholesky(source_cov + 1e-6 * torch.eye(source.size(1)))
        target_cov_sqrt = torch.linalg.cholesky(target_cov + 1e-6 * torch.eye(target.size(1)))
        
        # Whiten source
        whitened = torch.linalg.solve(source_cov_sqrt, source_centered.t()).t()
        
        # Color with target
        coral_aligned = torch.mm(whitened, target_cov_sqrt.t())
        
        # Add target mean
        coral_aligned = coral_aligned + torch.mean(target, dim=0, keepdim=True)
        
        return coral_aligned.numpy()
        
    except Exception as e:
        print(f"  ⚠️ CORAL alignment failed: {e}")
        print("  Using identity transformation")
        return source_features

def apply_coral_to_datasets(features, labels, datasets):
    """Apply CORAL alignment between BACH and BreakHis"""
    
    print("🔧 Applying CORAL domain adaptation...")
    
    # Extract domain-specific features
    bach_mask = np.array([d == 'bach' for d in datasets])
    breakhis_mask = np.array([d == 'breakhis' for d in datasets])
    
    bach_features = features[bach_mask]
    breakhis_features = features[breakhis_mask]
    
    print(f"Source domain (BACH): {len(bach_features)} samples")
    print(f"Target domain (BreakHis): {len(breakhis_features)} samples")
    
    # Apply CORAL: align BACH to BreakHis distribution
    print("Aligning BACH → BreakHis...")
    coral_bach_features = coral_alignment(bach_features, breakhis_features)
    
    # Reconstruct full feature matrix
    coral_features = features.copy()
    coral_features[bach_mask] = coral_bach_features
    
    print("✅ CORAL alignment complete")
    
    return coral_features

def test_coral_alignment_quality(original_features, coral_features, labels, datasets, filenames):
    """Test if CORAL improves cross-dataset alignment"""
    
    print("\n🧪 Testing CORAL alignment quality...")
    
    # Extract sample indices
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
    
    def test_individual_alignment(bach_indices, breakhis_benign_indices, breakhis_malignant_indices, 
                                 features, expected_match, label_type):
        """Test individual BACH samples"""
        correct_alignments = 0
        total_samples = len(bach_indices)
        
        for bach_idx in bach_indices:
            bach_feature = features[bach_idx]
            
            # Find best correlation with BreakHis benign
            best_benign_corr = -1
            for breakhis_idx in breakhis_benign_indices[:100]:  # Sample for speed
                try:
                    corr, _ = spearmanr(bach_feature, features[breakhis_idx])
                    if not np.isnan(corr) and corr > best_benign_corr:
                        best_benign_corr = corr
                except:
                    continue
            
            # Find best correlation with BreakHis malignant
            best_malignant_corr = -1
            for breakhis_idx in breakhis_malignant_indices[:100]:  # Sample for speed
                try:
                    corr, _ = spearmanr(bach_feature, features[breakhis_idx])
                    if not np.isnan(corr) and corr > best_malignant_corr:
                        best_malignant_corr = corr
                except:
                    continue
            
            # Check alignment
            if expected_match == 'benign':
                if best_benign_corr > best_malignant_corr:
                    correct_alignments += 1
            else:  # expected_match == 'malignant'
                if best_malignant_corr > best_benign_corr:
                    correct_alignments += 1
        
        accuracy = correct_alignments / total_samples if total_samples > 0 else 0
        print(f"  {label_type}: {correct_alignments}/{total_samples} ({accuracy:.1%}) correct")
        return accuracy
    
    # Test ORIGINAL features
    print("\n--- BEFORE CORAL ---")
    orig_benign_acc = test_individual_alignment(
        bach_benign_indices, breakhis_benign_indices, breakhis_malignant_indices,
        original_features, 'benign', 'BACH benign'
    )
    orig_invasive_acc = test_individual_alignment(
        bach_invasive_indices, breakhis_benign_indices, breakhis_malignant_indices, 
        original_features, 'malignant', 'BACH invasive'
    )
    orig_overall = (orig_benign_acc + orig_invasive_acc) / 2
    
    # Test CORAL-aligned features  
    print("\n--- AFTER CORAL ---")
    coral_benign_acc = test_individual_alignment(
        bach_benign_indices, breakhis_benign_indices, breakhis_malignant_indices,
        coral_features, 'benign', 'BACH benign'
    )
    coral_invasive_acc = test_individual_alignment(
        bach_invasive_indices, breakhis_benign_indices, breakhis_malignant_indices,
        coral_features, 'malignant', 'BACH invasive'  
    )
    coral_overall = (coral_benign_acc + coral_invasive_acc) / 2
    
    # Summary
    print(f"\n🎯 CORAL DOMAIN ADAPTATION RESULTS:")
    print(f"  Before CORAL: {orig_overall:.1%} overall alignment")
    print(f"  After CORAL:  {coral_overall:.1%} overall alignment")
    
    improvement = coral_overall - orig_overall
    if improvement > 0.05:
        print(f"  ✅ CORAL improved alignment by {improvement:.1%}")
        status = "IMPROVED"
    elif improvement > -0.05:
        print(f"  ⚖️ CORAL minimal change ({improvement:+.1%})")
        status = "NEUTRAL"
    else:
        print(f"  ❌ CORAL decreased alignment by {improvement:.1%}")
        status = "DECREASED"
    
    return {
        'before_coral': {'benign_acc': orig_benign_acc, 'invasive_acc': orig_invasive_acc, 'overall': orig_overall},
        'after_coral': {'benign_acc': coral_benign_acc, 'invasive_acc': coral_invasive_acc, 'overall': coral_overall},
        'improvement': improvement,
        'status': status
    }

def main():
    """Main CORAL domain adaptation test"""
    print("🚀 CORAL DOMAIN ADAPTATION FOR GIGAPATH FEATURES")
    print("=" * 70)
    print("Applying CORelation ALignment to reduce BACH ↔ BreakHis domain shift")
    print("=" * 70)
    
    # Load data
    features, labels, datasets, filenames = load_fresh_cache()
    
    # Apply CORAL alignment
    coral_features = apply_coral_to_datasets(features, labels, datasets)
    
    # Test alignment quality
    results = test_coral_alignment_quality(features, coral_features, labels, datasets, filenames)
    
    # Save CORAL-aligned cache
    coral_cache = {
        'combined': {
            'features': coral_features.tolist(),
            'labels': labels,
            'datasets': datasets,
            'filenames': filenames
        },
        'metadata': {
            'pipeline': 'Fresh Simple + CORAL Domain Adaptation',
            'coral_applied': True,
            'alignment_improvement': results['improvement']
        }
    }
    
    with open('/workspace/embeddings_cache_FRESH_SIMPLE_CORAL.pkl', 'wb') as f:
        pickle.dump(coral_cache, f)
    
    print(f"\n📁 CORAL-aligned cache saved: /workspace/embeddings_cache_FRESH_SIMPLE_CORAL.pkl")
    
    # Save results
    with open('/workspace/coral_alignment_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results

if __name__ == "__main__":
    main()