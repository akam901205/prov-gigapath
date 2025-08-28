#!/usr/bin/env python3
"""
Test All Available Caches for Cross-Dataset Alignment Performance
Ranks all cache files by BACH-BreakHis individual correlation test
"""
import numpy as np
import pickle
import os
from scipy.stats import spearmanr
import time

def get_all_cache_files():
    """Get all available cache files"""
    cache_files = {}
    
    # All cache files found on the system
    cache_paths = [
        "/workspace/embeddings_cache_FRESH_SIMPLE.pkl",
        "/workspace/embeddings_cache_FRESH_SIMPLE_CORAL.pkl",
        "/workspace/embeddings_cache_DIRECT_224.pkl",
        "/workspace/embeddings_cache_COMPLETE.pkl",
        "/workspace/embeddings_cache_COMPLETE_7STEP_VAHADANE.pkl",
        "/workspace/embeddings_cache_COMPLETE_TISSUE_MACENKO.pkl",
        "/workspace/embeddings_cache_MACENKO_NO_TISSUE.pkl",
        "/workspace/embeddings_cache_MPP_NO_MACENKO.pkl",
        "/workspace/embeddings_cache_L2_REPROCESSED.pkl",
        "/workspace/embeddings_cache_4_CLUSTERS_FIXED_TSNE.pkl",
        "/workspace/embeddings_cache_PURE_L2.pkl",
        "/workspace/embeddings_cache_ENHANCED_SEPARATION.pkl",
        "/workspace/embeddings_cache_BALANCED.pkl",
        "/workspace/embeddings_cache_BACH_OPTIMIZED.pkl",
        "/workspace/embeddings_cache_4_BIOLOGICAL_CLUSTERS.pkl",
        "/workspace/embeddings_cache_COMPREHENSIVE.pkl",
        "/workspace/embeddings_cache_FIXED_LABELS.pkl",
        "/workspace/FINAL_COMPLETE_PROCESSED.pkl"
    ]
    
    for cache_path in cache_paths:
        if os.path.exists(cache_path):
            cache_name = os.path.basename(cache_path).replace('embeddings_cache_', '').replace('.pkl', '')
            cache_files[cache_name] = cache_path
    
    return cache_files

def load_cache_safe(cache_path):
    """Safely load cache with different formats"""
    try:
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        
        # Handle different cache formats
        if 'combined' in cache:
            features = np.array(cache['combined']['features'])
            labels = cache['combined']['labels']
            datasets = cache['combined']['datasets']
            filenames = cache['combined'].get('filenames', [f"file_{i}" for i in range(len(labels))])
        else:
            features = np.array(cache['features'])
            labels = cache['labels']
            datasets = cache['datasets']
            filenames = cache.get('filenames', [f"file_{i}" for i in range(len(labels))])
        
        return features, labels, datasets, filenames, True
        
    except Exception as e:
        print(f"  ❌ Failed to load: {e}")
        return None, None, None, None, False

def extract_test_samples(features, labels, datasets):
    """Extract BACH and BreakHis test samples"""
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
    
    # Limit BACH samples to 100 each for consistency
    bach_benign_indices = bach_benign_indices[:100]
    bach_invasive_indices = bach_invasive_indices[:100]
    
    return {
        'bach_benign': bach_benign_indices,
        'bach_invasive': bach_invasive_indices,
        'breakhis_benign': breakhis_benign_indices,
        'breakhis_malignant': breakhis_malignant_indices
    }

def test_cache_alignment(features, labels, datasets, indices):
    """Test individual BACH sample alignment for a cache"""
    
    def test_bach_samples(bach_indices, expected_match):
        """Test BACH samples against BreakHis"""
        correct = 0
        total = len(bach_indices)
        
        if total == 0:
            return 0.0
        
        for bach_idx in bach_indices:
            bach_feature = features[bach_idx]
            
            # Test against BreakHis benign (sample 50 for speed)
            best_benign_corr = -1
            sample_benign = indices['breakhis_benign'][:50]
            for breakhis_idx in sample_benign:
                try:
                    corr, _ = spearmanr(bach_feature, features[breakhis_idx])
                    if not np.isnan(corr) and corr > best_benign_corr:
                        best_benign_corr = corr
                except:
                    continue
            
            # Test against BreakHis malignant (sample 50 for speed)
            best_malignant_corr = -1
            sample_malignant = indices['breakhis_malignant'][:50]
            for breakhis_idx in sample_malignant:
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
        
        return correct / total
    
    # Test both BACH categories
    benign_accuracy = test_bach_samples(indices['bach_benign'], 'benign')
    invasive_accuracy = test_bach_samples(indices['bach_invasive'], 'malignant')
    
    # Overall accuracy
    total_bach = len(indices['bach_benign']) + len(indices['bach_invasive'])
    if total_bach == 0:
        overall_accuracy = 0.0
    else:
        overall_accuracy = (benign_accuracy * len(indices['bach_benign']) + 
                           invasive_accuracy * len(indices['bach_invasive'])) / total_bach
    
    return {
        'benign_accuracy': benign_accuracy,
        'invasive_accuracy': invasive_accuracy,
        'overall_accuracy': overall_accuracy,
        'bach_benign_samples': len(indices['bach_benign']),
        'bach_invasive_samples': len(indices['bach_invasive']),
        'breakhis_benign_samples': len(indices['breakhis_benign']),
        'breakhis_malignant_samples': len(indices['breakhis_malignant'])
    }

def main():
    """Test all caches and rank by performance"""
    print("🚀 COMPREHENSIVE CACHE RANKING TEST")
    print("=" * 80)
    print("Testing all available caches using BACH-BreakHis individual correlation")
    print("=" * 80)
    
    cache_files = get_all_cache_files()
    print(f"Found {len(cache_files)} cache files to test:")
    for name, path in cache_files.items():
        file_size = os.path.getsize(path) / (1024*1024)  # MB
        print(f"  - {name}: {file_size:.1f}MB")
    
    results = []
    
    print(f"\n🔬 Testing each cache...")
    
    for i, (cache_name, cache_path) in enumerate(cache_files.items(), 1):
        print(f"\n[{i}/{len(cache_files)}] Testing: {cache_name}")
        print(f"  Path: {cache_path}")
        
        start_time = time.time()
        
        # Load cache
        features, labels, datasets, filenames, success = load_cache_safe(cache_path)
        if not success:
            continue
        
        print(f"  ✅ Loaded: {len(features)} samples")
        
        # Extract test samples
        indices = extract_test_samples(features, labels, datasets)
        
        if len(indices['bach_benign']) == 0 and len(indices['bach_invasive']) == 0:
            print(f"  ⚠️ No BACH samples found, skipping...")
            continue
        
        if len(indices['breakhis_benign']) == 0 and len(indices['breakhis_malignant']) == 0:
            print(f"  ⚠️ No BreakHis samples found, skipping...")
            continue
        
        # Test alignment
        alignment_results = test_cache_alignment(features, labels, datasets, indices)
        
        test_time = time.time() - start_time
        
        # Store results
        result = {
            'cache_name': cache_name,
            'cache_path': cache_path,
            'file_size_mb': os.path.getsize(cache_path) / (1024*1024),
            'total_samples': len(features),
            'test_time': test_time,
            **alignment_results
        }
        
        results.append(result)
        
        print(f"  📊 Results:")
        print(f"    - BACH benign alignment: {alignment_results['benign_accuracy']:.1%} ({alignment_results['bach_benign_samples']} samples)")
        print(f"    - BACH invasive alignment: {alignment_results['invasive_accuracy']:.1%} ({alignment_results['bach_invasive_samples']} samples)")
        print(f"    - Overall alignment: {alignment_results['overall_accuracy']:.1%}")
        print(f"    - Test time: {test_time:.1f}s")
    
    # Sort by overall accuracy
    results.sort(key=lambda x: x['overall_accuracy'], reverse=True)
    
    print(f"\n🏆 FINAL CACHE RANKING (by cross-dataset biological alignment)")
    print("=" * 80)
    
    for rank, result in enumerate(results, 1):
        status = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank:2d}."
        
        print(f"{status} {result['cache_name']:<30} | {result['overall_accuracy']:>6.1%} | "
              f"Benign: {result['benign_accuracy']:>5.1%} | Invasive: {result['invasive_accuracy']:>5.1%} | "
              f"({result['total_samples']:>4d} samples, {result['file_size_mb']:>5.1f}MB)")
    
    print(f"\n📊 PERFORMANCE CATEGORIES:")
    excellent = [r for r in results if r['overall_accuracy'] >= 0.70]
    good = [r for r in results if 0.60 <= r['overall_accuracy'] < 0.70]
    moderate = [r for r in results if 0.50 <= r['overall_accuracy'] < 0.60]
    poor = [r for r in results if r['overall_accuracy'] < 0.50]
    
    print(f"  ✅ EXCELLENT (≥70%): {len(excellent)} caches")
    print(f"  ⚠️ GOOD (60-69%): {len(good)} caches")  
    print(f"  ⚡ MODERATE (50-59%): {len(moderate)} caches")
    print(f"  ❌ POOR (<50%): {len(poor)} caches")
    
    if excellent:
        print(f"\n🎯 RECOMMENDED FOR PRODUCTION:")
        for result in excellent:
            print(f"  - {result['cache_name']} ({result['overall_accuracy']:.1%})")
    
    # Save results
    with open('/workspace/cache_ranking_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n📁 Detailed results saved: /workspace/cache_ranking_results.pkl")
    
    return results

if __name__ == "__main__":
    main()