#!/usr/bin/env python3
"""
Individual BACH Sample Correlation Test
For each BACH image, find best Spearman correlation with BreakHis samples
"""
import numpy as np
import pickle
from scipy.stats import spearmanr

def load_cache(cache_name="FRESH_SIMPLE"):
    """Load specified pipeline cache"""
    cache_files = {
        "FRESH_SIMPLE": "/workspace/embeddings_cache_FRESH_SIMPLE.pkl",
        "MACENKO": "/workspace/embeddings_cache_MACENKO_NO_TISSUE.pkl",
        "MPP_NO_MACENKO": "/workspace/embeddings_cache_MPP_NO_MACENKO.pkl"
    }
    
    cache_path = cache_files.get(cache_name, cache_files["FRESH_SIMPLE"])
    
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    
    if 'combined' in cache:
        features = np.array(cache['combined']['features'])
        labels = cache['combined']['labels']
        datasets = cache['combined']['datasets']
        filenames = cache['combined']['filenames']
    else:
        features = np.array(cache['features'])
        labels = cache['labels']
        datasets = cache['datasets']
        filenames = cache['filenames']
    
    print(f"✅ Loaded {cache_name} cache from {cache_path}")
    return features, labels, datasets, filenames

def extract_individual_samples(features, labels, datasets, filenames):
    """Extract individual BACH and BreakHis samples"""
    
    # Extract BACH samples
    bach_benign_indices = []
    bach_invasive_indices = []
    
    # Extract BreakHis samples
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
    
    # Limit to 100 samples each
    bach_benign_indices = bach_benign_indices[:100]
    bach_invasive_indices = bach_invasive_indices[:100]
    
    print(f"BACH benign samples: {len(bach_benign_indices)}")
    print(f"BACH invasive samples: {len(bach_invasive_indices)}")
    print(f"BreakHis benign samples: {len(breakhis_benign_indices)}")
    print(f"BreakHis malignant samples: {len(breakhis_malignant_indices)}")
    
    return {
        'bach_benign_indices': bach_benign_indices,
        'bach_invasive_indices': bach_invasive_indices,
        'breakhis_benign_indices': breakhis_benign_indices,
        'breakhis_malignant_indices': breakhis_malignant_indices
    }

def test_individual_bach_correlations(features, labels, datasets, filenames, indices):
    """For each BACH image, find best BreakHis correlation"""
    
    print("\n🔬 Testing individual BACH sample correlations...")
    
    results = {
        'bach_benign_results': [],
        'bach_invasive_results': []
    }
    
    # Test BACH benign samples
    print("\n=== TESTING 100 BACH BENIGN SAMPLES ===")
    for i, bach_idx in enumerate(indices['bach_benign_indices']):
        bach_feature = features[bach_idx]
        bach_filename = filenames[bach_idx]
        
        # Find best correlation with BreakHis benign
        best_benign_corr = -1
        best_benign_idx = None
        for breakhis_idx in indices['breakhis_benign_indices'][:200]:  # Sample 200 for speed
            try:
                corr, _ = spearmanr(bach_feature, features[breakhis_idx])
                if not np.isnan(corr) and corr > best_benign_corr:
                    best_benign_corr = corr
                    best_benign_idx = breakhis_idx
            except:
                continue
        
        # Find best correlation with BreakHis malignant
        best_malignant_corr = -1
        best_malignant_idx = None
        for breakhis_idx in indices['breakhis_malignant_indices'][:200]:  # Sample 200 for speed
            try:
                corr, _ = spearmanr(bach_feature, features[breakhis_idx])
                if not np.isnan(corr) and corr > best_malignant_corr:
                    best_malignant_corr = corr
                    best_malignant_idx = breakhis_idx
            except:
                continue
        
        # Determine which BreakHis label this BACH benign correlates with most
        if best_benign_corr > best_malignant_corr:
            best_match = 'breakhis_benign'
            best_corr = best_benign_corr
            best_idx = best_benign_idx
        else:
            best_match = 'breakhis_malignant'
            best_corr = best_malignant_corr
            best_idx = best_malignant_idx
        
        results['bach_benign_results'].append({
            'bach_filename': bach_filename,
            'best_breakhis_label': best_match,
            'best_correlation': best_corr,
            'best_breakhis_filename': filenames[best_idx] if best_idx else None,
            'correct_alignment': best_match == 'breakhis_benign'  # Should match benign
        })
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/100 BACH benign samples...")
    
    # Test BACH invasive samples
    print("\n=== TESTING 100 BACH INVASIVE SAMPLES ===")
    for i, bach_idx in enumerate(indices['bach_invasive_indices']):
        bach_feature = features[bach_idx]
        bach_filename = filenames[bach_idx]
        
        # Find best correlation with BreakHis benign
        best_benign_corr = -1
        best_benign_idx = None
        for breakhis_idx in indices['breakhis_benign_indices'][:200]:
            try:
                corr, _ = spearmanr(bach_feature, features[breakhis_idx])
                if not np.isnan(corr) and corr > best_benign_corr:
                    best_benign_corr = corr
                    best_benign_idx = breakhis_idx
            except:
                continue
        
        # Find best correlation with BreakHis malignant
        best_malignant_corr = -1
        best_malignant_idx = None
        for breakhis_idx in indices['breakhis_malignant_indices'][:200]:
            try:
                corr, _ = spearmanr(bach_feature, features[breakhis_idx])
                if not np.isnan(corr) and corr > best_malignant_corr:
                    best_malignant_corr = corr
                    best_malignant_idx = breakhis_idx
            except:
                continue
        
        # Determine which BreakHis label this BACH invasive correlates with most
        if best_malignant_corr > best_benign_corr:
            best_match = 'breakhis_malignant'
            best_corr = best_malignant_corr
            best_idx = best_malignant_idx
        else:
            best_match = 'breakhis_benign'
            best_corr = best_benign_corr
            best_idx = best_benign_idx
        
        results['bach_invasive_results'].append({
            'bach_filename': bach_filename,
            'best_breakhis_label': best_match,
            'best_correlation': best_corr,
            'best_breakhis_filename': filenames[best_idx] if best_idx else None,
            'correct_alignment': best_match == 'breakhis_malignant'  # Should match malignant
        })
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/100 BACH invasive samples...")
    
    return results

def analyze_individual_results(results):
    """Analyze individual correlation results"""
    
    print("\n📊 INDIVIDUAL BACH SAMPLE CORRELATION ANALYSIS")
    print("=" * 60)
    
    # Analyze BACH benign results
    benign_results = results['bach_benign_results']
    benign_correct = sum(1 for r in benign_results if r['correct_alignment'])
    benign_accuracy = benign_correct / len(benign_results) if benign_results else 0
    
    print(f"BACH BENIGN SAMPLES ({len(benign_results)} total):")
    print(f"  ✅ Correctly aligned to BreakHis benign: {benign_correct} ({benign_accuracy:.1%})")
    print(f"  ❌ Misaligned to BreakHis malignant: {len(benign_results) - benign_correct}")
    
    if benign_results:
        benign_correlations = [r['best_correlation'] for r in benign_results]
        print(f"  📊 Correlation strength: {np.mean(benign_correlations):.4f} ± {np.std(benign_correlations):.4f}")
    
    # Analyze BACH invasive results
    invasive_results = results['bach_invasive_results']
    invasive_correct = sum(1 for r in invasive_results if r['correct_alignment'])
    invasive_accuracy = invasive_correct / len(invasive_results) if invasive_results else 0
    
    print(f"\nBACH INVASIVE SAMPLES ({len(invasive_results)} total):")
    print(f"  ✅ Correctly aligned to BreakHis malignant: {invasive_correct} ({invasive_accuracy:.1%})")
    print(f"  ❌ Misaligned to BreakHis benign: {len(invasive_results) - invasive_correct}")
    
    if invasive_results:
        invasive_correlations = [r['best_correlation'] for r in invasive_results]
        print(f"  📊 Correlation strength: {np.mean(invasive_correlations):.4f} ± {np.std(invasive_correlations):.4f}")
    
    # Overall alignment accuracy
    total_samples = len(benign_results) + len(invasive_results)
    total_correct = benign_correct + invasive_correct
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    print(f"\n🎯 OVERALL CROSS-DATASET BIOLOGICAL ALIGNMENT: {overall_accuracy:.1%}")
    
    if overall_accuracy > 0.7:
        print("✅ EXCELLENT: Strong biological consistency across datasets")
        quality = "EXCELLENT"
    elif overall_accuracy > 0.6:
        print("⚠️ MODERATE: Some biological consistency present")
        quality = "MODERATE"
    else:
        print("❌ POOR: Low biological consistency - learning dataset artifacts")
        quality = "POOR"
    
    return {
        'benign_accuracy': benign_accuracy,
        'invasive_accuracy': invasive_accuracy,
        'overall_accuracy': overall_accuracy,
        'quality': quality
    }

def main(pipeline_name="FRESH_SIMPLE"):
    """Main execution"""
    print(f"🔬 INDIVIDUAL BACH-BREAKHIS CORRELATION TEST: {pipeline_name}")
    print("Testing each BACH image's best Spearman correlation with BreakHis")
    print("=" * 70)
    
    # Load data
    features, labels, datasets, filenames = load_cache(pipeline_name)
    
    # Extract sample indices
    indices = extract_individual_samples(features, labels, datasets, filenames)
    
    # Test individual correlations
    results = test_individual_bach_correlations(features, labels, datasets, filenames, indices)
    
    # Analyze results
    summary = analyze_individual_results(results)
    
    # Save detailed results
    output_data = {
        'individual_results': results,
        'summary': summary,
        'pipeline': f'{pipeline_name} Pipeline',
        'method': 'Individual Spearman correlation per BACH sample'
    }
    
    output_file = f'/workspace/individual_bach_correlation_{pipeline_name.lower()}.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"\n📁 Detailed results saved to: {output_file}")
    
    return output_data

if __name__ == "__main__":
    import sys
    pipeline_name = sys.argv[1] if len(sys.argv) > 1 else "FRESH_SIMPLE"
    main(pipeline_name)