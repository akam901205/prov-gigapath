#!/usr/bin/env python3
"""
Statistical Feature Analysis for GigaPath Embeddings
Computes skewness, chaos, and disorder metrics from L2 reprocessed embeddings
"""
import pickle
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_embeddings_cache():
    """Load the L2 reprocessed embeddings cache"""
    print("🔧 Loading L2 reprocessed embeddings cache...")
    
    cache_path = "/workspace/embeddings_cache_L2_REPROCESSED.pkl"
    try:
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        
        combined_data = cache['combined']
        features = np.array(combined_data['features'])
        labels = combined_data['labels']
        datasets = combined_data['datasets']
        filenames = combined_data['filenames']
        
        print(f"✅ Loaded {len(features)} feature vectors (1536-dim each)")
        print(f"📊 Datasets: {np.unique(datasets)}")
        print(f"🏷️ Labels: {np.unique(labels)}")
        
        return features, labels, datasets, filenames
    
    except FileNotFoundError:
        print("❌ L2 reprocessed cache not found!")
        return None, None, None, None

def compute_skewness_metrics(features):
    """Compute various skewness-based metrics"""
    print("📈 Computing skewness metrics...")
    
    n_samples, n_dims = features.shape
    skewness_metrics = []
    
    for i, feature_vector in enumerate(features):
        # Basic skewness
        skewness = stats.skew(feature_vector)
        
        # Kurtosis (tail heaviness)
        kurt = stats.kurtosis(feature_vector)
        
        # Asymmetry measures
        mean_val = np.mean(feature_vector)
        median_val = np.median(feature_vector)
        asymmetry_ratio = (mean_val - median_val) / (np.std(feature_vector) + 1e-10)
        
        # Distribution shape
        q75, q25 = np.percentile(feature_vector, [75, 25])
        q95, q05 = np.percentile(feature_vector, [95, 5])
        tail_asymmetry = (q95 - median_val) / (median_val - q05 + 1e-10)
        
        # Pearson skewness coefficients
        pearson_skew1 = (mean_val - median_val) / (np.std(feature_vector) + 1e-10)
        pearson_skew2 = 3 * (mean_val - median_val) / (np.std(feature_vector) + 1e-10)
        
        skewness_metrics.append({
            'sample_idx': i,
            'skewness': skewness,
            'kurtosis': kurt,
            'asymmetry_ratio': asymmetry_ratio,
            'tail_asymmetry': tail_asymmetry,
            'pearson_skew1': pearson_skew1,
            'pearson_skew2': pearson_skew2,
            'excess_kurtosis': kurt - 3,  # Excess over normal distribution
            'skewness_magnitude': abs(skewness)
        })
        
        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{n_samples} samples...")
    
    return pd.DataFrame(skewness_metrics)

def compute_chaos_disorder_metrics(features):
    """Compute chaos and disorder metrics"""
    print("🌪️ Computing chaos and disorder metrics...")
    
    n_samples, n_dims = features.shape
    chaos_metrics = []
    
    for i, feature_vector in enumerate(features):
        # Entropy-based disorder
        # Discretize features into bins for entropy calculation
        hist, _ = np.histogram(feature_vector, bins=50, density=True)
        hist = hist + 1e-10  # Avoid log(0)
        shannon_entropy = -np.sum(hist * np.log2(hist))
        
        # Variance-based disorder
        feature_variance = np.var(feature_vector)
        coefficient_of_variation = np.std(feature_vector) / (np.abs(np.mean(feature_vector)) + 1e-10)
        
        # Local variability (chaos measure)
        # Compute differences between consecutive features
        local_diffs = np.diff(feature_vector)
        local_volatility = np.std(local_diffs)
        local_chaos = np.mean(np.abs(local_diffs))
        
        # Fractal-like dimension approximation
        # Use box-counting on sorted features
        sorted_features = np.sort(feature_vector)
        scales = [2, 4, 8, 16, 32, 64]
        box_counts = []
        for scale in scales:
            bins = np.linspace(sorted_features.min(), sorted_features.max(), scale)
            count = np.histogram(sorted_features, bins)[0]
            non_empty_boxes = np.sum(count > 0)
            box_counts.append(non_empty_boxes)
        
        # Approximate fractal dimension
        if len(box_counts) > 1:
            log_scales = np.log(scales)
            log_counts = np.log(np.array(box_counts) + 1)
            fractal_dim = -np.polyfit(log_scales, log_counts, 1)[0] if len(log_counts) > 1 else 0
        else:
            fractal_dim = 0
        
        # Lyapunov-like exponent approximation
        # Measure sensitivity to initial conditions in feature space
        perturbation = 1e-6
        perturbed = feature_vector + perturbation * np.random.randn(len(feature_vector))
        divergence = np.linalg.norm(perturbed - feature_vector) / perturbation
        lyapunov_approx = np.log(divergence + 1e-10)
        
        # Range disorder
        feature_range = np.max(feature_vector) - np.min(feature_vector)
        iqr = np.percentile(feature_vector, 75) - np.percentile(feature_vector, 25)
        range_disorder = feature_range / (iqr + 1e-10)
        
        # Spatial disorder (treating feature vector as 1D signal)
        # Measure how features deviate from smooth patterns
        smoothed = np.convolve(feature_vector, np.ones(5)/5, mode='same')
        roughness = np.mean((feature_vector - smoothed)**2)
        
        # Information-theoretic measures
        # Approximate complexity using compression ratio
        feature_string = ''.join([str(int((f + 1) * 50)) for f in feature_vector[:100]])  # Sample first 100
        complexity_approx = len(set(feature_string)) / len(feature_string)
        
        chaos_metrics.append({
            'sample_idx': i,
            'shannon_entropy': shannon_entropy,
            'feature_variance': feature_variance,
            'coefficient_of_variation': coefficient_of_variation,
            'local_volatility': local_volatility,
            'local_chaos': local_chaos,
            'fractal_dimension': fractal_dim,
            'lyapunov_approx': lyapunov_approx,
            'range_disorder': range_disorder,
            'roughness': roughness,
            'complexity_approx': complexity_approx,
            'total_disorder': shannon_entropy * coefficient_of_variation * local_chaos
        })
        
        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{n_samples} samples...")
    
    return pd.DataFrame(chaos_metrics)

def analyze_by_dataset_label(skewness_df, chaos_df, labels, datasets):
    """Analyze metrics by dataset and label"""
    print("📊 Analyzing patterns by dataset and label...")
    
    # Add metadata to dataframes
    skewness_df['label'] = labels
    skewness_df['dataset'] = datasets
    chaos_df['label'] = labels
    chaos_df['dataset'] = datasets
    
    # Group analysis
    print("\n=== SKEWNESS ANALYSIS BY LABEL ===")
    skew_by_label = skewness_df.groupby('label').agg({
        'skewness': ['mean', 'std'],
        'kurtosis': ['mean', 'std'],
        'skewness_magnitude': ['mean', 'std']
    }).round(4)
    print(skew_by_label)
    
    print("\n=== CHAOS/DISORDER ANALYSIS BY LABEL ===")
    chaos_by_label = chaos_df.groupby('label').agg({
        'shannon_entropy': ['mean', 'std'],
        'total_disorder': ['mean', 'std'],
        'fractal_dimension': ['mean', 'std'],
        'local_chaos': ['mean', 'std']
    }).round(4)
    print(chaos_by_label)
    
    print("\n=== COMPARISON BY DATASET ===")
    dataset_comparison = chaos_df.groupby('dataset').agg({
        'shannon_entropy': 'mean',
        'total_disorder': 'mean',
        'coefficient_of_variation': 'mean'
    }).round(4)
    print(dataset_comparison)
    
    return skew_by_label, chaos_by_label, dataset_comparison

def save_analysis_results(skewness_df, chaos_df, labels, datasets):
    """Save analysis results to files"""
    print("💾 Saving analysis results...")
    
    # Combine all metrics
    combined_df = pd.DataFrame({
        'sample_idx': skewness_df['sample_idx'],
        'label': labels,
        'dataset': datasets,
        # Skewness metrics
        'skewness': skewness_df['skewness'],
        'kurtosis': skewness_df['kurtosis'],
        'excess_kurtosis': skewness_df['excess_kurtosis'],
        'asymmetry_ratio': skewness_df['asymmetry_ratio'],
        'skewness_magnitude': skewness_df['skewness_magnitude'],
        # Chaos/disorder metrics
        'shannon_entropy': chaos_df['shannon_entropy'],
        'feature_variance': chaos_df['feature_variance'],
        'coefficient_of_variation': chaos_df['coefficient_of_variation'],
        'local_chaos': chaos_df['local_chaos'],
        'fractal_dimension': chaos_df['fractal_dimension'],
        'total_disorder': chaos_df['total_disorder'],
        'roughness': chaos_df['roughness']
    })
    
    # Save to CSV
    combined_df.to_csv('/workspace/gigapath_statistical_features.csv', index=False)
    print("✅ Saved to: /workspace/gigapath_statistical_features.csv")
    
    # Save summary statistics
    summary_stats = {
        'skewness_by_label': combined_df.groupby('label')['skewness'].describe(),
        'chaos_by_label': combined_df.groupby('label')['total_disorder'].describe(),
        'entropy_by_dataset': combined_df.groupby('dataset')['shannon_entropy'].describe()
    }
    
    with open('/workspace/statistical_analysis_summary.pkl', 'wb') as f:
        pickle.dump(summary_stats, f)
    print("✅ Saved summary to: /workspace/statistical_analysis_summary.pkl")
    
    return combined_df

def main():
    """Main analysis pipeline"""
    print("🚀 Starting Statistical Feature Analysis for GigaPath Embeddings")
    print("=" * 70)
    
    # Load embeddings
    features, labels, datasets, filenames = load_embeddings_cache()
    if features is None:
        print("❌ Failed to load embeddings cache!")
        return
    
    # Compute skewness metrics
    print("\n" + "=" * 50)
    skewness_df = compute_skewness_metrics(features)
    
    # Compute chaos/disorder metrics
    print("\n" + "=" * 50) 
    chaos_df = compute_chaos_disorder_metrics(features)
    
    # Analyze patterns
    print("\n" + "=" * 50)
    analyze_by_dataset_label(skewness_df, chaos_df, labels, datasets)
    
    # Save results
    print("\n" + "=" * 50)
    combined_df = save_analysis_results(skewness_df, chaos_df, labels, datasets)
    
    print(f"\n🎉 Analysis complete! Processed {len(features)} samples.")
    print("📁 Results saved to CSV and summary files.")
    
    # Quick preview of interesting findings
    print("\n=== QUICK INSIGHTS ===")
    print(f"Highest chaos sample: {combined_df.loc[combined_df['total_disorder'].idxmax(), 'label']} "
          f"(disorder: {combined_df['total_disorder'].max():.4f})")
    print(f"Most skewed sample: {combined_df.loc[combined_df['skewness_magnitude'].idxmax(), 'label']} "
          f"(skewness: {combined_df.loc[combined_df['skewness_magnitude'].idxmax(), 'skewness']:.4f})")

if __name__ == "__main__":
    main()