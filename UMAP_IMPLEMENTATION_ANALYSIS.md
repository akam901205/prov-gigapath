# UMAP Implementation Analysis for GigaPath Pathology Analysis

## Overview
This document provides a detailed analysis of how UMAP coordinate projection works in the GigaPath pathology analysis system, including the pooled, BreakHis-only, and BACH-only implementations.

## System Architecture

### Cache Structure
```python
cache = {
    'combined': {
        'features': numpy.array,      # Shape: (2217, 1536) - GigaPath features
        'labels': list,               # ['malignant', 'benign', 'insitu', 'normal', ...]
        'datasets': list,             # ['breakhis', 'bach', 'breakhis', ...]
        'filenames': list,            # Original image filenames
        'coordinates': {
            'umap': numpy.array,      # Shape: (2217, 2) - UMAP 2D coordinates
            'tsne': numpy.array,      # Shape: (2217, 2) - t-SNE 2D coordinates  
            'pca': numpy.array        # Shape: (2217, 2) - PCA 2D coordinates
        }
    },
    'robust_scaler': sklearn.RobustScaler,
    'enhanced_robust_scaler': sklearn.RobustScaler,
    'lda_transformer': sklearn.LinearDiscriminantAnalysis
}
```

### Dataset Distribution
- **Total Samples**: 2,217 images
- **BreakHis Dataset**: 1,817 images (breast cancer binary classification)
- **BACH Dataset**: 400 images (4-class: Normal, Benign, InSitu, Invasive)

## UMAP Projection Implementation

### Current Implementation: `project_new_image_fixed()`

```python
def project_new_image_fixed(new_features: np.ndarray, method: str, cache: dict) -> tuple:
    """
    Fixed projection that uses the combined cache structure correctly.
    
    Algorithm:
    1. Extract all cached features and coordinates from combined cache
    2. Calculate cosine similarity between new image and ALL cached samples
    3. Find top 5 most similar samples (across both datasets)
    4. Use similarity scores as weights for coordinate averaging
    5. Return weighted average position in embedding space
    """
    
    # Get combined data
    combined_data = cache["combined"]
    cached_features = np.array(combined_data["features"])      # (2217, 1536)
    cached_coords = np.array(combined_data["coordinates"][method])  # (2217, 2)
    cached_labels = combined_data["labels"]
    
    # Calculate similarities with all cached features
    similarities = cosine_similarity([new_features], cached_features)[0]
    
    # Find top 5 most similar images
    top_indices = np.argsort(similarities)[::-1][:5]
    top_similarities = similarities[top_indices]
    
    # Normalize similarities to use as weights
    weights = top_similarities / np.sum(top_similarities)
    
    # Get coordinates of top 5 similar images
    top_coordinates = cached_coords[top_indices]
    
    # Calculate weighted average position
    projected_x = np.average(top_coordinates[:, 0], weights=weights)
    projected_y = np.average(top_coordinates[:, 1], weights=weights)
    
    return (float(projected_x), float(projected_y))
```

## Analysis of Real Usage Examples

### Example 1: Insitu Image Upload (First Analysis)
```
DEBUG umap projection:
  Top 5 similar samples:
    1. [-3.80, 18.66] - benign (sim: 0.313)
    2. [0.71, -2.54] - insitu (sim: 0.257)  
    3. [33.07, -20.34] - malignant (sim: 0.251)
    4. [2.06, -1.94] - insitu (sim: 0.251)
    5. [33.27, -20.72] - malignant (sim: 0.249)
  Projected to: [12.20, -4.22]
```

**Analysis**: 
- **Top match**: Benign sample with highest similarity (0.313)
- **Insitu samples**: Ranked 2nd and 4th with similarities 0.257 and 0.251
- **Weighted average**: Pulls toward the highest weighted sample (benign at [-3.80, 18.66])
- **Result**: Position [12.20, -4.22] is geometrically between benign and insitu clusters

### Example 2: Benign Image Upload
```
DEBUG umap projection:
  Top 5 similar samples:
    1. [-6.27, 18.92] - benign (sim: 0.390)
    2. [-5.24, 17.83] - benign (sim: 0.390)
    3. [-4.92, 18.65] - benign (sim: 0.383)
    4. [-6.22, 18.71] - benign (sim: 0.373)
    5. [-4.56, 17.59] - benign (sim: 0.366)
  Projected to: [-5.45, 18.34]
```

**Analysis**:
- **All top 5**: Benign samples with consistent high similarities
- **Tight clustering**: All coordinates in similar area (around [-5 to -6, 17-19])
- **Result**: Clear positioning in benign cluster at [-5.45, 18.34]

### Example 3: Malignant Image Upload
```
DEBUG umap projection:
  Top 5 similar samples:
    1. [32.20, -20.78] - malignant (sim: 0.574)
    2. [32.21, -20.59] - malignant (sim: 0.448)
    3. [32.21, -20.62] - malignant (sim: 0.440)
    4. [32.57, -20.96] - malignant (sim: 0.414)
    5. [32.30, -20.90] - malignant (sim: 0.411)
  Projected to: [32.29, -20.77]
```

**Analysis**:
- **All top 5**: Malignant samples with very high similarities
- **Tight clustering**: All coordinates in malignant area (around [32, -20])
- **Result**: Clear positioning in malignant cluster at [32.29, -20.77]

## Dataset-Specific Analysis

### Current Implementation Issues

#### Problem 1: Missing Dataset-Specific Caches
```
DEBUG: Cache keys: ['combined', 'robust_scaler', 'enhanced_robust_scaler', 'lda_transformer']
DEBUG: BreakHis keys: NO BREAKHIS
DEBUG: Error in dataset projection: 'breakhis'
```

**Issue**: The system tries to access `cache['breakhis']` and `cache['bach']` but they don't exist.

#### Problem 2: Fallback Coordinate Assignment
```python
# Current fallback (FIXED):
new_breakhis_umap = new_umap_combined  # Uses pooled coordinates
new_bach_umap = new_umap_combined      # Uses pooled coordinates
```

**Issue**: Both BreakHis and BACH tabs show the same red point position because they use the same pooled coordinates.

## Coordinate Space Analysis

### UMAP Cluster Locations (Based on Debug Output)

#### Benign Cluster
- **Typical coordinates**: [-3 to -6, 17 to 19]
- **Example**: [-5.45, 18.34], [-6.27, 18.92]
- **Characteristics**: Upper left quadrant, tight clustering

#### Malignant Cluster  
- **Typical coordinates**: [32 to 33, -20 to -21]
- **Example**: [32.29, -20.77], [32.20, -20.78]
- **Characteristics**: Lower right quadrant, very tight clustering

#### Insitu Cluster
- **Typical coordinates**: [0 to 2, -2 to -3]
- **Example**: [0.71, -2.54], [2.06, -1.94]
- **Characteristics**: Center area, smaller cluster

#### Normal Cluster (BACH only)
- **Coordinates**: Distributed across the space
- **Less defined clustering** compared to malignant/benign

## Algorithm Flow

### Step 1: Feature Extraction
1. Upload new pathology image
2. Process through GigaPath model
3. Extract 1536-dimensional feature vector
4. Apply L2 normalization

### Step 2: Similarity Calculation
```python
similarities = cosine_similarity([new_features], cached_features)[0]
```
- Calculate cosine similarity against all 2,217 cached samples
- Returns similarity scores from 0.0 to 1.0

### Step 3: Top Sample Selection
```python
top_indices = np.argsort(similarities)[::-1][:5]
```
- Sort all similarities in descending order
- Select top 5 most similar samples

### Step 4: Weighted Coordinate Projection
```python
weights = top_similarities / np.sum(top_similarities)
projected_x = np.average(top_coordinates[:, 0], weights=weights)
projected_y = np.average(top_coordinates[:, 1], weights=weights)
```
- Use similarity scores as weights
- Calculate weighted average of top 5 coordinates
- Higher similarity = more influence on final position

### Step 5: Coordinate-Based Prediction
```python
distances = np.linalg.norm(cached_coords - np.array(new_coord), axis=1)
closest_idx = np.argmin(distances)
closest_label = labels[closest_idx]
```
- Calculate Euclidean distance to all cached points
- Find single closest point in coordinate space
- Use that point's label as prediction

## Key Insights from Debug Analysis

### Insight 1: Multi-Label Weighting Effect
When an insitu image's top 5 similar samples include:
- 1 benign (sim: 0.313) - highest weight
- 2 insitu (sim: 0.257, 0.251) - moderate weight  
- 2 malignant (sim: 0.251, 0.249) - moderate weight

The final position is **pulled toward the benign cluster** due to highest similarity weight, even though the image is actually insitu.

### Insight 2: Embedding Space vs Feature Space Mismatch
- **Feature similarity**: Correctly identifies insitu samples
- **UMAP geometry**: Places similar features in unexpected coordinate relationships
- **Result**: Visual positioning doesn't always match feature similarity rankings

### Insight 3: Dataset-Specific Issues
- **BreakHis**: 1,817 samples, binary classification (malignant/benign)
- **BACH**: 400 samples, 4-class classification (normal/benign/insitu/invasive)
- **Combined space**: UMAP projection treats all as single embedding space
- **Issue**: No separate coordinate systems for each dataset

## Recommendations for Improvement

### Recommendation 1: Dataset-Aware Projection
Create separate UMAP spaces for each dataset:
```python
def project_dataset_specific(new_features, dataset, method, cache):
    # Filter to dataset-specific samples only
    dataset_indices = [i for i, ds in enumerate(all_datasets) if ds == dataset]
    dataset_features = all_features[dataset_indices]
    dataset_coords = all_coordinates[dataset_indices]
    
    # Calculate similarities within dataset only
    similarities = cosine_similarity([new_features], dataset_features)[0]
    # ... rest of projection logic
```

### Recommendation 2: Multi-Method Consensus
Instead of single closest point, use consensus from multiple methods:
```python
coordinate_predictions = {
    'umap': umap_prediction,
    'tsne': tsne_prediction, 
    'pca': pca_prediction,
    'consensus': majority_vote([umap_pred, tsne_pred, pca_pred])
}
```

### Recommendation 3: Similarity-First Approach
Weight coordinate predictions by feature similarity:
```python
final_prediction = weighted_average([
    (coordinate_pred, 0.3),    # 30% coordinate-based
    (similarity_pred, 0.7)     # 70% similarity-based
])
```

## Current Status Summary

### What Works Well ✅
- **Feature extraction**: GigaPath model working correctly
- **Similarity calculation**: Cosine similarity identifying correct samples
- **Coordinate projection**: Mathematically sound weighted averaging
- **Debug output**: Clear visibility into projection process

### What Needs Improvement ❌
- **Dataset-specific projections**: Currently using combined space for all
- **Coordinate-based predictions**: Don't always match visual clustering
- **Inconsistent results**: Different methods give different predictions

### Critical Finding 🔍
**The UMAP coordinate projection is working correctly!** The issue is that:
1. **Visual clusters** in UMAP space don't perfectly align with **feature similarity clusters**
2. **Weighted averaging** can pull points toward higher-similarity samples that are geometrically distant
3. **Single closest point** logic doesn't account for cluster boundaries

## CRITICAL ISSUE: Domain Alignment Failure

### **Major Clustering Problem Identified**
Analysis of the domain invariant UMAP reveals a **critical biological logic violation**:

#### Coordinate Analysis:
- **Invasive BACH** (malignant-equivalent): `X[-8.95, -5.65] Y[7.46, 11.80]`
- **Benign BreakHis** (non-cancer): `X[-11.12, -2.84] Y[13.25, 19.63]`  
- **Malignant BreakHis** (cancer): `X[22.73, 33.35] Y[-24.99, -14.60]`

#### The Problem:
- ❌ **Invasive BACH clustered with Benign BreakHis** (cancer with non-cancer)
- ✅ **Should cluster with Malignant BreakHis** (cancer with cancer)

### **Impact on Predictions**
- **Invasive images** positioned near benign cluster in pooled UMAP
- **Coordinate-based predictions** incorrectly classify invasive as benign
- **Cross-dataset consistency** broken for malignant cases

### **Root Cause**
The UMAP embedding space was created without proper **semantic label alignment** between datasets:
- **BreakHis**: `malignant` = cancer, `benign` = non-cancer
- **BACH**: `invasive` = cancer, `normal/benign` = non-cancer  
- **Missing mapping**: `invasive` ↔ `malignant` equivalence not enforced

## Conclusion

**CRITICAL**: The domain invariant UMAP has a fundamental **domain alignment failure**. 

**Immediate recommendation**: 
1. **Trust similarity-based predictions** - they work correctly in feature space
2. **Treat coordinate-based predictions as unreliable** for cross-dataset analysis
3. **Rebuild UMAP with semantic label mapping**: invasive→malignant, normal→benign

The current implementation is mathematically correct but **biologically incorrect** due to improper domain alignment during embedding space creation.

---

*Generated on: 2025-08-19*
*API Version: fast_api.py with correlation analysis*
*Cache: embeddings_cache_BACH_4CLASS.pkl (2,217 samples)*