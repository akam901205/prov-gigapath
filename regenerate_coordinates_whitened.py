#!/usr/bin/env python3
"""
Regenerate UMAP, t-SNE, PCA Coordinates for Prototype Whitened Embeddings
"""
import numpy as np
import pickle
import os
import umap.umap_ as umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def regenerate_whitened_coordinates():
    """Regenerate coordinates for whitened embeddings"""
    
    print("🚀 REGENERATING COORDINATES FOR WHITENED EMBEDDINGS")
    print("=" * 70)
    
    # Load prototype whitened cache
    with open("/workspace/embeddings_cache_PROTOTYPE_WHITENED.pkl", 'rb') as f:
        cache = pickle.load(f)
    
    whitened_features = np.array(cache['combined']['features'])
    labels = cache['combined']['labels']
    datasets = cache['combined']['datasets']
    filenames = cache['combined']['filenames']
    
    print(f"✅ Loaded whitened embeddings: {whitened_features.shape}")
    
    # Create binary labels for supervised learning
    binary_labels = []
    for label in labels:
        if label in ['benign', 'normal']:
            binary_labels.append(0)  # benign
        elif label in ['malignant', 'invasive', 'insitu']:
            binary_labels.append(1)  # malignant
        else:
            binary_labels.append(0)  # default to benign
    
    numeric_labels = np.array(binary_labels)
    
    print(f"📊 Label distribution: {np.bincount(numeric_labels)} (benign=0, malignant=1)")
    
    # 1. REGENERATE UMAP with whitened features
    print("\n🗺️ Computing UMAP on whitened embeddings...")
    umap_reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.0,
        n_components=2,
        metric='cosine',  # Perfect for L2 normalized whitened features
        target_metric='categorical',
        random_state=42
    )
    
    umap_coords = umap_reducer.fit_transform(whitened_features, y=numeric_labels)
    umap_score = silhouette_score(umap_coords, numeric_labels)
    print(f"   ✅ UMAP coordinates: {umap_coords.shape}")
    print(f"   📊 UMAP silhouette score: {umap_score:.4f}")
    
    # 2. REGENERATE t-SNE with whitened features
    print("\n📈 Computing t-SNE on whitened embeddings...")
    tsne_reducer = TSNE(
        n_components=2,
        perplexity=50,
        learning_rate='auto',
        max_iter=1000,
        random_state=42,
        metric='cosine',  # Perfect for L2 normalized whitened features
        init='pca'
    )
    
    tsne_coords = tsne_reducer.fit_transform(whitened_features)
    tsne_score = silhouette_score(tsne_coords, numeric_labels)
    print(f"   ✅ t-SNE coordinates: {tsne_coords.shape}")
    print(f"   📊 t-SNE silhouette score: {tsne_score:.4f}")
    
    # 3. REGENERATE PCA with whitened features
    print("\n📐 Computing PCA on whitened embeddings...")
    pca_reducer = PCA(n_components=2, random_state=42)
    pca_coords = pca_reducer.fit_transform(whitened_features)
    pca_score = silhouette_score(pca_coords, numeric_labels)
    print(f"   ✅ PCA coordinates: {pca_coords.shape}")
    print(f"   📊 PCA silhouette score: {pca_score:.4f}")
    print(f"   🔍 PCA explained variance: {pca_reducer.explained_variance_ratio_}")
    
    # 4. Update cache with new coordinates
    print("\n💾 Updating cache with new coordinates...")
    
    # Add coordinates to existing cache
    cache['combined']['coordinates'] = {
        'umap': umap_coords,
        'tsne': tsne_coords,
        'pca': pca_coords
    }
    
    # Update metadata
    cache['coordinate_metadata'] = {
        'umap_silhouette': float(umap_score),
        'tsne_silhouette': float(tsne_score), 
        'pca_silhouette': float(pca_score),
        'umap_params': {
            'n_neighbors': 15,
            'min_dist': 0.0,
            'metric': 'cosine',
            'supervised': True
        },
        'tsne_params': {
            'perplexity': 50,
            'metric': 'cosine',
            'init': 'pca'
        },
        'pca_explained_variance': pca_reducer.explained_variance_ratio_.tolist()
    }
    
    # Add binary labels for convenience
    cache['combined']['binary_labels'] = binary_labels
    
    # Save updated cache
    output_path = "/workspace/embeddings_cache_PROTOTYPE_WHITENED.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(cache, f)
    
    file_size = os.path.getsize(output_path) / (1024*1024)
    
    print(f"\n✅ COORDINATES REGENERATED SUCCESSFULLY!")
    print(f"   📁 Updated cache: {output_path}")
    print(f"   💾 Size: {file_size:.1f} MB")
    print(f"   🗺️ UMAP silhouette: {umap_score:.4f}")
    print(f"   📈 t-SNE silhouette: {tsne_score:.4f}")
    print(f"   📐 PCA silhouette: {pca_score:.4f}")
    
    # Quick visualization test
    print(f"\n🧪 Quick coordinate test...")
    test_indices = [200, 201, 0, 1]  # b001, b002, iv001, iv002
    
    for idx in test_indices:
        if idx < len(filenames):
            filename = filenames[idx]
            true_label = labels[idx]
            umap_pos = umap_coords[idx]
            
            print(f"   {filename} ({true_label}): UMAP=({umap_pos[0]:.3f}, {umap_pos[1]:.3f})")
    
    print(f"\n🎉 PROTOTYPE WHITENED CACHE WITH COORDINATES READY!")
    
    return cache

if __name__ == "__main__":
    regenerate_whitened_coordinates()