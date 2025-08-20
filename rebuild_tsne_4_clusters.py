#!/usr/bin/env python3
"""
Rebuild t-SNE with the same 4 biological clusters as the successful UMAP.

Strategy:
1. Load the successful 4-cluster UMAP cache
2. Rebuild only the t-SNE coordinates with better parameters
3. Keep the same biological clustering structure
"""

import pickle
import numpy as np
from sklearn.manifold import TSNE

def rebuild_tsne_with_4_clusters():
    """Rebuild t-SNE to match the successful UMAP 4-cluster structure."""
    
    print("🔄 Loading successful 4-cluster cache...")
    cache = pickle.load(open("embeddings_cache_4_BIOLOGICAL_CLUSTERS.pkl", "rb"))
    
    features = np.array(cache["combined"]["features"])
    labels = cache["combined"]["labels"]
    datasets = cache["combined"]["datasets"]
    
    # Use the same biological mapping that worked for UMAP
    biological_labels = []
    biological_numeric = []
    
    for label, dataset in zip(labels, datasets):
        if label == "normal" and dataset == "bach":
            biological_labels.append("normal")
            biological_numeric.append(0)
        elif (label == "benign" and dataset == "bach") or (label == "benign" and dataset == "breakhis"):
            biological_labels.append("benign")  
            biological_numeric.append(1)
        elif label == "insitu" and dataset == "bach":
            biological_labels.append("insitu")
            biological_numeric.append(2)
        elif (label == "invasive" and dataset == "bach") or (label == "malignant" and dataset == "breakhis"):
            biological_labels.append("malignant")
            biological_numeric.append(3)
        else:
            biological_labels.append("unknown")
            biological_numeric.append(-1)
    
    print(f"📊 Biological clustering for t-SNE:")
    for bio_class in ["normal", "benign", "insitu", "malignant"]:
        count = sum(1 for bl in biological_labels if bl == bio_class)
        print(f"   {bio_class}: {count} samples")
    
    print("🗺️  Creating 4-cluster biological t-SNE...")
    
    # Optimized t-SNE parameters for better 4-cluster separation
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=40,           # Optimal for 4 clusters with large dataset
        early_exaggeration=15,   # Stronger early separation
        learning_rate=300,       # Higher learning rate for better separation
        n_iter=1500,            # More iterations for convergence
        metric="cosine"         # Match UMAP metric
    )
    
    print("   Computing t-SNE... (this will take 3-5 minutes)")
    tsne_coords = tsne.fit_transform(features)
    
    # Update cache with new t-SNE coordinates
    cache["combined"]["coordinates"]["tsne"] = tsne_coords
    cache["tsne_optimization"] = {
        "method": "biological_4_cluster_tsne",
        "parameters": {
            "perplexity": 40,
            "early_exaggeration": 15,
            "learning_rate": 300,
            "n_iter": 1500
        }
    }
    
    # Save updated cache
    output_file = "embeddings_cache_4_CLUSTERS_FIXED_TSNE.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(cache, f)
    
    # Analyze new t-SNE clustering
    print(f"\\n🔍 4-cluster t-SNE analysis...")
    
    for bio_class in ["normal", "benign", "insitu", "malignant"]:
        bio_indices = [i for i, bl in enumerate(biological_labels) if bl == bio_class]
        if bio_indices:
            bio_coords = tsne_coords[bio_indices]
            center = np.mean(bio_coords, axis=0)
            print(f"   {bio_class.upper()}: center[{center[0]:.2f}, {center[1]:.2f}] ({len(bio_indices)} samples)")
            print(f"     Range: X[{bio_coords[:, 0].min():.2f}, {bio_coords[:, 0].max():.2f}] Y[{bio_coords[:, 1].min():.2f}, {bio_coords[:, 1].max():.2f}]")
    
    # Calculate inter-cluster distances
    centers = {}
    for bio_class in ["normal", "benign", "insitu", "malignant"]:
        bio_indices = [i for i, bl in enumerate(biological_labels) if bl == bio_class]
        if bio_indices:
            centers[bio_class] = np.mean(tsne_coords[bio_indices], axis=0)
    
    print(f"\\n📏 t-SNE 4-cluster distances:")
    bio_classes = ["normal", "benign", "insitu", "malignant"]
    for i, class1 in enumerate(bio_classes):
        for class2 in bio_classes[i+1:]:
            if class1 in centers and class2 in centers:
                dist = np.linalg.norm(centers[class1] - centers[class2])
                print(f"   {class1} ↔ {class2}: {dist:.2f}")
    
    print(f"\\n✅ 4-cluster t-SNE saved: {output_file}")
    return output_file

if __name__ == "__main__":
    rebuild_tsne_with_4_clusters()
