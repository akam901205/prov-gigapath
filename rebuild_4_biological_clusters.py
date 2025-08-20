#!/usr/bin/env python3
"""
Create 4 distinct biological clusters regardless of dataset origin:
1. NORMAL: BACH normal
2. BENIGN: BACH benign + BreakHis benign  
3. INSITU: BACH insitu
4. MALIGNANT: BACH invasive + BreakHis malignant
"""

import pickle
import numpy as np
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

def create_biological_4_clusters():
    """Create UMAP with 4 distinct biological clusters."""
    
    print("🔄 Loading original cache...")
    cache = pickle.load(open("embeddings_cache_BACH_4CLASS.pkl", "rb"))
    
    features = np.array(cache["combined"]["features"])
    labels = cache["combined"]["labels"]
    datasets = cache["combined"]["datasets"]
    filenames = cache["combined"]["filenames"]
    
    # Create biological clustering labels
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
    
    print(f"📊 Biological clustering:")
    for bio_class in ["normal", "benign", "insitu", "malignant"]:
        count = sum(1 for bl in biological_labels if bl == bio_class)
        print(f"   {bio_class}: {count} samples")
    
    print("🗺️  Creating 4-cluster biological UMAP...")
    umap_reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=1.0,         # High separation between clusters
        n_components=2,
        metric="cosine",
        random_state=42,
        target_metric="categorical",
        target_weight=0.8,    # Strong supervision for 4 clusters
        spread=3.0           # Increase spread between clusters
    )
    
    umap_coords = umap_reducer.fit_transform(features, y=biological_numeric)
    
    print("   Creating t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=50)
    tsne_coords = tsne.fit_transform(features)
    
    print("   Creating PCA...")
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(features)
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(scaled_features)
    
    # Save 4-cluster cache
    cluster_cache = {
        "combined": {
            "features": features,
            "labels": labels,  # Original labels
            "biological_labels": biological_labels,  # New biological labels
            "datasets": datasets,
            "filenames": filenames,
            "coordinates": {
                "umap": umap_coords,
                "tsne": tsne_coords,
                "pca": pca_coords
            }
        },
        "robust_scaler": scaler,
        "biological_mapping": {
            "normal": 0,    # BACH normal only
            "benign": 1,    # BACH benign + BreakHis benign
            "insitu": 2,    # BACH insitu only  
            "malignant": 3  # BACH invasive + BreakHis malignant
        },
        "cluster_info": {
            "method": "biological_4_cluster_supervised",
            "clusters": 4,
            "supervision_weight": 0.8
        }
    }
    
    output_file = "embeddings_cache_4_BIOLOGICAL_CLUSTERS.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(cluster_cache, f)
    
    # Analyze 4-cluster results
    print(f"\\n🔍 4-cluster biological analysis...")
    
    for bio_class in ["normal", "benign", "insitu", "malignant"]:
        bio_indices = [i for i, bl in enumerate(biological_labels) if bl == bio_class]
        if bio_indices:
            bio_coords = umap_coords[bio_indices]
            center = np.mean(bio_coords, axis=0)
            print(f"   {bio_class.upper()}: center[{center[0]:.2f}, {center[1]:.2f}] ({len(bio_indices)} samples)")
            
            # Show composition
            composition = {}
            for idx in bio_indices:
                orig_label = labels[idx]
                dataset = datasets[idx]
                key = f"{orig_label}_{dataset}"
                composition[key] = composition.get(key, 0) + 1
            
            comp_str = ", ".join([f"{k}:{v}" for k, v in composition.items()])
            print(f"     Composition: {comp_str}")
    
    # Calculate inter-cluster distances
    centers = {}
    for bio_class in ["normal", "benign", "insitu", "malignant"]:
        bio_indices = [i for i, bl in enumerate(biological_labels) if bl == bio_class]
        if bio_indices:
            centers[bio_class] = np.mean(umap_coords[bio_indices], axis=0)
    
    print(f"\\n📏 4-cluster distances:")
    bio_classes = ["normal", "benign", "insitu", "malignant"]
    for i, class1 in enumerate(bio_classes):
        for class2 in bio_classes[i+1:]:
            if class1 in centers and class2 in centers:
                dist = np.linalg.norm(centers[class1] - centers[class2])
                print(f"   {class1} ↔ {class2}: {dist:.2f}")
    
    print(f"\\n✅ 4-biological-cluster UMAP saved: {output_file}")
    return output_file

if __name__ == "__main__":
    create_biological_4_clusters()
