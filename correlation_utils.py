from scipy.stats import pearsonr, spearmanr
import numpy as np

def calculate_correlation_predictions(new_features, cached_features, labels, datasets):
    """Calculate both Pearson and Spearman correlation-based predictions."""
    
    # Flatten features if needed
    if len(new_features.shape) > 1:
        new_features_flat = new_features.flatten()
    else:
        new_features_flat = new_features
    
    pearson_similarities = []
    spearman_similarities = []
    
    # Calculate correlations with all cached embeddings
    for cached_emb in cached_features:
        if len(cached_emb.shape) > 1:
            cached_flat = cached_emb.flatten()
        else:
            cached_flat = cached_emb
        
        # Pearson correlation
        try:
            pearson_corr, _ = pearsonr(new_features_flat, cached_flat)
            if np.isnan(pearson_corr):
                pearson_corr = 0.0
            # Convert from -1,1 to 0,1 scale
            pearson_sim = (pearson_corr + 1) / 2
        except:
            pearson_sim = 0.0
        
        # Spearman correlation
        try:
            spearman_corr, _ = spearmanr(new_features_flat, cached_flat)
            if np.isnan(spearman_corr):
                spearman_corr = 0.0
            # Convert from -1,1 to 0,1 scale
            spearman_sim = (spearman_corr + 1) / 2
        except:
            spearman_sim = 0.0
            
        pearson_similarities.append(float(pearson_sim))
        spearman_similarities.append(float(spearman_sim))
    
    # Calculate predictions for both methods
    pearson_predictions = _calculate_correlation_method_predictions(
        pearson_similarities, labels, datasets, "pearson"
    )
    spearman_predictions = _calculate_correlation_method_predictions(
        spearman_similarities, labels, datasets, "spearman"
    )
    
    return {
        "pearson": pearson_predictions,
        "spearman": spearman_predictions
    }

def _calculate_correlation_method_predictions(similarities, labels, datasets, method_name):
    """Helper function to calculate predictions for a specific correlation method."""
    similarities = np.array(similarities)
    top_indices = np.argsort(similarities)[::-1]
    
    predictions = {}
    
    # For each dataset separately
    for dataset in ["breakhis", "bach"]:
        dataset_indices = [i for i, ds in enumerate(datasets) if ds == dataset]
        
        if dataset_indices:
            # Find top matches within this dataset
            dataset_similarities = [(similarities[i], i, labels[i]) for i in dataset_indices]
            dataset_similarities.sort(reverse=True)
            
            # Get top 5 matches in this dataset
            top_5_dataset = dataset_similarities[:5]
            
            if top_5_dataset:
                # Best match
                best_similarity, best_idx, best_label = top_5_dataset[0]
                
                # Consensus from top 5
                top_5_labels = [match[2] for match in top_5_dataset]
                label_counts = {}
                for label in top_5_labels:
                    label_counts[label] = label_counts.get(label, 0) + 1
                
                consensus_label = max(label_counts.items(), key=lambda x: x[1])[0]
                consensus_confidence = max(label_counts.values()) / 5.0
                
                predictions[dataset] = {
                    "best_match": {
                        "label": best_label,
                        "similarity": float(best_similarity),
                        "confidence": float(best_similarity)
                    },
                    "consensus": {
                        "label": consensus_label,
                        "confidence": float(consensus_confidence * best_similarity),
                        "vote_breakdown": label_counts
                    }
                }
    
    return {
        "method": method_name,
        "overall_top_similarity": float(max(similarities)) if len(similarities) > 0 else 0.0,
        "dataset_predictions": predictions
    }
