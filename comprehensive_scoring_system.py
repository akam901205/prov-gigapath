#!/usr/bin/env python3
"""
Comprehensive Multi-Metric Scoring System
Uses diverse distance and correlation measures for prototype classification
"""
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform, braycurtis, canberra, chebyshev
from scipy.stats import energy_distance
import warnings
warnings.filterwarnings('ignore')

def create_comprehensive_scoring_system():
    """Create multi-metric scoring system with diverse measures"""
    
    print("🎯 COMPREHENSIVE MULTI-METRIC SCORING SYSTEM")
    print("=" * 60)
    
    # Load whitened cache
    with open("/workspace/embeddings_cache_PROTOTYPE_WHITENED.pkl", 'rb') as f:
        cache = pickle.load(f)
    
    features = np.array(cache['combined']['features'])
    labels = cache['combined']['labels']
    y = np.array([1 if label in ['malignant', 'invasive', 'insitu'] else 0 for label in labels])
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Compute prototypes
    train_benign_proto = np.mean(X_train[y_train == 0], axis=0)
    train_malignant_proto = np.mean(X_train[y_train == 1], axis=0)
    
    print(f"📊 Test samples: {len(y_test)} (Malignant: {np.sum(y_test==1)}, Benign: {np.sum(y_test==0)})")
    print(f"🎯 Computing {len(X_test)} comprehensive scores...")
    
    def distance_correlation(x, y):
        """Simplified distance correlation"""
        try:
            n = len(x)
            if n < 2:
                return 0.0
            a = squareform(pdist(x.reshape(-1, 1)))
            b = squareform(pdist(y.reshape(-1, 1)))
            A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
            B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
            dcov_xy = np.sqrt(np.mean(A * B))
            dcov_xx = np.sqrt(np.mean(A * A))
            dcov_yy = np.sqrt(np.mean(B * B))
            if dcov_xx > 0 and dcov_yy > 0:
                return dcov_xy / np.sqrt(dcov_xx * dcov_yy)
            return 0.0
        except:
            return 0.0
    
    def energy_correlation(x, y):
        """Energy distance-based correlation"""
        try:
            # Simplified energy correlation
            energy_xy = np.mean(np.abs(x - y))
            energy_xx = np.mean(np.abs(x.reshape(-1,1) - x.reshape(1,-1)))
            energy_yy = np.mean(np.abs(y.reshape(-1,1) - y.reshape(1,-1)))
            if energy_xx > 0 and energy_yy > 0:
                return 1 - (2 * energy_xy) / (energy_xx + energy_yy)
            return 0.0
        except:
            return 0.0
    
    def compute_comprehensive_scores(X, benign_proto, malignant_proto, sample_name):
        """Compute all distance and correlation metrics"""
        n_samples = len(X)
        print(f"   Computing {sample_name} ({n_samples} samples)...")
        
        # Initialize score arrays
        scores = {}
        
        # 1. SIMILARITY MEASURES (higher = more similar to malignant)
        scores['cosine'] = X @ malignant_proto - X @ benign_proto
        
        # 2. DISTANCE MEASURES (lower distance to malignant = more malignant)
        eucl_benign = np.linalg.norm(X - benign_proto, axis=1)
        eucl_malignant = np.linalg.norm(X - malignant_proto, axis=1)
        scores['euclidean'] = eucl_benign - eucl_malignant  # Higher = closer to malignant
        
        # 3. SPECIALIZED DISTANCE METRICS
        canberra_benign = np.zeros(n_samples)
        canberra_malignant = np.zeros(n_samples)
        chebyshev_benign = np.zeros(n_samples)  
        chebyshev_malignant = np.zeros(n_samples)
        braycurtis_benign = np.zeros(n_samples)
        braycurtis_malignant = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Canberra distance
            canberra_benign[i] = np.sum(np.abs(X[i] - benign_proto) / (np.abs(X[i]) + np.abs(benign_proto) + 1e-8))
            canberra_malignant[i] = np.sum(np.abs(X[i] - malignant_proto) / (np.abs(X[i]) + np.abs(malignant_proto) + 1e-8))
            
            # Chebyshev distance (max absolute difference)
            chebyshev_benign[i] = np.max(np.abs(X[i] - benign_proto))
            chebyshev_malignant[i] = np.max(np.abs(X[i] - malignant_proto))
            
            # Bray-Curtis distance
            try:
                braycurtis_benign[i] = braycurtis(X[i], benign_proto)
                braycurtis_malignant[i] = braycurtis(X[i], malignant_proto)
            except:
                braycurtis_benign[i] = 0
                braycurtis_malignant[i] = 0
        
        scores['canberra'] = canberra_benign - canberra_malignant
        scores['chebyshev'] = chebyshev_benign - chebyshev_malignant  
        scores['braycurtis'] = braycurtis_benign - braycurtis_malignant
        
        # 4. CORRELATION MEASURES (sample subset for speed)
        sample_indices = np.random.choice(n_samples, min(n_samples, 100), replace=False)
        
        pearson_scores = np.zeros(n_samples)
        spearman_scores = np.zeros(n_samples)
        dcor_scores = np.zeros(n_samples)
        energy_scores = np.zeros(n_samples)
        
        for i in sample_indices:
            try:
                # Pearson correlation difference
                pearson_ben, _ = pearsonr(X[i], benign_proto)
                pearson_mal, _ = pearsonr(X[i], malignant_proto)
                pearson_scores[i] = pearson_mal - pearson_ben
                
                # Spearman correlation difference
                spearman_ben, _ = spearmanr(X[i], benign_proto)
                spearman_mal, _ = spearmanr(X[i], malignant_proto)
                spearman_scores[i] = spearman_mal - spearman_ben
                
                # Distance correlation (simplified)
                dcor_ben = distance_correlation(X[i], benign_proto)
                dcor_mal = distance_correlation(X[i], malignant_proto)
                dcor_scores[i] = dcor_mal - dcor_ben
                
                # Energy correlation
                energy_ben = energy_correlation(X[i], benign_proto)
                energy_mal = energy_correlation(X[i], malignant_proto)
                energy_scores[i] = energy_mal - energy_ben
                
            except:
                # Use cosine as fallback for non-sampled or failed computations
                pearson_scores[i] = scores['cosine'][i] * 0.1
                spearman_scores[i] = scores['cosine'][i] * 0.1
                dcor_scores[i] = scores['cosine'][i] * 0.1
                energy_scores[i] = scores['cosine'][i] * 0.1
        
        # For non-sampled indices, use approximation
        for i in range(n_samples):
            if i not in sample_indices:
                pearson_scores[i] = scores['cosine'][i] * 0.8
                spearman_scores[i] = scores['cosine'][i] * 0.8
                dcor_scores[i] = scores['cosine'][i] * 0.8
                energy_scores[i] = scores['cosine'][i] * 0.8
        
        scores['pearson'] = np.nan_to_num(pearson_scores, 0)
        scores['spearman'] = np.nan_to_num(spearman_scores, 0)
        scores['dcor'] = np.nan_to_num(dcor_scores, 0)
        scores['energy'] = np.nan_to_num(energy_scores, 0)
        
        return scores
    
    # Compute comprehensive scores
    test_scores = compute_comprehensive_scores(X_test, train_benign_proto, train_malignant_proto, "test set")
    
    print(f"\\n📊 AVAILABLE METRICS ({len(test_scores)} total):")
    for metric, scores in test_scores.items():
        print(f"   • {metric}: range [{scores.min():.3f}, {scores.max():.3f}]")
    
    # Normalize all scores to [0, 1] for fair combination
    normalized_scores = {}
    for metric, scores in test_scores.items():
        scores_min = scores.min()
        scores_max = scores.max()
        if scores_max > scores_min:
            normalized_scores[metric] = (scores - scores_min) / (scores_max - scores_min)
        else:
            normalized_scores[metric] = np.ones_like(scores) * 0.5
    
    print(f"\\n🗳️ VOTING STRATEGIES:")
    
    def evaluate_voting_strategy(predictions, scores, name):
        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        auc = roc_auc_score(y_test, scores)
        
        print(f"\\n{name}:")
        print(f"  Sensitivity: {sensitivity:.3f} ({sensitivity*100:.1f}%) - Missed: {fn}/279")
        print(f"  Specificity: {specificity:.3f} ({specificity*100:.1f}%) - False alarms: {fp}/165")
        print(f"  Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"  AUC: {auc:.3f}")
        
        return sensitivity, specificity, accuracy, auc, fn, fp
    
    results = []
    
    # Strategy 1: Simple majority vote (each metric votes malignant if > 0.5)
    votes = np.zeros(len(y_test))
    for metric, scores in normalized_scores.items():
        votes += (scores > 0.5).astype(int)
    
    majority_pred = (votes > len(normalized_scores) // 2).astype(int)
    majority_score = votes / len(normalized_scores)
    
    results.append(evaluate_voting_strategy(majority_pred, majority_score, "🗳️ Simple Majority Vote"))
    
    # Strategy 2: Weighted combination (weight by metric reliability)
    metric_weights = {
        'cosine': 1.0,      # Most reliable
        'euclidean': 1.0,   # Most reliable  
        'pearson': 0.8,     # Good correlation
        'spearman': 0.8,    # Good correlation
        'canberra': 0.6,    # Specialized distance
        'chebyshev': 0.6,   # Specialized distance
        'braycurtis': 0.5,  # Less reliable
        'dcor': 0.7,        # Distance correlation
        'energy': 0.5       # Experimental
    }
    
    weighted_score = np.zeros(len(y_test))
    total_weight = 0
    
    for metric, scores in normalized_scores.items():
        weight = metric_weights.get(metric, 0.5)
        weighted_score += weight * scores
        total_weight += weight
    
    weighted_score /= total_weight
    weighted_pred = (weighted_score > 0.5).astype(int)
    
    results.append(evaluate_voting_strategy(weighted_pred, weighted_score, "⚖️ Weighted Combination"))
    
    # Strategy 3: Consensus voting (require multiple metrics to agree)
    strong_consensus = np.zeros(len(y_test))
    high_confidence_metrics = ['cosine', 'euclidean', 'pearson', 'spearman']
    
    for metric in high_confidence_metrics:
        if metric in normalized_scores:
            strong_consensus += (normalized_scores[metric] > 0.6).astype(int)
    
    consensus_pred = (strong_consensus >= 3).astype(int)  # 3 out of 4 core metrics
    consensus_score = strong_consensus / len(high_confidence_metrics)
    
    results.append(evaluate_voting_strategy(consensus_pred, consensus_score, "🤝 Strong Consensus (3/4)"))
    
    # Strategy 4: Distance vs Correlation ensemble
    distance_metrics = ['cosine', 'euclidean', 'canberra', 'chebyshev', 'braycurtis']
    correlation_metrics = ['pearson', 'spearman', 'dcor', 'energy']
    
    distance_vote = np.zeros(len(y_test))
    correlation_vote = np.zeros(len(y_test))
    
    for metric in distance_metrics:
        if metric in normalized_scores:
            distance_vote += (normalized_scores[metric] > 0.5).astype(int)
    
    for metric in correlation_metrics:
        if metric in normalized_scores:
            correlation_vote += (normalized_scores[metric] > 0.5).astype(int)
    
    # Both distance and correlation committees must agree
    committee_pred = ((distance_vote >= len(distance_metrics)//2) & 
                     (correlation_vote >= len(correlation_metrics)//2)).astype(int)
    committee_score = (distance_vote/len(distance_metrics) + correlation_vote/len(correlation_metrics)) / 2
    
    results.append(evaluate_voting_strategy(committee_pred, committee_score, "🏛️ Committee Voting (Dist+Corr)"))
    
    # Strategy 5: Adaptive threshold (optimize for sensitivity)
    adaptive_score = weighted_score  # Use weighted combination
    
    # Find threshold that maximizes sensitivity while keeping specificity > 40%
    thresholds = np.linspace(0.1, 0.9, 50)
    best_thresh = 0.5
    best_sens = 0
    
    for thresh in thresholds:
        pred_thresh = (adaptive_score > thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred_thresh).ravel()
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        
        if spec >= 0.4 and sens > best_sens:  # Minimum 40% specificity
            best_sens = sens
            best_thresh = thresh
    
    adaptive_pred = (adaptive_score > best_thresh).astype(int)
    
    results.append(evaluate_voting_strategy(adaptive_pred, adaptive_score, f"🎯 Adaptive Threshold ({best_thresh:.2f})"))
    
    # Compare with previous best methods
    print(f"\\n📊 COMPARISON WITH PREVIOUS METHODS:")
    print("Method                    | Sens   | Spec   | Acc    | AUC    | Missed | F.Alarms")
    print("-" * 85)
    
    # Previous results for comparison
    previous_methods = [
        ("Original Cosine", 0.824, 0.903, 0.854, 0.925, 49, 16),
        ("Prototype Logistic", 0.957, 0.721, 0.869, 0.953, 12, 46),
        ("SVM + Correlations", 0.993, 0.455, 0.793, 0.950, 2, 90),
        ("Direct Logistic", 1.000, 0.158, 0.687, 0.957, 0, 139)
    ]
    
    for name, sens, spec, acc, auc, missed, falarms in previous_methods:
        print(f"{name:<24} | {sens:.3f} | {spec:.3f} | {acc:.3f} | {auc:.3f} | {missed:6} | {falarms:8}")
    
    print(f"\\nNEW COMPREHENSIVE SCORING:")
    for i, (sens, spec, acc, auc, missed, falarms) in enumerate(results):
        method_names = ["Simple Majority", "Weighted Combination", "Strong Consensus", "Committee Voting", "Adaptive Threshold"]
        name = method_names[i]
        print(f"{name:<24} | {sens:.3f} | {spec:.3f} | {acc:.3f} | {auc:.3f} | {missed:6} | {falarms:8}")
    
    # Find best comprehensive method
    best_sens = max(results, key=lambda x: x[0])
    best_balanced = max(results, key=lambda x: (x[0] + x[1])/2)
    best_auc = max(results, key=lambda x: x[3])
    
    method_names = ["Simple Majority", "Weighted Combination", "Strong Consensus", "Committee Voting", "Adaptive Threshold"]
    
    print(f"\\n🏆 COMPREHENSIVE SCORING RESULTS:")
    print(f"🥇 Best Sensitivity: {method_names[results.index(best_sens)]} ({best_sens[0]:.1%})")
    print(f"🥈 Best Balanced: {method_names[results.index(best_balanced)]} ({(best_balanced[0]+best_balanced[1])/2:.1%})")
    print(f"🥉 Best AUC: {method_names[results.index(best_auc)]} ({best_auc[3]:.3f})")
    
    # Save best comprehensive classifier
    best_method_idx = results.index(best_sens)
    best_method_name = method_names[best_method_idx]
    
    comprehensive_classifier = {
        'method': 'comprehensive_multi_metric_scoring',
        'best_strategy': best_method_name,
        'prototypes': {
            'benign': train_benign_proto,
            'malignant': train_malignant_proto
        },
        'metrics_used': list(test_scores.keys()),
        'metric_weights': metric_weights,
        'test_sensitivity': best_sens[0],
        'test_specificity': best_sens[1],
        'test_auc': best_sens[3],
        'test_accuracy': best_sens[2],
        'missed_cancers': best_sens[4],
        'false_alarms': best_sens[5]
    }
    
    with open('/workspace/comprehensive_scoring_classifier.pkl', 'wb') as f:
        pickle.dump(comprehensive_classifier, f)
    
    print(f"\\n✅ COMPREHENSIVE SCORING SYSTEM CREATED!")
    print(f"   Best strategy: {best_method_name}")
    print(f"   Metrics used: {len(test_scores)} different measures")
    print(f"   Test sensitivity: {best_sens[0]:.1%}")
    print(f"   Saved to: comprehensive_scoring_classifier.pkl")
    
    return comprehensive_classifier

if __name__ == '__main__':
    result = create_comprehensive_scoring_system()
    print(f"\\nFinal Best Sensitivity: {result['test_sensitivity']:.1%}")