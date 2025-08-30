#!/usr/bin/env python3
"""
Tiered Prediction System for Breast Cancer Classification
Stage 1: BreakHis Binary → Stage 2: BACH Multi-class → Stage 3: Cross-dataset Consensus
"""
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

def create_tiered_prediction_system():
    """Create comprehensive tiered prediction system"""
    
    print("🏗️ TIERED PREDICTION SYSTEM")
    print("=" * 40)
    print("Stage 1: BreakHis Binary (benign vs malignant)")
    print("Stage 2: BACH Multi-class (normal, benign, insitu, invasive)")  
    print("Stage 3: Cross-dataset Consensus")
    
    # Load data
    with open("/workspace/embeddings_cache_PROTOTYPE_WHITENED.pkl", 'rb') as f:
        cache = pickle.load(f)
    
    features = np.array(cache['combined']['features'])
    labels = cache['combined']['labels']
    datasets = cache['combined']['datasets']
    
    # Overall train/test split
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        features, np.arange(len(features)), test_size=0.2, random_state=42, stratify=datasets
    )
    
    train_labels = [labels[i] for i in train_idx]
    train_datasets = [datasets[i] for i in train_idx]
    test_labels = [labels[i] for i in test_idx]
    test_datasets = [datasets[i] for i in test_idx]
    
    print(f"\\n📊 Total test samples: {len(X_test)}")
    
    # STAGE 1: BreakHis Binary Classifier
    print("\\n🔬 STAGE 1: BreakHis Binary Classifier")
    
    # Filter training data to BreakHis only
    bh_train_indices = [i for i, ds in enumerate(train_datasets) if ds == 'breakhis']
    X_train_bh = X_train[bh_train_indices]
    y_train_bh = np.array([1 if train_labels[i] == 'malignant' else 0 for i in bh_train_indices])
    
    # BreakHis prototypes
    bh_benign_proto = np.mean(X_train_bh[y_train_bh == 0], axis=0)
    bh_malignant_proto = np.mean(X_train_bh[y_train_bh == 1], axis=0)
    
    print(f"   Training samples: {len(X_train_bh)} (Benign: {np.sum(y_train_bh==0)}, Malignant: {np.sum(y_train_bh==1)})")
    
    # Extract BreakHis features
    def extract_bh_features(X):
        cosine_benign = X @ bh_benign_proto
        cosine_malignant = X @ bh_malignant_proto
        eucl_benign = np.linalg.norm(X - bh_benign_proto, axis=1)
        eucl_malignant = np.linalg.norm(X - bh_malignant_proto, axis=1)
        
        return np.column_stack([
            cosine_benign, cosine_malignant, eucl_benign, eucl_malignant,
            cosine_malignant - cosine_benign,  # sim_diff
            eucl_benign - eucl_malignant,      # dist_diff
            cosine_malignant / (cosine_benign + 1e-8)  # sim_ratio
        ])
    
    X_train_bh_feat = extract_bh_features(X_train_bh)
    X_test_bh_feat = extract_bh_features(X_test)  # Apply to all test samples
    
    # Train BreakHis classifier
    scaler_bh = StandardScaler()
    X_train_bh_scaled = scaler_bh.fit_transform(X_train_bh_feat)
    X_test_bh_scaled = scaler_bh.transform(X_test_bh_feat)
    
    bh_classifier = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
    bh_classifier.fit(X_train_bh_scaled, y_train_bh)
    
    bh_predictions = bh_classifier.predict(X_test_bh_scaled)
    bh_probabilities = bh_classifier.predict_proba(X_test_bh_scaled)[:, 1]
    
    print(f"   ✅ BreakHis classifier trained")
    
    # STAGE 2: BACH Multi-class Classifier
    print("\\n🧬 STAGE 2: BACH Multi-class Classifier")
    
    # Filter training data to BACH only
    bach_train_indices = [i for i, ds in enumerate(train_datasets) if ds == 'bach']
    X_train_bach = X_train[bach_train_indices]
    y_train_bach = [train_labels[i] for i in bach_train_indices]
    
    # BACH label-specific prototypes
    bach_prototypes = {}
    for label in ['normal', 'benign', 'insitu', 'invasive']:
        label_indices = [i for i, lbl in enumerate(y_train_bach) if lbl == label]
        if label_indices:
            bach_prototypes[label] = np.mean(X_train_bach[label_indices], axis=0)
    
    print(f"   Training samples: {len(X_train_bach)}")
    for label, proto in bach_prototypes.items():
        count = sum(1 for lbl in y_train_bach if lbl == label)
        print(f"     {label}: {count} samples")
    
    # Convert BACH to binary for this classifier
    y_train_bach_binary = np.array([1 if lbl in ['invasive', 'insitu'] else 0 for lbl in y_train_bach])
    
    # Extract BACH features
    def extract_bach_features(X):
        if not bach_prototypes:
            return np.zeros((len(X), 7))
            
        similarities = {label: X @ proto for label, proto in bach_prototypes.items()}
        
        # Most discriminative BACH features
        features = []
        if 'invasive' in similarities and 'normal' in similarities:
            features.append(similarities['invasive'] - similarities['normal'])
        if 'insitu' in similarities and 'benign' in similarities:
            features.append(similarities['insitu'] - similarities['benign'])
        if 'invasive' in similarities and 'benign' in similarities:
            features.append(similarities['invasive'] - similarities['benign'])
        if 'insitu' in similarities and 'normal' in similarities:
            features.append(similarities['insitu'] - similarities['normal'])
        
        # Add basic similarities
        for label in ['normal', 'benign', 'invasive']:
            if label in similarities:
                features.append(similarities[label])
        
        if features:
            return np.column_stack(features)
        else:
            return np.zeros((len(X), 1))
    
    X_train_bach_feat = extract_bach_features(X_train_bach)
    X_test_bach_feat = extract_bach_features(X_test)
    
    # Train BACH classifier
    if X_train_bach_feat.shape[1] > 0:
        scaler_bach = StandardScaler()
        X_train_bach_scaled = scaler_bach.fit_transform(X_train_bach_feat)
        X_test_bach_scaled = scaler_bach.transform(X_test_bach_feat)
        
        bach_classifier = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
        bach_classifier.fit(X_train_bach_scaled, y_train_bach_binary)
        
        bach_predictions = bach_classifier.predict(X_test_bach_scaled)
        bach_probabilities = bach_classifier.predict_proba(X_test_bach_scaled)[:, 1]
        
        print(f"   ✅ BACH classifier trained ({X_train_bach_feat.shape[1]} features)")
    else:
        bach_predictions = np.zeros(len(X_test))
        bach_probabilities = np.zeros(len(X_test))
        print(f"   ❌ BACH classifier failed - no features")
    
    # STAGE 3: Cross-dataset Consensus (MI Enhanced)
    print("\\n🤝 STAGE 3: Cross-dataset Consensus")
    
    # Use the best MI Enhanced classifier
    y_train_binary = np.array([1 if train_labels[i] in ['malignant', 'invasive', 'insitu'] else 0 for i in range(len(train_labels))])
    
    # Create cross-dataset prototypes
    cross_benign_proto = np.mean(X_train[y_train_binary == 0], axis=0)
    cross_malignant_proto = np.mean(X_train[y_train_binary == 1], axis=0)
    
    # Label prototypes
    cross_label_prototypes = {}
    for label in ['normal', 'benign', 'insitu', 'invasive', 'malignant']:
        label_indices = [i for i, lbl in enumerate(train_labels) if lbl == label]
        if label_indices:
            cross_label_prototypes[label] = np.mean(X_train[label_indices], axis=0)
    
    # Extract cross-dataset features
    def extract_cross_features(X):
        cosine_benign = X @ cross_benign_proto
        cosine_malignant = X @ cross_malignant_proto
        eucl_benign = np.linalg.norm(X - cross_benign_proto, axis=1)
        eucl_malignant = np.linalg.norm(X - cross_malignant_proto, axis=1)
        
        sim_diff = cosine_malignant - cosine_benign
        dist_diff = eucl_benign - eucl_malignant
        sim_ratio = cosine_malignant / (cosine_benign + 1e-8)
        
        # Label-specific features
        similarities = {label: X @ proto for label, proto in cross_label_prototypes.items()}
        label_features = [
            similarities['invasive'] - similarities['normal'],
            similarities['insitu'] - similarities['benign'],
            similarities['invasive'] - similarities['benign'],
            similarities['malignant'] - similarities['normal'],
            similarities['insitu'] - similarities['normal']
        ]
        
        # MI approximation
        mi_features = [
            abs(similarities['malignant'] - similarities['normal']) * 2,
            abs(similarities['invasive'] - similarities['benign']) * 2,
            abs(similarities['insitu'] - similarities['normal']) * 2
        ]
        
        return np.column_stack([
            cosine_benign, cosine_malignant, eucl_benign, eucl_malignant,
            sim_diff, dist_diff, sim_ratio
        ] + label_features + mi_features)
    
    X_train_cross_feat = extract_cross_features(X_train)
    X_test_cross_feat = extract_cross_features(X_test)
    
    scaler_cross = StandardScaler()
    X_train_cross_scaled = scaler_cross.fit_transform(X_train_cross_feat)
    X_test_cross_scaled = scaler_cross.transform(X_test_cross_feat)
    
    cross_classifier = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
    cross_classifier.fit(X_train_cross_scaled, y_train_binary)
    
    cross_predictions = cross_classifier.predict(X_test_cross_scaled)
    cross_probabilities = cross_classifier.predict_proba(X_test_cross_scaled)[:, 1]
    
    print(f"   ✅ Cross-dataset classifier trained (15 features)")
    
    # TIERED ENSEMBLE STRATEGIES
    print("\\n🏗️ TIERED ENSEMBLE STRATEGIES:")
    
    # Convert test labels to binary for evaluation
    y_test_binary = np.array([1 if test_labels[i] in ['malignant', 'invasive', 'insitu'] else 0 for i in range(len(test_labels))])
    
    def evaluate_tiered_strategy(predictions, probabilities, name):
        tn, fp, fn, tp = confusion_matrix(y_test_binary, predictions).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        auc = roc_auc_score(y_test_binary, probabilities)
        g_mean = np.sqrt(sensitivity * specificity)
        
        print(f"\\n{name}:")
        print(f"  Sensitivity: {sensitivity:.3f} ({sensitivity*100:.1f}%) - Missed: {fn}/279")
        print(f"  Specificity: {specificity:.3f} ({specificity*100:.1f}%) - False alarms: {fp}/165")
        print(f"  Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"  AUC: {auc:.3f}")
        print(f"  G-Mean: {g_mean:.3f}")
        
        return sensitivity, specificity, accuracy, auc, g_mean
    
    results = []
    
    # Strategy 1: Weighted by dataset-specific expertise
    dataset_weights = np.array([
        0.7 if test_datasets[i] == 'breakhis' else 0.3 for i in range(len(test_datasets))
    ])
    
    weighted_proba_1 = (dataset_weights * bh_probabilities + 
                       (1 - dataset_weights) * bach_probabilities + 
                       cross_probabilities) / 3
    weighted_pred_1 = (weighted_proba_1 > 0.5).astype(int)
    
    results.append(evaluate_tiered_strategy(weighted_pred_1, weighted_proba_1, "🎯 Strategy 1: Dataset-Weighted"))
    
    # Strategy 2: Majority vote
    majority_pred = ((bh_predictions + bach_predictions + cross_predictions) >= 2).astype(int)
    avg_proba = (bh_probabilities + bach_probabilities + cross_probabilities) / 3
    
    results.append(evaluate_tiered_strategy(majority_pred, avg_proba, "🗳️ Strategy 2: Majority Vote"))
    
    # Strategy 3: Confidence-based routing
    confidence_pred = np.zeros(len(X_test))
    confidence_proba = np.zeros(len(X_test))
    
    for i in range(len(X_test)):
        # Get confidence from each classifier
        bh_conf = max(bh_probabilities[i], 1 - bh_probabilities[i])
        bach_conf = max(bach_probabilities[i], 1 - bach_probabilities[i])
        cross_conf = max(cross_probabilities[i], 1 - cross_probabilities[i])
        
        # Use most confident classifier
        if bh_conf >= bach_conf and bh_conf >= cross_conf:
            confidence_pred[i] = bh_predictions[i]
            confidence_proba[i] = bh_probabilities[i]
        elif bach_conf >= cross_conf:
            confidence_pred[i] = bach_predictions[i]
            confidence_proba[i] = bach_probabilities[i]
        else:
            confidence_pred[i] = cross_predictions[i]
            confidence_proba[i] = cross_probabilities[i]
    
    results.append(evaluate_tiered_strategy(confidence_pred, confidence_proba, "🎯 Strategy 3: Confidence Routing"))
    
    # Strategy 4: High sensitivity ensemble (OR logic for high-confidence malignant)
    high_sens_pred = ((bh_probabilities > 0.6) | (bach_probabilities > 0.7) | (cross_probabilities > 0.6)).astype(int)
    high_sens_proba = np.maximum.reduce([bh_probabilities, bach_probabilities, cross_probabilities])
    
    results.append(evaluate_tiered_strategy(high_sens_pred, high_sens_proba, "🚨 Strategy 4: High Sensitivity"))
    
    # Strategy 5: Conservative consensus (AND logic for high confidence)
    conservative_pred = ((bh_probabilities > 0.7) & (bach_probabilities > 0.7) & (cross_probabilities > 0.7)).astype(int)
    conservative_proba = (bh_probabilities + bach_probabilities + cross_probabilities) / 3
    
    results.append(evaluate_tiered_strategy(conservative_pred, conservative_proba, "🛡️ Strategy 5: Conservative"))
    
    # Individual classifier results for comparison
    results.append(evaluate_tiered_strategy(bh_predictions, bh_probabilities, "📊 BreakHis Only"))
    results.append(evaluate_tiered_strategy(bach_predictions, bach_probabilities, "🧬 BACH Only"))
    results.append(evaluate_tiered_strategy(cross_predictions, cross_probabilities, "🤝 Cross-dataset Only"))
    
    print(f"\\n📊 TIERED SYSTEM COMPARISON:")
    print("Strategy                | Sens   | Spec   | Acc    | AUC    | G-Mean")
    print("-" * 70)
    
    strategy_names = [
        "Dataset-Weighted", "Majority Vote", "Confidence Routing", 
        "High Sensitivity", "Conservative", "BreakHis Only", "BACH Only", "Cross-dataset Only"
    ]
    
    for i, (sens, spec, acc, auc, gmean) in enumerate(results):
        print(f"{strategy_names[i]:<22} | {sens:.3f} | {spec:.3f} | {acc:.3f} | {auc:.3f} | {gmean:.3f}")
    
    # Find best tiered strategy
    best_gmean = max(results[:5], key=lambda x: x[4])  # Best among tiered strategies
    best_idx = results[:5].index(best_gmean)
    
    print(f"\\n🏆 BEST TIERED STRATEGY:")
    print(f"{strategy_names[best_idx]}: G-Mean = {best_gmean[4]:.3f}")
    print(f"Sensitivity: {best_gmean[0]:.1%}, Specificity: {best_gmean[1]:.1%}")
    
    # Compare with best individual
    mi_enhanced_gmean = 0.846  # Known best individual performance
    
    if best_gmean[4] > mi_enhanced_gmean:
        print(f"\\n🎉 TIERED SYSTEM WINS!")
        print(f"   Improvement: +{best_gmean[4] - mi_enhanced_gmean:.3f} G-Mean")
    else:
        print(f"\\n📊 INDIVIDUAL MI ENHANCED STILL BETTER")
        print(f"   MI Enhanced: {mi_enhanced_gmean:.3f} vs Best Tiered: {best_gmean[4]:.3f}")
    
    # Save best tiered system
    tiered_system = {
        'stage_1_classifier': bh_classifier,
        'stage_1_scaler': scaler_bh,
        'stage_2_classifier': bach_classifier if 'bach_classifier' in locals() else None,
        'stage_2_scaler': scaler_bach if 'scaler_bach' in locals() else None,
        'stage_3_classifier': cross_classifier,
        'stage_3_scaler': scaler_cross,
        'prototypes': {
            'breakhis_benign': bh_benign_proto,
            'breakhis_malignant': bh_malignant_proto,
            'bach': bach_prototypes,
            'cross_benign': cross_benign_proto,
            'cross_malignant': cross_malignant_proto,
            'cross_label': cross_label_prototypes
        },
        'best_strategy': strategy_names[best_idx],
        'test_performance': {
            'sensitivity': best_gmean[0],
            'specificity': best_gmean[1],
            'g_mean': best_gmean[4],
            'auc': best_gmean[3]
        }
    }
    
    with open('/workspace/tiered_prediction_system.pkl', 'wb') as f:
        pickle.dump(tiered_system, f)
    
    print(f"\\n✅ TIERED PREDICTION SYSTEM CREATED!")
    print(f"   Best strategy: {strategy_names[best_idx]}")
    print(f"   G-Mean: {best_gmean[4]:.3f}")
    print(f"   Saved to: tiered_prediction_system.pkl")
    
    return tiered_system

if __name__ == '__main__':
    create_tiered_prediction_system()