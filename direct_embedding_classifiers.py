#!/usr/bin/env python3
"""
Direct Embedding Classifiers
Train ML models directly on full 1536-dimensional whitened embeddings
"""
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

def create_direct_embedding_classifiers():
    """Train classifiers directly on 1536-dimensional whitened embeddings"""
    
    print("🔢 DIRECT EMBEDDING CLASSIFIERS")
    print("=" * 45)
    print("Training ML models on FULL 1536-dimensional whitened embeddings")
    print("(NOT prototype-derived features)")
    
    # Load whitened cache
    with open("/workspace/embeddings_cache_PROTOTYPE_WHITENED.pkl", 'rb') as f:
        cache = pickle.load(f)
    
    features = np.array(cache['combined']['features'])  # Full 1536 dimensions
    labels = cache['combined']['labels']
    datasets = cache['combined']['datasets']
    
    # Convert to binary labels
    y = np.array([1 if label in ['malignant', 'invasive', 'insitu'] else 0 for label in labels])
    
    print(f"📊 Input: {features.shape} (full embeddings)")
    print(f"🏷️ Class distribution: Benign={np.sum(y==0)}, Malignant={np.sum(y==1)}")
    
    # Same train/test split as prototype methods for fair comparison
    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"🎯 Training: {X_train.shape}, Test: {X_test.shape}")
    
    def evaluate_direct_classifier(model, X_train, X_test, y_train, y_test, name):
        """Evaluate classifier and return medical metrics"""
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        if hasattr(model, 'predict_proba'):
            train_proba = model.predict_proba(X_train)[:, 1]
            test_proba = model.predict_proba(X_test)[:, 1]
        else:
            train_proba = model.decision_function(X_train)
            test_proba = model.decision_function(X_test)
        
        # Metrics
        train_acc = accuracy_score(y_train, train_pred)
        train_auc = roc_auc_score(y_train, train_proba)
        test_acc = accuracy_score(y_test, test_pred)
        test_auc = roc_auc_score(y_test, test_proba)
        
        # Medical metrics
        tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        ppv = tp / (tp + fp)
        npv = tn / (tn + fn)
        
        print(f"\\n{name}:")
        print(f"  Training: {train_acc:.3f} acc, {train_auc:.3f} AUC")
        print(f"  Test: {test_acc:.3f} acc, {test_auc:.3f} AUC")
        print(f"  Overfitting: {train_auc - test_auc:.3f} AUC")
        print(f"  🎯 Sensitivity: {sensitivity:.3f} ({sensitivity*100:.1f}%) - Missed: {fn}/279")
        print(f"  🛡️ Specificity: {specificity:.3f} ({specificity*100:.1f}%) - False alarms: {fp}/165")
        print(f"  📈 PPV: {ppv:.3f}, NPV: {npv:.3f}")
        
        return {
            'name': name,
            'test_auc': test_auc,
            'test_accuracy': test_acc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'missed_cancers': fn,
            'false_alarms': fp,
            'overfitting': train_auc - test_auc
        }
    
    results = []
    
    # 1. Direct Logistic Regression on full embeddings
    print("\\n📊 1. DIRECT LOGISTIC REGRESSION (1536 features)")
    lr_direct = LogisticRegression(
        random_state=42, 
        max_iter=1000, 
        C=1.0,
        solver='liblinear'  # Better for high-dimensional data
    )
    
    results.append(evaluate_direct_classifier(
        lr_direct, X_train, X_test, y_train, y_test, 
        "Direct Logistic (1536D)"
    ))
    
    # 2. Direct SVM on full embeddings (use subset for speed)
    print("\\n🔵 2. DIRECT SVM RBF (1536 features)")
    print("   Note: Using subset for computational efficiency")
    
    # Use subset for SVM (full 1536D SVM is computationally expensive)
    subset_size = min(1000, len(X_train))
    train_indices = np.random.choice(len(X_train), subset_size, replace=False)
    X_train_subset = X_train[train_indices]
    y_train_subset = y_train[train_indices]
    
    svm_direct = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        random_state=42
    )
    
    results.append(evaluate_direct_classifier(
        svm_direct, X_train_subset, X_test, y_train_subset, y_test,
        "Direct SVM (1536D subset)"
    ))
    
    # 3. Direct XGBoost on full embeddings
    print("\\n🌳 3. DIRECT XGBOOST (1536 features)")
    xgb_direct = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    results.append(evaluate_direct_classifier(
        xgb_direct, X_train, X_test, y_train, y_test,
        "Direct XGBoost (1536D)"
    ))
    
    # Summary comparison
    print(f"\\n📊 DIRECT EMBEDDING vs PROTOTYPE FEATURE COMPARISON:")
    print("Method                    | Input      | Sens   | Spec   | AUC    | Missed")
    print("-" * 80)
    
    # Add known prototype results for comparison
    prototype_results = [
        ("Prototype Logistic", "4-6 derived", 0.957, 0.721, 0.953, 12),
        ("SVM + Correlations", "8-13 derived", 0.993, 0.455, 0.950, 2),
        ("Original Cosine", "Direct similarity", 0.824, 0.903, 0.925, 49)
    ]
    
    for name, input_type, sens, spec, auc, missed in prototype_results:
        print(f"{name:<24} | {input_type:<10} | {sens:.3f} | {spec:.3f} | {auc:.3f} | {missed:6}")
    
    for result in results:
        input_type = "1536D full"
        print(f"{result['name']:<24} | {input_type:<10} | {result['sensitivity']:.3f} | {result['specificity']:.3f} | {result['test_auc']:.3f} | {result['missed_cancers']:6}")
    
    # Find best direct method
    best_direct = max(results, key=lambda x: x['sensitivity'])
    best_prototype_sens = 0.993  # SVM + Correlations
    
    print(f"\\n🏆 COMPARISON RESULTS:")
    print(f"Best Prototype Method: 99.3% sensitivity (SVM + Correlations)")
    print(f"Best Direct Method: {best_direct['sensitivity']:.1%} sensitivity ({best_direct['name']})")
    
    if best_direct['sensitivity'] > best_prototype_sens:
        print("🎉 DIRECT EMBEDDINGS WIN!")
    else:
        print("🏆 PROTOTYPE FEATURES WIN!")
        print(f"   Advantage: {best_prototype_sens - best_direct['sensitivity']:.3f} sensitivity points")
    
    # Save best direct classifier
    best_model = lr_direct if best_direct['name'] == "Direct Logistic (1536D)" else (xgb_direct if best_direct['name'] == "Direct XGBoost (1536D)" else svm_direct)
    
    direct_classifier = {
        'model': best_model,
        'method': 'direct_embeddings',
        'input_dimensions': 1536,
        'test_sensitivity': best_direct['sensitivity'],
        'test_specificity': best_direct['specificity'], 
        'test_auc': best_direct['test_auc'],
        'test_accuracy': best_direct['test_accuracy'],
        'missed_cancers': best_direct['missed_cancers'],
        'false_alarms': best_direct['false_alarms']
    }
    
    with open('/workspace/direct_embedding_classifier.pkl', 'wb') as f:
        pickle.dump(direct_classifier, f)
    
    print(f"\\n✅ BEST DIRECT CLASSIFIER SAVED:")
    print(f"   Method: {best_direct['name']}")
    print(f"   Sensitivity: {best_direct['sensitivity']:.1%}")
    print(f"   Uses: Full 1536-dimensional whitened embeddings")
    
    return results

if __name__ == '__main__':
    create_direct_embedding_classifiers()