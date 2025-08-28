#!/usr/bin/env python3
"""
Retrain All Classifiers on Whitened Features
Update BACH and BreakHis classifiers to use whitened embeddings
"""
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

def retrain_breakhis_classifiers():
    """Retrain BreakHis classifiers on whitened features"""
    
    print("🔥 Retraining BreakHis classifiers on whitened features...")
    
    # Load whitened cache
    with open("/workspace/embeddings_cache_PROTOTYPE_WHITENED.pkl", 'rb') as f:
        cache = pickle.load(f)
    
    features = np.array(cache['combined']['features'])
    labels = cache['combined']['labels']
    datasets = cache['combined']['datasets']
    
    # Extract BreakHis samples
    breakhis_indices = [i for i, ds in enumerate(datasets) if ds == 'breakhis']
    breakhis_features = features[breakhis_indices]
    breakhis_labels = [labels[i] for i in breakhis_indices]
    
    # Convert to binary
    breakhis_binary = [1 if label == 'malignant' else 0 for label in breakhis_labels]
    
    print(f"   BreakHis samples: {len(breakhis_features)}")
    print(f"   Label distribution: {np.bincount(breakhis_binary)}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        breakhis_features, breakhis_binary, 
        test_size=0.2, random_state=42, stratify=breakhis_binary
    )
    
    classifiers = {}
    
    # Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=2000)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_proba = lr_model.predict_proba(X_test)[:, 1]
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_auc = roc_auc_score(y_test, lr_proba)
    
    classifiers['logistic'] = {
        'model': lr_model,
        'accuracy': lr_acc,
        'auc': lr_auc
    }
    
    # SVM
    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    svm_proba = svm_model.predict_proba(X_test)[:, 1]
    svm_acc = accuracy_score(y_test, svm_pred)
    svm_auc = roc_auc_score(y_test, svm_proba)
    
    classifiers['svm'] = {
        'model': svm_model,
        'accuracy': svm_acc,
        'auc': svm_auc
    }
    
    # Random Forest
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_auc = roc_auc_score(y_test, rf_proba)
    
    classifiers['random_forest'] = {
        'model': rf_model,
        'accuracy': rf_acc,
        'auc': rf_auc
    }
    
    print(f"   ✅ Logistic Regression: {lr_acc:.1%} accuracy, {lr_auc:.3f} AUC")
    print(f"   ✅ SVM: {svm_acc:.1%} accuracy, {svm_auc:.3f} AUC") 
    print(f"   ✅ Random Forest: {rf_acc:.1%} accuracy, {rf_auc:.3f} AUC")
    
    # Save retrained BreakHis classifiers
    with open('/workspace/breakhis_classifiers_whitened.pkl', 'wb') as f:
        pickle.dump(classifiers, f)
    
    return classifiers

def retrain_bach_classifiers():
    """Retrain BACH classifiers on whitened features"""
    
    print("\n🎨 Retraining BACH classifiers on whitened features...")
    
    # Load whitened cache
    with open("/workspace/embeddings_cache_PROTOTYPE_WHITENED.pkl", 'rb') as f:
        cache = pickle.load(f)
    
    features = np.array(cache['combined']['features'])
    labels = cache['combined']['labels']
    datasets = cache['combined']['datasets']
    
    # Extract BACH samples
    bach_indices = [i for i, ds in enumerate(datasets) if ds == 'bach']
    bach_features = features[bach_indices]
    bach_labels = [labels[i] for i in bach_indices]
    
    # Binary BACH: normal/benign vs invasive/insitu
    bach_binary = []
    for label in bach_labels:
        if label in ['normal', 'benign']:
            bach_binary.append(0)  # benign
        else:  # invasive, insitu
            bach_binary.append(1)  # malignant
    
    print(f"   BACH samples: {len(bach_features)}")
    print(f"   Label distribution: {np.bincount(bach_binary)}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        bach_features, bach_binary,
        test_size=0.2, random_state=42, stratify=bach_binary
    )
    
    classifiers = {}
    
    # Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=2000)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_proba = lr_model.predict_proba(X_test)[:, 1]
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_auc = roc_auc_score(y_test, lr_proba)
    
    classifiers['logistic'] = {
        'model': lr_model,
        'accuracy': lr_acc,
        'auc': lr_auc
    }
    
    # SVM
    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    svm_proba = svm_model.predict_proba(X_test)[:, 1]
    svm_acc = accuracy_score(y_test, svm_pred)
    svm_auc = roc_auc_score(y_test, svm_proba)
    
    classifiers['svm'] = {
        'model': svm_model,
        'accuracy': svm_acc,
        'auc': svm_auc
    }
    
    print(f"   ✅ Logistic Regression: {lr_acc:.1%} accuracy, {lr_auc:.3f} AUC")
    print(f"   ✅ SVM: {svm_acc:.1%} accuracy, {svm_auc:.3f} AUC")
    
    # Save retrained BACH classifiers
    with open('/workspace/bach_classifiers_whitened.pkl', 'wb') as f:
        pickle.dump(classifiers, f)
    
    return classifiers

def update_correlation_methods():
    """Update correlation methods to use whitened embeddings"""
    
    print("\n📊 Updating correlation methods for whitened embeddings...")
    
    correlation_code = '''
def calculate_whitened_correlations(new_whitened_features, cached_features, cached_labels, method='pearson'):
    """Calculate correlations using whitened features"""
    from scipy.stats import pearsonr, spearmanr
    
    correlations = []
    
    for i, cached_feat in enumerate(cached_features):
        try:
            if method == 'pearson':
                corr, _ = pearsonr(new_whitened_features, cached_feat)
            elif method == 'spearman':
                corr, _ = spearmanr(new_whitened_features, cached_feat)
            else:
                # Cosine similarity (default)
                corr = np.dot(new_whitened_features, cached_feat)
            
            if not np.isnan(corr):
                correlations.append((corr, cached_labels[i], i))
        except:
            continue
    
    # Sort by correlation strength
    correlations.sort(key=lambda x: abs(x[0]), reverse=True)
    
    return correlations

def get_correlation_consensus(correlations, top_n=10):
    """Get correlation-based consensus from top matches"""
    if not correlations:
        return 'benign', 0.5
    
    # Take top N correlations
    top_correlations = correlations[:top_n]
    
    # Count label frequencies
    label_counts = {}
    for corr, label, idx in top_correlations:
        # Convert to binary
        binary_label = 'malignant' if label in ['malignant', 'invasive', 'insitu'] else 'benign'
        label_counts[binary_label] = label_counts.get(binary_label, 0) + 1
    
    # Consensus
    if label_counts.get('malignant', 0) > label_counts.get('benign', 0):
        consensus = 'malignant'
    else:
        consensus = 'benign'
    
    # Confidence based on agreement
    total = sum(label_counts.values())
    max_count = max(label_counts.values()) if label_counts else 1
    confidence = max_count / total if total > 0 else 0.5
    
    return consensus, confidence
'''
    
    with open('/workspace/whitened_correlation_methods.py', 'w') as f:
        f.write(correlation_code)
    
    print("✅ Created whitened_correlation_methods.py")

def main():
    """Complete classifier retraining and method updates"""
    
    print("🚀 RETRAINING ALL CLASSIFIERS FOR WHITENED EMBEDDINGS")
    print("=" * 70)
    
    # Retrain classifiers
    breakhis_classifiers = retrain_breakhis_classifiers()
    bach_classifiers = retrain_bach_classifiers()
    
    # Update correlation methods
    update_correlation_methods()
    
    print(f"\n🎉 CLASSIFIER RETRAINING COMPLETE!")
    print(f"📁 New classifier files:")
    print(f"   - /workspace/breakhis_classifiers_whitened.pkl")
    print(f"   - /workspace/bach_classifiers_whitened.pkl")
    print(f"   - /workspace/whitened_correlation_methods.py")
    
    print(f"\n📊 Performance on whitened features:")
    print(f"   BreakHis LR: {breakhis_classifiers['logistic']['accuracy']:.1%} acc, {breakhis_classifiers['logistic']['auc']:.3f} AUC")
    print(f"   BreakHis SVM: {breakhis_classifiers['svm']['accuracy']:.1%} acc, {breakhis_classifiers['svm']['auc']:.3f} AUC")
    print(f"   BACH LR: {bach_classifiers['logistic']['accuracy']:.1%} acc, {bach_classifiers['logistic']['auc']:.3f} AUC")
    print(f"   BACH SVM: {bach_classifiers['svm']['accuracy']:.1%} acc, {bach_classifiers['svm']['auc']:.3f} AUC")
    
    return True

if __name__ == "__main__":
    main()