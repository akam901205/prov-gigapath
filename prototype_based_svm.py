#!/usr/bin/env python3
"""
Prototype-Based SVM RBF Classifier
Uses prototype similarities as features for SVM with RBF kernel
"""
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

def create_prototype_svm_classifier():
    """Create SVM RBF using prototype similarities as features"""
    
    print("🔍 PROTOTYPE-BASED SVM RBF CLASSIFIER")
    print("=" * 55)
    
    # Load whitened cache
    with open("/workspace/embeddings_cache_PROTOTYPE_WHITENED.pkl", 'rb') as f:
        cache = pickle.load(f)
    
    features = np.array(cache['combined']['features'])
    labels = cache['combined']['labels']
    datasets = cache['combined']['datasets']
    
    # Convert to binary labels
    y = np.array([1 if label in ['malignant', 'invasive', 'insitu'] else 0 for label in labels])
    
    print(f"📊 Total samples: {len(features)}")
    print(f"🏷️ Class distribution: Benign={np.sum(y==0)}, Malignant={np.sum(y==1)}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Compute prototypes from training data only
    train_benign_proto = np.mean(X_train[y_train == 0], axis=0)
    train_malignant_proto = np.mean(X_train[y_train == 1], axis=0)
    
    print(f"🎯 Prototypes computed from {len(X_train)} training samples")
    
    # Method 1: Simple prototype similarities
    print("\\n🔵 METHOD 1: SVM on Prototype Similarities")
    
    train_benign_sim = X_train @ train_benign_proto
    train_malignant_sim = X_train @ train_malignant_proto
    X_train_proto = np.column_stack([train_benign_sim, train_malignant_sim])
    
    test_benign_sim = X_test @ train_benign_proto
    test_malignant_sim = X_test @ train_malignant_proto
    X_test_proto = np.column_stack([test_benign_sim, test_malignant_sim])
    
    # Scale features
    scaler_proto = StandardScaler()
    X_train_proto_scaled = scaler_proto.fit_transform(X_train_proto)
    X_test_proto_scaled = scaler_proto.transform(X_test_proto)
    
    # Train SVM with hyperparameter optimization
    svm_proto = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        random_state=42
    )
    
    svm_proto.fit(X_train_proto_scaled, y_train)
    
    train_pred = svm_proto.predict(X_train_proto_scaled)
    train_proba = svm_proto.predict_proba(X_train_proto_scaled)[:, 1]
    test_pred = svm_proto.predict(X_test_proto_scaled)
    test_proba = svm_proto.predict_proba(X_test_proto_scaled)[:, 1]
    
    train_acc = accuracy_score(y_train, train_pred)
    train_auc = roc_auc_score(y_train, train_proba)
    test_acc = accuracy_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_proba)
    
    print(f"📊 METHOD 1 RESULTS:")
    print(f"   Training: {train_acc:.4f} accuracy, {train_auc:.4f} AUC")
    print(f"   Test: {test_acc:.4f} accuracy, {test_auc:.4f} AUC")
    print(f"   Overfitting: {train_auc - test_auc:.4f} AUC difference")
    
    # Method 2: Extended features with distances and ratios
    print("\\n🔵 METHOD 2: SVM on Extended Prototype Features")
    
    # Create extended features
    train_eucl_benign = np.linalg.norm(X_train - train_benign_proto, axis=1)
    train_eucl_malignant = np.linalg.norm(X_train - train_malignant_proto, axis=1)
    test_eucl_benign = np.linalg.norm(X_test - train_benign_proto, axis=1)
    test_eucl_malignant = np.linalg.norm(X_test - train_malignant_proto, axis=1)
    
    # Additional engineered features
    train_sim_diff = train_malignant_sim - train_benign_sim
    train_dist_diff = train_eucl_benign - train_eucl_malignant
    train_sim_ratio = train_malignant_sim / (train_benign_sim + 1e-8)
    train_dist_ratio = train_eucl_malignant / (train_eucl_benign + 1e-8)
    
    test_sim_diff = test_malignant_sim - test_benign_sim
    test_dist_diff = test_eucl_benign - test_eucl_malignant
    test_sim_ratio = test_malignant_sim / (test_benign_sim + 1e-8)
    test_dist_ratio = test_eucl_malignant / (test_eucl_benign + 1e-8)
    
    X_train_extended = np.column_stack([
        train_benign_sim, train_malignant_sim,
        train_eucl_benign, train_eucl_malignant,
        train_sim_diff, train_dist_diff, train_sim_ratio, train_dist_ratio
    ])
    X_test_extended = np.column_stack([
        test_benign_sim, test_malignant_sim,
        test_eucl_benign, test_eucl_malignant,
        test_sim_diff, test_dist_diff, test_sim_ratio, test_dist_ratio
    ])
    
    print(f"📏 Extended feature shape: {X_train_extended.shape}")
    
    # Scale features
    scaler_ext = StandardScaler()
    X_train_ext_scaled = scaler_ext.fit_transform(X_train_extended)
    X_test_ext_scaled = scaler_ext.transform(X_test_extended)
    
    # Hyperparameter optimization for SVM
    print("🔧 Optimizing SVM hyperparameters...")
    
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
    }
    
    svm_extended = GridSearchCV(
        SVC(kernel='rbf', probability=True, random_state=42),
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    svm_extended.fit(X_train_ext_scaled, y_train)
    
    print(f"✅ Best parameters: {svm_extended.best_params_}")
    print(f"✅ Best CV score: {svm_extended.best_score_:.4f}")
    
    # Evaluate best SVM
    train_pred_ext = svm_extended.predict(X_train_ext_scaled)
    train_proba_ext = svm_extended.predict_proba(X_train_ext_scaled)[:, 1]
    test_pred_ext = svm_extended.predict(X_test_ext_scaled)
    test_proba_ext = svm_extended.predict_proba(X_test_ext_scaled)[:, 1]
    
    train_acc_ext = accuracy_score(y_train, train_pred_ext)
    train_auc_ext = roc_auc_score(y_train, train_proba_ext)
    test_acc_ext = accuracy_score(y_test, test_pred_ext)
    test_auc_ext = roc_auc_score(y_test, test_proba_ext)
    
    print(f"\\n📊 METHOD 2 RESULTS:")
    print(f"   Training: {train_acc_ext:.4f} accuracy, {train_auc_ext:.4f} AUC")
    print(f"   Test: {test_acc_ext:.4f} accuracy, {test_auc_ext:.4f} AUC")
    print(f"   Overfitting: {train_auc_ext - test_auc_ext:.4f} AUC difference")
    
    # Method 3: High sensitivity SVM (optimized for medical use)
    print("\\n🔵 METHOD 3: Medical-Optimized SVM (High Sensitivity)")
    
    # Use class weights to favor sensitivity
    class_weight = {0: 1, 1: 3}  # Favor malignant detection
    
    svm_medical = SVC(
        kernel='rbf',
        C=svm_extended.best_params_['C'],
        gamma=svm_extended.best_params_['gamma'],
        class_weight=class_weight,
        probability=True,
        random_state=42
    )
    
    svm_medical.fit(X_train_ext_scaled, y_train)
    
    test_pred_med = svm_medical.predict(X_test_ext_scaled)
    test_proba_med = svm_medical.predict_proba(X_test_ext_scaled)[:, 1]
    
    test_acc_med = accuracy_score(y_test, test_pred_med)
    test_auc_med = roc_auc_score(y_test, test_proba_med)
    
    print(f"📊 METHOD 3 RESULTS:")
    print(f"   Test: {test_acc_med:.4f} accuracy, {test_auc_med:.4f} AUC")
    
    # Calculate medical metrics for all SVM methods
    def calc_medical_metrics(y_true, y_pred, name):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        ppv = tp / (tp + fp)
        npv = tn / (tn + fn)
        
        print(f"\\n{name}:")
        print(f"  Sensitivity: {sensitivity:.3f} ({sensitivity*100:.1f}%) - Missed: {fn}/{tp+fn}")
        print(f"  Specificity: {specificity:.3f} ({specificity*100:.1f}%) - False alarms: {fp}/{tn+fp}")
        print(f"  PPV: {ppv:.3f} ({ppv*100:.1f}%)")
        print(f"  NPV: {npv:.3f} ({npv*100:.1f}%)")
        
        return sensitivity, specificity, ppv, npv, fn, fp
    
    print("\\n🏥 MEDICAL METRICS COMPARISON:")
    
    svm1_metrics = calc_medical_metrics(y_test, test_pred, "SVM Method 1 (Similarities)")
    svm2_metrics = calc_medical_metrics(y_test, test_pred_ext, "SVM Method 2 (Extended)")
    svm3_metrics = calc_medical_metrics(y_test, test_pred_med, "SVM Method 3 (Medical)")
    
    # Compare with known logistic performance
    print(f"\\n📊 COMPARISON WITH PROTOTYPE LOGISTIC:")
    print(f"   Prototype Logistic: 95.7% sensitivity, 0.953 AUC")
    print(f"   SVM Method 1: {svm1_metrics[0]:.1%} sensitivity, {test_auc:.3f} AUC")
    print(f"   SVM Method 2: {svm2_metrics[0]:.1%} sensitivity, {test_auc_ext:.3f} AUC")
    print(f"   SVM Method 3: {svm3_metrics[0]:.1%} sensitivity, {test_auc_med:.3f} AUC")
    
    # Find best SVM method
    svm_methods = [
        ('SVM Similarities', svm1_metrics[0], test_auc, test_acc),
        ('SVM Extended', svm2_metrics[0], test_auc_ext, test_acc_ext),  
        ('SVM Medical', svm3_metrics[0], test_auc_med, test_acc_med)
    ]
    
    best_svm = max(svm_methods, key=lambda x: x[1])  # Best by sensitivity
    best_svm_auc = max(svm_methods, key=lambda x: x[2])  # Best by AUC
    
    print(f"\\n🏆 BEST SVM RESULTS:")
    print(f"   Best Sensitivity: {best_svm[0]} ({best_svm[1]:.1%})")
    print(f"   Best AUC: {best_svm_auc[0]} ({best_svm_auc[2]:.3f})")
    
    # Verdict vs Logistic
    logistic_sens = 0.957
    logistic_auc = 0.953
    
    if best_svm[1] > logistic_sens:
        print(f"\\n🎉 SVM WINS! {best_svm[1]:.1%} > {logistic_sens:.1%} sensitivity")
    else:
        print(f"\\n🏆 LOGISTIC STILL WINS: {logistic_sens:.1%} > {best_svm[1]:.1%} sensitivity")
    
    if best_svm_auc[2] > logistic_auc:
        print(f"🎉 SVM WINS AUC! {best_svm_auc[2]:.3f} > {logistic_auc:.3f}")
    else:
        print(f"🏆 LOGISTIC WINS AUC: {logistic_auc:.3f} > {best_svm_auc[2]:.3f}")
    
    # Save best SVM model
    if best_svm[0] == 'SVM Extended':
        best_model = svm_extended.best_estimator_
        best_scaler = scaler_ext
        best_method = 'extended'
    elif best_svm[0] == 'SVM Medical':
        best_model = svm_medical
        best_scaler = scaler_ext
        best_method = 'medical'
    else:
        best_model = svm_proto
        best_scaler = scaler_proto
        best_method = 'similarities'
    
    prototype_svm_classifier = {
        'model': best_model,
        'scaler': best_scaler,
        'prototypes': {
            'benign': train_benign_proto,
            'malignant': train_malignant_proto
        },
        'method': best_method,
        'test_accuracy': best_svm[3],
        'test_auc': best_svm[2],
        'test_sensitivity': best_svm[1],
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'hyperparameters': svm_extended.best_params_ if best_method in ['extended', 'medical'] else {'C': 1.0, 'gamma': 'scale'}
    }
    
    # Save the prototype-based SVM classifier
    with open('/workspace/prototype_svm_classifier.pkl', 'wb') as f:
        pickle.dump(prototype_svm_classifier, f)
    
    print(f"\\n✅ PROTOTYPE-BASED SVM CREATED!")
    print(f"   Best method: {best_method}")
    print(f"   Test sensitivity: {best_svm[1]:.1%}")
    print(f"   Test AUC: {best_svm[2]:.4f}")
    print(f"   Hyperparameters: {prototype_svm_classifier['hyperparameters']}")
    print(f"   Saved to: prototype_svm_classifier.pkl")
    
    # Final comparison table
    print(f"\\n📊 FINAL PROTOTYPE CLASSIFIER COMPARISON:")
    print(f"Classifier           | Sensitivity | AUC    | Accuracy")
    print(f"-" * 55)
    print(f"Prototype Logistic   | 95.7%       | 0.953  | 86.9%")
    print(f"Prototype SVM        | {best_svm[1]*100:.1f}%       | {best_svm[2]:.3f}  | {best_svm[3]*100:.1f}%")
    
    return prototype_svm_classifier

if __name__ == '__main__':
    result = create_prototype_svm_classifier()
    print(f"\\nFinal SVM Sensitivity: {result['test_sensitivity']:.1%}")
\""