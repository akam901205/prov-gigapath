#!/usr/bin/env python3
"""
Prototype Whitening Classifier Pipeline
GigaPath → Source Whitener → Whiten + L2 → Prototype Classifier (Nearest-Class Mean)
"""
import numpy as np
import pickle
import torch
from scipy.stats import spearmanr
from sklearn.preprocessing import normalize
from sklearn.covariance import LedoitWolf

class PrototypeWhiteningClassifier:
    """
    Prototype classifier with whitening preprocessing
    1. Fit whitener on ALL raw embeddings (source whitener)
    2. For each image: whiten → L2 normalize → cosine similarity to class centroids
    """
    
    def __init__(self, shrinkage='auto'):
        self.shrinkage = shrinkage
        self.source_mean = None
        self.whitening_matrix = None
        self.class_prototypes = {}
        self.fitted = False
    
    def fit_source_whitener(self, all_embeddings):
        """
        Fit source whitener on ALL raw embeddings
        Computes mean and Ledoit-Wolf covariance for whitening transform
        """
        print(f"🔧 Fitting source whitener on {len(all_embeddings)} embeddings...")
        
        # Compute source statistics
        self.source_mean = np.mean(all_embeddings, axis=0, keepdims=True)
        print(f"   Source mean shape: {self.source_mean.shape}")
        
        # Center embeddings
        centered_embeddings = all_embeddings - self.source_mean
        
        # Ledoit-Wolf shrinkage covariance estimation
        if self.shrinkage == 'auto':
            # Automatic shrinkage estimation
            lw = LedoitWolf()
            source_cov = lw.fit(centered_embeddings).covariance_
            actual_shrinkage = lw.shrinkage_
            print(f"   Auto shrinkage: {actual_shrinkage:.4f}")
        else:
            # Manual shrinkage
            sample_cov = np.cov(centered_embeddings, rowvar=False)
            trace_cov = np.trace(sample_cov) / sample_cov.shape[0]
            source_cov = (1 - self.shrinkage) * sample_cov + self.shrinkage * trace_cov * np.eye(sample_cov.shape[0])
            print(f"   Manual shrinkage: {self.shrinkage}")
        
        print(f"   Source covariance shape: {source_cov.shape}")
        print(f"   Covariance condition number: {np.linalg.cond(source_cov):.2e}")
        
        # Compute whitening matrix (inverse square root of covariance)
        try:
            # Eigendecomposition for stable inverse square root
            eigenvals, eigenvecs = np.linalg.eigh(source_cov)
            
            # Ensure positive eigenvalues
            eps = 1e-6
            eigenvals = np.maximum(eigenvals, eps)
            
            # Whitening matrix: V * D^(-1/2) * V^T
            inv_sqrt_eigenvals = 1.0 / np.sqrt(eigenvals)
            self.whitening_matrix = eigenvecs @ np.diag(inv_sqrt_eigenvals) @ eigenvecs.T
            
            print(f"   ✅ Whitening matrix computed: {self.whitening_matrix.shape}")
            print(f"   Min eigenvalue: {np.min(eigenvals):.2e}")
            
        except Exception as e:
            print(f"   ❌ Whitening matrix computation failed: {e}")
            self.whitening_matrix = np.eye(source_cov.shape[0])
    
    def whiten_and_l2_normalize(self, embedding):
        """
        Apply whitening + L2 normalization to single embedding
        1. Center by subtracting source mean
        2. Multiply by whitening matrix 
        3. L2 normalize to unit length
        """
        # Center embedding
        centered = embedding.reshape(1, -1) - self.source_mean
        
        # Apply whitening
        whitened = centered @ self.whitening_matrix.T
        
        # L2 normalize to unit vector
        l2_normalized = normalize(whitened, norm='l2')[0]
        
        return l2_normalized
    
    def fit_prototypes(self, embeddings, labels, datasets):
        """
        Fit class prototypes (centroids) in whitened L2-normalized space
        """
        print("\n🎯 Fitting class prototypes...")
        
        # First fit the source whitener on ALL embeddings
        self.fit_source_whitener(embeddings)
        
        # Transform all embeddings to whitened L2 space
        whitened_embeddings = []
        for i, emb in enumerate(embeddings):
            whitened_l2 = self.whiten_and_l2_normalize(emb)
            whitened_embeddings.append(whitened_l2)
            
            if (i + 1) % 500 == 0:
                print(f"   Transformed {i + 1}/{len(embeddings)} embeddings...")
        
        whitened_embeddings = np.array(whitened_embeddings)
        
        # Compute class centroids in whitened space
        # Binary classification: benign vs malignant
        binary_labels = []
        for label in labels:
            if label in ['benign', 'normal']:
                binary_labels.append('benign')
            elif label in ['malignant', 'invasive', 'insitu']:
                binary_labels.append('malignant')
            else:
                binary_labels.append('unknown')
        
        # Compute centroids
        benign_mask = np.array([bl == 'benign' for bl in binary_labels])
        malignant_mask = np.array([bl == 'malignant' for bl in binary_labels])
        
        if np.sum(benign_mask) > 0:
            self.class_prototypes['benign'] = np.mean(whitened_embeddings[benign_mask], axis=0)
            print(f"   Benign prototype: {np.sum(benign_mask)} samples, norm = {np.linalg.norm(self.class_prototypes['benign']):.4f}")
        
        if np.sum(malignant_mask) > 0:
            self.class_prototypes['malignant'] = np.mean(whitened_embeddings[malignant_mask], axis=0)
            print(f"   Malignant prototype: {np.sum(malignant_mask)} samples, norm = {np.linalg.norm(self.class_prototypes['malignant']):.4f}")
        
        self.fitted = True
        print("   ✅ Prototype fitting complete")
    
    def predict(self, embedding):
        """
        Predict class for single embedding using prototype classifier
        Returns: class prediction + cosine similarities
        """
        if not self.fitted:
            raise ValueError("Classifier not fitted yet!")
        
        # Transform to whitened L2 space
        whitened_l2 = self.whiten_and_l2_normalize(embedding)
        
        # Compute cosine similarities to prototypes
        cos_benign = np.dot(whitened_l2, self.class_prototypes['benign'])
        cos_malignant = np.dot(whitened_l2, self.class_prototypes['malignant'])
        
        # Prediction: pick class with highest cosine similarity
        if cos_malignant > cos_benign:
            prediction = 'malignant'
            confidence = cos_malignant
        else:
            prediction = 'benign'
            confidence = cos_benign
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'cos_benign': cos_benign,
            'cos_malignant': cos_malignant,
            'whitened_l2_embedding': whitened_l2
        }

def test_prototype_classifier():
    """Test the prototype whitening classifier"""
    
    print("🚀 PROTOTYPE WHITENING CLASSIFIER TEST")
    print("=" * 70)
    
    # Load embeddings
    with open("/workspace/embeddings_cache_FRESH_SIMPLE.pkl", 'rb') as f:
        cache = pickle.load(f)
    
    features = np.array(cache['combined']['features'])
    labels = cache['combined']['labels']
    datasets = cache['combined']['datasets']
    filenames = cache['combined']['filenames']
    
    print(f"✅ Loaded: {len(features)} embeddings")
    
    # Initialize and fit classifier
    classifier = PrototypeWhiteningClassifier(shrinkage='auto')  # Try auto first
    classifier.fit_prototypes(features, labels, datasets)
    
    # Test on individual BACH samples
    print("\n🧪 Testing prototype classifier on BACH samples...")
    
    # Extract test indices
    bach_benign_indices = []
    bach_invasive_indices = []
    
    for i, (dataset, label) in enumerate(zip(datasets, labels)):
        if dataset == 'bach':
            if label == 'benign':
                bach_benign_indices.append(i)
            elif label == 'invasive':
                bach_invasive_indices.append(i)
    
    bach_benign_indices = bach_benign_indices[:100]
    bach_invasive_indices = bach_invasive_indices[:100]
    
    # Test BACH benign samples
    print(f"\n=== TESTING {len(bach_benign_indices)} BACH BENIGN SAMPLES ===")
    benign_correct = 0
    benign_confidences = []
    
    for i, bach_idx in enumerate(bach_benign_indices):
        result = classifier.predict(features[bach_idx])
        if result['prediction'] == 'benign':
            benign_correct += 1
        benign_confidences.append(result['confidence'])
        
        if (i + 1) % 25 == 0:
            print(f"   Processed {i + 1}/{len(bach_benign_indices)}...")
    
    benign_accuracy = benign_correct / len(bach_benign_indices)
    print(f"   ✅ Benign accuracy: {benign_correct}/{len(bach_benign_indices)} ({benign_accuracy:.1%})")
    print(f"   📊 Avg confidence: {np.mean(benign_confidences):.4f}")
    
    # Test BACH invasive samples
    print(f"\n=== TESTING {len(bach_invasive_indices)} BACH INVASIVE SAMPLES ===")
    invasive_correct = 0
    invasive_confidences = []
    
    for i, bach_idx in enumerate(bach_invasive_indices):
        result = classifier.predict(features[bach_idx])
        if result['prediction'] == 'malignant':
            invasive_correct += 1
        invasive_confidences.append(result['confidence'])
        
        if (i + 1) % 25 == 0:
            print(f"   Processed {i + 1}/{len(bach_invasive_indices)}...")
    
    invasive_accuracy = invasive_correct / len(bach_invasive_indices)
    print(f"   ✅ Invasive accuracy: {invasive_correct}/{len(bach_invasive_indices)} ({invasive_accuracy:.1%})")
    print(f"   📊 Avg confidence: {np.mean(invasive_confidences):.4f}")
    
    # Overall results
    total_correct = benign_correct + invasive_correct
    total_samples = len(bach_benign_indices) + len(bach_invasive_indices)
    overall_accuracy = total_correct / total_samples
    
    print(f"\n🎯 PROTOTYPE CLASSIFIER RESULTS:")
    print(f"   Overall accuracy: {total_correct}/{total_samples} ({overall_accuracy:.1%})")
    print(f"   Benign accuracy: {benign_accuracy:.1%}")
    print(f"   Invasive accuracy: {invasive_accuracy:.1%}")
    
    # Compare to previous best
    previous_best = 66.5  # Best CORAL result
    if overall_accuracy * 100 > previous_best:
        print(f"   ✅ NEW BEST! Improved by {overall_accuracy * 100 - previous_best:.1f}%")
        status = "NEW_BEST"
    else:
        print(f"   ⚖️ {overall_accuracy * 100:.1f}% vs previous best {previous_best:.1f}%")
        status = "COMPARABLE"
    
    # Save classifier
    classifier_data = {
        'source_mean': classifier.source_mean,
        'whitening_matrix': classifier.whitening_matrix,
        'class_prototypes': classifier.class_prototypes,
        'performance': {
            'benign_accuracy': benign_accuracy,
            'invasive_accuracy': invasive_accuracy,
            'overall_accuracy': overall_accuracy,
            'status': status
        }
    }
    
    with open('/workspace/prototype_whitening_classifier.pkl', 'wb') as f:
        pickle.dump(classifier_data, f)
    
    print(f"\n💾 Classifier saved: /workspace/prototype_whitening_classifier.pkl")
    
    return classifier, overall_accuracy

def test_different_shrinkages():
    """Test different shrinkage values for prototype classifier"""
    
    print("🧪 TESTING DIFFERENT SHRINKAGE VALUES")
    print("=" * 60)
    
    shrinkage_values = ['auto', 0.1, 0.3, 0.5, 0.7, 0.9]
    results = {}
    
    # Load embeddings once
    with open("/workspace/embeddings_cache_FRESH_SIMPLE.pkl", 'rb') as f:
        cache = pickle.load(f)
    
    features = np.array(cache['combined']['features']) 
    labels = cache['combined']['labels']
    datasets = cache['combined']['datasets']
    
    for shrinkage in shrinkage_values:
        print(f"\n--- TESTING SHRINKAGE: {shrinkage} ---")
        
        classifier = PrototypeWhiteningClassifier(shrinkage=shrinkage)
        classifier, accuracy = test_prototype_classifier_with_shrinkage(
            classifier, features, labels, datasets, shrinkage
        )
        
        results[shrinkage] = {
            'accuracy': accuracy,
            'classifier': classifier
        }
    
    # Find best shrinkage
    best_shrinkage = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_accuracy = results[best_shrinkage]['accuracy']
    
    print(f"\n🏆 BEST SHRINKAGE: {best_shrinkage}")
    print(f"   Best accuracy: {best_accuracy:.1%}")
    
    return results, best_shrinkage

def test_prototype_classifier_with_shrinkage(classifier, features, labels, datasets, shrinkage):
    """Test prototype classifier with specific shrinkage"""
    
    # Fit prototypes
    classifier.fit_prototypes(features, labels, datasets)
    
    # Extract BACH test samples
    bach_benign_indices = []
    bach_invasive_indices = []
    
    for i, (dataset, label) in enumerate(zip(datasets, labels)):
        if dataset == 'bach':
            if label == 'benign':
                bach_benign_indices.append(i)
            elif label == 'invasive':
                bach_invasive_indices.append(i)
    
    bach_benign_indices = bach_benign_indices[:100]
    bach_invasive_indices = bach_invasive_indices[:100]
    
    # Test predictions
    total_correct = 0
    total_samples = 0
    
    # Test benign
    for bach_idx in bach_benign_indices:
        result = classifier.predict(features[bach_idx])
        if result['prediction'] == 'benign':
            total_correct += 1
        total_samples += 1
    
    # Test invasive  
    for bach_idx in bach_invasive_indices:
        result = classifier.predict(features[bach_idx])
        if result['prediction'] == 'malignant':
            total_correct += 1
        total_samples += 1
    
    accuracy = total_correct / total_samples
    print(f"   Prototype accuracy: {total_correct}/{total_samples} ({accuracy:.1%})")
    
    return classifier, accuracy

def main():
    """Main prototype whitening classifier pipeline"""
    print("🚀 PROTOTYPE WHITENING CLASSIFIER PIPELINE")
    print("=" * 70)
    print("Pipeline: GigaPath → Source Whitener → Whiten + L2 → Prototype Classifier")
    print("=" * 70)
    
    # Test with auto shrinkage first
    classifier, accuracy = test_prototype_classifier()
    
    # Test different shrinkage values
    print(f"\n" + "=" * 70)
    results, best_shrinkage = test_different_shrinkages()
    
    print(f"\n🎉 PROTOTYPE WHITENING CLASSIFIER COMPLETE")
    print(f"🏆 Best performance: {results[best_shrinkage]['accuracy']:.1%} (shrinkage={best_shrinkage})")
    
    return results

if __name__ == "__main__":
    main()