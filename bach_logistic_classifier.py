"""
BACH 4-Class Logistic Regression Classifier
Trains on GigaPath features to predict: normal, benign, insitu, invasive
Uses stratified cross-validation and generates ROC plots
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import base64
import io
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
import warnings
warnings.filterwarnings('ignore')

class BACHLogisticClassifier:
    def __init__(self, cache_file='/workspace/embeddings_cache_REAL_GIGAPATH.pkl'):
        self.cache_file = cache_file
        self.model = None
        self.svm_model = None
        self.label_encoder = None
        self.cv_scores = None
        self.svm_cv_scores = None
        self.test_scores = None
        self.svm_test_scores = None
        self.roc_data = None
        self.svm_roc_data = None
        self.test_roc_data = None
        self.svm_test_roc_data = None
        self.data_splits = None
        self.class_names = ['normal', 'benign', 'insitu', 'invasive']
        
    def load_bach_data(self):
        """Load BACH dataset features and labels from cache"""
        print("Loading BACH data from cache...")
        
        with open(self.cache_file, 'rb') as f:
            cache = pickle.load(f)
        
        # Extract BACH data only
        combined_data = cache['combined']
        bach_indices = [i for i, ds in enumerate(combined_data['datasets']) if ds == 'bach']
        
        if not bach_indices:
            raise ValueError("No BACH data found in cache")
        
        # Get BACH features and labels
        features = np.array([combined_data['features'][i] for i in bach_indices])
        labels = [combined_data['labels'][i] for i in bach_indices]
        filenames = [combined_data['filenames'][i] for i in bach_indices]
        
        print(f"Loaded BACH data: {len(features)} samples")
        print(f"Classes: {set(labels)}")
        print(f"Class distribution: {[(label, labels.count(label)) for label in set(labels)]}")
        
        return features, labels, filenames
    
    def train_classifier(self):
        """Train classifiers with proper train/validation/test splits"""
        features, labels, filenames = self.load_bach_data()
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(labels)
        
        print(f"Total samples: {len(features)}")
        print(f"Label encoding: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        print(f"Class distribution: {[(cls, sum(y_encoded == i)) for i, cls in enumerate(self.label_encoder.classes_)]}")
        
        # PROPER TRAIN/VALIDATION/TEST SPLITS
        # First split: 80% train+val, 20% test (stratified)
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
        )
        
        # Second split: 75% train, 25% validation from remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
        )
        
        # Store split information
        self.data_splits = {
            'train_size': len(X_train),
            'val_size': len(X_val), 
            'test_size': len(X_test),
            'train_distribution': [(cls, sum(y_train == i)) for i, cls in enumerate(self.label_encoder.classes_)],
            'val_distribution': [(cls, sum(y_val == i)) for i, cls in enumerate(self.label_encoder.classes_)],
            'test_distribution': [(cls, sum(y_test == i)) for i, cls in enumerate(self.label_encoder.classes_)]
        }
        
        print(f"\\n📊 DATA SPLITS:")
        print(f"  Train: {len(X_train)} samples - {self.data_splits['train_distribution']}")
        print(f"  Validation: {len(X_val)} samples - {self.data_splits['val_distribution']}")
        print(f"  Test: {len(X_test)} samples - {self.data_splits['test_distribution']}")
        
        # LOGISTIC REGRESSION TRAINING
        print("\\n🔥 Training Logistic Regression...")
        self.model = LogisticRegression(
            multi_class='ovr',
            solver='liblinear',
            C=1.0,
            random_state=42,
            max_iter=1000
        )
        
        # Train on training set only
        self.model.fit(X_train, y_train)
        
        # Validation performance
        val_pred_proba = self.model.predict_proba(X_val)
        val_pred = self.model.predict(X_val)
        val_accuracy = (val_pred == y_val).mean()
        
        # TEST PERFORMANCE (held-out)
        test_pred_proba = self.model.predict_proba(X_test)
        test_pred = self.model.predict(X_test)
        test_accuracy = (test_pred == y_test).mean()
        
        print(f"  Validation Accuracy: {val_accuracy:.3f}")
        print(f"  🎯 TEST Accuracy: {test_accuracy:.3f}")
        
        # Generate TEST ROC curves (honest evaluation)
        self.generate_test_roc_curves(y_test, test_pred_proba, "Logistic Regression")
        
        # SVM RBF TRAINING
        print("\\n🔥 Training SVM RBF...")
        self.svm_model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        )
        
        # Train on training set only
        self.svm_model.fit(X_train, y_train)
        
        # Validation performance
        svm_val_pred_proba = self.svm_model.predict_proba(X_val)
        svm_val_pred = self.svm_model.predict(X_val)
        svm_val_accuracy = (svm_val_pred == y_val).mean()
        
        # TEST PERFORMANCE (held-out)
        svm_test_pred_proba = self.svm_model.predict_proba(X_test)
        svm_test_pred = self.svm_model.predict(X_test)
        svm_test_accuracy = (svm_test_pred == y_test).mean()
        
        print(f"  Validation Accuracy: {svm_val_accuracy:.3f}")
        print(f"  🎯 TEST Accuracy: {svm_test_accuracy:.3f}")
        
        # Generate SVM TEST ROC curves
        self.generate_svm_test_roc_curves(y_test, svm_test_pred_proba, "SVM RBF")
        
        # Store test scores (what we actually report)
        self.test_scores = {'accuracy': test_accuracy, 'predictions': test_pred, 'probabilities': test_pred_proba}
        self.svm_test_scores = {'accuracy': svm_test_accuracy, 'predictions': svm_test_pred, 'probabilities': svm_test_pred_proba}
        
        # Print HONEST test performance
        print(f"\\n🎯 HONEST TEST PERFORMANCE:")
        print(f"  Logistic Regression Test Accuracy: {test_accuracy:.3f}")
        print(f"  SVM RBF Test Accuracy: {svm_test_accuracy:.3f}")
        
        print("\\n📈 Test Set Classification Reports:")
        print("\\nLogistic Regression (Test Set):")
        print(classification_report(y_test, test_pred, target_names=self.label_encoder.classes_))
        
        print("\\nSVM RBF (Test Set):")
        print(classification_report(y_test, svm_test_pred, target_names=self.label_encoder.classes_))
        
        return self.model, self.label_encoder
    
    def generate_roc_curves(self, y_true, y_pred_proba):
        """Generate ROC curves for all classes"""
        n_classes = len(self.label_encoder.classes_)
        
        # Binarize labels for ROC computation
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true, classes=range(n_classes))
        
        # Compute ROC curve for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve
        fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_pred_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Store ROC data
        self.roc_data = {
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'class_names': self.label_encoder.classes_.tolist()
        }
        
        print(f"ROC AUC scores: {[(self.label_encoder.classes_[i], roc_auc[i]) for i in range(n_classes)]}")
        print(f"Micro-average ROC AUC: {roc_auc['micro']:.3f}")
        
        return self.roc_data
    
    def generate_svm_roc_curves(self, y_true, y_pred_proba):
        """Generate ROC curves for SVM classifier"""
        n_classes = len(self.label_encoder.classes_)
        
        # Binarize labels for ROC computation
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true, classes=range(n_classes))
        
        # Compute ROC curve for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve
        fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_pred_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Store SVM ROC data
        self.svm_roc_data = {
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'class_names': self.label_encoder.classes_.tolist()
        }
        
        print(f"SVM ROC AUC scores: {[(self.label_encoder.classes_[i], roc_auc[i]) for i in range(n_classes)]}")
        print(f"SVM Micro-average ROC AUC: {roc_auc['micro']:.3f}")
        
        return self.svm_roc_data
    
    def generate_test_roc_curves(self, y_true, y_pred_proba, classifier_name):
        """Generate ROC curves from TEST SET predictions (honest evaluation)"""
        n_classes = len(self.label_encoder.classes_)
        
        # Binarize labels for ROC computation
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true, classes=range(n_classes))
        
        # Compute ROC curve for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve
        fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_pred_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Store TEST ROC data
        self.test_roc_data = {
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'class_names': self.label_encoder.classes_.tolist(),
            'classifier': classifier_name,
            'evaluation_type': 'TEST_SET'
        }
        
        print(f"{classifier_name} TEST ROC AUC scores: {[(self.label_encoder.classes_[i], roc_auc[i]) for i in range(n_classes)]}")
        print(f"{classifier_name} TEST Micro-average ROC AUC: {roc_auc['micro']:.3f}")
        
        return self.test_roc_data
    
    def generate_svm_test_roc_curves(self, y_true, y_pred_proba, classifier_name):
        """Generate SVM ROC curves from TEST SET predictions"""
        n_classes = len(self.label_encoder.classes_)
        
        # Binarize labels for ROC computation
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true, classes=range(n_classes))
        
        # Compute ROC curve for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve
        fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_pred_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Store SVM TEST ROC data
        self.svm_test_roc_data = {
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'class_names': self.label_encoder.classes_.tolist(),
            'classifier': classifier_name,
            'evaluation_type': 'TEST_SET'
        }
        
        print(f"{classifier_name} TEST ROC AUC scores: {[(self.label_encoder.classes_[i], roc_auc[i]) for i in range(n_classes)]}")
        print(f"{classifier_name} TEST Micro-average ROC AUC: {roc_auc['micro']:.3f}")
        
        return self.svm_test_roc_data
    
    def plot_roc_curves(self, save_path=None, return_base64=False):
        """Generate ROC curve plots"""
        if not self.roc_data:
            raise ValueError("ROC data not available. Train model first.")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot ROC curve for each class
        colors = ['red', 'blue', 'orange', 'purple']
        for i, (class_name, color) in enumerate(zip(self.roc_data['class_names'], colors)):
            ax.plot(
                self.roc_data['fpr'][i], 
                self.roc_data['tpr'][i],
                color=color,
                lw=2,
                label=f'{class_name.upper()} (AUC = {self.roc_data["roc_auc"][i]:.3f})'
            )
        
        # Plot micro-average ROC curve
        ax.plot(
            self.roc_data['fpr']["micro"], 
            self.roc_data['tpr']["micro"],
            color='black',
            linestyle='--',
            lw=2,
            label=f'Micro-avg (AUC = {self.roc_data["roc_auc"]["micro"]:.3f})'
        )
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('BACH 4-Class ROC Curves\\nLogistic Regression on GigaPath Features', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC plot saved to: {save_path}")
        
        if return_base64:
            # Convert plot to base64 for frontend
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=200, bbox_inches='tight')
            buffer.seek(0)
            plot_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            return plot_base64
        
        return fig
    
    def predict(self, features):
        """Predict class probabilities for new features"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_classifier() first.")
        
        # Ensure features are 2D
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Get probabilities for each class
        probabilities = self.model.predict_proba(features)[0]
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = self.label_encoder.classes_[predicted_class_idx]
        confidence = probabilities[predicted_class_idx]
        
        # Create result dictionary
        result = {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(self.label_encoder.classes_, probabilities)
            },
            'class_ranking': [
                {
                    'class': self.label_encoder.classes_[i],
                    'probability': float(probabilities[i])
                }
                for i in np.argsort(probabilities)[::-1]  # Sort by probability descending
            ]
        }
        
        return result
    
    def predict_svm(self, features):
        """Predict using SVM RBF classifier"""
        if self.svm_model is None:
            raise ValueError("SVM model not trained. Call train_classifier() first.")
        
        # Ensure features are 2D
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Get probabilities for each class
        probabilities = self.svm_model.predict_proba(features)[0]
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = self.label_encoder.classes_[predicted_class_idx]
        confidence = probabilities[predicted_class_idx]
        
        # Create result dictionary
        result = {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(self.label_encoder.classes_, probabilities)
            },
            'class_ranking': [
                {
                    'class': self.label_encoder.classes_[i],
                    'probability': float(probabilities[i])
                }
                for i in np.argsort(probabilities)[::-1]  # Sort by probability descending
            ]
        }
        
        return result
    
    def save_model(self, model_path='/workspace/bach_logistic_model.pkl'):
        """Save trained model and label encoder"""
        model_data = {
            'model': self.model,
            'svm_model': self.svm_model,
            'label_encoder': self.label_encoder,
            'cv_scores': self.cv_scores,
            'svm_cv_scores': self.svm_cv_scores,
            'test_scores': self.test_scores,
            'svm_test_scores': self.svm_test_scores,
            'roc_data': self.roc_data,
            'svm_roc_data': self.svm_roc_data,
            'test_roc_data': self.test_roc_data,
            'svm_test_roc_data': self.svm_test_roc_data,
            'data_splits': self.data_splits,
            'class_names': self.class_names
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {model_path}")
        return model_path
    
    def load_model(self, model_path='/workspace/bach_logistic_model.pkl'):
        """Load pre-trained model"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.svm_model = model_data.get('svm_model', None)
            self.label_encoder = model_data['label_encoder']
            self.cv_scores = model_data.get('cv_scores', None)
            self.svm_cv_scores = model_data.get('svm_cv_scores', None)
            self.test_scores = model_data.get('test_scores', None)
            self.svm_test_scores = model_data.get('svm_test_scores', None)
            self.roc_data = model_data.get('roc_data', None)
            self.svm_roc_data = model_data.get('svm_roc_data', None)
            self.test_roc_data = model_data.get('test_roc_data', None)
            self.svm_test_roc_data = model_data.get('svm_test_roc_data', None)
            self.data_splits = model_data.get('data_splits', None)
            self.class_names = model_data['class_names']
            
            print(f"Model loaded from: {model_path}")
            print(f"CV Accuracy: {self.cv_scores.mean():.3f} ± {self.cv_scores.std():.3f}")
            return True
            
        except FileNotFoundError:
            print(f"Model file not found: {model_path}")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

def train_bach_classifier():
    """Main function to train BACH classifier"""
    classifier = BACHLogisticClassifier()
    
    # Train the classifier
    model, label_encoder = classifier.train_classifier()
    
    # Save the model
    model_path = classifier.save_model()
    
    # Generate and save ROC plot
    roc_plot_path = '/workspace/bach_roc_plot.png'
    classifier.plot_roc_curves(save_path=roc_plot_path)
    
    print(f"\\n✅ BACH Classifier Training Complete!")
    print(f"📊 Model saved: {model_path}")
    print(f"📈 ROC plot saved: {roc_plot_path}")
    
    return classifier

if __name__ == "__main__":
    # Train the classifier
    classifier = train_bach_classifier()
    
    # Test with some sample predictions
    print("\\n🔍 Testing classifier...")
    features, labels, filenames = classifier.load_bach_data()
    
    # Test on first 3 samples
    for i in range(3):
        result = classifier.predict(features[i])
        actual = labels[i]
        print(f"Sample {i+1}: Predicted={result['predicted_class']} (conf: {result['confidence']:.3f}), Actual={actual}")