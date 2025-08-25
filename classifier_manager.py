"""
Classifier Manager - Handles all ML model predictions safely
Provides isolated prediction functions with robust error handling
"""

import traceback
import os
from bach_logistic_classifier import BACHLogisticClassifier
from breakhis_binary_classifier import BreakHisBinaryClassifier
from bach_normal_benign_classifier import BACHNormalBenignClassifier
from bach_invasive_insitu_classifier import BACHInvasiveInsituClassifier

class ClassifierManager:
    def __init__(self):
        self.bach_classifier = None
        self.breakhis_classifier = None
        self.bach_normal_benign_classifier = None
        self.bach_invasive_insitu_classifier = None
        self._bach_loaded = False
        self._breakhis_loaded = False
        self._normal_benign_loaded = False
        self._invasive_insitu_loaded = False
    
    def load_bach_classifier(self):
        """Load BACH classifier on demand"""
        if not self._bach_loaded:
            try:
                print("🔥 Loading BACH classifier...")
                self.bach_classifier = BACHLogisticClassifier(cache_file='/workspace/embeddings_cache_L2_REPROCESSED.pkl')
                
                # Try retrained model first
                retrained_path = '/workspace/bach_logistic_model_L2_RETRAINED.pkl'
                fallback_path = '/workspace/bach_logistic_model.pkl'
                
                if os.path.exists(retrained_path):
                    success = self.bach_classifier.load_model(retrained_path)
                    print(f"🔥 Using L2 RETRAINED BACH model: {success}")
                else:
                    success = self.bach_classifier.load_model(fallback_path)
                    print(f"🔥 Using OLD BACH model: {success}")
                
                self._bach_loaded = success
                if success:
                    print(f"🔥 BACH classes: {self.bach_classifier.class_names}")
                    print(f"🔥 BACH models available: LR={self.bach_classifier.model is not None}, SVM={self.bach_classifier.svm_model is not None}, XGB={hasattr(self.bach_classifier, 'xgb_model') and self.bach_classifier.xgb_model is not None}")
            except Exception as e:
                print(f"❌ BACH classifier loading failed: {e}")
                self._bach_loaded = False
        return self._bach_loaded
    
    def load_breakhis_classifier(self):
        """Load BreakHis classifier on demand"""
        if not self._breakhis_loaded:
            try:
                print("🔥 Loading BreakHis classifier...")
                self.breakhis_classifier = BreakHisBinaryClassifier(cache_file='/workspace/embeddings_cache_L2_REPROCESSED.pkl')
                
                # Try retrained model first
                retrained_path = '/workspace/breakhis_binary_model_L2_RETRAINED.pkl'
                fallback_path = '/workspace/breakhis_binary_model.pkl'
                
                if os.path.exists(retrained_path):
                    success = self.breakhis_classifier.load_model(retrained_path)
                    print(f"🔥 Using L2 RETRAINED BreakHis model: {success}")
                else:
                    success = self.breakhis_classifier.load_model(fallback_path)
                    print(f"🔥 Using OLD BreakHis model: {success}")
                
                self._breakhis_loaded = success
                if success:
                    print(f"🔥 BreakHis classes: {self.breakhis_classifier.class_names}")
                    print(f"🔥 BreakHis models available: LR={self.breakhis_classifier.lr_model is not None}, SVM={self.breakhis_classifier.svm_model is not None}, XGB={hasattr(self.breakhis_classifier, 'xgb_model') and self.breakhis_classifier.xgb_model is not None}")
            except Exception as e:
                print(f"❌ BreakHis classifier loading failed: {e}")
                self._breakhis_loaded = False
        return self._breakhis_loaded
    
    def load_normal_benign_classifier(self):
        """Load BACH normal vs benign classifier on demand"""
        if not self._normal_benign_loaded:
            try:
                print("🟦 Loading SPECIALIZED BACH Normal vs Benign classifier...")
                self.bach_normal_benign_classifier = BACHNormalBenignClassifier(cache_file='/workspace/embeddings_cache_L2_REPROCESSED.pkl')
                
                # Try retrained model first
                retrained_path = '/workspace/bach_normal_benign_model_L2_RETRAINED.pkl'
                fallback_path = '/workspace/bach_normal_benign_model.pkl'
                
                if os.path.exists(retrained_path):
                    success = self.bach_normal_benign_classifier.load_model(retrained_path)
                    print(f"🟦 Using L2 RETRAINED Normal vs Benign model: {success}")
                else:
                    success = self.bach_normal_benign_classifier.load_model(fallback_path)
                    print(f"🟦 Using OLD Normal vs Benign model: {success}")
                
                self._normal_benign_loaded = success
                if success:
                    print(f"🟦 Normal vs Benign model classes: {self.bach_normal_benign_classifier.class_names}")
            except Exception as e:
                print(f"❌ SPECIALIZED Normal vs Benign classifier loading failed: {e}")
                import traceback
                traceback.print_exc()
                self._normal_benign_loaded = False
        return self._normal_benign_loaded
    
    def load_invasive_insitu_classifier(self):
        """Load BACH invasive vs insitu classifier on demand"""
        if not self._invasive_insitu_loaded:
            try:
                print("🟪 Loading SPECIALIZED BACH Invasive vs InSitu classifier...")
                self.bach_invasive_insitu_classifier = BACHInvasiveInsituClassifier(cache_file='/workspace/embeddings_cache_L2_REPROCESSED.pkl')
                
                # Try retrained model first
                retrained_path = '/workspace/bach_invasive_insitu_model_L2_RETRAINED.pkl'
                fallback_path = '/workspace/bach_invasive_insitu_model.pkl'
                
                if os.path.exists(retrained_path):
                    success = self.bach_invasive_insitu_classifier.load_model(retrained_path)
                    print(f"🟪 Using L2 RETRAINED Invasive vs InSitu model: {success}")
                else:
                    success = self.bach_invasive_insitu_classifier.load_model(fallback_path)
                    print(f"🟪 Using OLD Invasive vs InSitu model: {success}")
                
                self._invasive_insitu_loaded = success
                if success:
                    print(f"🟪 Invasive vs InSitu model classes: {self.bach_invasive_insitu_classifier.class_names}")
            except Exception as e:
                print(f"❌ SPECIALIZED Invasive vs InSitu classifier loading failed: {e}")
                import traceback
                traceback.print_exc()
                self._invasive_insitu_loaded = False
        return self._invasive_insitu_loaded
    
    # BACH Prediction Methods
    def predict_bach_lr(self, features):
        """Safely predict with BACH Logistic Regression"""
        try:
            if not self.load_bach_classifier() or not self.bach_classifier.model:
                return None
            result = self.bach_classifier.predict(features)
            print(f"🔥 BACH LR: {result['predicted_class']} ({result['confidence']:.3f})")
            return result
        except Exception as e:
            print(f"❌ BACH LR prediction failed: {e}")
            return None
    
    def predict_bach_svm(self, features):
        """Safely predict with BACH SVM"""
        try:
            if not self.load_bach_classifier() or not self.bach_classifier.svm_model:
                return None
            result = self.bach_classifier.predict_svm(features)
            print(f"🔥 BACH SVM: {result['predicted_class']} ({result['confidence']:.3f})")
            return result
        except Exception as e:
            print(f"❌ BACH SVM prediction failed: {e}")
            return None
    
    def predict_bach_xgb(self, features):
        """Safely predict with BACH XGBoost"""
        try:
            if not self.load_bach_classifier() or not hasattr(self.bach_classifier, 'xgb_model') or not self.bach_classifier.xgb_model:
                return None
            result = self.bach_classifier.predict_xgb(features)
            print(f"🔥 BACH XGBoost: {result['predicted_class']} ({result['confidence']:.3f})")
            return result
        except Exception as e:
            print(f"❌ BACH XGBoost prediction failed: {e}")
            return None
    
    # BreakHis Prediction Methods
    def predict_breakhis_lr(self, features):
        """Safely predict with BreakHis Logistic Regression"""
        try:
            if not self.load_breakhis_classifier() or not self.breakhis_classifier.lr_model:
                return None
            result = self.breakhis_classifier.predict_lr(features)
            print(f"🔥 BreakHis LR: {result['predicted_class']} ({result['confidence']:.3f})")
            return result
        except Exception as e:
            print(f"❌ BreakHis LR prediction failed: {e}")
            return None
    
    def predict_breakhis_svm(self, features):
        """Safely predict with BreakHis SVM"""
        try:
            if not self.load_breakhis_classifier() or not self.breakhis_classifier.svm_model:
                return None
            result = self.breakhis_classifier.predict_svm(features)
            print(f"🔥 BreakHis SVM: {result['predicted_class']} ({result['confidence']:.3f})")
            return result
        except Exception as e:
            print(f"❌ BreakHis SVM prediction failed: {e}")
            return None
    
    def predict_breakhis_xgb(self, features):
        """Safely predict with BreakHis XGBoost"""
        try:
            if not self.load_breakhis_classifier() or not hasattr(self.breakhis_classifier, 'xgb_model') or not self.breakhis_classifier.xgb_model:
                return None
            result = self.breakhis_classifier.predict_xgb(features)
            print(f"🔥 BreakHis XGBoost: {result['predicted_class']} ({result['confidence']:.3f})")
            return result
        except Exception as e:
            print(f"❌ BreakHis XGBoost prediction failed: {e}")
            return None
    
    # Utility Methods
    def generate_roc_plot(self, classifier_type='bach'):
        """Generate ROC plot safely"""
        try:
            classifier = self.bach_classifier if classifier_type == 'bach' else self.breakhis_classifier
            if classifier and hasattr(classifier, 'plot_roc_curves'):
                return classifier.plot_roc_curves(return_base64=True)
        except Exception as e:
            print(f"⚠️ ROC plot failed for {classifier_type}: {e}")
        return None
    
    def get_model_info(self, classifier_type='bach'):
        """Get model performance info safely"""
        try:
            classifier = self.bach_classifier if classifier_type == 'bach' else self.breakhis_classifier
            if not classifier:
                return {"status": f"{classifier_type} not loaded"}
                
            info = {
                "classifier": classifier_type.upper(),
                "classes": getattr(classifier, 'class_names', []),
                "evaluation_type": "HELD_OUT_TEST_SET"
            }
            
            # Add test accuracies
            if hasattr(classifier, 'test_scores') and classifier.test_scores:
                info["test_accuracy_lr"] = float(classifier.test_scores['accuracy'])
            if hasattr(classifier, 'svm_test_scores') and classifier.svm_test_scores:
                info["test_accuracy_svm"] = float(classifier.svm_test_scores['accuracy'])
            if hasattr(classifier, 'xgb_test_scores') and classifier.xgb_test_scores:
                info["test_accuracy_xgb"] = float(classifier.xgb_test_scores['accuracy'])
                
            # Add ROC AUCs
            if hasattr(classifier, 'test_roc_data') and classifier.test_roc_data:
                info["test_roc_auc_lr"] = float(classifier.test_roc_data['roc_auc'].get('micro', 0))
            if hasattr(classifier, 'svm_test_roc_data') and classifier.svm_test_roc_data:
                info["test_roc_auc_svm"] = float(classifier.svm_test_roc_data['roc_auc'].get('micro', 0))
            if hasattr(classifier, 'xgb_test_roc_data') and classifier.xgb_test_roc_data:
                info["test_roc_auc_xgb"] = float(classifier.xgb_test_roc_data['roc_auc'].get('micro', 0))
                
            if hasattr(classifier, 'data_splits') and classifier.data_splits:
                info["data_splits"] = classifier.data_splits
                
            return info
        except Exception as e:
            print(f"⚠️ Model info failed for {classifier_type}: {e}")
            return {"status": f"{classifier_type} info failed", "error": str(e)}
    
    # Tiered Prediction System
    def predict_tiered(self, features):
        """
        Tiered prediction system:
        1. BreakHis binary (malignant vs benign)
        2a. If benign → BACH normal vs benign
        2b. If malignant → BACH invasive vs insitu
        """
        print("🏥 Starting tiered prediction system...")
        
        # Stage 1: BreakHis binary classification
        breakhis_lr = self.predict_breakhis_lr(features)
        breakhis_svm = self.predict_breakhis_svm(features)
        breakhis_xgb = self.predict_breakhis_xgb(features)
        
        # Determine BreakHis consensus
        breakhis_predictions = [r['predicted_class'] for r in [breakhis_lr, breakhis_svm, breakhis_xgb] if r]
        malignant_votes = sum(1 for pred in breakhis_predictions if pred == 'malignant')
        breakhis_consensus = 'malignant' if malignant_votes >= 2 else 'benign'
        
        print(f"🏥 Stage 1 - BreakHis consensus: {breakhis_consensus} ({malignant_votes}/{len(breakhis_predictions)} malignant votes)")
        
        # Stage 2: Deploy appropriate BACH classifier
        bach_tiered_results = None
        if breakhis_consensus == 'benign':
            print("🏥 Stage 2a - Deploying BACH Normal vs Benign classifier...")
            bach_tiered_results = self._predict_normal_vs_benign(features)
        else:
            print("🏥 Stage 2b - Deploying BACH Invasive vs InSitu classifier...")
            bach_tiered_results = self._predict_invasive_vs_insitu(features)
        
        return {
            'stage_1_breakhis': {
                'consensus': breakhis_consensus,
                'vote_breakdown': {'malignant': malignant_votes, 'benign': len(breakhis_predictions) - malignant_votes},
                'classifiers': {
                    'logistic_regression': breakhis_lr,
                    'svm_rbf': breakhis_svm,
                    'xgboost': breakhis_xgb
                }
            },
            'stage_2_bach_specialized': bach_tiered_results,
            'tiered_final_prediction': bach_tiered_results['consensus'] if bach_tiered_results else breakhis_consensus,
            'clinical_pathway': f"BreakHis → {'Normal/Benign' if breakhis_consensus == 'benign' else 'Invasive/InSitu'}"
        }
    
    def _predict_normal_vs_benign(self, features):
        """Predict using normal vs benign specialized classifiers"""
        if not self.load_normal_benign_classifier():
            return None
            
        try:
            lr_result = self.bach_normal_benign_classifier.predict_lr(features) if self.bach_normal_benign_classifier.lr_model else None
            svm_result = self.bach_normal_benign_classifier.predict_svm(features) if self.bach_normal_benign_classifier.svm_model else None
            xgb_result = self.bach_normal_benign_classifier.predict_xgb(features) if hasattr(self.bach_normal_benign_classifier, 'xgb_model') and self.bach_normal_benign_classifier.xgb_model else None
            
            # Consensus among available classifiers
            predictions = [r['predicted_class'] for r in [lr_result, svm_result, xgb_result] if r]
            normal_votes = sum(1 for pred in predictions if pred == 'normal')
            consensus = 'normal' if normal_votes >= len(predictions) / 2 else 'benign'
            
            return {
                'task': 'Normal vs Benign',
                'consensus': consensus,
                'classifiers': {
                    'logistic_regression': lr_result,
                    'svm_rbf': svm_result,
                    'xgboost': xgb_result
                },
                'vote_breakdown': {'normal': normal_votes, 'benign': len(predictions) - normal_votes}
            }
        except Exception as e:
            print(f"❌ Normal vs Benign prediction failed: {e}")
            return None
    
    def _predict_invasive_vs_insitu(self, features):
        """Predict using invasive vs insitu specialized classifiers"""
        if not self.load_invasive_insitu_classifier():
            return None
            
        try:
            lr_result = self.bach_invasive_insitu_classifier.predict_lr(features) if self.bach_invasive_insitu_classifier.lr_model else None
            svm_result = self.bach_invasive_insitu_classifier.predict_svm(features) if self.bach_invasive_insitu_classifier.svm_model else None
            xgb_result = self.bach_invasive_insitu_classifier.predict_xgb(features) if hasattr(self.bach_invasive_insitu_classifier, 'xgb_model') and self.bach_invasive_insitu_classifier.xgb_model else None
            
            # Consensus among available classifiers
            predictions = [r['predicted_class'] for r in [lr_result, svm_result, xgb_result] if r]
            invasive_votes = sum(1 for pred in predictions if pred == 'invasive')
            consensus = 'invasive' if invasive_votes >= len(predictions) / 2 else 'insitu'
            
            return {
                'task': 'Invasive vs InSitu',
                'consensus': consensus,
                'classifiers': {
                    'logistic_regression': lr_result,
                    'svm_rbf': svm_result,
                    'xgboost': xgb_result
                },
                'vote_breakdown': {'invasive': invasive_votes, 'insitu': len(predictions) - invasive_votes}
            }
        except Exception as e:
            print(f"❌ Invasive vs InSitu prediction failed: {e}")
            return None

# Global classifier manager instance
classifier_manager = ClassifierManager()