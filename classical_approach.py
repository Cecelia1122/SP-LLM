"""
2_classical_approach.py - Classical signal processing approach
Purpose: MFCC feature extraction + traditional ML classifiers (SVM, Random Forest, Neural Network)
"""

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
from tqdm import tqdm

from utils import (load_audio_file, extract_classical_features, plot_confusion_matrix, 
                   plot_roc_curves, PerformanceTracker, KEYWORD)
from data_preparation import load_data_splits

class ClassicalKeywordSpotter:
    def __init__(self, keyword: str = KEYWORD):
        self.keyword = keyword
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_dim = None
        self.performance_tracker = PerformanceTracker()
        
    def extract_features_from_files(self, file_paths, labels):
        """Extract classical features from audio files"""
        print("🎵 Extracting classical features (MFCC + spectral)...")
        
        features = []
        valid_labels = []
        
        for i, (file_path, label) in enumerate(tqdm(zip(file_paths, labels), 
                                                  total=len(file_paths), 
                                                  desc="Processing files")):
            # Load audio
            audio, sr = load_audio_file(file_path)
            
            if audio is not None:
                # Extract features
                feature_vector = extract_classical_features(audio, sr)
                
                if feature_vector is not None:
                    features.append(feature_vector)
                    valid_labels.append(label)
        
        features = np.array(features)
        valid_labels = np.array(valid_labels)
        
        print(f"✅ Extracted features from {len(features)} files")
        print(f"📊 Feature dimension: {features.shape[1]}")
        print(f"📊 Positive samples: {sum(valid_labels)}")
        print(f"📊 Negative samples: {len(valid_labels) - sum(valid_labels)}")
        
        self.feature_dim = features.shape[1]
        
        return features, valid_labels
    
    def train_svm_classifier(self, X_train, y_train, X_val, y_val):
        """Train SVM classifier with hyperparameter tuning"""
        print("🤖 Training SVM classifier...")
        
        # Hyperparameter tuning
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'poly']
        }
        
        svm = SVC(probability=True, random_state=42, class_weight='balanced')
        grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
        
        # Fit on training data
        grid_search.fit(X_train, y_train)
        
        best_svm = grid_search.best_estimator_
        print(f"✅ Best SVM parameters: {grid_search.best_params_}")
        
        # Evaluate on validation set
        val_accuracy = best_svm.score(X_val, y_val)
        val_predictions = best_svm.predict(X_val)
        val_probabilities = best_svm.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_probabilities)
        
        # Calculate false positive rate
        cm = confusion_matrix(y_val, val_predictions)
        fpr = cm[0,1] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
        
        print(f"📊 SVM Validation Accuracy: {val_accuracy:.3f}")
        print(f"📊 SVM Validation AUC: {val_auc:.3f}")
        print(f"📊 SVM False Positive Rate: {fpr:.3f}")
        
        return {
            'model': best_svm,
            'accuracy': val_accuracy,
            'auc': val_auc,
            'false_positive_rate': fpr,
            'val_predictions': val_predictions,
            'val_probabilities': val_probabilities,
            'val_labels': y_val
        }
    
    def train_random_forest_classifier(self, X_train, y_train, X_val, y_val):
        """Train Random Forest classifier"""
        print("🌳 Training Random Forest classifier...")
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
        
        # Fit on training data
        grid_search.fit(X_train, y_train)
        
        best_rf = grid_search.best_estimator_
        print(f"✅ Best RF parameters: {grid_search.best_params_}")
        
        # Evaluate on validation set
        val_accuracy = best_rf.score(X_val, y_val)
        val_predictions = best_rf.predict(X_val)
        val_probabilities = best_rf.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_probabilities)
        
        # Calculate false positive rate
        cm = confusion_matrix(y_val, val_predictions)
        fpr = cm[0,1] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
        
        print(f"📊 RF Validation Accuracy: {val_accuracy:.3f}")
        print(f"📊 RF Validation AUC: {val_auc:.3f}")
        print(f"📊 RF False Positive Rate: {fpr:.3f}")
        
        return {
            'model': best_rf,
            'accuracy': val_accuracy,
            'auc': val_auc,
            'false_positive_rate': fpr,
            'val_predictions': val_predictions,
            'val_probabilities': val_probabilities,
            'val_labels': y_val,
            'feature_importance': best_rf.feature_importances_
        }
    
    def train_neural_network_classifier(self, X_train, y_train, X_val, y_val):
        """Train Neural Network classifier"""
        print("🧠 Training Neural Network classifier...")
        
        # Hyperparameter tuning
        param_grid = {
            'hidden_layer_sizes': [(64, 32), (128, 64, 32), (256, 128, 64)],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'alpha': [0.0001, 0.001, 0.01]
        }
        
        nn = MLPClassifier(
            max_iter=2000, 
            random_state=42, 
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
        
        grid_search = GridSearchCV(nn, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
        
        # Fit on training data
        grid_search.fit(X_train, y_train)
        
        best_nn = grid_search.best_estimator_
        print(f"✅ Best NN parameters: {grid_search.best_params_}")
        
        # Evaluate on validation set
        val_accuracy = best_nn.score(X_val, y_val)
        val_predictions = best_nn.predict(X_val)
        val_probabilities = best_nn.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_probabilities)
        
        # Calculate false positive rate
        cm = confusion_matrix(y_val, val_predictions)
        fpr = cm[0,1] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
        
        print(f"📊 NN Validation Accuracy: {val_accuracy:.3f}")
        print(f"📊 NN Validation AUC: {val_auc:.3f}")
        print(f"📊 NN False Positive Rate: {fpr:.3f}")
        
        return {
            'model': best_nn,
            'accuracy': val_accuracy,
            'auc': val_auc,
            'false_positive_rate': fpr,
            'val_predictions': val_predictions,
            'val_probabilities': val_probabilities,
            'val_labels': y_val
        }
    
    def train_all_models(self, data_splits):
        """Train all classical models"""
        print("🚀 Starting classical approach training...\n")
        
        # Load and prepare data
        train_data = data_splits['train']
        val_data = data_splits['validation']
        
        # Extract features for training set
        X_train, y_train = self.extract_features_from_files(
            train_data['files'], train_data['labels']
        )
        
        # Extract features for validation set
        X_val, y_val = self.extract_features_from_files(
            val_data['files'], val_data['labels']
        )
        
        # Normalize features
        print("📏 Normalizing features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train individual models
        print("\n" + "="*50)
        self.models['SVM'] = self.train_svm_classifier(X_train_scaled, y_train, X_val_scaled, y_val)
        
        print("\n" + "="*50)
        self.models['Random Forest'] = self.train_random_forest_classifier(X_train_scaled, y_train, X_val_scaled, y_val)
        
        print("\n" + "="*50)
        self.models['Neural Network'] = self.train_neural_network_classifier(X_train_scaled, y_train, X_val_scaled, y_val)
        
        # Select best model
        self.select_best_model()
        
        # Track performance
        for model_name, metrics in self.models.items():
            self.performance_tracker.add_experiment(
                f"classical_{model_name.lower().replace(' ', '_')}", 
                metrics
            )
        
        return self.models
    
    def select_best_model(self):
        """Select the best performing model"""
        best_score = 0
        best_name = None
        
        print("\n📊 MODEL COMPARISON:")
        print("-" * 60)
        
        for name, metrics in self.models.items():
            # Composite score: AUC with penalty for high false positive rate
            score = metrics['auc'] * (1 - metrics['false_positive_rate'])
            
            print(f"{name}:")
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  AUC: {metrics['auc']:.3f}")
            print(f"  False Positive Rate: {metrics['false_positive_rate']:.3f}")
            print(f"  Composite Score: {score:.3f}")
            print()
            
            if score > best_score:
                best_score = score
                best_name = name
        
        self.best_model = best_name
        print(f"🏆 Best classical model: {self.best_model} (Score: {best_score:.3f})")
    
    def evaluate_on_test_set(self, data_splits):
        """Evaluate the best model on test set"""
        if not self.best_model:
            print("❌ No best model selected!")
            return None
        
        print(f"\n🧪 Evaluating {self.best_model} on test set...")
        
        # Load test data
        test_data = data_splits['test']
        X_test, y_test = self.extract_features_from_files(
            test_data['files'], test_data['labels']
        )
        X_test_scaled = self.scaler.transform(X_test)
        
        # Get best model
        best_model_data = self.models[self.best_model]
        model = best_model_data['model']
        
        # Evaluate
        test_accuracy = model.score(X_test_scaled, y_test)
        test_predictions = model.predict(X_test_scaled)
        test_probabilities = model.predict_proba(X_test_scaled)[:, 1]
        test_auc = roc_auc_score(y_test, test_probabilities)
        
        # Calculate detailed metrics
        cm = confusion_matrix(y_test, test_predictions)
        fpr = cm[0,1] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
        
        print(f"📊 Test Accuracy: {test_accuracy:.3f}")
        print(f"📊 Test AUC: {test_auc:.3f}")
        print(f"📊 Test False Positive Rate: {fpr:.3f}")
        
        print("\nDetailed Test Results:")
        print(classification_report(y_test, test_predictions, 
                                  target_names=['Not Keyword', 'Keyword']))
        
        # Store test results
        self.models[self.best_model].update({
            'test_accuracy': test_accuracy,
            'test_auc': test_auc,
            'test_fpr': fpr,
            'test_predictions': test_predictions,
            'test_probabilities': test_probabilities,
            'test_labels': y_test
        })
        
        return {
            'accuracy': test_accuracy,
            'auc': test_auc,
            'false_positive_rate': fpr,
            'predictions': test_predictions,
            'probabilities': test_probabilities,
            'labels': y_test
        }
    
    def create_visualizations(self):
        """Create performance visualizations"""
        print("📈 Creating visualizations...")
        
        os.makedirs("results", exist_ok=True)
        
        # 1. Model comparison bar plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        model_names = list(self.models.keys())
        accuracies = [self.models[name]['accuracy'] for name in model_names]
        aucs = [self.models[name]['auc'] for name in model_names]
        fprs = [self.models[name]['false_positive_rate'] for name in model_names]
        
        # Accuracy comparison
        bars1 = ax1.bar(model_names, accuracies, alpha=0.8, color='skyblue')
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # AUC comparison
        bars2 = ax2.bar(model_names, aucs, alpha=0.8, color='lightgreen')
        ax2.set_title('Model AUC Comparison')
        ax2.set_ylabel('AUC')
        ax2.set_ylim(0, 1)
        for bar, auc in zip(bars2, aucs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{auc:.3f}', ha='center', va='bottom')
        
        # False Positive Rate comparison
        bars3 = ax3.bar(model_names, fprs, alpha=0.8, color='salmon')
        ax3.set_title('False Positive Rate Comparison')
        ax3.set_ylabel('False Positive Rate')
        ax3.set_ylim(0, max(fprs) * 1.2)
        for bar, fpr in zip(bars3, fprs):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{fpr:.3f}', ha='center', va='bottom')
        
        # ROC curves
        for model_name in model_names:
            data = self.models[model_name]
            if 'val_labels' in data and 'val_probabilities' in data:
                fpr, tpr, _ = roc_curve(data['val_labels'], data['val_probabilities'])
                ax4.plot(fpr, tpr, label=f'{model_name} (AUC: {data["auc"]:.3f})')
        
        ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax4.set_xlabel('False Positive Rate')
        ax4.set_ylabel('True Positive Rate')
        ax4.set_title('ROC Curves - Classical Models')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/classical_models_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Confusion matrix for best model
        if self.best_model and 'test_labels' in self.models[self.best_model]:
            best_data = self.models[self.best_model]
            fig = plot_confusion_matrix(
                best_data['test_labels'], 
                best_data['test_predictions'],
                f'Confusion Matrix - {self.best_model} (Test Set)'
            )
            fig.savefig('results/classical_best_model_confusion_matrix.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
        
        # 3. Feature importance (if Random Forest is available)
        if 'Random Forest' in self.models and 'feature_importance' in self.models['Random Forest']:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            importance = self.models['Random Forest']['feature_importance']
            feature_names = [f'Feature_{i+1}' for i in range(len(importance))]
            
            # Sort by importance
            sorted_idx = np.argsort(importance)[::-1]
            top_features = sorted_idx[:20]  # Top 20 features
            
            ax.bar(range(len(top_features)), importance[top_features])
            ax.set_title('Top 20 Feature Importances (Random Forest)')
            ax.set_xlabel('Features')
            ax.set_ylabel('Importance')
            ax.set_xticks(range(len(top_features)))
            ax.set_xticklabels([feature_names[i] for i in top_features], rotation=45)
            
            plt.tight_layout()
            plt.savefig('results/classical_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def save_model(self, filename: str = "models/classical_keyword_spotter.pkl"):
        """Save the trained classical model"""
        if not self.best_model:
            print("❌ No model to save!")
            return
        
        os.makedirs("models", exist_ok=True)
        
        model_data = {
            'model': self.models[self.best_model]['model'],
            'scaler': self.scaler,
            'keyword': self.keyword,
            'model_type': self.best_model,
            'feature_dim': self.feature_dim,
            'approach': 'classical',
            'performance_metrics': {
                name: {k: v for k, v in data.items() 
                      if k in ['accuracy', 'auc', 'false_positive_rate']}
                for name, data in self.models.items()
            }
        }
        
        joblib.dump(model_data, filename)
        print(f"💾 Classical model saved: {filename}")
        
        return filename
    
    def load_model(self, filename: str = "models/classical_keyword_spotter.pkl"):
        """Load a trained classical model"""
        try:
            model_data = joblib.load(filename)
            
            # Restore model components
            best_model_name = model_data['model_type']
            self.models[best_model_name] = {
                'model': model_data['model'],
                'accuracy': model_data['performance_metrics'][best_model_name]['accuracy'],
                'auc': model_data['performance_metrics'][best_model_name]['auc'],
                'false_positive_rate': model_data['performance_metrics'][best_model_name]['false_positive_rate']
            }
            
            self.scaler = model_data['scaler']
            self.keyword = model_data['keyword']
            self.best_model = best_model_name
            self.feature_dim = model_data['feature_dim']
            
            print(f"✅ Classical model loaded: {filename}")
            print(f"🎯 Model type: {self.best_model}")
            print(f"🎵 Keyword: {self.keyword}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def predict_audio_file(self, file_path: str, confidence_threshold: float = 0.7):
        """Predict keyword from audio file"""
        if not self.best_model:
            print("❌ No trained model available!")
            return None
        
        # Load and extract features
        audio, sr = load_audio_file(file_path)
        if audio is None:
            return None
        
        features = extract_classical_features(audio, sr)
        if features is None:
            return None
        
        # Scale and predict
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        model = self.models[self.best_model]['model']
        
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        
        return {
            'prediction': prediction,
            'probability': probability,
            'detected': probability > confidence_threshold,
            'confidence_level': 'High' if probability > 0.8 else 'Medium' if probability > 0.6 else 'Low',
            'model_type': f'Classical_{self.best_model}',
            'approach': 'classical'
        }

def main():
    """Main training function for classical approach"""
    print("=== CLASSICAL KEYWORD SPOTTER TRAINING ===\n")
    
    try:
        # Load data splits
        print("📂 Loading data splits...")
        data_splits = load_data_splits()
        
        # Initialize classical spotter
        spotter = ClassicalKeywordSpotter(keyword="yes")
        
        # Train all models
        models = spotter.train_all_models(data_splits)
        
        # Evaluate on test set
        test_results = spotter.evaluate_on_test_set(data_splits)
        
        # Create visualizations
        spotter.create_visualizations()
        
        # Save model
        model_path = spotter.save_model()
        
        # Save performance tracking
        spotter.performance_tracker.save_to_json("results/classical_performance_tracking.json")
        
        print("\n🎉 Classical approach training completed!")
        print(f"🏆 Best model: {spotter.best_model}")
        print(f"📊 Test accuracy: {test_results['accuracy']:.3f}")
        print(f"📊 Test AUC: {test_results['auc']:.3f}")
        print(f"💾 Model saved: {model_path}")
        
        return spotter, test_results
        
    except Exception as e:
        print(f"❌ Classical training failed: {e}")
        raise

if __name__ == "__main__":
    main()