"""
3_speech_llm_approach.py - Modern Speech LLM approach (Phase 8)
Purpose: Fine-tune Wav2Vec2 model for keyword detection
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
from transformers import (
    Wav2Vec2ForSequenceClassification, 
    Wav2Vec2Processor, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
import time

from utils import load_audio_file, PerformanceTracker, KEYWORD, SAMPLE_RATE
from data_preparation import load_data_splits

class SpeechCommandsDataset(Dataset):
    """Dataset class for Speech Commands"""
    
    def __init__(self, file_paths, labels, processor, max_length=16000):
        self.file_paths = file_paths
        self.labels = labels
        self.processor = processor
        self.max_length = max_length
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load audio
        audio, sr = load_audio_file(self.file_paths[idx], target_sr=SAMPLE_RATE)
        
        if audio is None:
            # Return zero audio if loading fails
            audio = np.zeros(self.max_length)
        
        # Ensure consistent length
        if len(audio) > self.max_length:
            audio = audio[:self.max_length]
        else:
            audio = np.pad(audio, (0, self.max_length - len(audio)), mode='constant')
        
        # Process audio
        inputs = self.processor(
            audio, 
            sampling_rate=SAMPLE_RATE, 
            return_tensors="pt", 
            padding=True,
            max_length=self.max_length,
            truncation=True
        )
        
        return {
            'input_values': inputs.input_values.flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class SpeechLLMKeywordSpotter:
    def __init__(self, keyword: str = KEYWORD, model_name: str = "facebook/wav2vec2-base"):
        self.keyword = keyword
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🔥 Using device: {self.device}")
        print(f"🎯 Target keyword: {self.keyword}")
        print(f"🤖 Base model: {self.model_name}")
        
        # Initialize processor and model
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model = None
        self.training_history = {}
        self.performance_tracker = PerformanceTracker()
        
    def initialize_model(self, num_labels: int = 2):
        """Initialize the model for classification"""
        print(f"🔧 Initializing model with {num_labels} classes...")
        
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        
        # Move to device
        self.model.to(self.device)
        
        # Freeze some layers for more stable training
        for param in self.model.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False
        
        print(f"✅ Model initialized and moved to {self.device}")
        
        return self.model
    
    def create_data_loaders(self, data_splits, batch_size: int = 16):
        """Create data loaders for training"""
        print("📦 Creating data loaders...")
        
        # Create datasets
        train_dataset = SpeechCommandsDataset(
            data_splits['train']['files'],
            data_splits['train']['labels'],
            self.processor
        )
        
        val_dataset = SpeechCommandsDataset(
            data_splits['validation']['files'],
            data_splits['validation']['labels'],
            self.processor
        )
        
        test_dataset = SpeechCommandsDataset(
            data_splits['test']['files'],
            data_splits['test']['labels'],
            self.processor
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"📊 Training samples: {len(train_dataset)}")
        print(f"📊 Validation samples: {len(val_dataset)}")
        print(f"📊 Test samples: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        acc = accuracy_score(labels, predictions)
        
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train_model(self, data_splits, num_epochs: int = 10, learning_rate: float = 3e-5, 
                   batch_size: int = 16, output_dir: str = "models/speech_llm_model"):
        """Train the Wav2Vec2 model"""
        print("🚀 Starting Speech LLM training...\n")
        
        start_time = time.time()
        
        # Initialize model
        self.initialize_model(num_labels=2)
        
        # Create data loaders
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = self.create_data_loaders(
            data_splits, batch_size
        )
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            save_total_limit=3,
            learning_rate=learning_rate,
            lr_scheduler_type="linear",
            report_to=None  # Disable wandb logging
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        print("🏃‍♂️ Starting training...")
        
        # Train the model
        train_result = trainer.train()
        
        training_time = time.time() - start_time
        
        print(f"✅ Training completed in {training_time/60:.1f} minutes")
        
        # Save training history
        self.training_history = {
            'train_loss': train_result.training_loss,
            'training_time': training_time,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size
        }
        
        # Evaluate on validation set
        print("📊 Evaluating on validation set...")
        val_results = trainer.evaluate()
        
        print("Validation Results:")
        for key, value in val_results.items():
            print(f"  {key}: {value:.4f}")
        
        # Evaluate on test set
        print("📊 Evaluating on test set...")
        test_results = trainer.predict(test_dataset)
        
        # Calculate test metrics
        test_predictions = np.argmax(test_results.predictions, axis=1)
        test_labels = test_results.label_ids
        test_probabilities = torch.softmax(torch.tensor(test_results.predictions), dim=1)[:, 1].numpy()
        
        test_accuracy = accuracy_score(test_labels, test_predictions)
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
            test_labels, test_predictions, average='binary'
        )
        test_auc = roc_auc_score(test_labels, test_probabilities)
        
        print("Test Results:")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print(f"  Test Precision: {test_precision:.4f}")
        print(f"  Test Recall: {test_recall:.4f}")
        print(f"  Test F1: {test_f1:.4f}")
        print(f"  Test AUC: {test_auc:.4f}")
        
        # Store results
        self.test_results = {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1,
            'auc': test_auc,
            'predictions': test_predictions,
            'probabilities': test_probabilities,
            'labels': test_labels,
            'training_time': training_time
        }
        
        # Track performance
        self.performance_tracker.add_experiment(
            "speech_llm_wav2vec2",
            {
                'test_accuracy': test_accuracy,
                'test_auc': test_auc,
                'test_f1': test_f1,
                'training_time_minutes': training_time/60
            }
        )
        
        # Save model
        print("💾 Saving trained model...")
        trainer.save_model()
        self.processor.save_pretrained(output_dir)
        
        return {
            'trainer': trainer,
            'validation_results': val_results,
            'test_results': self.test_results,
            'training_history': self.training_history
        }
    
    def load_model(self, model_dir: str = "models/speech_llm_model"):
        """Load a trained model"""
        try:
            print(f"📂 Loading Speech LLM model from {model_dir}...")
            
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_dir)
            self.processor = Wav2Vec2Processor.from_pretrained(model_dir)
            self.model.to(self.device)
            self.model.eval()
            
            print("✅ Speech LLM model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load Speech LLM model: {e}")
            return False
    
    def predict_audio_file(self, file_path: str, confidence_threshold: float = 0.7):
        """Predict keyword from audio file using Speech LLM"""
        if self.model is None:
            print("❌ No trained model available!")
            return None
        
        # Load audio
        audio, sr = load_audio_file(file_path, target_sr=SAMPLE_RATE)
        if audio is None:
            return None
        
        # Process audio
        inputs = self.processor(
            audio, 
            sampling_rate=SAMPLE_RATE, 
            return_tensors="pt", 
            padding=True,
            max_length=16000,
            truncation=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)
            
        probability = predictions[0][1].cpu().item()  # Probability of positive class
        prediction = 1 if probability > confidence_threshold else 0
        
        return {
            'prediction': prediction,
            'probability': probability,
            'detected': prediction == 1,
            'confidence_level': 'High' if probability > 0.8 else 'Medium' if probability > 0.6 else 'Low',
            'model_type': 'Speech_LLM_Wav2Vec2',
            'approach': 'speech_llm'
        }
    
    def create_training_visualizations(self):
        """Create visualizations for training results"""
        print("📈 Creating Speech LLM visualizations...")
        
        os.makedirs("results", exist_ok=True)
        
        if not hasattr(self, 'test_results'):
            print("⚠️ No test results available for visualization")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Test metrics bar plot
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
        values = [
            self.test_results['accuracy'],
            self.test_results['precision'],
            self.test_results['recall'],
            self.test_results['f1'],
            self.test_results['auc']
        ]
        
        bars = ax1.bar(metrics, values, alpha=0.8, color='lightcoral')
        ax1.set_title('Speech LLM Test Performance')
        ax1.set_ylabel('Score')
        ax1.set_ylim(0, 1)
        
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 2. Confusion matrix
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(self.test_results['labels'], self.test_results['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_title('Confusion Matrix - Speech LLM')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        # 3. ROC Curve
        from sklearn.metrics import roc_curve
        
        fpr, tpr, _ = roc_curve(self.test_results['labels'], self.test_results['probabilities'])
        ax3.plot(fpr, tpr, label=f'Wav2Vec2 (AUC: {self.test_results["auc"]:.3f})', 
                linewidth=2, color='red')
        ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('ROC Curve - Speech LLM')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Training time comparison (placeholder - would be populated by comparison script)
        training_time_minutes = self.test_results['training_time'] / 60
        ax4.bar(['Speech LLM\n(Wav2Vec2)'], [training_time_minutes], alpha=0.8, color='orange')
        ax4.set_title('Training Time')
        ax4.set_ylabel('Time (minutes)')
        ax4.text(0, training_time_minutes + 0.5, f'{training_time_minutes:.1f} min', 
                ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('results/speech_llm_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Speech LLM visualizations saved")
    
    def benchmark_inference_speed(self, test_files, num_samples: int = 100):
        """Benchmark inference speed"""
        if self.model is None:
            print("❌ No model loaded for benchmarking!")
            return
        
        print(f"⏱️ Benchmarking inference speed on {num_samples} samples...")
        
        # Select random test files
        np.random.seed(42)
        selected_files = np.random.choice(test_files, min(num_samples, len(test_files)), replace=False)
        
        # Time inference
        start_time = time.time()
        predictions = []
        
        for file_path in tqdm(selected_files, desc="Inference"):
            result = self.predict_audio_file(file_path)
            if result:
                predictions.append(result)
        
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_sample = total_time / len(predictions)
        samples_per_second = len(predictions) / total_time
        
        print(f"📊 Inference Benchmark Results:")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Average time per sample: {avg_time_per_sample*1000:.1f} ms")
        print(f"  Samples per second: {samples_per_second:.1f}")
        
        return {
            'total_time': total_time,
            'avg_time_per_sample': avg_time_per_sample,
            'samples_per_second': samples_per_second,
            'num_samples': len(predictions)
        }
    
    def save_model_info(self, filename: str = "models/speech_llm_info.pkl"):
        """Save model information and results"""
        if not hasattr(self, 'test_results'):
            print("⚠️ No test results to save")
            return
        
        model_info = {
            'keyword': self.keyword,
            'model_name': self.model_name,
            'approach': 'speech_llm',
            'test_results': self.test_results,
            'training_history': self.training_history,
            'device': str(self.device)
        }
        
        joblib.dump(model_info, filename)
        print(f"💾 Speech LLM info saved: {filename}")
        
        return filename

def main():
    """Main training function for Speech LLM approach"""
    print("=== SPEECH LLM KEYWORD SPOTTER TRAINING ===\n")
    
    try:
        # Check for GPU availability
        if torch.cuda.is_available():
            print(f"🔥 GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print("💻 Using CPU (training will be slower)")
        
        # Load data splits
        print("📂 Loading data splits...")
        data_splits = load_data_splits()
        
        # Initialize Speech LLM spotter
        spotter = SpeechLLMKeywordSpotter(keyword="yes")
        
        # Train model
        results = spotter.train_model(
            data_splits,
            num_epochs=15,  # Adjust based on your computational resources
            learning_rate=3e-5,
            batch_size=8 if torch.cuda.is_available() else 4  # Smaller batch for CPU
        )
        
        # Create visualizations
        spotter.create_training_visualizations()
        
        # Benchmark inference speed
        benchmark_results = spotter.benchmark_inference_speed(
            data_splits['test']['files'],
            num_samples=50
        )
        
        # Save model info
        spotter.save_model_info()
        
        # Save performance tracking
        spotter.performance_tracker.save_to_json("results/speech_llm_performance_tracking.json")
        
        print("\n🎉 Speech LLM approach training completed!")
        print(f"📊 Test accuracy: {results['test_results']['accuracy']:.3f}")
        print(f"📊 Test AUC: {results['test_results']['auc']:.3f}")
        print(f"⏱️ Training time: {results['test_results']['training_time']/60:.1f} minutes")
        print(f"⚡ Inference speed: {benchmark_results['avg_time_per_sample']*1000:.1f} ms per sample")
        
        return spotter, results
        
    except Exception as e:
        print(f"❌ Speech LLM training failed: {e}")
        raise

if __name__ == "__main__":
    main()