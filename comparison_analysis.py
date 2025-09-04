"""
5_comparison_analysis.py - Compare classical vs Speech LLM approaches
Purpose: Comprehensive comparison and analysis of both approaches
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import time
from pathlib import Path

from utils import save_results_summary, plot_confusion_matrix, plot_roc_curves
from data_preparation import get_dataset_info, load_data_splits
from classical_approach import ClassicalKeywordSpotter
from speech_llm_approach import SpeechLLMKeywordSpotter

class ModelComparator:
    def __init__(self):
        self.classical_results = {}
        self.speech_llm_results = {}
        self.comparison_results = {}
        
    def load_classical_results(self, model_path: str = "models/classical_keyword_spotter.pkl"):
        """Load classical model results"""
        print("📊 Loading classical model results...")
        
        try:
            # Load saved performance tracking
            import json
            with open("results/classical_performance_tracking.json", 'r') as f:
                classical_tracking = json.load(f)
            
            # Load model
            classical_spotter = ClassicalKeywordSpotter()
            if classical_spotter.load_model(model_path):
                print("✅ Classical model loaded successfully")
            else:
                raise Exception("Failed to load classical model")
            
            self.classical_results = {
                'model': classical_spotter,
                'performance_tracking': classical_tracking,
                'approach': 'classical'
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load classical results: {e}")
            return False
    
    def load_speech_llm_results(self, 
                               model_path: str = "models/speech_llm_model",
                               info_path: str = "models/speech_llm_info.pkl"):
        """Load Speech LLM results"""
        print("📊 Loading Speech LLM results...")
        
        try:
            # Load model info
            speech_llm_info = joblib.load(info_path)
            
            # Load performance tracking
            import json
            with open("results/speech_llm_performance_tracking.json", 'r') as f:
                speech_llm_tracking = json.load(f)
            
            # Load model
            speech_llm_spotter = SpeechLLMKeywordSpotter()
            if speech_llm_spotter.load_model(model_path):
                print("✅ Speech LLM model loaded successfully")
            else:
                raise Exception("Failed to load Speech LLM model")
            
            self.speech_llm_results = {
                'model': speech_llm_spotter,
                'info': speech_llm_info,
                'performance_tracking': speech_llm_tracking,
                'approach': 'speech_llm'
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load Speech LLM results: {e}")
            return False
    
    def evaluate_models_on_test_set(self, data_splits):
        """Evaluate both models on the same test set"""
        print("🧪 Evaluating both models on test set...")
        
        test_files = data_splits['test']['files']
        test_labels = data_splits['test']['labels']
        
        print(f"📊 Test set size: {len(test_files)} samples")
        
        # Prepare results storage
        classical_predictions = []
        classical_probabilities = []
        speech_llm_predictions = []
        speech_llm_probabilities = []
        
        # Timing
        classical_times = []
        speech_llm_times = []
        
        print("🔍 Testing Classical approach...")
        for i, file_path in enumerate(test_files):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(test_files)}")
            
            # Classical prediction
            start_time = time.time()
            if self.classical_results:
                result = self.classical_results['model'].predict_audio_file(file_path, confidence_threshold=0.5)
                if result:
                    classical_predictions.append(result['prediction'])
                    classical_probabilities.append(result['probability'])
                else:
                    classical_predictions.append(0)
                    classical_probabilities.append(0.0)
            else:
                classical_predictions.append(0)
                classical_probabilities.append(0.0)
            classical_times.append(time.time() - start_time)
        
        print("🤖 Testing Speech LLM approach...")
        for i, file_path in enumerate(test_files):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(test_files)}")
            
            # Speech LLM prediction
            start_time = time.time()
            if self.speech_llm_results:
                result = self.speech_llm_results['model'].predict_audio_file(file_path, confidence_threshold=0.5)
                if result:
                    speech_llm_predictions.append(result['prediction'])
                    speech_llm_probabilities.append(result['probability'])
                else:
                    speech_llm_predictions.append(0)
                    speech_llm_probabilities.append(0.0)
            else:
                speech_llm_predictions.append(0)
                speech_llm_probabilities.append(0.0)
            speech_llm_times.append(time.time() - start_time)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        # Classical metrics
        classical_accuracy = accuracy_score(test_labels, classical_predictions)
        classical_precision = precision_score(test_labels, classical_predictions, zero_division=0)
        classical_recall = recall_score(test_labels, classical_predictions, zero_division=0)
        classical_f1 = f1_score(test_labels, classical_predictions, zero_division=0)
        classical_auc = roc_auc_score(test_labels, classical_probabilities)
        
        # Speech LLM metrics
        speech_llm_accuracy = accuracy_score(test_labels, speech_llm_predictions)
        speech_llm_precision = precision_score(test_labels, speech_llm_predictions, zero_division=0)
        speech_llm_recall = recall_score(test_labels, speech_llm_predictions, zero_division=0)
        speech_llm_f1 = f1_score(test_labels, speech_llm_predictions, zero_division=0)
        speech_llm_auc = roc_auc_score(test_labels, speech_llm_probabilities)
        
        # Store comparison results
        self.comparison_results = {
            'test_labels': test_labels,
            'classical': {
                'predictions': classical_predictions,
                'probabilities': classical_probabilities,
                'accuracy': classical_accuracy,
                'precision': classical_precision,
                'recall': classical_recall,
                'f1': classical_f1,
                'auc': classical_auc,
                'avg_inference_time': np.mean(classical_times)
            },
            'speech_llm': {
                'predictions': speech_llm_predictions,
                'probabilities': speech_llm_probabilities,
                'accuracy': speech_llm_accuracy,
                'precision': speech_llm_precision,
                'recall': speech_llm_recall,
                'f1': speech_llm_f1,
                'auc': speech_llm_auc,
                'avg_inference_time': np.mean(speech_llm_times)
            }
        }
        
        # Print results
        print("\n" + "="*60)
        print("📊 COMPARISON RESULTS:")
        print("="*60)
        
        print(f"\n🔍 Classical Approach:")
        print(f"  Accuracy: {classical_accuracy:.3f}")
        print(f"  Precision: {classical_precision:.3f}")
        print(f"  Recall: {classical_recall:.3f}")
        print(f"  F1: {classical_f1:.3f}")
        print(f"  AUC: {classical_auc:.3f}")
        print(f"  Avg Inference Time: {np.mean(classical_times)*1000:.1f} ms")
        
        print(f"\n🤖 Speech LLM Approach:")
        print(f"  Accuracy: {speech_llm_accuracy:.3f}")
        print(f"  Precision: {speech_llm_precision:.3f}")
        print(f"  Recall: {speech_llm_recall:.3f}")
        print(f"  F1: {speech_llm_f1:.3f}")
        print(f"  AUC: {speech_llm_auc:.3f}")
        print(f"  Avg Inference Time: {np.mean(speech_llm_times)*1000:.1f} ms")
        
        # Determine winner
        if speech_llm_accuracy > classical_accuracy:
            winner = "Speech LLM"
            performance_diff = speech_llm_accuracy - classical_accuracy
        elif classical_accuracy > speech_llm_accuracy:
            winner = "Classical"
            performance_diff = classical_accuracy - speech_llm_accuracy
        else:
            winner = "Tie"
            performance_diff = 0.0
        
        print(f"\n🏆 Winner: {winner}")
        if performance_diff > 0:
            print(f"   Performance advantage: {performance_diff:.3f} ({performance_diff*100:.1f}%)")
        
        return self.comparison_results
    
    def create_comprehensive_visualizations(self):
        """Create comprehensive comparison visualizations"""
        print("📈 Creating comprehensive visualizations...")
        
        if not self.comparison_results:
            print("❌ No comparison results available!")
            return
        
        os.makedirs("results", exist_ok=True)
        
        # Create main comparison figure
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Performance Metrics Comparison (Top Left)
        ax1 = plt.subplot(3, 3, 1)
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
        classical_values = [
            self.comparison_results['classical']['accuracy'],
            self.comparison_results['classical']['precision'],
            self.comparison_results['classical']['recall'],
            self.comparison_results['classical']['f1'],
            self.comparison_results['classical']['auc']
        ]
        speech_llm_values = [
            self.comparison_results['speech_llm']['accuracy'],
            self.comparison_results['speech_llm']['precision'],
            self.comparison_results['speech_llm']['recall'],
            self.comparison_results['speech_llm']['f1'],
            self.comparison_results['speech_llm']['auc']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, classical_values, width, label='Classical', alpha=0.8, color='skyblue')
        bars2 = ax1.bar(x + width/2, speech_llm_values, width, label='Speech LLM', alpha=0.8, color='lightcoral')
        
        ax1.set_ylabel('Score')
        ax1.set_title('Performance Metrics Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics, rotation=45)
        ax1.legend()
        ax1.set_ylim(0, 1.1)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. ROC Curves Comparison (Top Center)
        ax2 = plt.subplot(3, 3, 2)
        
        # Classical ROC
        fpr_classical, tpr_classical, _ = roc_curve(
            self.comparison_results['test_labels'], 
            self.comparison_results['classical']['probabilities']
        )
        
        # Speech LLM ROC
        fpr_speech_llm, tpr_speech_llm, _ = roc_curve(
            self.comparison_results['test_labels'], 
            self.comparison_results['speech_llm']['probabilities']
        )
        
        ax2.plot(fpr_classical, tpr_classical, 
                label=f'Classical (AUC: {self.comparison_results["classical"]["auc"]:.3f})', 
                linewidth=2, color='blue')
        ax2.plot(fpr_speech_llm, tpr_speech_llm, 
                label=f'Speech LLM (AUC: {self.comparison_results["speech_llm"]["auc"]:.3f})', 
                linewidth=2, color='red')
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curves Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Inference Time Comparison (Top Right)
        ax3 = plt.subplot(3, 3, 3)
        
        classical_time = self.comparison_results['classical']['avg_inference_time'] * 1000
        speech_llm_time = self.comparison_results['speech_llm']['avg_inference_time'] * 1000
        
        bars = ax3.bar(['Classical', 'Speech LLM'], [classical_time, speech_llm_time], 
                      alpha=0.8, color=['skyblue', 'lightcoral'])
        ax3.set_ylabel('Time (ms)')
        ax3.set_title('Average Inference Time')
        
        for bar, time_val in zip(bars, [classical_time, speech_llm_time]):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(classical_time, speech_llm_time)*0.02,
                    f'{time_val:.1f}ms', ha='center', va='bottom')
        
        # 4. Confusion Matrix - Classical (Middle Left)
        ax4 = plt.subplot(3, 3, 4)
        
        cm_classical = confusion_matrix(
            self.comparison_results['test_labels'], 
            self.comparison_results['classical']['predictions']
        )
        
        sns.heatmap(cm_classical, annot=True, fmt='d', cmap='Blues', ax=ax4)
        ax4.set_title('Confusion Matrix - Classical')
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('Actual')
        
        # 5. Confusion Matrix - Speech LLM (Middle Center)
        ax5 = plt.subplot(3, 3, 5)
        
        cm_speech_llm = confusion_matrix(
            self.comparison_results['test_labels'], 
            self.comparison_results['speech_llm']['predictions']
        )
        
        sns.heatmap(cm_speech_llm, annot=True, fmt='d', cmap='Reds', ax=ax5)
        ax5.set_title('Confusion Matrix - Speech LLM')
        ax5.set_xlabel('Predicted')
        ax5.set_ylabel('Actual')
        
        # 6. Probability Distribution Comparison (Middle Right)
        ax6 = plt.subplot(3, 3, 6)
        
        ax6.hist(self.comparison_results['classical']['probabilities'], bins=30, alpha=0.7, 
                label='Classical', color='blue', density=True)
        ax6.hist(self.comparison_results['speech_llm']['probabilities'], bins=30, alpha=0.7, 
                label='Speech LLM', color='red', density=True)
        ax6.set_xlabel('Prediction Probability')
        ax6.set_ylabel('Density')
        ax6.set_title('Prediction Probability Distribution')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Model Complexity Comparison (Bottom Left)
        ax7 = plt.subplot(3, 3, 7)
        
        # Approximate model sizes (you might want to calculate these precisely)
        classical_params = 0.01  # MB (much smaller)
        speech_llm_params = 95   # MB (Wav2Vec2 base is ~95MB)
        
        bars = ax7.bar(['Classical\n(MFCC+SVM)', 'Speech LLM\n(Wav2Vec2)'], 
                      [classical_params, speech_llm_params], 
                      alpha=0.8, color=['skyblue', 'lightcoral'])
        ax7.set_ylabel('Model Size (MB)')
        ax7.set_title('Model Complexity Comparison')
        ax7.set_yscale('log')
        
        for bar, size in zip(bars, [classical_params, speech_llm_params]):
            ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.5,
                    f'{size:.2f}MB', ha='center', va='bottom')
        
        # 8. Training Requirements (Bottom Center)
        ax8 = plt.subplot(3, 3, 8)
        
        # Approximate training times (in minutes)
        classical_train_time = 5    # Classical is much faster
        speech_llm_train_time = 120 # Speech LLM takes longer
        
        bars = ax8.bar(['Classical', 'Speech LLM'], [classical_train_time, speech_llm_train_time], 
                      alpha=0.8, color=['skyblue', 'lightcoral'])
        ax8.set_ylabel('Training Time (minutes)')
        ax8.set_title('Training Requirements')
        
        for bar, time_val in zip(bars, [classical_train_time, speech_llm_train_time]):
            ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(classical_train_time, speech_llm_train_time)*0.02,
                    f'{time_val}min', ha='center', va='bottom')
        
        # 9. Summary Table (Bottom Right)
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('tight')
        ax9.axis('off')
        
        # Create summary table
        table_data = [
            ['Metric', 'Classical', 'Speech LLM', 'Winner'],
            ['Accuracy', f'{self.comparison_results["classical"]["accuracy"]:.3f}', 
             f'{self.comparison_results["speech_llm"]["accuracy"]:.3f}', 
             'Speech LLM' if self.comparison_results["speech_llm"]["accuracy"] > self.comparison_results["classical"]["accuracy"] else 'Classical'],
            ['AUC', f'{self.comparison_results["classical"]["auc"]:.3f}', 
             f'{self.comparison_results["speech_llm"]["auc"]:.3f}', 
             'Speech LLM' if self.comparison_results["speech_llm"]["auc"] > self.comparison_results["classical"]["auc"] else 'Classical'],
            ['Speed (ms)', f'{classical_time:.1f}', f'{speech_llm_time:.1f}', 
             'Classical' if classical_time < speech_llm_time else 'Speech LLM'],
            ['Model Size', '~0.01MB', '~95MB', 'Classical'],
            ['Training Time', '~5min', '~120min', 'Classical']
        ]
        
        table = ax9.table(cellText=table_data, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Color header row
        for i in range(4):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax9.set_title('Summary Comparison', pad=20)
        
        plt.tight_layout()
        plt.savefig('results/comprehensive_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Comprehensive visualizations saved!")
    
    def generate_detailed_report(self):
        """Generate detailed comparison report"""
        print("📝 Generating detailed comparison report...")
        
        if not self.comparison_results:
            print("❌ No comparison results available!")
            return
        
        # Get dataset info
        dataset_info = get_dataset_info()
        
        # Prepare report data
        report_data = {
            'dataset_info': dataset_info,
            'classical_models': {
                'best_model': self.classical_results['model'].best_model if self.classical_results else 'N/A',
                'accuracy': self.comparison_results['classical']['accuracy'],
                'auc': self.comparison_results['classical']['auc'],
                'false_positive_rate': 1 - self.comparison_results['classical']['precision'] if self.comparison_results['classical']['precision'] > 0 else 0
            },
            'speech_llm': {
                'test_accuracy': self.comparison_results['speech_llm']['accuracy'],
                'test_auc': self.comparison_results['speech_llm']['auc'],
                'training_time': self.speech_llm_results['info']['training_history']['training_time'] / 60 if self.speech_llm_results else 0
            },
            'comparison': {
                'best_classical': self.classical_results['model'].best_model if self.classical_results else 'N/A',
                'best_classical_accuracy': self.comparison_results['classical']['accuracy'],
                'speech_llm_accuracy': self.comparison_results['speech_llm']['accuracy'],
                'winner': 'Speech LLM' if self.comparison_results['speech_llm']['accuracy'] > self.comparison_results['classical']['accuracy'] else 'Classical',
                'performance_difference': abs(self.comparison_results['speech_llm']['accuracy'] - self.comparison_results['classical']['accuracy'])
            }
        }
        
        # Save detailed report
        save_results_summary(report_data, "results/detailed_comparison_report.txt")
        
        # Create CSV summary
        summary_df = pd.DataFrame({
            'Model': ['Classical', 'Speech LLM'],
            'Accuracy': [self.comparison_results['classical']['accuracy'], 
                        self.comparison_results['speech_llm']['accuracy']],
            'Precision': [self.comparison_results['classical']['precision'], 
                         self.comparison_results['speech_llm']['precision']],
            'Recall': [self.comparison_results['classical']['recall'], 
                      self.comparison_results['speech_llm']['recall']],
            'F1': [self.comparison_results['classical']['f1'], 
                   self.comparison_results['speech_llm']['f1']],
            'AUC': [self.comparison_results['classical']['auc'], 
                    self.comparison_results['speech_llm']['auc']],
            'Inference_Time_ms': [self.comparison_results['classical']['avg_inference_time']*1000, 
                                 self.comparison_results['speech_llm']['avg_inference_time']*1000]
        })
        
        summary_df.to_csv('results/model_comparison_summary.csv', index=False)
        
        print("✅ Detailed report generated!")
        print("📄 Files created:")
        print("  - results/detailed_comparison_report.txt")
        print("  - results/model_comparison_summary.csv")
        print("  - results/comprehensive_comparison.png")
        
        return report_data
    
    def run_complete_comparison(self):
        """Run complete comparison analysis"""
        print("🔍 Starting complete comparison analysis...\n")
        
        try:
            # Load data splits
            data_splits = load_data_splits()
            
            # Load model results
            classical_loaded = self.load_classical_results()
            speech_llm_loaded = self.load_speech_llm_results()
            
            if not classical_loaded and not speech_llm_loaded:
                raise Exception("No models could be loaded!")
            
            if not classical_loaded:
                print("⚠️ Classical model not available - comparison will be limited")
            if not speech_llm_loaded:
                print("⚠️ Speech LLM model not available - comparison will be limited")
            
            # Evaluate models on test set
            comparison_results = self.evaluate_models_on_test_set(data_splits)
            
            # Create visualizations
            self.create_comprehensive_visualizations()
            
            # Generate detailed report
            report_data = self.generate_detailed_report()
            
            print("\n🎉 Complete comparison analysis finished!")
            
            return comparison_results, report_data
            
        except Exception as e:
            print(f"❌ Comparison analysis failed: {e}")
            raise

def main():
    """Main function for comparison analysis"""
    print("=== CLASSICAL vs SPEECH LLM COMPARISON ===\n")
    
    comparator = ModelComparator()
    
    try:
        results, report = comparator.run_complete_comparison()
        
        # Print final summary
        print("\n" + "="*80)
        print("🏁 FINAL COMPARISON SUMMARY")
        print("="*80)
        
        if 'classical' in results and 'speech_llm' in results:
            classical_acc = results['classical']['accuracy']
            speech_llm_acc = results['speech_llm']['accuracy']
            classical_time = results['classical']['avg_inference_time'] * 1000
            speech_llm_time = results['speech_llm']['avg_inference_time'] * 1000
            
            print(f"\n📊 ACCURACY:")
            print(f"   Classical: {classical_acc:.3f} ({classical_acc*100:.1f}%)")
            print(f"   Speech LLM: {speech_llm_acc:.3f} ({speech_llm_acc*100:.1f}%)")
            
            if speech_llm_acc > classical_acc:
                print(f"   🏆 Winner: Speech LLM (+{(speech_llm_acc-classical_acc)*100:.1f}%)")
            elif classical_acc > speech_llm_acc:
                print(f"   🏆 Winner: Classical (+{(classical_acc-speech_llm_acc)*100:.1f}%)")
            else:
                print(f"   🤝 Tie!")
            
            print(f"\n⚡ SPEED:")
            print(f"   Classical: {classical_time:.1f} ms/sample")
            print(f"   Speech LLM: {speech_llm_time:.1f} ms/sample")
            print(f"   🚀 Speed advantage: Classical is {speech_llm_time/classical_time:.1f}x faster")
            
            print(f"\n🎯 PRACTICAL RECOMMENDATIONS:")
            if classical_acc > 0.85 and classical_time < speech_llm_time:
                print("   ✅ Use Classical approach for production (good accuracy + fast inference)")
            elif speech_llm_acc - classical_acc > 0.05:
                print("   ✅ Use Speech LLM approach if accuracy is critical")
            else:
                print("   ⚖️ Both approaches viable - choose based on deployment constraints")
        
        print(f"\n📁 Results saved in 'results/' directory")
        
        return results
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        return None

if __name__ == "__main__":
    main()