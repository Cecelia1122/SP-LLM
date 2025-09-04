"""
run_complete_project.py - Main execution script for the entire project
Purpose: Run the complete keyword spotter project pipeline
"""

import sys
import os
import time
from pathlib import Path

# Add src directory to path
sys.path.append('src')

# Import all project modules
try:
    from utils import create_directories
    from data_preparation import DatasetPreparator, get_dataset_info
    from classical_approach import ClassicalKeywordSpotter, main as classical_main
    from speech_llm_approach import SpeechLLMKeywordSpotter, main as speech_llm_main
    from comparison_analysis import ModelComparator, main as comparison_main
    from realtime_detector import RealTimeKeywordDetector
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure all files are in the correct locations")
    sys.exit(1)

class ProjectRunner:
    def __init__(self):
        self.project_start_time = time.time()
        self.results = {}
        
    def print_header(self, title, emoji="🚀"):
        """Print formatted header"""
        print(f"\n{emoji} " + "="*60)
        print(f"{emoji} {title}")
        print(f"{emoji} " + "="*60)
    
    def print_phase_complete(self, phase_name, duration):
        """Print phase completion message"""
        print(f"✅ {phase_name} completed in {duration:.1f} minutes")
        print("-" * 60)
    
    def phase_1_data_preparation(self):
        """Phase 1: Download and prepare dataset"""
        self.print_header("PHASE 1: DATA PREPARATION", "📂")
        
        phase_start = time.time()
        
        try:
            # Create project structure
            create_directories()
            
            # Initialize dataset preparator
            preparator = DatasetPreparator()
            
            # Check if dataset already exists
            if preparator.dataset_dir.exists():
                print("✅ Dataset already exists, skipping download")
                
                # Just create splits if they don't exist
                if not (preparator.data_dir / "splits").exists():
                    print("📊 Creating data splits...")
                    splits = preparator.create_balanced_splits(max_samples_per_class=2000)
                    self.results['data_preparation'] = {
                        'splits_info': {name: len(files) for name, files in splits.items()}
                    }
                else:
                    print("✅ Data splits already exist")
            else:
                # Full dataset preparation
                result = preparator.prepare_complete_dataset(
                    max_samples_per_class=2000,
                    cleanup=True
                )
                self.results['data_preparation'] = result
            
            # Get dataset info
            dataset_info = get_dataset_info()
            if dataset_info:
                print(f"📊 Dataset ready: {dataset_info['total_samples']} samples")
                self.results['dataset_info'] = dataset_info
            
            phase_duration = (time.time() - phase_start) / 60
            self.print_phase_complete("Data Preparation", phase_duration)
            
            return True
            
        except Exception as e:
            print(f"❌ Phase 1 failed: {e}")
            return False
    
    def phase_2_classical_training(self):
        """Phase 2: Train classical approach"""
        self.print_header("PHASE 2: CLASSICAL APPROACH TRAINING", "🔍")
        
        phase_start = time.time()
        
        try:
            # Check if model already exists
            classical_model_path = Path("models/classical_keyword_spotter.pkl")
            
            if classical_model_path.exists():
                print("✅ Classical model already exists")
                
                # Load existing model for results
                spotter = ClassicalKeywordSpotter()
                if spotter.load_model(str(classical_model_path)):
                    print(f"📊 Loaded model: {spotter.best_model}")
                    self.results['classical_approach'] = {'model_loaded': True}
                else:
                    print("⚠️ Failed to load existing model, retraining...")
                    spotter, test_results = classical_main()
                    self.results['classical_approach'] = test_results
            else:
                # Train new model
                spotter, test_results = classical_main()
                self.results['classical_approach'] = test_results
            
            phase_duration = (time.time() - phase_start) / 60
            self.print_phase_complete("Classical Training", phase_duration)
            
            return True
            
        except Exception as e:
            print(f"❌ Phase 2 failed: {e}")
            return False
    
    def phase_3_speech_llm_training(self):
        """Phase 3: Train Speech LLM approach"""
        self.print_header("PHASE 3: SPEECH LLM APPROACH TRAINING", "🤖")
        
        phase_start = time.time()
        
        try:
            import torch
            
            # Check if model already exists
            speech_llm_model_path = Path("models/speech_llm_model")
            
            if speech_llm_model_path.exists() and (speech_llm_model_path / "config.json").exists():
                print("✅ Speech LLM model already exists")
                
                # Load existing model for results
                spotter = SpeechLLMKeywordSpotter()
                if spotter.load_model(str(speech_llm_model_path)):
                    print("📊 Loaded existing Speech LLM model")
                    self.results['speech_llm_approach'] = {'model_loaded': True}
                else:
                    print("⚠️ Failed to load existing model, retraining...")
                    spotter, results = speech_llm_main()
                    self.results['speech_llm_approach'] = results
            else:
                # Train new model
                if torch.cuda.is_available():
                    print(f"🔥 GPU available: {torch.cuda.get_device_name(0)}")
                    print("⚡ Training will be faster with GPU acceleration")
                else:
                    print("💻 Training with CPU (will take longer)")
                    print("💡 Consider using Google Colab for faster GPU training")
                
                spotter, results = speech_llm_main()
                self.results['speech_llm_approach'] = results
            
            phase_duration = (time.time() - phase_start) / 60
            self.print_phase_complete("Speech LLM Training", phase_duration)
            
            return True
            
        except Exception as e:
            print(f"❌ Phase 3 failed: {e}")
            print("💡 Speech LLM training requires significant computational resources")
            print("💡 Consider training this component separately or using a pre-trained model")
            return False
    
    def phase_4_comparison_analysis(self):
        """Phase 4: Compare both approaches"""
        self.print_header("PHASE 4: COMPARATIVE ANALYSIS", "📊")
        
        phase_start = time.time()
        
        try:
            comparison_results = comparison_main()
            self.results['comparison'] = comparison_results
            
            phase_duration = (time.time() - phase_start) / 60
            self.print_phase_complete("Comparison Analysis", phase_duration)
            
            return True
            
        except Exception as e:
            print(f"❌ Phase 4 failed: {e}")
            return False
    
    def phase_5_real_time_demo(self):
        """Phase 5: Demonstrate real-time detection"""
        self.print_header("PHASE 5: REAL-TIME DEMONSTRATION", "🎤")
        
        try:
            print("🎯 Real-time detection demo available")
            print("💡 Run 'python 4_real_time_detector.py' to try it")
            
            # Optional: Ask user if they want to try real-time detection now
            user_input = input("\n🎤 Would you like to try real-time detection now? (y/n): ").strip().lower()
            
            if user_input == 'y':
                print("\nStarting real-time detection...")
                detector = RealTimeKeywordDetector(
                    confidence_threshold=0.7,
                    detection_cooldown=2.0
                )
                
                print("💡 You can switch models and test both approaches")
                detector.start_detection()
            else:
                print("✅ Real-time demo skipped")
            
            return True
            
        except Exception as e:
            print(f"❌ Phase 5 failed: {e}")
            return False
    
    def generate_project_summary(self):
        """Generate final project summary"""
        self.print_header("PROJECT COMPLETION SUMMARY", "🎉")
        
        total_duration = (time.time() - self.project_start_time) / 60
        
        print(f"⏱️ Total execution time: {total_duration:.1f} minutes")
        print()
        
        # Dataset summary
        if 'dataset_info' in self.results:
            dataset = self.results['dataset_info']
            print(f"📊 Dataset: {dataset['total_samples']} samples")
            print(f"   - Positive (keyword): {dataset['positive_samples']}")
            print(f"   - Negative (other): {dataset['negative_samples']}")
            print()
        
        # Model performance summary
        if 'comparison' in self.results and self.results['comparison']:
            comp = self.results['comparison']
            if 'classical' in comp and 'speech_llm' in comp:
                print("🏆 Model Performance:")
                print(f"   Classical:  {comp['classical']['accuracy']:.3f} accuracy")
                print(f"   Speech LLM: {comp['speech_llm']['accuracy']:.3f} accuracy")
                
                if comp['speech_llm']['accuracy'] > comp['classical']['accuracy']:
                    winner = "Speech LLM"
                    advantage = comp['speech_llm']['accuracy'] - comp['classical']['accuracy']
                else:
                    winner = "Classical"
                    advantage = comp['classical']['accuracy'] - comp['speech_llm']['accuracy']
                
                print(f"   Winner: {winner} (+{advantage:.3f})")
                print()
        
        # Files created
        print("📁 Files Created:")
        
        expected_files = [
            "data/splits/",
            "models/classical_keyword_spotter.pkl",
            "models/speech_llm_model/",
            "results/comprehensive_comparison.png",
            "results/detailed_comparison_report.txt",
            "results/model_comparison_summary.csv"
        ]
        
        for file_path in expected_files:
            path = Path(file_path)
            if path.exists():
                print(f"   ✅ {file_path}")
            else:
                print(f"   ❌ {file_path} (not created)")
        
        print()
        
        # Next steps
        print("🚀 Next Steps:")
        print("   1. Review results in 'results/' directory")
        print("   2. Try real-time detection: python 4_real_time_detector.py")
        print("   3. Test with your own audio files")
        print("   4. Customize models for your specific use case")
        print("   5. Deploy the best model for your application")
        print()
        
        # CV/Resume points
        print("📝 For Your CV/Resume:")
        print("   • Implemented speech signal processing system using MFCC feature extraction")
        print("   • Developed classical ML approach (SVM, Random Forest, Neural Networks)")
        print("   • Fine-tuned Speech LLM (Wav2Vec2) for keyword detection")
        print("   • Achieved X% accuracy with real-time processing capabilities")
        print("   • Comparative analysis of classical vs modern deep learning approaches")
        print("   • Built complete ML pipeline from data preparation to deployment")
        
        return True
    
    def run_complete_project(self, 
                           skip_data_prep=False,
                           skip_classical=False, 
                           skip_speech_llm=False,
                           skip_comparison=False,
                           skip_demo=False):
        """Run the complete project pipeline"""
        
        print("🎯 ENHANCED KEYWORD SPOTTER PROJECT")
        print("📚 Classical Signal Processing vs Modern Speech LLMs")
        print(f"⏱️ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        success_count = 0
        total_phases = 5
        
        try:
            # Phase 1: Data Preparation
            if not skip_data_prep:
                if self.phase_1_data_preparation():
                    success_count += 1
            else:
                print("⏭️ Skipping data preparation")
                success_count += 1
            
            # Phase 2: Classical Training
            if not skip_classical:
                if self.phase_2_classical_training():
                    success_count += 1
            else:
                print("⏭️ Skipping classical training")
            
            # Phase 3: Speech LLM Training  
            if not skip_speech_llm:
                if self.phase_3_speech_llm_training():
                    success_count += 1
            else:
                print("⏭️ Skipping Speech LLM training")
            
            # Phase 4: Comparison Analysis
            if not skip_comparison:
                if self.phase_4_comparison_analysis():
                    success_count += 1
            else:
                print("⏭️ Skipping comparison analysis")
            
            # Phase 5: Real-time Demo
            if not skip_demo:
                if self.phase_5_real_time_demo():
                    success_count += 1
            else:
                print("⏭️ Skipping real-time demo")
            
            # Generate summary
            self.generate_project_summary()
            
            print(f"\n🎉 PROJECT COMPLETED: {success_count}/{total_phases} phases successful")
            
            if success_count == total_phases:
                print("🏆 Perfect run! All phases completed successfully.")
            elif success_count >= 3:
                print("✅ Good run! Core functionality implemented.")
            else:
                print("⚠️ Partial completion. Some phases need attention.")
            
            return self.results
            
        except KeyboardInterrupt:
            print("\n⚠️ Project interrupted by user")
            return self.results
        except Exception as e:
            print(f"\n❌ Project failed: {e}")
            return self.results

def main():
    """Main execution function with command line options"""
    
    # Parse command line arguments for selective execution
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Enhanced Keyword Spotter Project")
    parser.add_argument("--skip-data", action="store_true", help="Skip data preparation")
    parser.add_argument("--skip-classical", action="store_true", help="Skip classical training")
    parser.add_argument("--skip-speech-llm", action="store_true", help="Skip Speech LLM training")
    parser.add_argument("--skip-comparison", action="store_true", help="Skip comparison analysis")
    parser.add_argument("--skip-demo", action="store_true", help="Skip real-time demo")
    parser.add_argument("--quick", action="store_true", help="Quick run (skip Speech LLM)")
    
    args = parser.parse_args()
    
    # Quick mode skips Speech LLM training (computationally intensive)
    if args.quick:
        args.skip_speech_llm = True
        print("🚀 Quick mode: Skipping Speech LLM training")
    
    # Initialize and run project
    runner = ProjectRunner()
    
    results = runner.run_complete_project(
        skip_data_prep=args.skip_data,
        skip_classical=args.skip_classical,
        skip_speech_llm=args.skip_speech_llm,
        skip_comparison=args.skip_comparison,
        skip_demo=args.skip_demo
    )
    
    return results

if __name__ == "__main__":
    main()