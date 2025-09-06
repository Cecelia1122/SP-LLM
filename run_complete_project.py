"""
run_complete_project.py - Main execution script for the entire project
Purpose: Run the complete keyword spotter project pipeline

"""

import sys
import os
import time
from pathlib import Path

# Prevent matplotlib from opening windows in any imported module
os.environ.setdefault("MPLBACKEND", "Agg")

# Add src directory to path (if you keep code in src/)
sys.path.append('src')

# Import project modules (keep RealTimeKeywordDetector lazy for Phase 5)
try:
    from utils import create_directories
    from data_preparation import DatasetPreparator, get_dataset_info
    from classical_approach import ClassicalKeywordSpotter, main as classical_main
    from speech_llm_approach import SpeechLLMKeywordSpotter, main as speech_llm_main
    from comparison_analysis import ModelComparator, main as comparison_main
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure all files are in the correct locations")
    sys.exit(1)


class ProjectRunner:
    def __init__(self):
        self.project_start_time = time.time()
        self.results = {}

    def print_header(self, title, emoji="🚀"):
        print(f"\n{emoji} " + "="*60)
        print(f"{emoji} {title}")
        print(f"{emoji} " + "="*60)

    def print_phase_complete(self, phase_name, duration):
        print(f"✅ {phase_name} completed in {duration:.1f} minutes")
        print("-" * 60)

    def phase_1_data_preparation(self):
        self.print_header("PHASE 1: DATA PREPARATION", "📂")
        phase_start = time.time()
        try:
            create_directories()
            preparator = DatasetPreparator()
            if preparator.dataset_dir.exists():
                print("✅ Dataset already exists, skipping download")
                if not (preparator.data_dir / "splits").exists():
                    print("📊 Creating data splits...")
                    splits = preparator.create_balanced_splits(max_samples_per_class=2000)
                    self.results['data_preparation'] = {
                        'splits_info': {name: len(files) for name, files in splits.items()}
                    }
                else:
                    print("✅ Data splits already exist")
            else:
                result = preparator.prepare_complete_dataset(
                    max_samples_per_class=2000,
                    cleanup=True
                )
                self.results['data_preparation'] = result

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
        self.print_header("PHASE 2: CLASSICAL APPROACH TRAINING", "🔍")
        phase_start = time.time()
        try:
            classical_model_path = Path("models/classical_keyword_spotter.pkl")
            if classical_model_path.exists():
                print("✅ Classical model already exists")
                spotter = ClassicalKeywordSpotter()
                if spotter.load_model(str(classical_model_path)):
                    loaded_name = getattr(spotter, "best_model", None)
                    print(f"📊 Loaded model: {loaded_name if loaded_name else 'SVM'}")
                    self.results['classical_approach'] = {'model_loaded': True}
                else:
                    print("⚠️ Failed to load existing model, retraining...")
                    spotter, test_results = classical_main()
                    self.results['classical_approach'] = test_results
            else:
                spotter, test_results = classical_main()
                self.results['classical_approach'] = test_results

            phase_duration = (time.time() - phase_start) / 60
            self.print_phase_complete("Classical Training", phase_duration)
            return True
        except Exception as e:
            print(f"❌ Phase 2 failed: {e}")
            return False

    def phase_3_speech_llm_training(self):
        self.print_header("PHASE 3: SPEECH LLM APPROACH TRAINING", "🤖")
        phase_start = time.time()
        try:
            import torch
            speech_llm_model_path = Path("models/speech_llm_model")
            if speech_llm_model_path.exists() and (speech_llm_model_path / "config.json").exists():
                print("✅ Speech LLM model already exists")
                spotter = SpeechLLMKeywordSpotter()
                if spotter.load_model(str(speech_llm_model_path)):
                    print("📊 Loaded existing Speech LLM model")
                    self.results['speech_llm_approach'] = {'model_loaded': True}
                else:
                    print("⚠️ Failed to load existing model, retraining...")
                    spotter, results = speech_llm_main()
                    self.results['speech_llm_approach'] = results
            else:
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
        self.print_header("PHASE 4: COMPARATIVE ANALYSIS", "📊")
        phase_start = time.time()
        try:
            comparison_results = comparison_main()
            if isinstance(comparison_results, tuple) and len(comparison_results) >= 1:
                comparison_results = comparison_results[0]
            self.results['comparison'] = comparison_results
            phase_duration = (time.time() - phase_start) / 60
            self.print_phase_complete("Comparison Analysis", phase_duration)
            return True
        except Exception as e:
            print(f"❌ Phase 4 failed: {e}")
            return False

    def phase_5_real_time_demo(self):
        self.print_header("PHASE 5: REAL-TIME DEMONSTRATION", "🎤")
        try:
            # Auto-skip in non-interactive sessions (e.g., CI)
            if not sys.stdin.isatty():
                print("ℹ️ Non-interactive session detected. Skipping real-time demo.")
                return True

            print("🎯 Real-time detection demo (in-process)")
            print("💡 Behavior matches the standalone realtime_detector.py (same parameters)")

            # Tell user if LLM model is missing (common with --quick)
            llm_present = (Path("models/speech_llm_model") / "config.json").exists()
            if not llm_present:
                print("ℹ️ Speech LLM model not found (expected if you used --quick).")
                print("   Only the Classical model will be available in the demo.")

            # Lazy import to avoid import-time failures if audio deps are missing
            try:
                from realtime_detector import RealTimeKeywordDetector
            except Exception as e:
                print(f"⚠️ Real-time demo unavailable: {e}")
                print("💡 Install audio dependencies (e.g., PyAudio) to enable this feature.")
                return True

            user_input = input("\n🎤 Start real-time detection now? (y/n): ").strip().lower()
            if user_input != 'y':
                print("✅ Real-time demo skipped")
                return True

            # Use the same parameters as the standalone detector's defaults
            detector = RealTimeKeywordDetector(
                confidence_threshold_classical=0.60,
                confidence_threshold_llm=0.70,
                detection_cooldown=1.5,
                warmup_seconds=1.5,
                min_rms_dbfs=None  # auto-calibrate energy gate
            )
            print("💡 Commands: type c/s/m/q + Enter to switch/toggle/quit")
            detector.start_detection()
            print("✅ Real-time demo finished")
            return True

        except Exception as e:
            print(f"❌ Phase 5 failed: {e}")
            return False

    def generate_project_summary(self):
        self.print_header("PROJECT COMPLETION SUMMARY", "🎉")
        total_duration = (time.time() - self.project_start_time) / 60
        print(f"⏱️ Total execution time: {total_duration:.1f} minutes")
        print()

        if 'dataset_info' in self.results:
            dataset = self.results['dataset_info']
            print(f"📊 Dataset: {dataset['total_samples']} samples")
            print(f"   - Positive (keyword): {dataset['positive_samples']}")
            print(f"   - Negative (other): {dataset['negative_samples']}")
            print()

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

    
        return True

    def run_complete_project(self,
                             skip_data_prep=False,
                             skip_classical=False,
                             skip_speech_llm=False,
                             skip_comparison=False,
                             skip_demo=False):
        print("🎯 ENHANCED KEYWORD SPOTTER PROJECT")
        print("📚 Classical Signal Processing vs Modern Speech LLMs")
        print(f"⏱️ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Optional perf consistency
        try:
            import torch
            torch.set_num_threads(min(8, max(2, os.cpu_count() or 4)))
            print(f"[Perf] torch.get_num_threads() = {torch.get_num_threads()}")
        except Exception:
            pass

        success_count = 0
        total_phases = 5

        try:
            if not skip_data_prep:
                if self.phase_1_data_preparation():
                    success_count += 1
            else:
                print("⏭️ Skipping data preparation")
                success_count += 1

            if not skip_classical:
                if self.phase_2_classical_training():
                    success_count += 1
            else:
                print("⏭️ Skipping classical training")

            if not skip_speech_llm:
                if self.phase_3_speech_llm_training():
                    success_count += 1
            else:
                print("⏭️ Skipping Speech LLM training")

            if not skip_comparison:
                if self.phase_4_comparison_analysis():
                    success_count += 1
            else:
                print("⏭️ Skipping comparison analysis")

            if not skip_demo:
                if self.phase_5_real_time_demo():
                    success_count += 1
            else:
                print("⏭️ Skipping real-time demo")

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
    import argparse
    parser = argparse.ArgumentParser(description="Run Enhanced Keyword Spotter Project")
    parser.add_argument("--skip-data", action="store_true", help="Skip data preparation")
    parser.add_argument("--skip-classical", action="store_true", help="Skip classical training")
    parser.add_argument("--skip-speech-llm", action="store_true", help="Skip Speech LLM training")
    parser.add_argument("--skip-comparison", action="store_true", help="Skip comparison analysis")
    parser.add_argument("--skip-demo", action="store_true", help="Skip real-time demo")
    parser.add_argument("--quick", action="store_true", help="Quick run (skip Speech LLM)")
    args = parser.parse_args()

    if args.quick:
        args.skip_speech_llm = True
        args.skip_comparison = True
        print("🚀 Quick mode: Skipping Speech LLM training and comparison")

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