"""
1_data_preparation.py - Dataset download and preparation
Purpose: Download Google Speech Commands dataset and organize for training
"""

import os
import urllib.request
import tarfile
import shutil
from pathlib import Path
import numpy as np
from utils import create_directories, validate_audio_file, print_progress_bar

class DatasetPreparator:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.dataset_dir = self.data_dir / "speech_commands_dataset"
        self.url = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
        self.archive_path = self.data_dir / "speech_commands.tar.gz"
        
        # Target words for our binary classification
        self.keyword = "yes"
        self.negative_words = [
            "no", "up", "down", "left", "right", "on", "off", "stop", "go", 
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"
        ]
        
    def download_dataset(self):
        """Download Google Speech Commands dataset"""
        print("📥 Downloading Google Speech Commands dataset...")
        print("⚠️  This is ~2GB and may take several minutes")
        
        # Create data directory
        self.data_dir.mkdir(exist_ok=True)
        
        if self.archive_path.exists():
            print("✅ Dataset archive already exists")
            return
        
        try:
            # Download with progress tracking
            def progress_hook(block_num, block_size, total_size):
                if total_size > 0:
                    progress = (block_num * block_size) / total_size
                    print_progress_bar(int(progress * 100), 100, 
                                     prefix='Download Progress:', suffix='Complete')
            
            urllib.request.urlretrieve(self.url, self.archive_path, progress_hook)
            print("\n✅ Dataset downloaded successfully")
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            raise
    
    def extract_dataset(self):
        """Extract the downloaded dataset"""
        print("📂 Extracting dataset...")
        
        if self.dataset_dir.exists():
            print("✅ Dataset already extracted")
            return
        
        try:
            with tarfile.open(self.archive_path, "r:gz") as tar:
                tar.extractall(self.dataset_dir)
            
            print("✅ Dataset extracted successfully")
            
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            raise
    
    def validate_dataset(self):
        """Validate the extracted dataset"""
        print("🔍 Validating dataset...")
        
        if not self.dataset_dir.exists():
            raise FileNotFoundError("Dataset directory not found")
        
        # Check for required directories
        required_dirs = [self.keyword] + self.negative_words[:5]  # Check subset
        missing_dirs = []
        
        for dir_name in required_dirs:
            dir_path = self.dataset_dir / dir_name
            if not dir_path.exists():
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            print(f"⚠️  Missing directories: {missing_dirs}")
        else:
            print("✅ All required directories found")
        
        # Count files in each directory
        stats = {}
        for dir_name in [self.keyword] + self.negative_words:
            dir_path = self.dataset_dir / dir_name
            if dir_path.exists():
                wav_files = list(dir_path.glob("*.wav"))
                stats[dir_name] = len(wav_files)
                
                # Validate a few files
                valid_files = 0
                for wav_file in wav_files[:5]:  # Check first 5 files
                    if validate_audio_file(str(wav_file)):
                        valid_files += 1
                
                print(f"  {dir_name}: {len(wav_files)} files ({valid_files}/5 validated)")
        
        return stats
    
    def create_balanced_splits(self, max_samples_per_class: int = 2000):
        """Create balanced training splits"""
        print("⚖️  Creating balanced dataset splits...")
        
        # Create splits directory
        splits_dir = self.data_dir / "splits"
        splits_dir.mkdir(exist_ok=True)
        
        # Collect positive samples (keyword)
        keyword_dir = self.dataset_dir / self.keyword
        positive_files = list(keyword_dir.glob("*.wav"))
        np.random.shuffle(positive_files)
        
        # Limit positive samples
        positive_files = positive_files[:max_samples_per_class]
        
        print(f"📊 Positive samples ({self.keyword}): {len(positive_files)}")
        
        # Collect negative samples (balanced across all negative words)
        negative_files = []
        samples_per_negative_word = max_samples_per_class // len(self.negative_words)
        
        for word in self.negative_words:
            word_dir = self.dataset_dir / word
            if word_dir.exists():
                word_files = list(word_dir.glob("*.wav"))
                np.random.shuffle(word_files)
                negative_files.extend(word_files[:samples_per_negative_word])
        
        np.random.shuffle(negative_files)
        negative_files = negative_files[:max_samples_per_class]  # Ensure balance
        
        print(f"📊 Negative samples: {len(negative_files)}")
        
        # Combine and create final splits
        all_files = [(f, 1) for f in positive_files] + [(f, 0) for f in negative_files]
        np.random.shuffle(all_files)
        
        # Split into train/validation/test
        n_total = len(all_files)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        
        train_files = all_files[:n_train]
        val_files = all_files[n_train:n_train + n_val]
        test_files = all_files[n_train + n_val:]
        
        # Save splits
        splits = {
            'train': train_files,
            'validation': val_files,
            'test': test_files
        }
        
        for split_name, files in splits.items():
            split_file = splits_dir / f"{split_name}_files.txt"
            with open(split_file, 'w') as f:
                for file_path, label in files:
                    f.write(f"{file_path}\t{label}\n")
        
        # Print statistics
        for split_name, files in splits.items():
            n_pos = sum(1 for _, label in files if label == 1)
            n_neg = len(files) - n_pos
            print(f"  {split_name}: {len(files)} total ({n_pos} positive, {n_neg} negative)")
        
        return splits
    
    def cleanup_temp_files(self):
        """Remove temporary files to save space"""
        print("🧹 Cleaning up temporary files...")
        
        if self.archive_path.exists():
            self.archive_path.unlink()
            print("✅ Removed archive file")
    
    def prepare_complete_dataset(self, max_samples_per_class: int = 2000, cleanup: bool = True):
        """Complete dataset preparation pipeline"""
        print("🚀 Starting complete dataset preparation...\n")
        
        try:
            # Step 1: Create directories
            create_directories()
            
            # Step 2: Download dataset
            self.download_dataset()
            
            # Step 3: Extract dataset
            self.extract_dataset()
            
            # Step 4: Validate dataset
            stats = self.validate_dataset()
            
            # Step 5: Create balanced splits
            splits = self.create_balanced_splits(max_samples_per_class)
            
            # Step 6: Cleanup (optional)
            if cleanup:
                self.cleanup_temp_files()
            
            print("\n🎉 Dataset preparation completed successfully!")
            print(f"📍 Dataset location: {self.dataset_dir}")
            print(f"📍 Splits location: {self.data_dir / 'splits'}")
            
            return {
                'dataset_dir': str(self.dataset_dir),
                'splits_dir': str(self.data_dir / "splits"),
                'stats': stats,
                'splits_info': {name: len(files) for name, files in splits.items()}
            }
            
        except Exception as e:
            print(f"❌ Dataset preparation failed: {e}")
            raise

def load_data_splits(data_dir: str = "data"):
    """
    Load the prepared data splits
    
    Returns:
        Dictionary with train, validation, test file paths and labels
    """
    splits_dir = Path(data_dir) / "splits"
    
    if not splits_dir.exists():
        raise FileNotFoundError("Data splits not found. Run data preparation first.")
    
    splits = {}
    for split_name in ['train', 'validation', 'test']:
        split_file = splits_dir / f"{split_name}_files.txt"
        if split_file.exists():
            files = []
            labels = []
            
            with open(split_file, 'r') as f:
                for line in f:
                    file_path, label = line.strip().split('\t')
                    files.append(file_path)
                    labels.append(int(label))
            
            splits[split_name] = {
                'files': files,
                'labels': labels
            }
    
    return splits

def get_dataset_info(data_dir: str = "data"):
    """Get information about the prepared dataset"""
    try:
        splits = load_data_splits(data_dir)
        
        total_samples = sum(len(split['files']) for split in splits.values())
        total_positive = sum(sum(split['labels']) for split in splits.values())
        total_negative = total_samples - total_positive
        
        info = {
            'total_samples': total_samples,
            'positive_samples': total_positive,
            'negative_samples': total_negative,
            'splits': {name: len(data['files']) for name, data in splits.items()}
        }
        
        return info
        
    except FileNotFoundError:
        return None

if __name__ == "__main__":
    # Example usage
    print("=== DATASET PREPARATION ===\n")
    
    # Initialize preparator
    preparator = DatasetPreparator()
    
    # Prepare complete dataset
    try:
        result = preparator.prepare_complete_dataset(
            max_samples_per_class=2000,  # Adjust based on your needs
            cleanup=True  # Set to False if you want to keep the archive
        )
        
        print("\n=== DATASET SUMMARY ===")
        for key, value in result['stats'].items():
            print(f"{key}: {value} files")
            
    except Exception as e:
        print(f"Failed to prepare dataset: {e}")
        exit(1)
    
    # Test loading splits
    print("\n=== TESTING DATA LOADING ===")
    try:
        splits = load_data_splits()
        info = get_dataset_info()
        
        print("Dataset splits loaded successfully:")
        for split_name, split_data in splits.items():
            n_pos = sum(split_data['labels'])
            n_neg = len(split_data['labels']) - n_pos
            print(f"  {split_name}: {len(split_data['files'])} files ({n_pos} positive, {n_neg} negative)")
        
        print(f"\nTotal dataset: {info['total_samples']} samples")
        print(f"  Positive: {info['positive_samples']}")
        print(f"  Negative: {info['negative_samples']}")
        
    except Exception as e:
        print(f"Failed to load dataset: {e}")