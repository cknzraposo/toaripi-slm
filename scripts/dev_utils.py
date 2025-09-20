#!/usr/bin/env python3
"""
Development utilities for Toaripi SLM project.
Provides common tasks like data validation, model conversion, etc.
"""

import argparse
import json
import csv
from pathlib import Path
import sys
import subprocess


def validate_parallel_data(csv_path: str) -> bool:
    """Validate parallel English-Toaripi data format"""
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        if not rows:
            print(f"❌ Empty file: {csv_path}")
            return False
            
        required_columns = {'english', 'toaripi'}
        actual_columns = set(rows[0].keys())
        
        if not required_columns.issubset(actual_columns):
            missing = required_columns - actual_columns
            print(f"❌ Missing columns: {missing}")
            return False
            
        # Check for empty values
        empty_count = 0
        for i, row in enumerate(rows):
            if not row['english'].strip() or not row['toaripi'].strip():
                empty_count += 1
                if empty_count <= 5:  # Show first 5 examples
                    print(f"⚠️  Row {i+1}: Empty value(s)")
        
        print(f"✅ Validated {len(rows)} parallel sentences")
        if empty_count > 0:
            print(f"⚠️  Found {empty_count} rows with empty values")
            
        return True
        
    except Exception as e:
        print(f"❌ Error validating {csv_path}: {e}")
        return False


def create_training_data_split(input_path: str, output_dir: str, train_ratio: float = 0.8):
    """Split parallel data into train/validation sets"""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(input_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # Shuffle for better distribution
        import random
        random.seed(42)
        random.shuffle(rows)
        
        # Split data
        split_idx = int(len(rows) * train_ratio)
        train_data = rows[:split_idx]
        val_data = rows[split_idx:]
        
        # Write train set
        train_path = output_path / "train.csv"
        with open(train_path, 'w', newline='', encoding='utf-8') as f:
            if train_data:
                writer = csv.DictWriter(f, fieldnames=train_data[0].keys())
                writer.writeheader()
                writer.writerows(train_data)
        
        # Write validation set
        val_path = output_path / "validation.csv"
        with open(val_path, 'w', newline='', encoding='utf-8') as f:
            if val_data:
                writer = csv.DictWriter(f, fieldnames=val_data[0].keys())
                writer.writeheader()
                writer.writerows(val_data)
        
        print(f"✅ Created training split:")
        print(f"   Train: {len(train_data)} samples → {train_path}")
        print(f"   Validation: {len(val_data)} samples → {val_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating data split: {e}")
        return False


def check_model_compatibility(model_name: str):
    """Check if a model is compatible with our training setup"""
    compatible_models = [
        "microsoft/DialoGPT-small",
        "microsoft/DialoGPT-medium", 
        "microsoft/DialoGPT-large",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "microsoft/phi-2",
        "facebook/opt-1.3b",
        "gpt2",
        "gpt2-medium"
    ]
    
    print(f"🔍 Checking model: {model_name}")
    
    if model_name in compatible_models:
        print(f"✅ {model_name} is known to be compatible")
    else:
        print(f"⚠️  {model_name} not in tested model list")
        print("   Supported models:")
        for model in compatible_models:
            print(f"   - {model}")
    
    # Try to get model info (requires transformers)
    try:
        import requests
        url = f"https://huggingface.co/api/models/{model_name}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            model_info = response.json()
            print(f"✅ Model exists on Hugging Face Hub")
            print(f"   Downloads: {model_info.get('downloads', 'N/A')}")
            print(f"   License: {model_info.get('license', 'N/A')}")
        else:
            print(f"❌ Model not found on Hugging Face Hub")
            
    except Exception as e:
        print(f"⚠️  Could not fetch model info: {e}")


def estimate_training_requirements(model_name: str, dataset_size: int):
    """Estimate training time and resource requirements"""
    # Rough estimates based on model size
    model_sizes = {
        "gpt2": 117,  # Million parameters
        "gpt2-medium": 345,
        "microsoft/DialoGPT-small": 117,
        "microsoft/DialoGPT-medium": 345,
        "microsoft/DialoGPT-large": 762,
        "microsoft/phi-2": 2700,
        "mistralai/Mistral-7B-Instruct-v0.2": 7000,
        "facebook/opt-1.3b": 1300
    }
    
    params = model_sizes.get(model_name, 1000)  # Default estimate
    
    print(f"📊 Training estimates for {model_name}:")
    print(f"   Model parameters: ~{params}M")
    print(f"   Dataset size: {dataset_size} samples")
    
    # GPU memory estimates (rough)
    if params < 500:
        gpu_memory = "4-8 GB"
        cpu_fallback = "Possible"
    elif params < 2000:
        gpu_memory = "8-16 GB" 
        cpu_fallback = "Slow but possible"
    else:
        gpu_memory = "16-24 GB"
        cpu_fallback = "Not recommended"
    
    print(f"   Estimated GPU memory: {gpu_memory}")
    print(f"   CPU training: {cpu_fallback}")
    
    # Time estimates (very rough)
    if dataset_size < 1000:
        time_est = "1-3 hours"
    elif dataset_size < 10000:
        time_est = "3-12 hours"
    else:
        time_est = "12+ hours"
    
    print(f"   Estimated training time: {time_est} (with GPU)")


def generate_quick_start_guide():
    """Generate a quick start guide for the project"""
    guide = """
# Toaripi SLM Quick Start Guide

## 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\\Scripts\\activate     # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Verify setup
python scripts/verify_setup.py
```

## 2. Prepare Your Data
```bash
# Validate existing sample data
python scripts/dev_utils.py validate-data data/samples/parallel_data.csv

# Create train/validation split
python scripts/dev_utils.py split-data data/samples/parallel_data.csv data/processed --train-ratio 0.8
```

## 3. Configure Training
Edit `configs/training/base_config.yaml`:
- Adjust `model_name` for your target model
- Set `output_dir` for saving models
- Modify `batch_size` based on your GPU memory

## 4. Start Training
```bash
# Basic training
python scripts/train.py --config configs/training/base_config.yaml

# LoRA fine-tuning (recommended for large models)
python scripts/train.py --config configs/training/lora_config.yaml
```

## 5. Test Your Model
```bash
# Run model inference
python scripts/inference.py --model models/hf/toaripi-finetuned --prompt "Tell me a story about fishing"

# Start web interface
python -m app.main
```

## 6. Deploy for Edge
```bash
# Convert to GGUF for CPU deployment
python scripts/convert_to_gguf.py --model models/hf/toaripi-finetuned --output models/gguf/
```

## Need Help?
- Check `docs/` for detailed documentation
- Run `python scripts/verify_setup.py` to diagnose issues
- Review sample data in `data/samples/`
"""
    
    with open("QUICKSTART.md", "w", encoding="utf-8") as f:
        f.write(guide.strip())
    
    print("✅ Generated QUICKSTART.md")


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Toaripi SLM Development Utilities")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Validate data command
    validate_parser = subparsers.add_parser("validate-data", help="Validate parallel data CSV")
    validate_parser.add_argument("csv_path", help="Path to CSV file")
    
    # Split data command
    split_parser = subparsers.add_parser("split-data", help="Split data into train/validation")
    split_parser.add_argument("input_path", help="Input CSV file")
    split_parser.add_argument("output_dir", help="Output directory")
    split_parser.add_argument("--train-ratio", type=float, default=0.8, help="Training data ratio")
    
    # Model compatibility command
    model_parser = subparsers.add_parser("check-model", help="Check model compatibility")
    model_parser.add_argument("model_name", help="Hugging Face model name")
    
    # Training estimates command
    estimate_parser = subparsers.add_parser("estimate-training", help="Estimate training requirements")
    estimate_parser.add_argument("model_name", help="Model name")
    estimate_parser.add_argument("dataset_size", type=int, help="Number of training samples")
    
    # Quick start guide command
    subparsers.add_parser("quickstart", help="Generate quick start guide")
    
    args = parser.parse_args()
    
    if args.command == "validate-data":
        validate_parallel_data(args.csv_path)
    elif args.command == "split-data":
        create_training_data_split(args.input_path, args.output_dir, args.train_ratio)
    elif args.command == "check-model":
        check_model_compatibility(args.model_name)
    elif args.command == "estimate-training":
        estimate_training_requirements(args.model_name, args.dataset_size)
    elif args.command == "quickstart":
        generate_quick_start_guide()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()