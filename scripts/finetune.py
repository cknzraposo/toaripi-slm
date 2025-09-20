#!/usr/bin/env python3
"""
Fine-tuning script for Toaripi SLM.
Trains language models on Toaripi educational content.
"""

import argparse
import sys
from pathlib import Path
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from toaripi_slm.core import ToaripiTrainer
from toaripi_slm.utils import setup_logging, load_config

def main():
    """Main entry point for fine-tuning."""
    parser = argparse.ArgumentParser(description="Fine-tune Toaripi language model")
    parser.add_argument("--config", required=True, help="Training configuration file")
    parser.add_argument("--data", required=True, help="Training data directory or file")
    parser.add_argument("--model", help="Base model override")
    parser.add_argument("--output", required=True, help="Output model directory")
    parser.add_argument("--resume", help="Resume from checkpoint directory")
    parser.add_argument("--dry-run", action="store_true", help="Show estimates without training")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    
    print(f"🚀 Toaripi SLM Fine-tuning")
    print(f"Config: {args.config}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    
    try:
        # Load configuration
        print("\n📋 Loading configuration...")
        config = load_config(args.config)
        
        # Override model if specified
        if args.model:
            config["model"]["name"] = args.model
            print(f"Model override: {args.model}")
        
        # Initialize trainer
        print("\n🔧 Initializing trainer...")
        trainer = ToaripiTrainer(config)
        
        # Load model
        print("\n📦 Loading base model...")
        trainer.load_model()
        
        # Setup LoRA
        print("\n⚙️ Setting up LoRA for efficient fine-tuning...")
        trainer.setup_lora()
        
        # Prepare training dataset
        print("\n📊 Preparing training dataset...")
        data_path = Path(args.data)
        
        # Determine if we have split data or single file
        if data_path.is_dir():
            train_file = data_path / "train.csv"
            val_file = data_path / "validation.csv"
            
            if not train_file.exists():
                print(f"❌ Training file not found: {train_file}")
                return 1
            
            train_dataset = trainer.prepare_dataset(train_file)
            val_dataset = trainer.prepare_dataset(val_file) if val_file.exists() else None
            
        else:
            # Single file - use all for training
            train_dataset = trainer.prepare_dataset(data_path)
            val_dataset = None
        
        print(f"Training samples: {len(train_dataset)}")
        if val_dataset:
            print(f"Validation samples: {len(val_dataset)}")
        
        # Training time estimates
        estimates = trainer.estimate_training_time(len(train_dataset))
        print(f"\n⏱️ Training estimates:")
        print(f"  Total steps: {estimates['total_steps']}")
        print(f"  Estimated time: {estimates['estimated_hours']:.1f} hours")
        print(f"  Device: {estimates['device']}")
        
        if args.dry_run:
            print("\n🏁 Dry run completed - no actual training performed")
            return 0
        
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Start training
        print(f"\n🏋️ Starting training...")
        trainer.train(train_dataset, val_dataset, output_dir)
        
        # Save final model
        print(f"\n💾 Saving final model...")
        final_model_dir = output_dir / "final"
        trainer.save_model(final_model_dir)
        
        print(f"\n🎉 Fine-tuning completed successfully!")
        print(f"   Model saved to: {final_model_dir}")
        print(f"   Ready for inference and GGUF export")
        
        # Save training summary
        summary = {
            "config_file": str(args.config),
            "data_path": str(args.data),
            "base_model": config["model"]["name"],
            "training_samples": len(train_dataset),
            "validation_samples": len(val_dataset) if val_dataset else 0,
            "output_directory": str(final_model_dir),
            "estimates": estimates
        }
        
        summary_file = output_dir / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"   Training summary: {summary_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during fine-tuning: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())