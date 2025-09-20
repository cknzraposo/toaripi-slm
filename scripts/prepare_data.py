#!/usr/bin/env python3
"""
Data preparation script for Toaripi SLM.
Processes parallel English-Toaripi data for training.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from toaripi_slm.data import DataProcessor
from toaripi_slm.utils import setup_logging, load_config

def main():
    """Main entry point for data preparation."""
    parser = argparse.ArgumentParser(description="Prepare Toaripi training data")
    parser.add_argument("--input", required=True, help="Input parallel data file (CSV)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--config", help="Configuration file (optional)")
    parser.add_argument("--split", action="store_true", help="Split data into train/val/test")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training data ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation data ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test data ratio")
    parser.add_argument("--min-length", type=int, default=5, help="Minimum text length")
    parser.add_argument("--max-length", type=int, default=500, help="Maximum text length")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    
    print(f"📊 Toaripi Data Preparation Tool")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    
    try:
        # Load configuration if provided
        config = {}
        if args.config:
            config = load_config(args.config)
            print(f"Config: {args.config}")
        
        # Initialize data processor
        processor = DataProcessor(config)
        
        # Load parallel data
        print("\n🔄 Loading parallel data...")
        data = processor.load_parallel_data(args.input)
        
        # Validate data
        print("\n✅ Validating data...")
        stats = processor.validate_data(data)
        print(f"  Total rows: {stats['total_rows']}")
        print(f"  English avg length: {stats['avg_english_length']:.1f}")
        print(f"  Toaripi avg length: {stats['avg_toaripi_length']:.1f}")
        print(f"  Unique pairs: {stats['unique_english']}")
        
        # Filter by length
        print(f"\n🔍 Filtering by length ({args.min_length}-{args.max_length} chars)...")
        filtered_data = processor.filter_by_length(
            data, 
            min_length=args.min_length,
            max_length=args.max_length
        )
        
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.split:
            # Split data
            print(f"\n📂 Splitting data...")
            train_df, val_df, test_df = processor.split_data(
                filtered_data,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio
            )
            
            # Save splits
            processor.save_splits(train_df, val_df, test_df, output_dir)
        else:
            # Save processed data
            output_file = output_dir / "processed_data.csv"
            filtered_data.to_csv(output_file, index=False, encoding='utf-8')
            print(f"✅ Saved processed data: {output_file}")
        
        # Create educational prompts
        print("\n📝 Creating educational prompts...")
        prompts = processor.create_educational_prompts(filtered_data)
        
        # Save prompts
        prompts_file = output_dir / "educational_prompts.json"
        import json
        with open(prompts_file, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved educational prompts: {prompts_file}")
        
        # Save final statistics
        final_stats = processor.validate_data(filtered_data)
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if hasattr(obj, 'item'):  # numpy scalars
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            else:
                return obj
        
        final_stats = convert_types(final_stats)
        stats_file = output_dir / "data_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(final_stats, f, indent=2)
        print(f"✅ Saved data statistics: {stats_file}")
        
        print(f"\n🎉 Data preparation completed successfully!")
        print(f"   Output directory: {output_dir}")
        print(f"   Final dataset size: {len(filtered_data)} rows")
        print(f"   Educational prompts: {len(prompts)}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during data preparation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())