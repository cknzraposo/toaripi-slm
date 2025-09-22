#!/usr/bin/env python3
"""
Prepare parallel data for Toaripi SLM training.

This script processes raw parallel English-Toaripi data and prepares it for training.
"""

import click
import pandas as pd
import yaml
from pathlib import Path
import sys
import os
from typing import Optional, Tuple

def load_config(config_path: Optional[Path] = None) -> dict:
    """Load preprocessing configuration."""
    if config_path is None:
        config_path = Path("configs/data/preprocessing_config.yaml")
    
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    else:
        # Default configuration
        return {
            "min_length": 3,
            "max_length": 512,
            "remove_duplicates": True,
            "clean_text": True
        }

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if pd.isna(text):
        return ""
    
    # Convert to string and strip whitespace
    text = str(text).strip()
    
    # Remove multiple spaces
    import re
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing quotes if present
    text = text.strip('"').strip("'")
    
    return text

def validate_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Validate and clean the dataset."""
    stats = {
        "initial_rows": len(df),
        "removed_empty": 0,
        "removed_too_short": 0,
        "removed_too_long": 0,
        "removed_duplicates": 0,
        "final_rows": 0
    }
    
    # Check for required columns
    required_cols = ["english", "toaripi"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Clean text
    df["english"] = df["english"].apply(clean_text)
    df["toaripi"] = df["toaripi"].apply(clean_text)
    
    # Remove empty rows
    initial_len = len(df)
    df = df[(df["english"] != "") & (df["toaripi"] != "")]
    stats["removed_empty"] = initial_len - len(df)
    
    # Remove rows that are too short or too long
    config = load_config()
    min_length = config.get("min_length", 3)
    max_length = config.get("max_length", 512)
    
    initial_len = len(df)
    df = df[
        (df["english"].str.len() >= min_length) & 
        (df["toaripi"].str.len() >= min_length) &
        (df["english"].str.len() <= max_length) & 
        (df["toaripi"].str.len() <= max_length)
    ]
    stats["removed_too_short"] = initial_len - len(df)
    
    # Remove duplicates if configured
    if config.get("remove_duplicates", True):
        initial_len = len(df)
        df = df.drop_duplicates(subset=["english", "toaripi"])
        stats["removed_duplicates"] = initial_len - len(df)
    
    stats["final_rows"] = len(df)
    
    return df, stats

@click.command()
@click.option('--config', type=Path, help='Configuration YAML file')
@click.option('--input', 'input_path', type=Path, help='Input CSV file path')
@click.option('--output', type=Path, help='Output processed CSV file')
@click.option('--validate', is_flag=True, help='Validate data after processing')
@click.option('--split', is_flag=True, help='Split into train/validation sets')
@click.option('--train-ratio', type=float, default=0.8, help='Training set ratio (default: 0.8)')
def main(config, input_path, output, validate, split, train_ratio):
    """Prepare parallel data for training."""
    
    # Default paths if not specified
    if not input_path:
        input_path = Path("data/raw/Full_bible_english_toaripi.csv")
    if not output:
        output = Path("data/processed/toaripi_parallel.csv")
    
    click.echo(f"🔄 Processing data from {input_path}")
    click.echo(f"📁 Output will be saved to {output}")
    
    try:
        # Load input data with encoding detection
        if not input_path.exists():
            click.echo(f"❌ Input file not found: {input_path}")
            sys.exit(1)
        
        # Try different encodings
        encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        df = None
        used_encoding = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(input_path, encoding=encoding)
                used_encoding = encoding
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            click.echo(f"❌ Could not read file with any encoding")
            sys.exit(1)
            
        click.echo(f"✅ Loaded {len(df)} rows from input file (encoding: {used_encoding})")
        click.echo(f"📊 Columns: {list(df.columns)}")
        
        # Create output directory
        output.parent.mkdir(parents=True, exist_ok=True)
        
        # Validate and clean data
        df_clean, stats = validate_data(df)
        
        click.echo(f"🧹 Data cleaning results:")
        click.echo(f"   Initial rows: {stats['initial_rows']}")
        click.echo(f"   Removed empty: {stats['removed_empty']}")
        click.echo(f"   Removed too short/long: {stats['removed_too_short']}")
        click.echo(f"   Removed duplicates: {stats['removed_duplicates']}")
        click.echo(f"   Final rows: {stats['final_rows']}")
        
        # Save processed data
        df_clean.to_csv(output, index=False, encoding='utf-8')
        click.echo(f"💾 Saved processed data to {output}")
        
        # Split data if requested
        if split:
            train_size = int(len(df_clean) * train_ratio)
            df_train = df_clean.iloc[:train_size]
            df_val = df_clean.iloc[train_size:]
            
            train_path = output.parent / "train.csv"
            val_path = output.parent / "validation.csv"
            
            df_train.to_csv(train_path, index=False, encoding='utf-8')
            df_val.to_csv(val_path, index=False, encoding='utf-8')
            
            click.echo(f"📊 Split data:")
            click.echo(f"   Training set: {len(df_train)} rows → {train_path}")
            click.echo(f"   Validation set: {len(df_val)} rows → {val_path}")
        
        # Validation if requested
        if validate:
            click.echo("🔍 Validating processed data...")
            
            # Check if we have enough data
            if len(df_clean) < 100:
                click.echo("⚠️  Warning: Dataset is quite small (< 100 samples)")
            
            # Check language balance
            avg_en_len = df_clean['english'].str.len().mean()
            avg_to_len = df_clean['toaripi'].str.len().mean()
            
            click.echo(f"📈 Dataset statistics:")
            click.echo(f"   Average English length: {avg_en_len:.1f} characters")
            click.echo(f"   Average Toaripi length: {avg_to_len:.1f} characters")
            click.echo(f"   Length ratio (Toaripi/English): {avg_to_len/avg_en_len:.2f}")
            
            # Show sample rows
            click.echo(f"\n📋 Sample data:")
            for i, row in df_clean.head(3).iterrows():
                click.echo(f"   EN: {row['english'][:80]}...")
                click.echo(f"   TO: {row['toaripi'][:80]}...")
                click.echo("")
            
            click.echo(f"✅ Validation completed: {len(df_clean)} valid parallel sentences")
            
    except Exception as e:
        click.echo(f"❌ Error processing data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()