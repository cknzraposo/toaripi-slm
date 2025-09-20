#!/usr/bin/env python3
"""
Create train/validation/test splits for Toaripi SLM training data.

This script splits the processed Toaripi data into training, validation, and test sets
following the preprocessing configuration specifications.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load preprocessing configuration."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found. Using defaults.")
        return {
            'validation': {
                'test_split': 0.1,
                'dev_split': 0.05,
                'random_seed': 42,
                'stratify_by': 'book'
            }
        }

def create_splits(df: pd.DataFrame, config: dict) -> tuple:
    """Create train/validation/test splits."""
    validation_config = config.get('validation', {})
    
    test_size = validation_config.get('test_split', 0.1)
    dev_size = validation_config.get('dev_split', 0.05)
    random_seed = validation_config.get('random_seed', 42)
    stratify_by = validation_config.get('stratify_by', None)
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Calculate actual validation size relative to remaining data after test split
    # If we want 5% dev and 10% test from total, then:
    # After removing 10% for test, we need dev_size/(1-test_size) from remaining
    val_size_adjusted = dev_size / (1 - test_size)
    
    logger.info(f"Creating splits: train={1-test_size-dev_size:.1%}, dev={dev_size:.1%}, test={test_size:.1%}")
    
    stratify_column = None
    if stratify_by and stratify_by in df.columns:
        # Only stratify if we have enough samples per class
        class_counts = df[stratify_by].value_counts()
        min_samples_needed = max(2, int(1 / min(test_size, val_size_adjusted)))
        
        if class_counts.min() >= min_samples_needed:
            stratify_column = df[stratify_by]
            logger.info(f"Stratifying by {stratify_by}")
        else:
            logger.warning(f"Not enough samples per {stratify_by} class for stratification. Proceeding without stratification.")
    
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify_column
    )
    
    # Second split: separate validation from training
    stratify_train_val = None
    if stratify_column is not None:
        stratify_train_val = train_val_df[stratify_by]
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        random_state=random_seed,
        stratify=stratify_train_val
    )
    
    return train_df, val_df, test_df

def prepare_for_slm_training(df: pd.DataFrame, output_file: str):
    """Prepare data for SLM training by creating educational prompts."""
    logger.info(f"Preparing educational prompts for {len(df)} verses")
    
    # Create educational prompts based on the Copilot instructions
    educational_prompts = []
    
    for _, row in df.iterrows():
        toaripi_text = row['toaripi']
        verse_id = row['verse_id']
        book = row['book']
        chapter = row['chapter']
        
        # Create educational content prompt
        prompt = f"""Create educational content in Toaripi language.

Context: {book} Chapter {chapter}
Text: {toaripi_text}

Generate educational content suitable for primary school students learning Toaripi language."""
        
        # Create different types of educational content
        educational_prompts.append({
            'prompt': prompt,
            'toaripi': toaripi_text,
            'verse_id': verse_id,
            'book': book,
            'chapter': chapter,
            'content_type': 'reading_comprehension'
        })
        
        # Vocabulary exercise prompt
        vocab_prompt = f"""Create vocabulary exercises from this Toaripi text:

Text: {toaripi_text}

Generate vocabulary list with:
- Important Toaripi words
- Simple explanations
- Example sentences
- Suitable for primary learners"""
        
        educational_prompts.append({
            'prompt': vocab_prompt,
            'toaripi': toaripi_text,
            'verse_id': f"{verse_id}_vocab",
            'book': book,
            'chapter': chapter,
            'content_type': 'vocabulary'
        })
    
    # Create DataFrame and save
    prompt_df = pd.DataFrame(educational_prompts)
    prompt_df.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"Saved {len(prompt_df)} educational prompts to {output_file}")
    
    return prompt_df

def main():
    """Main function to create data splits."""
    # Setup paths
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / "configs" / "data" / "preprocessing_config.yaml"
    input_file = base_dir / "data" / "processed" / "toaripi_processed.csv"
    output_dir = base_dir / "data" / "processed"
    
    # Load configuration
    config = load_config(str(config_path))
    
    # Load processed data
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} verses")
    
    # Create splits
    train_df, val_df, test_df = create_splits(df, config)
    
    logger.info(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # Save standard splits
    train_df.to_csv(output_dir / "train.csv", index=False, encoding='utf-8')
    val_df.to_csv(output_dir / "validation.csv", index=False, encoding='utf-8')
    test_df.to_csv(output_dir / "test.csv", index=False, encoding='utf-8')
    
    logger.info("Saved standard train/validation/test splits")
    
    # Create educational prompts for SLM training
    prepare_for_slm_training(train_df, str(output_dir / "train_educational.csv"))
    prepare_for_slm_training(val_df, str(output_dir / "validation_educational.csv"))
    prepare_for_slm_training(test_df, str(output_dir / "test_educational.csv"))
    
    # Create summary statistics
    stats = {
        'total_verses': len(df),
        'train_size': len(train_df),
        'validation_size': len(val_df),
        'test_size': len(test_df),
        'train_percentage': len(train_df) / len(df) * 100,
        'validation_percentage': len(val_df) / len(df) * 100,
        'test_percentage': len(test_df) / len(df) * 100,
        'books_in_train': train_df['book'].nunique(),
        'books_in_validation': val_df['book'].nunique(),
        'books_in_test': test_df['book'].nunique(),
        'avg_verse_length_train': train_df['toaripi'].str.len().mean(),
        'avg_verse_length_val': val_df['toaripi'].str.len().mean(),
        'avg_verse_length_test': test_df['toaripi'].str.len().mean(),
    }
    
    # Save statistics
    with open(output_dir / "split_stats.yaml", 'w') as f:
        yaml.dump(stats, f, default_flow_style=False)
    
    logger.info("Data splitting completed successfully!")
    logger.info(f"Statistics: {stats}")

if __name__ == "__main__":
    main()