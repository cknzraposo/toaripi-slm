#!/usr/bin/env python3
"""
Data preparation script for Toaripi SLM.

This script processes raw parallel English-Toaripi data and prepares it for training.
"""

import click
import pandas as pd
from pathlib import Path
from loguru import logger

from toaripi_slm import DataProcessor, load_config, setup_logging
from toaripi_slm.data.processor import DataProcessingConfig


@click.command()
@click.option(
    '--input', 
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Input CSV file with parallel English-Toaripi data'
)
@click.option(
    '--output-dir',
    type=click.Path(path_type=Path),
    required=True, 
    help='Output directory for processed data'
)
@click.option(
    '--config',
    type=click.Path(exists=True, path_type=Path),
    default='configs/data/preprocessing_config.yaml',
    help='Path to preprocessing configuration file'
)
@click.option(
    '--content-types',
    multiple=True,
    default=['story', 'vocabulary', 'qa', 'dialogue'],
    help='Content types to generate prompts for'
)
@click.option(
    '--age-groups', 
    multiple=True,
    default=['primary', 'secondary'],
    help='Age groups to target'
)
@click.option(
    '--test-split',
    type=float,
    default=0.1,
    help='Fraction of data to use for testing'
)
@click.option(
    '--validation-split',
    type=float, 
    default=0.1,
    help='Fraction of training data to use for validation'
)
@click.option(
    '--log-level',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
    default='INFO',
    help='Logging level'
)
@click.option(
    '--log-file',
    type=click.Path(path_type=Path),
    help='Optional log file path'
)
def prepare_data(
    input: Path,
    output_dir: Path,
    config: Path,
    content_types: tuple,
    age_groups: tuple,
    test_split: float,
    validation_split: float,
    log_level: str,
    log_file: Path
):
    """Prepare parallel English-Toaripi data for training."""
    
    # Setup logging
    setup_logging(level=log_level, log_file=str(log_file) if log_file else None)
    
    logger.info("Starting data preparation")
    logger.info(f"Input file: {input}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Config file: {config}")
    
    try:
        # Load configuration
        if config.exists():
            preprocessing_config = load_config(str(config))
            logger.info(f"Loaded preprocessing config from {config}")
        else:
            logger.warning(f"Config file {config} not found, using defaults")
            preprocessing_config = {}
        
        # Create data processing config
        data_config = DataProcessingConfig(
            min_length=preprocessing_config.get('min_length', 10),
            max_length=preprocessing_config.get('max_length', 512),
            remove_duplicates=preprocessing_config.get('remove_duplicates', True),
            test_split=test_split,
            dev_split=validation_split
        )
        
        # Initialize data processor with config
        processor = DataProcessor(config=data_config)
        
        # Load parallel data
        logger.info("Loading parallel data...")
        df = processor.load_parallel_csv(str(input))
        logger.info(f"Loaded {len(df)} parallel sentences")
        
        # Clean data
        logger.info("Cleaning data...")
        df_clean = processor.clean_data(df)
        logger.info(f"After cleaning: {len(df_clean)} sentences")
        
        # Create educational prompts
        logger.info("Creating educational prompts...")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create educational prompts using the processor
        prompts_df = processor.create_educational_prompts(df_clean)
        logger.info(f"Generated {len(prompts_df)} educational prompts")
        
        # Add metadata for content types and age groups requested
        # The processor creates all types, so we can filter or add metadata
        prompts_df['age_group'] = 'primary'  # Default to primary
        
        logger.info(f"Total prompts generated: {len(prompts_df)}")
        
        # Split data
        logger.info("Splitting data into train/validation/test sets...")
        train_df, dev_df, test_df = processor.split_dataset(
            prompts_df,
            test_size=test_split,
            dev_size=validation_split
        )
        
        # Create splits dictionary for consistency with the rest of the script
        splits = {
            'train': train_df,
            'validation': dev_df,
            'test': test_df
        }
        
        # Save splits
        for split_name, split_df in splits.items():
            output_file = output_dir / f"{split_name}.csv"
            split_df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"Saved {split_name} set: {len(split_df)} samples -> {output_file}")
        
        # Save metadata
        metadata = {
            'input_file': str(input),
            'total_parallel_sentences': len(df),
            'cleaned_sentences': len(df_clean),
            'total_prompts': len(prompts_df),
            'content_types': list(content_types),
            'age_groups': list(age_groups),
            'splits': {name: len(split_df) for name, split_df in splits.items()},
            'config': preprocessing_config
        }
        
        metadata_file = output_dir / 'metadata.json'
        import json
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved metadata to {metadata_file}")
        logger.success("Data preparation completed successfully!")
        
        # Print summary
        click.echo("\n" + "="*50)
        click.echo("DATA PREPARATION SUMMARY")
        click.echo("="*50)
        click.echo(f"Input sentences: {len(df)}")
        click.echo(f"Cleaned sentences: {len(df_clean)}")
        click.echo(f"Total prompts: {len(prompts_df)}")
        click.echo(f"Training prompts: {len(splits['train'])}")
        click.echo(f"Validation prompts: {len(splits['validation'])}")
        click.echo(f"Test prompts: {len(splits['test'])}")
        click.echo(f"Output directory: {output_dir}")
        click.echo("="*50)
        
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        raise click.ClickException(f"Data preparation failed: {e}")


if __name__ == '__main__':
    prepare_data()