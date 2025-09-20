"""
Data processing module for Toaripi SLM.

This module provides the DataProcessor class for handling parallel English-Toaripi
data, including alignment, preprocessing, and dataset preparation for training.
"""

import os
import re
import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
import unicodedata

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from loguru import logger


@dataclass
class DataProcessingConfig:
    """Configuration for data processing."""
    
    # Text cleaning
    min_length: int = 10
    max_length: int = 512
    remove_duplicates: bool = True
    normalize_unicode: bool = True
    strip_whitespace: bool = True
    
    # Language filtering
    filter_non_text: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    
    # Dataset splitting
    test_split: float = 0.1
    dev_split: float = 0.05
    random_seed: int = 42
    stratify_column: Optional[str] = None
    
    # Output format
    output_format: str = "csv"  # csv, json, tsv
    encoding: str = "utf-8"


class DataProcessor:
    """
    Processor for handling Toaripi-English parallel data and educational content.
    
    This class handles:
    - Loading parallel text data from various sources
    - Text cleaning and normalization
    - Data alignment and quality checking
    - Dataset splitting for training/validation/test
    - Educational prompt creation
    - Data format conversion and export
    
    Example:
        >>> processor = DataProcessor()
        >>> processor.load_parallel_csv("data/raw/toaripi_bible.csv")
        >>> processor.clean_data()
        >>> processor.create_educational_prompts()
        >>> processor.split_dataset()
        >>> processor.save_processed_data("data/processed/")
    """
    
    def __init__(self, config: Optional[DataProcessingConfig] = None):
        """
        Initialize the data processor.
        
        Args:
            config: Data processing configuration
        """
        self.config = config or DataProcessingConfig()
        self.raw_data = None
        self.processed_data = None
        self.train_data = None
        self.dev_data = None
        self.test_data = None
        
        logger.info("Initialized DataProcessor")
    
    def load_parallel_csv(
        self,
        file_path: str,
        english_col: str = "english",
        toaripi_col: str = "toaripi",
        **kwargs
    ) -> pd.DataFrame:
        """
        Load parallel text data from CSV file.
        
        Args:
            file_path: Path to CSV file
            english_col: Name of English text column
            toaripi_col: Name of Toaripi text column
            **kwargs: Additional pandas.read_csv parameters
            
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading parallel data from: {file_path}")
        
        # Load CSV
        df = pd.read_csv(file_path, encoding=self.config.encoding, **kwargs)
        
        # Validate required columns
        if english_col not in df.columns:
            raise ValueError(f"English column '{english_col}' not found. Available: {list(df.columns)}")
        if toaripi_col not in df.columns:
            raise ValueError(f"Toaripi column '{toaripi_col}' not found. Available: {list(df.columns)}")
        
        # Standardize column names
        df = df.rename(columns={english_col: 'english', toaripi_col: 'toaripi'})
        
        # Store raw data
        self.raw_data = df.copy()
        
        logger.info(f"Loaded {len(df)} parallel text pairs")
        logger.info(f"Columns: {list(df.columns)}")
        
        return df
    
    def load_parallel_json(self, file_path: str) -> pd.DataFrame:
        """
        Load parallel text data from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading parallel data from: {file_path}")
        
        with open(file_path, 'r', encoding=self.config.encoding) as f:
            data = json.load(f)
        
        # Convert to DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and 'data' in data:
            df = pd.DataFrame(data['data'])
        else:
            df = pd.DataFrame([data])
        
        # Validate structure
        if 'english' not in df.columns or 'toaripi' not in df.columns:
            raise ValueError("JSON data must contain 'english' and 'toaripi' fields")
        
        self.raw_data = df.copy()
        
        logger.info(f"Loaded {len(df)} parallel text pairs from JSON")
        return df
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize individual text string.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Unicode normalization
        if self.config.normalize_unicode:
            text = unicodedata.normalize('NFC', text)
        
        # Remove URLs
        if self.config.remove_urls:
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        if self.config.remove_emails:
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Clean whitespace
        if self.config.strip_whitespace:
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        
        # Remove control characters
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C')
        
        return text
    
    def clean_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Clean the loaded parallel data.
        
        Args:
            df: DataFrame to clean (uses self.raw_data if None)
            
        Returns:
            Cleaned DataFrame
        """
        if df is None:
            if self.raw_data is None:
                raise ValueError("No data loaded. Call load_parallel_* first.")
            df = self.raw_data.copy()
        
        logger.info("Cleaning parallel data...")
        initial_count = len(df)
        
        # Clean text columns
        df['english'] = df['english'].apply(self.clean_text)
        df['toaripi'] = df['toaripi'].apply(self.clean_text)
        
        # Filter by length
        df = df[
            (df['english'].str.len() >= self.config.min_length) & 
            (df['english'].str.len() <= self.config.max_length) &
            (df['toaripi'].str.len() >= self.config.min_length) & 
            (df['toaripi'].str.len() <= self.config.max_length)
        ]
        
        # Remove empty entries
        df = df[
            (df['english'].str.strip() != '') & 
            (df['toaripi'].str.strip() != '')
        ]
        
        # Remove duplicates
        if self.config.remove_duplicates:
            df = df.drop_duplicates(subset=['english', 'toaripi'])
        
        # Filter non-text content
        if self.config.filter_non_text:
            # Remove entries with too many numbers or special characters
            df = df[
                ~df['english'].str.contains(r'^\d+$|^[^\w\s]+$', regex=True, na=False) &
                ~df['toaripi'].str.contains(r'^\d+$|^[^\w\s]+$', regex=True, na=False)
            ]
        
        self.processed_data = df.reset_index(drop=True)
        
        logger.info(f"Cleaned data: {initial_count} -> {len(df)} pairs "
                   f"({100 * len(df) / initial_count:.1f}% retained)")
        
        return self.processed_data
    
    def create_educational_prompts(
        self,
        df: Optional[pd.DataFrame] = None,
        prompt_templates: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Create educational training prompts from parallel data.
        
        Args:
            df: DataFrame with parallel data
            prompt_templates: Custom prompt templates
            
        Returns:
            DataFrame with educational prompts
        """
        if df is None:
            if self.processed_data is None:
                raise ValueError("No processed data. Call clean_data() first.")
            df = self.processed_data.copy()
        
        logger.info("Creating educational prompts...")
        
        # Default prompt templates
        if prompt_templates is None:
            prompt_templates = {
                'story': """Create a simple story in Toaripi for primary school students.

English example: {english}
Toaripi translation: {toaripi}

Write a new educational story in Toaripi language:""",
                
                'vocabulary': """Create vocabulary exercises in Toaripi.

English reference: {english}
Toaripi reference: {toaripi}

Generate vocabulary words and examples in Toaripi:""",
                
                'qa': """Create questions and answers in Toaripi for students.

English context: {english}
Toaripi context: {toaripi}

Generate educational Q&A in Toaripi:""",
                
                'general': """Generate educational content in Toaripi language.

English reference: {english}
Toaripi translation: {toaripi}

Create content suitable for primary school students:"""
            }
        
        # Create different types of prompts
        educational_data = []
        
        for _, row in df.iterrows():
            english_text = row['english']
            toaripi_text = row['toaripi']
            
            # Create multiple prompt variations
            for prompt_type, template in prompt_templates.items():
                prompt = template.format(english=english_text, toaripi=toaripi_text)
                
                educational_data.append({
                    'prompt': prompt,
                    'completion': toaripi_text,
                    'english_reference': english_text,
                    'toaripi_reference': toaripi_text,
                    'prompt_type': prompt_type,
                    'source_index': row.name if hasattr(row, 'name') else len(educational_data)
                })
        
        # Create DataFrame
        educational_df = pd.DataFrame(educational_data)
        
        # Add combined text for training (prompt + completion)
        educational_df['text'] = educational_df['prompt'] + '\n\n' + educational_df['completion']
        
        self.processed_data = educational_df
        
        logger.info(f"Created {len(educational_df)} educational prompts "
                   f"({len(educational_df) // len(df)} variations per original pair)")
        
        return educational_df
    
    def split_dataset(
        self,
        df: Optional[pd.DataFrame] = None,
        test_size: Optional[float] = None,
        dev_size: Optional[float] = None,
        stratify_col: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train/dev/test sets.
        
        Args:
            df: DataFrame to split
            test_size: Size of test set (0.0-1.0)
            dev_size: Size of dev set (0.0-1.0)
            stratify_col: Column to use for stratified splitting
            
        Returns:
            Tuple of (train_df, dev_df, test_df)
        """
        if df is None:
            if self.processed_data is None:
                raise ValueError("No processed data available.")
            df = self.processed_data.copy()
        
        test_size = test_size or self.config.test_split
        dev_size = dev_size or self.config.dev_split
        stratify_col = stratify_col or self.config.stratify_column
        
        logger.info(f"Splitting dataset: train/dev/test = "
                   f"{1-test_size-dev_size:.1%}/{dev_size:.1%}/{test_size:.1%}")
        
        # Prepare stratification column
        stratify = df[stratify_col] if stratify_col and stratify_col in df.columns else None
        
        # First split: separate test set
        train_dev_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=self.config.random_seed,
            stratify=stratify
        )
        
        # Second split: separate dev from train
        if dev_size > 0:
            # Adjust dev size relative to remaining data
            adjusted_dev_size = dev_size / (1 - test_size)
            
            # Prepare stratification for second split
            train_dev_stratify = None
            if stratify is not None:
                train_dev_stratify = train_dev_df[stratify_col]
            
            train_df, dev_df = train_test_split(
                train_dev_df,
                test_size=adjusted_dev_size,
                random_state=self.config.random_seed,
                stratify=train_dev_stratify
            )
        else:
            train_df = train_dev_df
            dev_df = pd.DataFrame(columns=df.columns)  # Empty dev set
        
        # Store splits
        self.train_data = train_df.reset_index(drop=True)
        self.dev_data = dev_df.reset_index(drop=True)
        self.test_data = test_df.reset_index(drop=True)
        
        logger.info(f"Dataset split: Train={len(train_df)}, Dev={len(dev_df)}, Test={len(test_df)}")
        
        return self.train_data, self.dev_data, self.test_data
    
    def analyze_data(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Analyze the dataset and return statistics.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary of statistics
        """
        if df is None:
            if self.processed_data is None:
                raise ValueError("No data to analyze.")
            df = self.processed_data
        
        logger.info("Analyzing dataset...")
        
        stats = {
            'total_pairs': len(df),
            'english_stats': {
                'avg_length': df['english'].str.len().mean(),
                'max_length': df['english'].str.len().max(),
                'min_length': df['english'].str.len().min(),
                'total_words': df['english'].str.split().str.len().sum()
            },
            'toaripi_stats': {
                'avg_length': df['toaripi'].str.len().mean(),
                'max_length': df['toaripi'].str.len().max(),
                'min_length': df['toaripi'].str.len().min(),
                'total_words': df['toaripi'].str.split().str.len().sum()
            }
        }
        
        # Additional analysis if columns exist
        if 'prompt_type' in df.columns:
            stats['prompt_types'] = df['prompt_type'].value_counts().to_dict()
        
        if 'book' in df.columns:
            stats['books'] = df['book'].value_counts().to_dict()
        
        # Language diversity metrics
        stats['english_unique_words'] = len(set(' '.join(df['english']).split()))
        stats['toaripi_unique_words'] = len(set(' '.join(df['toaripi']).split()))
        
        logger.info(f"Analysis complete: {stats['total_pairs']} pairs, "
                   f"{stats['english_unique_words']} unique English words, "
                   f"{stats['toaripi_unique_words']} unique Toaripi words")
        
        return stats
    
    def save_processed_data(
        self,
        output_dir: str,
        save_splits: bool = True,
        save_analysis: bool = True
    ) -> None:
        """
        Save processed data to files.
        
        Args:
            output_dir: Output directory
            save_splits: Whether to save train/dev/test splits
            save_analysis: Whether to save analysis report
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving processed data to: {output_dir}")
        
        # Save main processed dataset
        if self.processed_data is not None:
            if self.config.output_format == 'csv':
                self.processed_data.to_csv(
                    output_path / "processed_data.csv",
                    index=False,
                    encoding=self.config.encoding
                )
            elif self.config.output_format == 'json':
                self.processed_data.to_json(
                    output_path / "processed_data.json",
                    orient='records',
                    indent=2
                )
            elif self.config.output_format == 'tsv':
                self.processed_data.to_csv(
                    output_path / "processed_data.tsv",
                    sep='\t',
                    index=False,
                    encoding=self.config.encoding
                )
        
        # Save dataset splits
        if save_splits and all([self.train_data is not None, self.test_data is not None]):
            for split_name, split_data in [
                ('train', self.train_data),
                ('dev', self.dev_data),
                ('test', self.test_data)
            ]:
                if len(split_data) > 0:
                    if self.config.output_format == 'csv':
                        split_data.to_csv(
                            output_path / f"{split_name}.csv",
                            index=False,
                            encoding=self.config.encoding
                        )
                    elif self.config.output_format == 'json':
                        split_data.to_json(
                            output_path / f"{split_name}.json",
                            orient='records',
                            indent=2
                        )
        
        # Save analysis report
        if save_analysis:
            try:
                analysis = self.analyze_data()
                with open(output_path / "analysis_report.json", 'w') as f:
                    json.dump(analysis, f, indent=2, default=str)
            except Exception as e:
                logger.warning(f"Failed to save analysis report: {e}")
        
        # Save configuration
        config_dict = {
            'min_length': self.config.min_length,
            'max_length': self.config.max_length,
            'test_split': self.config.test_split,
            'dev_split': self.config.dev_split,
            'output_format': self.config.output_format,
            'encoding': self.config.encoding
        }
        
        with open(output_path / "processing_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info("Data saving completed")
    
    def validate_alignment(self, df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Validate the quality of English-Toaripi alignment.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary of validation metrics
        """
        if df is None:
            df = self.processed_data or self.raw_data
        
        if df is None:
            raise ValueError("No data available for validation.")
        
        logger.info("Validating data alignment...")
        
        metrics = {}
        
        # Length ratio analysis
        df['length_ratio'] = df['toaripi'].str.len() / df['english'].str.len()
        metrics['avg_length_ratio'] = df['length_ratio'].mean()
        metrics['length_ratio_std'] = df['length_ratio'].std()
        
        # Word count ratio
        df['word_ratio'] = (df['toaripi'].str.split().str.len() / 
                           df['english'].str.split().str.len())
        metrics['avg_word_ratio'] = df['word_ratio'].mean()
        
        # Quality indicators
        metrics['empty_pairs'] = ((df['english'].str.strip() == '') | 
                                 (df['toaripi'].str.strip() == '')).sum()
        
        metrics['very_short_pairs'] = ((df['english'].str.len() < 5) | 
                                      (df['toaripi'].str.len() < 5)).sum()
        
        metrics['very_long_pairs'] = ((df['english'].str.len() > 500) | 
                                     (df['toaripi'].str.len() > 500)).sum()
        
        # Suspicious patterns
        metrics['number_only_pairs'] = (df['english'].str.match(r'^\d+$').fillna(False) | 
                                       df['toaripi'].str.match(r'^\d+$').fillna(False)).sum()
        
        logger.info(f"Validation complete. Average length ratio: {metrics['avg_length_ratio']:.2f}")
        
        return metrics
    
    @classmethod
    def from_config_file(cls, config_path: str) -> "DataProcessor":
        """
        Create DataProcessor from configuration file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configured DataProcessor instance
        """
        import yaml
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Handle nested config structure
        if 'preprocessing' in config_dict:
            config_dict = config_dict['preprocessing']
        
        config = DataProcessingConfig(**config_dict)
        return cls(config)
    
    def __repr__(self) -> str:
        data_info = f"data={len(self.processed_data) if self.processed_data is not None else 'None'}"
        return f"DataProcessor({data_info})"