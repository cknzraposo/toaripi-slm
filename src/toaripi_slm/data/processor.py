"""
Data processing and loading utilities for Toaripi SLM.
Handles parallel English-Toaripi text data and educational content formatting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import csv
from sklearn.model_selection import train_test_split
from loguru import logger
import re


class DataError(Exception):
    """Data processing-related errors."""
    pass


class DataProcessor:
    """
    Process parallel English-Toaripi data for training.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize data processor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.required_columns = ["english", "toaripi"]
        self.data = None
        
    def load_parallel_data(self, data_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load parallel English-Toaripi data from CSV.
        
        Args:
            data_path: Path to CSV file with parallel data
            
        Returns:
            DataFrame with parallel text data
            
        Raises:
            DataError: If file format is invalid
        """
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise DataError(f"Data file not found: {data_path}")
        
        try:
            df = pd.read_csv(data_path, encoding='utf-8')
        except Exception as e:
            raise DataError(f"Error reading CSV file {data_path}: {e}")
        
        # Validate required columns
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            raise DataError(f"Missing required columns: {missing_cols}")
        
        # Basic data validation
        if len(df) == 0:
            raise DataError("Data file is empty")
        
        # Check for empty entries
        for col in self.required_columns:
            empty_count = df[col].isna().sum() + (df[col] == "").sum()
            if empty_count > 0:
                logger.warning(f"Found {empty_count} empty entries in column '{col}'")
        
        # Remove rows with empty required fields
        df = df.dropna(subset=self.required_columns)
        df = df[df[self.required_columns].ne("").all(axis=1)]
        
        logger.info(f"Loaded {len(df)} parallel text pairs from {data_path}")
        self.data = df
        return df
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return str(text)
        
        # Basic cleaning
        text = text.strip()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters but preserve unicode
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', text)
        
        return text
    
    def create_educational_prompts(self, df: pd.DataFrame, 
                                 prompt_templates: Optional[Dict] = None) -> List[Dict]:
        """
        Create educational prompts from parallel data.
        
        Args:
            df: DataFrame with parallel data
            prompt_templates: Optional custom prompt templates
            
        Returns:
            List of educational prompts
        """
        if prompt_templates is None:
            prompt_templates = {
                "story": "Write a simple story in Toaripi based on this English text: {english}\nToaripi reference: {toaripi}",
                "vocabulary": "Create vocabulary exercises using these words from English: {english}\nToaripi: {toaripi}",
                "comprehension": "Create reading comprehension questions for this Toaripi text: {toaripi}\nEnglish meaning: {english}",
                "dialogue": "Create a dialogue in Toaripi inspired by: {english}\nExample Toaripi: {toaripi}"
            }
        
        prompts = []
        
        for _, row in df.iterrows():
            english_text = self.clean_text(row['english'])
            toaripi_text = self.clean_text(row['toaripi'])
            
            for content_type, template in prompt_templates.items():
                prompt = {
                    "type": content_type,
                    "english": english_text,
                    "toaripi": toaripi_text,
                    "prompt": template.format(english=english_text, toaripi=toaripi_text),
                    "age_group": "primary"  # Default to primary school
                }
                prompts.append(prompt)
        
        logger.info(f"Created {len(prompts)} educational prompts")
        return prompts
    
    def split_data(self, df: pd.DataFrame, 
                   train_ratio: float = 0.8,
                   val_ratio: float = 0.1,
                   test_ratio: float = 0.1,
                   random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets.
        
        Args:
            df: Input DataFrame
            train_ratio: Training data ratio
            val_ratio: Validation data ratio 
            test_ratio: Test data ratio
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise DataError("Train, validation, and test ratios must sum to 1.0")
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df, 
            train_size=train_ratio,
            random_state=random_seed,
            shuffle=True
        )
        
        # Second split: val vs test
        if val_ratio > 0 and test_ratio > 0:
            val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
            val_df, test_df = train_test_split(
                temp_df,
                train_size=val_ratio_adjusted,
                random_state=random_seed,
                shuffle=True
            )
        elif val_ratio > 0:
            val_df, test_df = temp_df, pd.DataFrame()
        else:
            val_df, test_df = pd.DataFrame(), temp_df
        
        logger.info(f"Data split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        return train_df, val_df, test_df
    
    def save_splits(self, train_df: pd.DataFrame, 
                   val_df: pd.DataFrame,
                   test_df: pd.DataFrame,
                   output_dir: Union[str, Path]) -> None:
        """
        Save data splits to files.
        
        Args:
            train_df: Training data
            val_df: Validation data  
            test_df: Test data
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save splits
        if len(train_df) > 0:
            train_path = output_dir / "train.csv"
            train_df.to_csv(train_path, index=False, encoding='utf-8')
            logger.info(f"Saved training data: {train_path}")
        
        if len(val_df) > 0:
            val_path = output_dir / "validation.csv"
            val_df.to_csv(val_path, index=False, encoding='utf-8')
            logger.info(f"Saved validation data: {val_path}")
        
        if len(test_df) > 0:
            test_path = output_dir / "test.csv"
            test_df.to_csv(test_path, index=False, encoding='utf-8')
            logger.info(f"Saved test data: {test_path}")
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate data quality and return statistics.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        stats = {
            "total_rows": len(df),
            "columns": list(df.columns),
            "missing_data": df.isnull().sum().to_dict(),
            "avg_english_length": df['english'].str.len().mean() if 'english' in df.columns else 0,
            "avg_toaripi_length": df['toaripi'].str.len().mean() if 'toaripi' in df.columns else 0,
            "unique_english": df['english'].nunique() if 'english' in df.columns else 0,
            "unique_toaripi": df['toaripi'].nunique() if 'toaripi' in df.columns else 0,
        }
        
        # Check for duplicate pairs
        if 'english' in df.columns and 'toaripi' in df.columns:
            duplicates = df.duplicated(subset=['english', 'toaripi']).sum()
            stats["duplicate_pairs"] = duplicates
        
        # Check text length distribution
        if 'english' in df.columns:
            eng_lengths = df['english'].str.len()
            stats["english_length_stats"] = {
                "min": eng_lengths.min(),
                "max": eng_lengths.max(), 
                "median": eng_lengths.median()
            }
        
        if 'toaripi' in df.columns:
            toa_lengths = df['toaripi'].str.len()
            stats["toaripi_length_stats"] = {
                "min": toa_lengths.min(),
                "max": toa_lengths.max(),
                "median": toa_lengths.median()
            }
        
        return stats
    
    def filter_by_length(self, df: pd.DataFrame,
                        min_length: int = 5,
                        max_length: int = 500) -> pd.DataFrame:
        """
        Filter data by text length.
        
        Args:
            df: Input DataFrame
            min_length: Minimum text length
            max_length: Maximum text length
            
        Returns:
            Filtered DataFrame
        """
        original_count = len(df)
        
        # Filter by length
        mask = (
            (df['english'].str.len() >= min_length) & 
            (df['english'].str.len() <= max_length) &
            (df['toaripi'].str.len() >= min_length) &
            (df['toaripi'].str.len() <= max_length)
        )
        
        filtered_df = df[mask].copy()
        
        logger.info(f"Filtered by length: {original_count} -> {len(filtered_df)} rows")
        return filtered_df