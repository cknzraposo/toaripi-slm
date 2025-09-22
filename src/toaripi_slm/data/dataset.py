"""
Custom dataset classes for Toaripi parallel text data.

This module implements defensive dataset loading and processing
for educational content generation in Toaripi language.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
import json
from dataclasses import dataclass

from .prompts import create_educational_prompt, ContentType, AgeGroup

logger = logging.getLogger(__name__)


@dataclass
class DataSample:
    """Defensive data sample representation."""
    english: str
    toaripi: str
    prompt: str
    target: str
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate sample after initialization."""
        if not self.english or not self.english.strip():
            raise ValueError("English text cannot be empty")
        if not self.toaripi or not self.toaripi.strip():
            raise ValueError("Toaripi text cannot be empty")
        if not self.prompt or not self.prompt.strip():
            raise ValueError("Prompt cannot be empty")
        if not self.target or not self.target.strip():
            raise ValueError("Target cannot be empty")
    
    def validate_content(self) -> bool:
        """Validate educational appropriateness of content."""
        inappropriate_patterns = [
            'violence', 'inappropriate', 'adult', 'harmful'
        ]
        
        all_text = f"{self.english} {self.toaripi} {self.prompt} {self.target}".lower()
        
        for pattern in inappropriate_patterns:
            if pattern in all_text:
                logger.warning(f"Potentially inappropriate content detected: {pattern}")
                return False
        
        return True


class ToaripiParallelDataset(Dataset):
    """
    Defensive dataset class for Toaripi parallel text data.
    
    Implements comprehensive validation and educational content formatting
    for safe and effective training.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        content_type: ContentType = ContentType.STORY,
        age_group: AgeGroup = AgeGroup.PRIMARY_MIDDLE,
        validation_split: Optional[float] = None,
        is_validation: bool = False,
        filter_inappropriate: bool = True
    ):
        """
        Initialize dataset with defensive validation.
        
        Args:
            data_path: Path to CSV file with parallel data
            tokenizer: Pre-trained tokenizer for text encoding
            max_length: Maximum sequence length for tokenization
            content_type: Type of educational content to generate
            age_group: Target age group for content
            validation_split: Fraction of data to use for validation
            is_validation: Whether this is the validation split
            filter_inappropriate: Whether to filter inappropriate content
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.content_type = content_type
        self.age_group = age_group
        self.filter_inappropriate = filter_inappropriate
        
        # Validation
        self._validate_inputs()
        
        # Load and process data
        self.samples = self._load_and_process_data(validation_split, is_validation)
        
        logger.info(f"Loaded {len(self.samples)} samples for {'validation' if is_validation else 'training'}")
    
    def _validate_inputs(self):
        """Validate constructor inputs."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        if not self.data_path.suffix.lower() == '.csv':
            raise ValueError(f"Expected CSV file, got: {self.data_path.suffix}")
        
        if self.max_length <= 0 or self.max_length > 4096:
            raise ValueError(f"max_length must be between 1 and 4096, got: {self.max_length}")
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer cannot be None")
        
        # Test tokenizer
        try:
            test_encoding = self.tokenizer("Test", return_tensors="pt", max_length=10, truncation=True)
            if 'input_ids' not in test_encoding:
                raise ValueError("Tokenizer does not return input_ids")
        except Exception as e:
            raise ValueError(f"Tokenizer validation failed: {e}")
    
    def _load_and_process_data(
        self, 
        validation_split: Optional[float], 
        is_validation: bool
    ) -> List[DataSample]:
        """Load and process data with defensive validation."""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(self.data_path, encoding=encoding)
                    logger.info(f"Successfully loaded data with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError(f"Could not read CSV file with any encoding")
            
            # Validate required columns
            required_columns = {'english', 'toaripi'}
            if not required_columns.issubset(df.columns):
                missing = required_columns - set(df.columns)
                raise ValueError(f"Missing required columns: {missing}")
            
            # Clean data
            df = self._clean_dataframe(df)
            
            # Split data if needed
            if validation_split is not None:
                df = self._split_data(df, validation_split, is_validation)
            
            # Create samples with defensive processing
            samples = []
            failed_samples = 0
            
            for idx, row in df.iterrows():
                try:
                    sample = self._create_sample(row, idx)
                    if sample and (not self.filter_inappropriate or sample.validate_content()):
                        samples.append(sample)
                    else:
                        failed_samples += 1
                except Exception as e:
                    logger.warning(f"Failed to create sample {idx}: {e}")
                    failed_samples += 1
            
            if failed_samples > 0:
                logger.warning(f"Failed to process {failed_samples} samples")
            
            if len(samples) == 0:
                raise ValueError("No valid samples found in dataset")
            
            return samples
            
        except Exception as e:
            logger.error(f"Failed to load data from {self.data_path}: {e}")
            raise
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataframe with defensive validation."""
        initial_count = len(df)
        
        # Remove rows with empty values
        df = df.dropna(subset=['english', 'toaripi'])
        df = df[df['english'].str.strip() != '']
        df = df[df['toaripi'].str.strip() != '']
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['english', 'toaripi'])
        
        # Length filtering
        min_length = 3
        max_length = 1000
        
        df = df[
            (df['english'].str.len() >= min_length) & 
            (df['english'].str.len() <= max_length) &
            (df['toaripi'].str.len() >= min_length) & 
            (df['toaripi'].str.len() <= max_length)
        ]
        
        final_count = len(df)
        logger.info(f"Cleaned data: {initial_count} → {final_count} rows")
        
        return df.reset_index(drop=True)
    
    def _split_data(
        self, 
        df: pd.DataFrame, 
        validation_split: float, 
        is_validation: bool
    ) -> pd.DataFrame:
        """Split data for training/validation."""
        if not 0 < validation_split < 1:
            raise ValueError(f"validation_split must be between 0 and 1, got: {validation_split}")
        
        # Deterministic split
        split_idx = int(len(df) * (1 - validation_split))
        
        if is_validation:
            return df.iloc[split_idx:].reset_index(drop=True)
        else:
            return df.iloc[:split_idx].reset_index(drop=True)
    
    def _create_sample(self, row: pd.Series, idx: int) -> Optional[DataSample]:
        """Create a data sample with defensive validation."""
        try:
            english_text = str(row['english']).strip()
            toaripi_text = str(row['toaripi']).strip()
            
            # Create educational prompt
            prompt = create_educational_prompt(
                english_text=english_text,
                toaripi_text=toaripi_text,
                content_type=self.content_type,
                age_group=self.age_group,
                topic="daily life",
                length="2-3"
            )
            
            # Target is the Toaripi text for generation
            target = toaripi_text
            
            # Metadata
            metadata = {
                'source_index': idx,
                'english_length': len(english_text),
                'toaripi_length': len(toaripi_text),
                'content_type': self.content_type.value,
                'age_group': self.age_group.value
            }
            
            # Add additional metadata if available
            for col in ['book_id', 'chapter_id', 'verse_id']:
                if col in row and pd.notna(row[col]):
                    metadata[col] = row[col]
            
            return DataSample(
                english=english_text,
                toaripi=toaripi_text,
                prompt=prompt,
                target=target,
                metadata=metadata
            )
            
        except Exception as e:
            logger.warning(f"Failed to create sample from row {idx}: {e}")
            return None
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get tokenized sample with defensive validation.
        
        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")
        
        sample = self.samples[idx]
        
        try:
            # Tokenize prompt and target
            prompt_encoding = self.tokenizer(
                sample.prompt,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            target_encoding = self.tokenizer(
                sample.target,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Create labels (target tokens for generation)
            labels = target_encoding['input_ids'].clone()
            
            # Set padding tokens to -100 (ignored in loss calculation)
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            return {
                'input_ids': prompt_encoding['input_ids'].squeeze(0),
                'attention_mask': prompt_encoding['attention_mask'].squeeze(0),
                'labels': labels.squeeze(0),
                'metadata': sample.metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to tokenize sample {idx}: {e}")
            # Return a safe fallback
            return self._create_fallback_sample()
    
    def _create_fallback_sample(self) -> Dict[str, torch.Tensor]:
        """Create a safe fallback sample for error cases."""
        fallback_text = "Generate educational content in Toaripi."
        
        encoding = self.tokenizer(
            fallback_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0),
            'metadata': {'fallback': True}
        }
    
    def get_sample_metadata(self, idx: int) -> Dict[str, Any]:
        """Get metadata for a specific sample."""
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range")
        
        return self.samples[idx].metadata or {}
    
    def get_content_statistics(self) -> Dict[str, Any]:
        """Get statistics about the dataset content."""
        if not self.samples:
            return {}
        
        english_lengths = [len(sample.english) for sample in self.samples]
        toaripi_lengths = [len(sample.toaripi) for sample in self.samples]
        
        return {
            'total_samples': len(self.samples),
            'content_type': self.content_type.value,
            'age_group': self.age_group.value,
            'english_length_stats': {
                'min': min(english_lengths),
                'max': max(english_lengths),
                'avg': sum(english_lengths) / len(english_lengths)
            },
            'toaripi_length_stats': {
                'min': min(toaripi_lengths),
                'max': max(toaripi_lengths),
                'avg': sum(toaripi_lengths) / len(toaripi_lengths)
            }
        }
    
    def save_statistics(self, output_path: str):
        """Save dataset statistics to JSON file."""
        try:
            stats = self.get_content_statistics()
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            logger.info(f"Dataset statistics saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save statistics: {e}")


def create_dataloaders(
    train_dataset: ToaripiParallelDataset,
    val_dataset: Optional[ToaripiParallelDataset] = None,
    batch_size: int = 4,
    num_workers: int = 0,
    shuffle_train: bool = True
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create data loaders with defensive validation.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Optional validation dataset
        batch_size: Batch size for training
        num_workers: Number of worker processes
        shuffle_train: Whether to shuffle training data
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got: {batch_size}")
    
    if num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got: {num_workers}")
    
    try:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True  # Avoid issues with batch size variations
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False
            )
        
        logger.info(f"Created data loaders: train_size={len(train_dataset)}, val_size={len(val_dataset) if val_dataset else 0}")
        
        return train_loader, val_loader
        
    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        raise


# Export main interfaces
__all__ = [
    "DataSample",
    "ToaripiParallelDataset", 
    "create_dataloaders"
]