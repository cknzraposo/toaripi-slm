"""
Data preprocessing utilities for Toaripi SLM.

This module provides defensive data cleaning and validation utilities
for educational content processing.
"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class TextCleaner:
    """Defensive text cleaning utilities for educational content."""
    
    def __init__(self):
        """Initialize text cleaner with educational content rules."""
        # Patterns for inappropriate content detection
        self.inappropriate_patterns = [
            r'\b(violence|violent|fight|kill|death|blood|weapon|gun|knife)\b',
            r'\b(alcohol|drug|beer|wine|smoke|smoking)\b',
            r'\b(hate|stupid|idiot|damn|hell)\b',
            r'\b(adult|mature|inappropriate)\b'
        ]
        
        # Patterns for text normalization
        self.normalization_patterns = [
            (r'\s+', ' '),  # Multiple spaces to single space
            (r'["""]', '"'),  # Normalize quotes
            (r"[''']", "'"),  # Normalize apostrophes
            (r'^\s+|\s+$', ''),  # Strip leading/trailing whitespace
        ]
        
        # Educational content keywords (positive indicators)
        self.educational_keywords = [
            'learn', 'teach', 'school', 'student', 'education',
            'family', 'friend', 'home', 'village', 'community',
            'help', 'share', 'kind', 'good', 'happy', 'safe'
        ]
    
    def clean_text(self, text: str) -> str:
        """
        Clean text with defensive validation.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
            
        Raises:
            ValueError: If text is invalid or inappropriate
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        
        # Normalize text
        cleaned_text = text
        for pattern, replacement in self.normalization_patterns:
            cleaned_text = re.sub(pattern, replacement, cleaned_text)
        
        # Validate content appropriateness
        if not self.is_appropriate_content(cleaned_text):
            raise ValueError("Text contains inappropriate content for educational use")
        
        # Final validation
        if len(cleaned_text.strip()) == 0:
            raise ValueError("Text becomes empty after cleaning")
        
        return cleaned_text.strip()
    
    def is_appropriate_content(self, text: str) -> bool:
        """
        Check if content is appropriate for educational use.
        
        Args:
            text: Text to validate
            
        Returns:
            True if content is appropriate
        """
        if not text:
            return False
        
        text_lower = text.lower()
        
        # Check for inappropriate patterns
        for pattern in self.inappropriate_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                logger.warning(f"Inappropriate content detected: {pattern}")
                return False
        
        return True
    
    def calculate_educational_score(self, text: str) -> float:
        """
        Calculate educational value score for text.
        
        Args:
            text: Text to evaluate
            
        Returns:
            Score between 0 and 1 (higher is better)
        """
        if not text:
            return 0.0
        
        text_lower = text.lower()
        words = text_lower.split()
        
        if len(words) == 0:
            return 0.0
        
        # Count educational keywords
        educational_count = sum(1 for word in words 
                              if any(keyword in word for keyword in self.educational_keywords))
        
        # Basic score calculation
        score = min(educational_count / len(words) * 2, 1.0)
        
        # Bonus for appropriate length
        if 10 <= len(words) <= 100:
            score += 0.1
        
        # Penalty for very short or very long texts
        if len(words) < 3 or len(words) > 200:
            score = max(score - 0.3, 0.0)
        
        return min(score, 1.0)


class DataValidator:
    """Defensive data validation for parallel text datasets."""
    
    def __init__(self, min_length: int = 3, max_length: int = 1000):
        """
        Initialize validator with defensive parameters.
        
        Args:
            min_length: Minimum text length
            max_length: Maximum text length
        """
        if min_length <= 0:
            raise ValueError("min_length must be positive")
        if max_length <= min_length:
            raise ValueError("max_length must be greater than min_length")
        
        self.min_length = min_length
        self.max_length = max_length
        self.text_cleaner = TextCleaner()
    
    def validate_parallel_pair(self, english: str, toaripi: str) -> Tuple[bool, List[str]]:
        """
        Validate a parallel text pair with defensive checks.
        
        Args:
            english: English text
            toaripi: Toaripi text
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Basic validation
        if not english or not isinstance(english, str):
            issues.append("English text is invalid")
        if not toaripi or not isinstance(toaripi, str):
            issues.append("Toaripi text is invalid")
        
        if issues:
            return False, issues
        
        # Length validation
        if len(english.strip()) < self.min_length:
            issues.append(f"English text too short: {len(english)} < {self.min_length}")
        if len(toaripi.strip()) < self.min_length:
            issues.append(f"Toaripi text too short: {len(toaripi)} < {self.min_length}")
        
        if len(english) > self.max_length:
            issues.append(f"English text too long: {len(english)} > {self.max_length}")
        if len(toaripi) > self.max_length:
            issues.append(f"Toaripi text too long: {len(toaripi)} > {self.max_length}")
        
        # Content appropriateness
        if not self.text_cleaner.is_appropriate_content(english):
            issues.append("English text contains inappropriate content")
        if not self.text_cleaner.is_appropriate_content(toaripi):
            issues.append("Toaripi text contains inappropriate content")
        
        # Educational value check
        en_score = self.text_cleaner.calculate_educational_score(english)
        to_score = self.text_cleaner.calculate_educational_score(toaripi)
        
        if en_score < 0.1:
            issues.append(f"English text has low educational value: {en_score:.2f}")
        if to_score < 0.1:
            issues.append(f"Toaripi text has low educational value: {to_score:.2f}")
        
        return len(issues) == 0, issues
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate entire dataset with comprehensive checks.
        
        Args:
            df: DataFrame with 'english' and 'toaripi' columns
            
        Returns:
            Validation report dictionary
        """
        if df is None or df.empty:
            return {
                'is_valid': False,
                'total_rows': 0,
                'valid_rows': 0,
                'issues': ['Dataset is empty']
            }
        
        # Check required columns
        required_cols = ['english', 'toaripi']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return {
                'is_valid': False,
                'total_rows': len(df),
                'valid_rows': 0,
                'issues': [f'Missing required columns: {missing_cols}']
            }
        
        total_rows = len(df)
        valid_rows = 0
        all_issues = []
        
        # Validate each row
        for idx, row in df.iterrows():
            try:
                english = str(row['english']) if pd.notna(row['english']) else ""
                toaripi = str(row['toaripi']) if pd.notna(row['toaripi']) else ""
                
                is_valid, issues = self.validate_parallel_pair(english, toaripi)
                
                if is_valid:
                    valid_rows += 1
                else:
                    all_issues.extend([f"Row {idx}: {issue}" for issue in issues])
                    
            except Exception as e:
                all_issues.append(f"Row {idx}: Validation error - {e}")
        
        # Calculate statistics
        validity_ratio = valid_rows / total_rows if total_rows > 0 else 0.0
        
        return {
            'is_valid': validity_ratio >= 0.5,  # At least 50% valid rows
            'total_rows': total_rows,
            'valid_rows': valid_rows,
            'validity_ratio': validity_ratio,
            'issues': all_issues[:100],  # Limit issues for readability
            'issue_count': len(all_issues)
        }


def preprocess_dataset(
    input_path: str,
    output_path: str,
    min_length: int = 3,
    max_length: int = 1000,
    remove_duplicates: bool = True,
    educational_threshold: float = 0.1
) -> Dict[str, Any]:
    """
    Preprocess dataset with comprehensive validation and cleaning.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output processed CSV file
        min_length: Minimum text length
        max_length: Maximum text length
        remove_duplicates: Whether to remove duplicate pairs
        educational_threshold: Minimum educational value score
        
    Returns:
        Processing report dictionary
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If processing fails
    """
    try:
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load data with encoding detection
        encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(input_path, encoding=encoding)
                logger.info(f"Loaded data with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise ValueError("Could not read input file with any encoding")
        
        initial_count = len(df)
        logger.info(f"Initial dataset size: {initial_count} rows")
        
        # Initialize validator and cleaner
        validator = DataValidator(min_length, max_length)
        cleaner = TextCleaner()
        
        # Clean and validate data
        valid_rows = []
        
        for idx, row in df.iterrows():
            try:
                english = str(row['english']) if pd.notna(row['english']) else ""
                toaripi = str(row['toaripi']) if pd.notna(row['toaripi']) else ""
                
                # Clean texts
                english_clean = cleaner.clean_text(english)
                toaripi_clean = cleaner.clean_text(toaripi)
                
                # Validate pair
                is_valid, issues = validator.validate_parallel_pair(english_clean, toaripi_clean)
                
                if is_valid:
                    # Check educational value
                    en_score = cleaner.calculate_educational_score(english_clean)
                    to_score = cleaner.calculate_educational_score(toaripi_clean)
                    
                    if en_score >= educational_threshold and to_score >= educational_threshold:
                        # Create cleaned row
                        clean_row = row.copy()
                        clean_row['english'] = english_clean
                        clean_row['toaripi'] = toaripi_clean
                        valid_rows.append(clean_row)
                
            except Exception as e:
                logger.warning(f"Failed to process row {idx}: {e}")
                continue
        
        # Create cleaned DataFrame
        if not valid_rows:
            raise ValueError("No valid rows found after processing")
        
        df_clean = pd.DataFrame(valid_rows)
        
        # Remove duplicates if requested
        if remove_duplicates:
            df_clean = df_clean.drop_duplicates(subset=['english', 'toaripi'])
        
        final_count = len(df_clean)
        
        # Save processed data
        df_clean.to_csv(output_path, index=False, encoding='utf-8')
        
        # Generate report
        report = {
            'input_path': str(input_path),
            'output_path': str(output_path),
            'initial_rows': initial_count,
            'final_rows': final_count,
            'removed_rows': initial_count - final_count,
            'removal_ratio': (initial_count - final_count) / initial_count if initial_count > 0 else 0,
            'processing_success': True,
            'parameters': {
                'min_length': min_length,
                'max_length': max_length,
                'remove_duplicates': remove_duplicates,
                'educational_threshold': educational_threshold
            }
        }
        
        logger.info(f"Processing complete: {initial_count} → {final_count} rows")
        return report
        
    except Exception as e:
        logger.error(f"Dataset preprocessing failed: {e}")
        raise


# Export main interfaces
__all__ = [
    "TextCleaner",
    "DataValidator", 
    "ToaripiPreprocessor",
    "preprocess_dataset"
]


class ToaripiPreprocessor:
    """Main preprocessor class for Toaripi educational content."""
    
    def __init__(self, min_length: int = 10, max_length: int = 500, 
                 validate_cultural: bool = True):
        """
        Initialize the Toaripi preprocessor.
        
        Args:
            min_length: Minimum text length
            max_length: Maximum text length
            validate_cultural: Enable cultural appropriateness validation
        """
        self.min_length = min_length
        self.max_length = max_length
        self.validate_cultural = validate_cultural
        
        self.text_cleaner = TextCleaner()
        self.data_validator = DataValidator(min_length, max_length)
    
    def load_parallel_data(self, file_path: Path) -> 'Dataset':
        """
        Load parallel data from file and create a Dataset object.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Dataset object with loaded data
        """
        from ..models.dataset import Dataset, ParallelText
        from ..models.enums import ContentType, AgeGroup
        from ..utils.helpers import get_file_hash
        import pandas as pd
        
        # Load data
        df = pd.read_csv(file_path)
        
        # Validate structure
        if 'english' not in df.columns or 'toaripi' not in df.columns:
            raise ValueError("Data file must contain 'english' and 'toaripi' columns")
        
        # Create parallel texts
        parallel_texts = []
        for _, row in df.iterrows():
            if pd.notna(row['english']) and pd.notna(row['toaripi']):
                parallel_text = ParallelText(
                    english=str(row['english']),
                    toaripi=str(row['toaripi']),
                    content_type=ContentType.STORY,  # Default
                    age_group=AgeGroup.PRIMARY_LOWER,  # Default
                    metadata={}
                )
                parallel_texts.append(parallel_text)
        
        # Calculate file info
        file_size_mb = file_path.stat().st_size / (1024 * 1024)  # Convert to MB
        file_hash = get_file_hash(file_path)
        
        # Create dataset
        dataset = Dataset(
            dataset_id=file_hash[:16],  # Use first 16 chars of hash as ID
            name=file_path.stem,
            file_path=file_path,
            file_hash=file_hash,
            file_size_mb=file_size_mb,
            total_samples=len(parallel_texts),
            parallel_texts=parallel_texts,
            metadata={
                "source_file": str(file_path),
                "loaded_rows": len(parallel_texts),
                "original_rows": len(df)
            }
        )
        
        return dataset
    
    def prepare_educational_dataset(self, dataset: 'Dataset', 
                                  content_type: 'ContentType', 
                                  age_group: 'AgeGroup') -> 'Dataset':
        """
        Prepare dataset for specific educational content and age group.
        
        Args:
            dataset: Input dataset
            content_type: Target content type
            age_group: Target age group
            
        Returns:
            Processed dataset
        """
        from ..models.dataset import Dataset, ParallelText
        
        processed_texts = []
        
        for parallel_text in dataset.parallel_texts:
            try:
                # Clean texts
                english_clean = self.text_cleaner.clean_text(parallel_text.english)
                toaripi_clean = self.text_cleaner.clean_text(parallel_text.toaripi)
                
                # Validate pair
                is_valid, issues = self.data_validator.validate_parallel_pair(
                    english_clean, toaripi_clean
                )
                
                if is_valid:
                    # Create new parallel text with target settings
                    new_parallel_text = ParallelText(
                        english=english_clean,
                        toaripi=toaripi_clean,
                        content_type=content_type,
                        age_group=age_group,
                        metadata=parallel_text.metadata.copy()
                    )
                    processed_texts.append(new_parallel_text)
                    
            except Exception as e:
                logger.warning(f"Failed to process parallel text: {e}")
                continue
        
        # Create processed dataset
        processed_dataset = Dataset(
            dataset_id=f"{dataset.dataset_id}_{content_type.value}_{age_group.value}",
            name=f"{dataset.name}_{content_type.value}_{age_group.value}",
            file_path=dataset.file_path,  # Same source file
            file_hash=dataset.file_hash,
            file_size_mb=dataset.file_size_mb,
            total_samples=len(processed_texts),
            parallel_texts=processed_texts,
            metadata={
                **dataset.metadata,
                "content_type": content_type.value,
                "age_group": age_group.value,
                "processed_count": len(processed_texts),
                "original_count": len(dataset.parallel_texts)
            }
        )
        
        return processed_dataset
    
    def validate_data_quality(self, dataset: 'Dataset') -> Dict[str, Any]:
        """
        Validate overall data quality of a dataset.
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            Validation results dictionary
        """
        import pandas as pd
        
        # Convert to DataFrame for validation
        data = []
        for pt in dataset.parallel_texts:
            data.append({
                'english': pt.english,
                'toaripi': pt.toaripi
            })
        
        df = pd.DataFrame(data)
        return self.data_validator.validate_dataset(df)
    
    def check_cultural_appropriateness(self, dataset: 'Dataset') -> Dict[str, Any]:
        """
        Check cultural appropriateness of dataset content.
        
        Args:
            dataset: Dataset to check
            
        Returns:
            Cultural appropriateness results
        """
        issues = []
        passed_count = 0
        
        for i, pt in enumerate(dataset.parallel_texts):
            # Check both English and Toaripi texts
            if not self.text_cleaner.is_appropriate_content(pt.english):
                issues.append(f"Text {i}: English content inappropriate")
            elif not self.text_cleaner.is_appropriate_content(pt.toaripi):
                issues.append(f"Text {i}: Toaripi content inappropriate")
            else:
                passed_count += 1
        
        total_count = len(dataset.parallel_texts)
        passed_ratio = passed_count / total_count if total_count > 0 else 0
        
        return {
            "passed": passed_ratio >= 0.9,  # 90% threshold
            "passed_count": passed_count,
            "total_count": total_count,
            "passed_ratio": passed_ratio,
            "issues": issues[:20]  # Limit issues
        }
    
    def check_educational_suitability(self, dataset: 'Dataset') -> Dict[str, Any]:
        """
        Check educational suitability of dataset content.
        
        Args:
            dataset: Dataset to check
            
        Returns:
            Educational suitability results
        """
        issues = []
        suitable_count = 0
        
        for i, pt in enumerate(dataset.parallel_texts):
            # Calculate educational scores
            en_score = self.text_cleaner.calculate_educational_score(pt.english)
            to_score = self.text_cleaner.calculate_educational_score(pt.toaripi)
            
            if en_score < 0.1:
                issues.append(f"Text {i}: English educational value too low ({en_score:.2f})")
            elif to_score < 0.1:
                issues.append(f"Text {i}: Toaripi educational value too low ({to_score:.2f})")
            else:
                suitable_count += 1
        
        total_count = len(dataset.parallel_texts)
        suitable_ratio = suitable_count / total_count if total_count > 0 else 0
        
        return {
            "passed": suitable_ratio >= 0.7,  # 70% threshold
            "suitable_count": suitable_count,
            "total_count": total_count,
            "suitable_ratio": suitable_ratio,
            "issues": issues[:20]  # Limit issues
        }