#!/usr/bin/env python3
"""
Validate the processed Toaripi dataset for SLM training quality.

This script performs comprehensive validation of the processed data to ensure
it meets the requirements for educational content generation and SLM training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
import re
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ToaripiDataValidator:
    """Validator for Toaripi SLM training data."""
    
    def __init__(self):
        self.validation_results = {}
        
    def validate_text_quality(self, df: pd.DataFrame) -> dict:
        """Validate text quality metrics."""
        logger.info("Validating text quality...")
        
        results = {}
        
        # Length distribution
        lengths = df['toaripi'].str.len()
        results['length_stats'] = {
            'mean': float(lengths.mean()),
            'median': float(lengths.median()),
            'std': float(lengths.std()),
            'min': int(lengths.min()),
            'max': int(lengths.max()),
            'within_range': int(((lengths >= 10) & (lengths <= 512)).sum())
        }
        
        # Word count distribution
        word_counts = df['toaripi'].str.split().str.len()
        results['word_stats'] = {
            'mean_words': float(word_counts.mean()),
            'median_words': float(word_counts.median()),
            'min_words': int(word_counts.min()),
            'max_words': int(word_counts.max())
        }
        
        # Character diversity
        all_text = ' '.join(df['toaripi'])
        unique_chars = set(all_text)
        results['character_diversity'] = {
            'unique_characters': len(unique_chars),
            'total_characters': len(all_text),
            'diversity_ratio': len(unique_chars) / len(all_text)
        }
        
        return results
    
    def validate_toaripi_language_features(self, df: pd.DataFrame) -> dict:
        """Validate Toaripi language-specific features."""
        logger.info("Validating Toaripi language features...")
        
        results = {}
        
        # Common Toaripi words and patterns
        toaripi_indicators = [
            r'\b(Ualare|Iehova|karu|vita|soka|lei|voa|reha)\b',
            r'\b(foromai|eavia|leipe|meiape|la|sa|ita)\b',
            r'\b(arero|aeata|kofa|eta|tau|firu)\b'
        ]
        
        # Count verses with Toaripi indicators
        toaripi_verse_count = 0
        for _, row in df.iterrows():
            text = row['toaripi'].lower()
            has_toaripi = any(re.search(pattern, text, re.IGNORECASE) 
                            for pattern in toaripi_indicators)
            if has_toaripi:
                toaripi_verse_count += 1
        
        results['language_authenticity'] = {
            'verses_with_toaripi_features': toaripi_verse_count,
            'percentage_authentic': (toaripi_verse_count / len(df)) * 100,
            'total_verses': len(df)
        }
        
        # Word frequency analysis
        all_words = []
        for text in df['toaripi']:
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        most_common = word_freq.most_common(20)
        
        results['vocabulary_stats'] = {
            'total_words': len(all_words),
            'unique_words': len(word_freq),
            'vocabulary_richness': len(word_freq) / len(all_words),
            'most_common_words': most_common
        }
        
        return results
    
    def validate_educational_suitability(self, df: pd.DataFrame) -> dict:
        """Validate suitability for educational content generation."""
        logger.info("Validating educational suitability...")
        
        results = {}
        
        # Check for appropriate content length for primary education
        suitable_length = df['toaripi'].str.len().between(20, 300)
        results['educational_length'] = {
            'suitable_for_primary': int(suitable_length.sum()),
            'percentage_suitable': (suitable_length.sum() / len(df)) * 100
        }
        
        # Check sentence structure (simple sentences are better for education)
        sentence_complexity = []
        for text in df['toaripi']:
            # Count sentences (periods, exclamations, questions)
            sentences = len(re.findall(r'[.!?]+', text))
            words = len(text.split())
            if sentences > 0:
                words_per_sentence = words / sentences
                sentence_complexity.append(words_per_sentence)
        
        if sentence_complexity:
            results['sentence_complexity'] = {
                'avg_words_per_sentence': float(np.mean(sentence_complexity)),
                'median_words_per_sentence': float(np.median(sentence_complexity)),
                'simple_sentences': int(sum(1 for x in sentence_complexity if x <= 15))
            }
        
        # Check for problematic content (should be minimal in biblical text)
        problematic_patterns = [
            r'\b(violence|adult|inappropriate)\b',
            r'\b(kill|death|war|fight)\b'
        ]
        
        problematic_count = 0
        for text in df['toaripi']:
            for pattern in problematic_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    problematic_count += 1
                    break
        
        results['content_appropriateness'] = {
            'potentially_problematic': problematic_count,
            'percentage_appropriate': ((len(df) - problematic_count) / len(df)) * 100
        }
        
        return results
    
    def validate_data_splits(self, base_dir: Path) -> dict:
        """Validate train/validation/test splits."""
        logger.info("Validating data splits...")
        
        results = {}
        
        # Load all splits
        splits = {}
        for split_name in ['train', 'validation', 'test']:
            file_path = base_dir / f"{split_name}.csv"
            if file_path.exists():
                splits[split_name] = pd.read_csv(file_path)
        
        if splits:
            total_size = sum(len(df) for df in splits.values())
            
            results['split_distribution'] = {
                split: {
                    'size': len(df),
                    'percentage': (len(df) / total_size) * 100
                }
                for split, df in splits.items()
            }
            
            # Check for data leakage (overlapping verses)
            all_verse_ids = set()
            overlaps = {}
            for split_name, df in splits.items():
                verse_ids = set(df['verse_id'])
                overlap = all_verse_ids.intersection(verse_ids)
                overlaps[split_name] = len(overlap)
                all_verse_ids.update(verse_ids)
            
            results['data_leakage'] = overlaps
            results['no_leakage'] = all(count == 0 for count in overlaps.values())
        
        return results
    
    def validate_educational_prompts(self, base_dir: Path) -> dict:
        """Validate educational prompt files."""
        logger.info("Validating educational prompts...")
        
        results = {}
        
        educational_files = list(base_dir.glob("*_educational.csv"))
        
        if educational_files:
            total_prompts = 0
            content_types = Counter()
            
            for file_path in educational_files:
                df = pd.read_csv(file_path)
                total_prompts += len(df)
                
                if 'content_type' in df.columns:
                    content_types.update(df['content_type'])
            
            results['educational_prompts'] = {
                'total_prompts': total_prompts,
                'files_created': len(educational_files),
                'content_type_distribution': dict(content_types)
            }
        
        return results
    
    def generate_validation_report(self, base_dir: Path) -> dict:
        """Generate comprehensive validation report."""
        logger.info("Generating validation report...")
        
        # Load main processed data
        main_file = base_dir / "toaripi_processed.csv"
        if not main_file.exists():
            raise FileNotFoundError(f"Main processed file not found: {main_file}")
        
        df = pd.read_csv(main_file)
        
        # Run all validations
        report = {
            'dataset_overview': {
                'total_verses': len(df),
                'unique_books': df['book'].nunique(),
                'books': df['book'].value_counts().to_dict()
            },
            'text_quality': self.validate_text_quality(df),
            'language_features': self.validate_toaripi_language_features(df),
            'educational_suitability': self.validate_educational_suitability(df),
            'data_splits': self.validate_data_splits(base_dir),
            'educational_prompts': self.validate_educational_prompts(base_dir)
        }
        
        return report
    
    def save_report(self, report: dict, output_path: Path):
        """Save validation report to file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(report, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Validation report saved to {output_path}")


def main():
    """Main validation function."""
    # Setup paths
    base_dir = Path(__file__).parent.parent / "data" / "processed"
    report_path = base_dir / "validation_report.yaml"
    
    # Initialize validator
    validator = ToaripiDataValidator()
    
    try:
        # Generate validation report
        report = validator.generate_validation_report(base_dir)
        
        # Save report
        validator.save_report(report, report_path)
        
        # Print summary
        logger.info("=== VALIDATION SUMMARY ===")
        overview = report['dataset_overview']
        logger.info(f"Total verses: {overview['total_verses']}")
        logger.info(f"Books: {overview['unique_books']}")
        
        text_quality = report['text_quality']
        logger.info(f"Average verse length: {text_quality['length_stats']['mean']:.1f} characters")
        logger.info(f"Verses within range: {text_quality['length_stats']['within_range']}")
        
        lang_features = report['language_features']
        auth_pct = lang_features['language_authenticity']['percentage_authentic']
        logger.info(f"Toaripi language authenticity: {auth_pct:.1f}%")
        
        vocab_stats = lang_features['vocabulary_stats']
        logger.info(f"Vocabulary richness: {vocab_stats['vocabulary_richness']:.4f}")
        
        if 'data_splits' in report and 'split_distribution' in report['data_splits']:
            splits = report['data_splits']['split_distribution']
            for split, info in splits.items():
                logger.info(f"{split.capitalize()}: {info['size']} verses ({info['percentage']:.1f}%)")
        
        if 'educational_prompts' in report:
            prompts = report['educational_prompts']
            if 'total_prompts' in prompts:
                logger.info(f"Educational prompts: {prompts['total_prompts']}")
        
        logger.info("Validation completed successfully!")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise


if __name__ == "__main__":
    main()