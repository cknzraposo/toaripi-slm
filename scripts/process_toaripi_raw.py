#!/usr/bin/env python3
"""
Process raw Toaripi Bible text into structured training data for SLM.

This script converts the raw Toaripi Bible text into a structured CSV format
suitable for parallel corpus creation and SLM fine-tuning.
"""

import re
import csv
import unicodedata
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ToaripiTextProcessor:
    """Processes raw Toaripi Bible text into structured format."""
    
    def __init__(self, config_path: str = "configs/data/preprocessing_config.yaml"):
        """Initialize processor with configuration."""
        self.config = self._load_config(config_path)
        
        # Book name mappings from Toaripi to English
        self.book_mappings = {
            "GENESE": "Genesis",
            "Genese": "Genesis", 
            "ESODO": "Exodus",
            "LEVITIKA": "Leviticus",
            "NUMERA": "Numbers",
            "DEUTERONOMI": "Deuteronomy",
            "TOSUA": "Joshua",
            "LOHIO KARU": "Judges",
            "RUTA": "Ruth",
            "SAMUELA": "1 Samuel",
            "KING KARU": "1 Kings",
            "BASILEIA VE FARI": "1 Chronicles",
            # Add more mappings as needed
        }
        
    def _load_config(self, config_path: str) -> Dict:
        """Load preprocessing configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Return default configuration."""
        return {
            'preprocessing': {
                'text_cleaning': {
                    'min_length': 10,
                    'max_length': 512,
                    'remove_duplicates': True,
                    'normalize_unicode': True,
                    'strip_whitespace': True
                }
            }
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text according to config."""
        if not text:
            return ""
            
        # Normalize Unicode
        if self.config['preprocessing']['text_cleaning']['normalize_unicode']:
            text = unicodedata.normalize('NFKC', text)
        
        # Remove excessive whitespace
        if self.config['preprocessing']['text_cleaning']['strip_whitespace']:
            text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    def is_book_header(self, line: str) -> Tuple[bool, Optional[str]]:
        """Check if line is a book header and return book name."""
        line = line.strip()
        
        # Look for book headers like "GENESE 1, 2" or "EVERA TESTAMENTA GENESE"
        book_patterns = [
            r'^(GENESE|ESODO|LEVITIKA|NUMERA|DEUTERONOMI|TOSUA|LOHIO KARU|RUTA)\s*\d*',
            r'TESTAMENTA\s+(GENESE|ESODO|LEVITIKA)',
        ]
        
        for pattern in book_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                book_name = match.group(1) if len(match.groups()) >= 1 else match.group(0)
                return True, book_name.upper()
        
        return False, None
    
    def extract_verse_number(self, line: str) -> Tuple[Optional[int], str]:
        """Extract verse number from line and return clean text."""
        line = line.strip()
        
        # Pattern for verse numbers at start of line: "1 Soka Ualare..."
        verse_pattern = r'^(\d+)\s+(.+)$'
        match = re.match(verse_pattern, line)
        
        if match:
            verse_num = int(match.group(1))
            text = match.group(2)
            return verse_num, text
        
        return None, line
    
    def is_content_line(self, line: str) -> bool:
        """Check if line contains actual biblical content."""
        line = line.strip()
        
        # Skip empty lines
        if not line:
            return False
            
        # Skip lines that are clearly formatting/headers
        skip_patterns = [
            r'^[A-Z\s]+$',  # All caps headers
            r'Scale \d+:\d+',  # Map scales
            r'^\d+\s*$',  # Just numbers
            r'Page \d+',  # Page numbers
            r'©.*?$',  # Copyright
            r'Kilometre|Miles',  # Map measurements
            r'^[^\w\s]*$',  # Non-word characters only
        ]
        
        for pattern in skip_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return False
        
        # Must have actual Toaripi words (check for common patterns)
        toaripi_indicators = [
            r'\b(Ualare|Iehova|karu|vita|soka|lei|voa|reha)\b',
            r'\b(foromai|eavia|leipe|meiape)\b'
        ]
        
        for pattern in toaripi_indicators:
            if re.search(pattern, line, re.IGNORECASE):
                return True
        
        # If line has reasonable length and word structure, consider it content
        words = re.findall(r'\b\w+\b', line)
        return len(words) >= 3 and len(line) >= 10
    
    def process_file(self, input_path: str) -> List[Dict]:
        """Process the raw Toaripi file into structured records."""
        logger.info(f"Processing file: {input_path}")
        
        records = []
        current_book = None
        current_chapter = 1
        verse_counter = 1
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # Check for book headers
                is_book, book_name = self.is_book_header(line)
                if is_book and book_name:
                    current_book = self.book_mappings.get(book_name, book_name)
                    current_chapter = 1
                    verse_counter = 1
                    logger.info(f"Found book: {current_book}")
                    continue
                
                # Skip non-content lines
                if not self.is_content_line(line):
                    continue
                
                # Extract verse and clean text
                verse_num, text = self.extract_verse_number(line)
                
                # Clean the text
                clean_text = self.clean_text(text)
                
                # Skip if text is too short or too long
                min_len = self.config['preprocessing']['text_cleaning']['min_length']
                max_len = self.config['preprocessing']['text_cleaning']['max_length']
                
                if len(clean_text) < min_len or len(clean_text) > max_len:
                    continue
                
                # Create record
                if current_book:
                    verse_id = f"{current_book}_{current_chapter}_{verse_num or verse_counter}"
                    
                    record = {
                        'toaripi': clean_text,
                        'verse_id': verse_id,
                        'book': current_book,
                        'chapter': current_chapter,
                        'verse': verse_num or verse_counter,
                        'line_number': line_num
                    }
                    
                    records.append(record)
                    
                    if verse_num:
                        verse_counter = verse_num + 1
                    else:
                        verse_counter += 1
        
        logger.info(f"Processed {len(records)} verses from {len(set(r['book'] for r in records if r['book']))} books")
        return records
    
    def save_to_csv(self, records: List[Dict], output_path: str):
        """Save processed records to CSV file."""
        logger.info(f"Saving {len(records)} records to: {output_path}")
        
        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Define columns in desired order
        columns = ['toaripi', 'verse_id', 'book', 'chapter', 'verse', 'line_number']
        
        df = pd.DataFrame(records)
        
        # Remove duplicates if configured
        if self.config['preprocessing']['text_cleaning'].get('remove_duplicates', True):
            initial_count = len(df)
            df = df.drop_duplicates(subset=['toaripi'])
            logger.info(f"Removed {initial_count - len(df)} duplicate verses")
        
        # Save to CSV
        df[columns].to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Successfully saved processed data to {output_path}")
    
    def generate_stats(self, records: List[Dict]) -> Dict:
        """Generate statistics about the processed data."""
        if not records:
            return {}
        
        df = pd.DataFrame(records)
        
        stats = {
            'total_verses': len(df),
            'unique_books': df['book'].nunique(),
            'books': df['book'].value_counts().to_dict(),
            'avg_verse_length': df['toaripi'].str.len().mean(),
            'min_verse_length': df['toaripi'].str.len().min(),
            'max_verse_length': df['toaripi'].str.len().max(),
            'total_characters': df['toaripi'].str.len().sum(),
            'total_words': df['toaripi'].str.split().str.len().sum()
        }
        
        return stats


def main():
    """Main processing function."""
    # Setup paths
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / "data" / "samples" / "toaripi_raw.txt"
    output_file = base_dir / "data" / "processed" / "toaripi_processed.csv"
    stats_file = base_dir / "data" / "processed" / "processing_stats.yaml"
    
    # Initialize processor
    processor = ToaripiTextProcessor()
    
    # Process the file
    try:
        records = processor.process_file(str(input_file))
        
        if not records:
            logger.error("No records processed. Check input file and processing logic.")
            return
        
        # Save processed data
        processor.save_to_csv(records, str(output_file))
        
        # Generate and save statistics
        stats = processor.generate_stats(records)
        with open(stats_file, 'w') as f:
            yaml.dump(stats, f, default_flow_style=False)
        
        logger.info("Processing completed successfully!")
        logger.info(f"Statistics: {stats}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    main()