"""
CSV validation service for training data quality checking
"""

import re
import logging
from typing import List, Set

logger = logging.getLogger(__name__)

class CSVValidator:
    """Validator for CSV training data quality and format"""
    
    def __init__(self):
        # Toaripi orthography - basic character set
        # This would be expanded with full Toaripi language specification
        self.toaripi_chars = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        self.toaripi_chars.update('.,!?;:-\'\"()[]')
        self.toaripi_chars.update('àáâãäåæçèéêëìíîïñòóôõöøùúûüý')  # Extended chars
        
        # Prohibited content patterns
        self.prohibited_patterns = [
            r'\b(adult|sex|violence|drug|weapon)\b',
            r'\b(theology|doctrine|religious)\b',
            r'\b(politics|political|government)\b',
            r'\b(inappropriate|offensive|harmful)\b'
        ]
        
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) 
                                 for pattern in self.prohibited_patterns]
    
    def validate_toaripi_characters(self, text: str) -> bool:
        """
        Validate that Toaripi text uses appropriate character set
        
        Args:
            text: Toaripi text to validate
            
        Returns:
            bool: True if characters are valid
        """
        try:
            text_chars = set(text)
            invalid_chars = text_chars - self.toaripi_chars
            
            # Log invalid characters for debugging
            if invalid_chars:
                logger.debug(f"Invalid Toaripi characters found: {invalid_chars}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Character validation error: {e}")
            return False
    
    def check_prohibited_content(self, text: str) -> List[str]:
        """
        Check for prohibited content patterns
        
        Args:
            text: Text to check
            
        Returns:
            List of matched prohibited patterns
        """
        matches = []
        
        try:
            for pattern in self.compiled_patterns:
                if pattern.search(text):
                    matches.append(pattern.pattern)
            
            return matches
            
        except Exception as e:
            logger.error(f"Prohibited content check error: {e}")
            return []
    
    def validate_text_quality(self, english: str, toaripi: str) -> dict:
        """
        Comprehensive text quality validation
        
        Args:
            english: English text
            toaripi: Toaripi text
            
        Returns:
            Dict with validation results
        """
        results = {
            'valid': True,
            'issues': [],
            'warnings': []
        }
        
        try:
            # Check text length balance
            english_words = len(english.split())
            toaripi_words = len(toaripi.split())
            
            # Flag if word count difference is too large
            if abs(english_words - toaripi_words) > max(english_words, toaripi_words) * 0.5:
                results['warnings'].append("Large word count difference between languages")
            
            # Check for repeated patterns
            if self._has_repetitive_patterns(english):
                results['warnings'].append("Repetitive patterns in English text")
            
            if self._has_repetitive_patterns(toaripi):
                results['warnings'].append("Repetitive patterns in Toaripi text")
            
            # Check character validation
            if not self.validate_toaripi_characters(toaripi):
                results['issues'].append("Invalid characters in Toaripi text")
                results['valid'] = False
            
            # Check prohibited content
            prohibited_english = self.check_prohibited_content(english)
            prohibited_toaripi = self.check_prohibited_content(toaripi)
            
            if prohibited_english or prohibited_toaripi:
                results['issues'].append("Prohibited content detected")
                results['valid'] = False
            
            return results
            
        except Exception as e:
            logger.error(f"Text quality validation error: {e}")
            results['valid'] = False
            results['issues'].append("Validation error")
            return results
    
    def _has_repetitive_patterns(self, text: str) -> bool:
        """Check for repetitive patterns in text"""
        try:
            words = text.lower().split()
            if len(words) < 3:
                return False
            
            # Check for repeated sequences
            for i in range(len(words) - 2):
                for j in range(i + 2, len(words) - 1):
                    if words[i:i+2] == words[j:j+2]:
                        return True
            
            return False
            
        except Exception:
            return False
    
    def estimate_difficulty_level(self, english: str, toaripi: str) -> str:
        """
        Estimate difficulty level based on text complexity
        
        Args:
            english: English text
            toaripi: Toaripi text
            
        Returns:
            Difficulty level: 'beginner', 'intermediate', 'advanced'
        """
        try:
            # Simple heuristics for difficulty
            english_words = english.split()
            avg_word_length = sum(len(word) for word in english_words) / len(english_words)
            sentence_length = len(english_words)
            
            # Beginner: short sentences, simple words
            if sentence_length <= 5 and avg_word_length <= 5:
                return 'beginner'
            
            # Advanced: long sentences, complex words
            elif sentence_length > 10 or avg_word_length > 7:
                return 'advanced'
            
            # Everything else is intermediate
            else:
                return 'intermediate'
                
        except Exception as e:
            logger.error(f"Difficulty estimation error: {e}")
            return 'intermediate'
    
    def get_content_category(self, english: str, toaripi: str) -> str:
        """
        Automatically categorize content based on keywords
        
        Args:
            english: English text
            toaripi: Toaripi text
            
        Returns:
            Content category
        """
        try:
            text = (english + " " + toaripi).lower()
            
            # Define category keywords
            categories = {
                'greetings': ['hello', 'goodbye', 'morning', 'evening', 'welcome'],
                'family': ['mother', 'father', 'child', 'family', 'parent'],
                'nature': ['tree', 'water', 'bird', 'fish', 'river', 'forest'],
                'daily_life': ['work', 'house', 'food', 'cooking', 'market'],
                'education': ['school', 'learn', 'teach', 'student', 'book'],
                'culture': ['tradition', 'story', 'dance', 'song', 'ceremony']
            }
            
            # Find best matching category
            best_category = 'general'
            best_score = 0
            
            for category, keywords in categories.items():
                score = sum(1 for keyword in keywords if keyword in text)
                if score > best_score:
                    best_score = score
                    best_category = category
            
            return best_category
            
        except Exception as e:
            logger.error(f"Category detection error: {e}")
            return 'general'