"""
Safety checking service for content validation and constitutional compliance
"""

import asyncio
import logging
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)

class SafetyChecker:
    """Content safety and constitutional compliance checker"""
    
    def __init__(self):
        # Constitutional constraints for Toaripi SLM
        self.constitutional_rules = {
            'educational_only': True,
            'age_appropriate': True,
            'cultural_sensitive': True,
            'no_theology': True,
            'no_adult_content': True,
            'no_violence': True
        }
        
        # Prohibited content patterns
        self.prohibited_patterns = {
            'theological': [
                r'\b(god|allah|buddha|christ|jesus|holy|sacred|divine|worship|pray|religion)\b',
                r'\b(church|mosque|temple|shrine|prayer|blessing|salvation|sin)\b',
                r'\b(bible|quran|torah|scripture|doctrine|theology|preacher|priest)\b'
            ],
            'adult_content': [
                r'\b(sex|sexual|adult|mature|intimate|romantic|dating|love)\b',
                r'\b(naked|nude|body|physical|touching|kissing|marriage)\b'
            ],
            'violence': [
                r'\b(fight|fighting|hit|hitting|hurt|pain|kill|death|blood)\b',
                r'\b(weapon|gun|knife|sword|war|battle|attack|violence)\b',
                r'\b(angry|mad|hate|hurt|damage|destroy|break)\b'
            ],
            'inappropriate': [
                r'\b(drug|alcohol|smoke|drinking|drunk|high|pill|medicine)\b',
                r'\b(money|rich|poor|expensive|cheap|buy|sell|cost)\b',
                r'\b(politics|government|leader|president|vote|election)\b'
            ]
        }
        
        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for category, patterns in self.prohibited_patterns.items():
            self.compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
        
        # Age-appropriate vocabulary for different levels
        self.age_appropriate_words = {
            'primary': {
                'allowed': [
                    'child', 'children', 'play', 'game', 'fun', 'happy', 'smile',
                    'family', 'mother', 'father', 'house', 'home', 'food', 'eat',
                    'water', 'tree', 'bird', 'fish', 'sun', 'moon', 'day', 'night',
                    'learn', 'school', 'friend', 'help', 'work', 'walk', 'run'
                ],
                'complexity_limit': 2  # Max syllables
            },
            'secondary': {
                'allowed': [
                    'education', 'knowledge', 'understand', 'explain', 'community',
                    'tradition', 'culture', 'responsibility', 'cooperation',
                    'environment', 'nature', 'conservation', 'future', 'development'
                ],
                'complexity_limit': 4  # Max syllables
            }
        }
    
    async def check_content_safety(self, english_text: str, toaripi_text: str) -> float:
        """
        Check content safety and return safety score (0.0-1.0)
        
        Args:
            english_text: English text to check
            toaripi_text: Toaripi text to check
            
        Returns:
            Safety score (1.0 = completely safe, 0.0 = unsafe)
        """
        try:
            # Combine texts for checking
            combined_text = f"{english_text} {toaripi_text}".lower()
            
            # Initialize safety score
            safety_score = 1.0
            
            # Check each prohibited category
            for category, patterns in self.compiled_patterns.items():
                for pattern in patterns:
                    if pattern.search(combined_text):
                        # Severe penalties for constitutional violations
                        if category == 'theological':
                            safety_score -= 0.8  # Major violation
                        elif category == 'adult_content':
                            safety_score -= 0.7  # Major violation
                        elif category == 'violence':
                            safety_score -= 0.6  # Major violation
                        else:
                            safety_score -= 0.3  # Minor violation
                        
                        logger.debug(f"Safety violation detected: {category}")
                        break  # One violation per category is enough
            
            # Additional checks
            safety_score -= await self._check_complexity_appropriateness(english_text)
            safety_score -= await self._check_cultural_sensitivity(english_text, toaripi_text)
            
            # Ensure score stays in valid range
            return max(0.0, min(1.0, safety_score))
            
        except Exception as e:
            logger.error(f"Safety check error: {e}")
            return 0.5  # Default to moderate safety if check fails
    
    def get_safety_flags(self, english_text: str, toaripi_text: str) -> List[str]:
        """
        Get specific safety flags for content
        
        Args:
            english_text: English text to check
            toaripi_text: Toaripi text to check
            
        Returns:
            List of safety flag strings
        """
        flags = []
        
        try:
            combined_text = f"{english_text} {toaripi_text}".lower()
            
            # Check each category
            for category, patterns in self.compiled_patterns.items():
                for pattern in patterns:
                    if pattern.search(combined_text):
                        flags.append(category.upper())
                        break
            
            # Check for other issues
            if self._is_too_complex(english_text):
                flags.append('COMPLEXITY_HIGH')
            
            if self._has_cultural_issues(english_text, toaripi_text):
                flags.append('CULTURAL_SENSITIVITY')
            
            return flags
            
        except Exception as e:
            logger.error(f"Safety flags error: {e}")
            return ['ERROR']
    
    async def validate_constitutional_compliance(self, content: str) -> dict:
        """
        Comprehensive constitutional compliance check
        
        Args:
            content: Content to validate
            
        Returns:
            Compliance report dict
        """
        try:
            report = {
                'compliant': True,
                'violations': [],
                'warnings': [],
                'score': 1.0
            }
            
            content_lower = content.lower()
            
            # Check each constitutional rule
            for rule, enabled in self.constitutional_rules.items():
                if not enabled:
                    continue
                
                violation = None
                
                if rule == 'educational_only':
                    if not self._is_educational_content(content_lower):
                        violation = "Content is not educational in nature"
                
                elif rule == 'age_appropriate':
                    if not self._is_age_appropriate(content_lower):
                        violation = "Content not appropriate for target age group"
                
                elif rule == 'cultural_sensitive':
                    if self._has_cultural_issues(content, content):
                        violation = "Content may not be culturally sensitive"
                
                elif rule == 'no_theology':
                    if self._has_theological_content(content_lower):
                        violation = "Theological content is prohibited"
                
                elif rule == 'no_adult_content':
                    if self._has_adult_content(content_lower):
                        violation = "Adult content is prohibited"
                
                elif rule == 'no_violence':
                    if self._has_violent_content(content_lower):
                        violation = "Violent content is prohibited"
                
                if violation:
                    report['violations'].append({
                        'rule': rule,
                        'description': violation,
                        'severity': 'high' if rule in ['no_theology', 'no_adult_content'] else 'medium'
                    })
                    report['compliant'] = False
                    report['score'] -= 0.2
            
            # Ensure score stays valid
            report['score'] = max(0.0, report['score'])
            
            return report
            
        except Exception as e:
            logger.error(f"Constitutional compliance check error: {e}")
            return {
                'compliant': False,
                'violations': [{'rule': 'system', 'description': 'Compliance check failed'}],
                'warnings': [],
                'score': 0.0
            }
    
    async def _check_complexity_appropriateness(self, text: str) -> float:
        """Check if text complexity is appropriate for educational use"""
        try:
            words = text.split()
            if not words:
                return 0.0
            
            # Calculate average word length
            avg_word_length = sum(len(word) for word in words) / len(words)
            
            # Penalty for overly complex text
            if avg_word_length > 8:
                return 0.3  # High complexity penalty
            elif avg_word_length > 6:
                return 0.1  # Moderate complexity penalty
            
            return 0.0  # No penalty
            
        except Exception:
            return 0.0
    
    async def _check_cultural_sensitivity(self, english_text: str, toaripi_text: str) -> float:
        """Check for cultural sensitivity issues"""
        try:
            # Simple check for potentially insensitive content
            sensitive_topics = [
                'primitive', 'backward', 'uncivilized', 'tribal', 'savage',
                'exotic', 'strange', 'weird', 'different', 'foreign'
            ]
            
            combined_text = f"{english_text} {toaripi_text}".lower()
            
            for topic in sensitive_topics:
                if topic in combined_text:
                    return 0.2  # Cultural sensitivity penalty
            
            return 0.0  # No issues found
            
        except Exception:
            return 0.0
    
    def _is_educational_content(self, text: str) -> bool:
        """Check if content is educational"""
        educational_indicators = [
            'learn', 'teach', 'know', 'understand', 'explain', 'show',
            'example', 'practice', 'skill', 'knowledge', 'education',
            'children', 'student', 'school', 'lesson', 'study'
        ]
        
        return any(indicator in text for indicator in educational_indicators)
    
    def _is_age_appropriate(self, text: str) -> bool:
        """Check if content is age-appropriate"""
        inappropriate_for_children = [
            'adult', 'mature', 'complex', 'difficult', 'advanced',
            'serious', 'tragic', 'sad', 'scary', 'frightening'
        ]
        
        return not any(word in text for word in inappropriate_for_children)
    
    def _has_cultural_issues(self, english_text: str, toaripi_text: str) -> bool:
        """Check for cultural sensitivity issues"""
        return False  # Placeholder - would implement sophisticated checking
    
    def _has_theological_content(self, text: str) -> bool:
        """Check for theological content"""
        for pattern in self.compiled_patterns['theological']:
            if pattern.search(text):
                return True
        return False
    
    def _has_adult_content(self, text: str) -> bool:
        """Check for adult content"""
        for pattern in self.compiled_patterns['adult_content']:
            if pattern.search(text):
                return True
        return False
    
    def _has_violent_content(self, text: str) -> bool:
        """Check for violent content"""
        for pattern in self.compiled_patterns['violence']:
            if pattern.search(text):
                return True
        return False
    
    def _is_too_complex(self, text: str) -> bool:
        """Check if text is too complex"""
        words = text.split()
        if not words:
            return False
        
        avg_word_length = sum(len(word) for word in words) / len(words)
        return avg_word_length > 8