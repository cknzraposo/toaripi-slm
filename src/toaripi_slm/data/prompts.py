"""
Educational prompt templates for Toaripi language learning.

This module defines prompt templates that ensure educational value
and cultural appropriateness for Toaripi language content generation.
"""

from enum import Enum
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Educational content types with validation."""
    STORY = "story"
    VOCABULARY = "vocabulary"
    DIALOGUE = "dialogue"
    QUESTION_ANSWER = "qa"
    READING_COMPREHENSION = "comprehension"
    CULTURAL_NOTE = "cultural"


class AgeGroup(Enum):
    """Age groups for educational content targeting."""
    PRIMARY_EARLY = "primary_early"  # Ages 5-7
    PRIMARY_MIDDLE = "primary_middle"  # Ages 8-10
    PRIMARY_LATE = "primary_late"  # Ages 11-12
    SECONDARY = "secondary"  # Ages 13+


@dataclass
class PromptTemplate:
    """Defensive prompt template with validation."""
    template: str
    content_type: ContentType
    age_group: AgeGroup
    required_fields: List[str]
    max_length: int = 512
    cultural_guidelines: Optional[str] = None
    
    def __post_init__(self):
        """Validate template after initialization."""
        if not self.template or not self.template.strip():
            raise ValueError("Template cannot be empty")
        
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        
        # Check for required field placeholders
        for field in self.required_fields:
            placeholder = f"{{{field}}}"
            if placeholder not in self.template:
                raise ValueError(f"Template missing required field: {field}")
    
    def format(self, **kwargs) -> str:
        """Safely format template with validation."""
        try:
            # Check all required fields are provided
            missing_fields = [field for field in self.required_fields 
                            if field not in kwargs or kwargs[field] is None]
            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")
            
            # Validate content appropriateness
            for key, value in kwargs.items():
                if not self._validate_content(str(value)):
                    raise ValueError(f"Inappropriate content detected in field: {key}")
            
            formatted = self.template.format(**kwargs)
            
            # Length validation
            if len(formatted) > self.max_length:
                logger.warning(f"Formatted prompt exceeds max_length: {len(formatted)} > {self.max_length}")
            
            return formatted
            
        except KeyError as e:
            raise ValueError(f"Template formatting error: {e}")
    
    def _validate_content(self, content: str) -> bool:
        """Validate content for cultural appropriateness and educational value."""
        if not content or not content.strip():
            return False
        
        # Basic inappropriate content check
        inappropriate_words = [
            'violence', 'war', 'fight', 'kill', 'death', 'blood',
            'alcohol', 'drug', 'weapon', 'gun', 'knife'
        ]
        
        content_lower = content.lower()
        for word in inappropriate_words:
            if word in content_lower:
                logger.warning(f"Potentially inappropriate content detected: {word}")
                return False
        
        return True


class ToaripiPromptTemplates:
    """Collection of educational prompt templates for Toaripi SLM."""
    
    def __init__(self):
        """Initialize with defensive validation."""
        self._templates: Dict[str, PromptTemplate] = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize all educational prompt templates."""
        try:
            # Story generation templates
            self._templates["simple_story"] = PromptTemplate(
                template="""Create a simple educational story in Toaripi language.

English context: {english_text}
Toaripi reference: {toaripi_text}

Generate a story about {topic} suitable for {age_group} students.
The story should:
- Use simple vocabulary appropriate for young learners
- Include cultural elements relevant to Papua New Guinea
- Be educational and engaging
- Be approximately {length} sentences long

Story in Toaripi:""",
                content_type=ContentType.STORY,
                age_group=AgeGroup.PRIMARY_EARLY,
                required_fields=["english_text", "toaripi_text", "topic", "age_group", "length"],
                max_length=800
            )
            
            # Vocabulary exercise templates
            self._templates["vocabulary_list"] = PromptTemplate(
                template="""Create a vocabulary exercise in Toaripi language.

English reference: {english_text}
Toaripi reference: {toaripi_text}

Generate {count} vocabulary words about {topic} for {age_group} students.
Format as:
1. Toaripi word - English translation - Example sentence

Topic: {topic}
Age group: {age_group}

Vocabulary list:""",
                content_type=ContentType.VOCABULARY,
                age_group=AgeGroup.PRIMARY_MIDDLE,
                required_fields=["english_text", "toaripi_text", "topic", "age_group", "count"],
                max_length=600
            )
            
            # Dialogue templates
            self._templates["simple_dialogue"] = PromptTemplate(
                template="""Create a simple dialogue in Toaripi language.

English context: {english_text}
Toaripi reference: {toaripi_text}

Create a dialogue between {participants} about {topic}.
Suitable for {age_group} students learning Toaripi.
Include cultural context and everyday expressions.

Dialogue:""",
                content_type=ContentType.DIALOGUE,
                age_group=AgeGroup.PRIMARY_LATE,
                required_fields=["english_text", "toaripi_text", "participants", "topic", "age_group"],
                max_length=700
            )
            
            # Question and answer templates
            self._templates["comprehension_qa"] = PromptTemplate(
                template="""Create comprehension questions for this Toaripi text.

English text: {english_text}
Toaripi text: {toaripi_text}

Generate {question_count} questions suitable for {age_group} students.
Include both simple recall and basic comprehension questions.

Questions and answers:""",
                content_type=ContentType.QUESTION_ANSWER,
                age_group=AgeGroup.PRIMARY_MIDDLE,
                required_fields=["english_text", "toaripi_text", "question_count", "age_group"],
                max_length=500
            )
            
            logger.info(f"Initialized {len(self._templates)} prompt templates")
            
        except Exception as e:
            logger.error(f"Failed to initialize prompt templates: {e}")
            raise
    
    def get_template(self, template_name: str) -> Optional[PromptTemplate]:
        """Get template by name with validation."""
        if not template_name or not isinstance(template_name, str):
            raise ValueError("Template name must be a non-empty string")
        
        template = self._templates.get(template_name)
        if template is None:
            logger.warning(f"Template not found: {template_name}")
            
        return template
    
    def list_templates(self) -> List[str]:
        """List all available template names."""
        return list(self._templates.keys())
    
    def get_templates_by_type(self, content_type: ContentType) -> List[str]:
        """Get templates filtered by content type."""
        return [name for name, template in self._templates.items() 
                if template.content_type == content_type]
    
    def get_templates_by_age_group(self, age_group: AgeGroup) -> List[str]:
        """Get templates filtered by age group."""
        return [name for name, template in self._templates.items() 
                if template.age_group == age_group]
    
    def validate_template_data(self, template_name: str, **kwargs) -> bool:
        """Validate data for a specific template."""
        template = self.get_template(template_name)
        if template is None:
            return False
        
        try:
            template.format(**kwargs)
            return True
        except ValueError as e:
            logger.error(f"Template validation failed for {template_name}: {e}")
            return False


def create_educational_prompt(
    english_text: str,
    toaripi_text: str,
    content_type: ContentType = ContentType.STORY,
    age_group: AgeGroup = AgeGroup.PRIMARY_MIDDLE,
    **kwargs: Any
) -> str:
    """
    Create an educational prompt with defensive validation.
    
    Args:
        english_text: English reference text
        toaripi_text: Toaripi reference text
        content_type: Type of educational content to generate
        age_group: Target age group for the content
        **kwargs: Additional template-specific parameters
    
    Returns:
        Formatted educational prompt
        
    Raises:
        ValueError: If inputs are invalid or template formatting fails
    """
    if not english_text or not english_text.strip():
        raise ValueError("English text cannot be empty")
    
    if not toaripi_text or not toaripi_text.strip():
        raise ValueError("Toaripi text cannot be empty")
    
    templates = ToaripiPromptTemplates()
    
    # Select appropriate template based on content type
    template_map = {
        ContentType.STORY: "simple_story",
        ContentType.VOCABULARY: "vocabulary_list",
        ContentType.DIALOGUE: "simple_dialogue",
        ContentType.QUESTION_ANSWER: "comprehension_qa"
    }
    
    template_name = template_map.get(content_type)
    if not template_name:
        raise ValueError(f"No template available for content type: {content_type}")
    
    template = templates.get_template(template_name)
    if template is None:
        raise ValueError(f"Template not found: {template_name}")
    
    # Prepare template data with defaults
    template_data = {
        "english_text": english_text,
        "toaripi_text": toaripi_text,
        "age_group": age_group.value,
        **kwargs
    }
    
    # Add sensible defaults for common fields
    if "topic" not in template_data:
        template_data["topic"] = "daily life"
    if "length" not in template_data:
        template_data["length"] = "3-5"
    if "count" not in template_data:
        template_data["count"] = "5"
    if "participants" not in template_data:
        template_data["participants"] = "two children"
    if "question_count" not in template_data:
        template_data["question_count"] = "3"
    
    return template.format(**template_data)


# Export main interfaces
__all__ = [
    "ContentType",
    "AgeGroup", 
    "PromptTemplate",
    "ToaripiPromptTemplates",
    "create_educational_prompt"
]