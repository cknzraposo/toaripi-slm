"""
Content generation module for Toaripi SLM.

This module provides the ToaripiGenerator class for generating educational
content in Toaripi language, including stories, vocabulary, Q&A, and dialogues.
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
from loguru import logger


class ContentType(Enum):
    """Types of educational content that can be generated."""
    STORY = "story"
    VOCABULARY = "vocabulary" 
    QA = "qa"
    DIALOGUE = "dialogue"
    COMPREHENSION = "comprehension"


class AgeGroup(Enum):
    """Target age groups for educational content."""
    PRIMARY = "primary"    # Ages 6-12
    SECONDARY = "secondary"  # Ages 13-18
    ADULT = "adult"       # Adult learners


@dataclass
class GenerationConfig:
    """Configuration for content generation."""
    
    # Generation parameters
    max_length: int = 200
    min_length: int = 20
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    
    # Content parameters
    content_type: ContentType = ContentType.STORY
    age_group: AgeGroup = AgeGroup.PRIMARY
    topic: str = ""
    context: str = ""
    
    # Safety and quality
    filter_inappropriate: bool = True
    ensure_educational: bool = True
    max_retries: int = 3


class ToaripiGenerator:
    """
    Generator for creating educational content in Toaripi language.
    
    This class handles:
    - Loading trained Toaripi models
    - Generating educational stories, vocabulary, Q&A, and dialogues
    - Ensuring content appropriateness for target age groups
    - Applying cultural sensitivity filters
    - Supporting both online and offline inference
    
    Example:
        >>> generator = ToaripiGenerator.load("models/hf/toaripi-mistral")
        >>> story = generator.generate_story(
        ...     prompt="Children helping with fishing",
        ...     age_group=AgeGroup.PRIMARY,
        ...     max_length=150
        ... )
        >>> print(story)
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        load_in_8bit: bool = False
    ):
        """
        Initialize the Toaripi content generator.
        
        Args:
            model_path: Path to trained model directory
            device: Device to run inference on ('cpu', 'cuda', 'auto')
            load_in_8bit: Whether to load model in 8-bit precision
        """
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_in_8bit = load_in_8bit
        
        self.model = None
        self.tokenizer = None
        self.is_lora_model = False
        
        # Content templates and filters
        self._load_content_templates()
        self._setup_content_filters()
        
        logger.info(f"Initialized ToaripiGenerator (device: {self.device})")
        
        if model_path:
            self.load_model(model_path)
    
    def _load_content_templates(self):
        """Load content generation templates."""
        self.templates = {
            ContentType.STORY: {
                AgeGroup.PRIMARY: "Write a simple story in Toaripi about {topic}. Use 3-5 sentences with easy words that primary school children can understand. The story should teach about {context}.",
                AgeGroup.SECONDARY: "Create a story in Toaripi about {topic}. Use 5-8 sentences with appropriate vocabulary for teenagers. Include {context} as a theme.",
                AgeGroup.ADULT: "Develop a narrative in Toaripi focusing on {topic}. Write 8-12 sentences that incorporate {context} and cultural elements."
            },
            ContentType.VOCABULARY: {
                AgeGroup.PRIMARY: "Create a vocabulary list in Toaripi for {topic}. Provide 5-10 simple words with English translations and one example sentence for each word.",
                AgeGroup.SECONDARY: "Generate a vocabulary set for {topic} in Toaripi. Include 10-15 words with definitions and example sentences suitable for secondary students.",
                AgeGroup.ADULT: "Compile advanced vocabulary for {topic} in Toaripi. Present 15-20 words with detailed explanations and contextual examples."
            },
            ContentType.QA: {
                AgeGroup.PRIMARY: "Create 3-5 simple questions and answers in Toaripi about {topic}. Use basic question words and short, clear answers.",
                AgeGroup.SECONDARY: "Develop 5-8 questions and answers in Toaripi related to {topic}. Include different question types and detailed responses.",
                AgeGroup.ADULT: "Generate 8-10 comprehensive questions and answers in Toaripi covering {topic}. Include analytical and discussion questions."
            },
            ContentType.DIALOGUE: {
                AgeGroup.PRIMARY: "Write a simple dialogue in Toaripi between two children talking about {topic}. Use 4-6 exchanges with everyday language.",
                AgeGroup.SECONDARY: "Create a conversation in Toaripi between teenagers discussing {topic}. Include 6-10 exchanges with natural expressions.",
                AgeGroup.ADULT: "Develop a dialogue in Toaripi between adults about {topic}. Write 8-12 exchanges with formal and informal registers."
            },
            ContentType.COMPREHENSION: {
                AgeGroup.PRIMARY: "Write a short paragraph in Toaripi about {topic}, then create 3 simple comprehension questions.",
                AgeGroup.SECONDARY: "Create a passage in Toaripi about {topic} followed by 5 comprehension questions of varying difficulty.",
                AgeGroup.ADULT: "Develop a detailed text in Toaripi on {topic} with 6-8 analytical comprehension questions."
            }
        }
    
    def _setup_content_filters(self):
        """Setup content appropriateness filters."""
        # Inappropriate content keywords (basic filtering)
        self.inappropriate_keywords = {
            'violence', 'war', 'death', 'blood', 'kill', 'fight', 'hurt', 'pain',
            'adult', 'sexual', 'drug', 'alcohol', 'smoke', 'gambling', 'crime'
        }
        
        # Educational content indicators
        self.educational_indicators = {
            'learn', 'teach', 'school', 'study', 'practice', 'example', 'lesson',
            'children', 'family', 'community', 'culture', 'tradition', 'help'
        }
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained Toaripi model.
        
        Args:
            model_path: Path to model directory or HuggingFace model name
        """
        model_dir = Path(model_path)
        self.model_path = model_path
        
        logger.info(f"Loading model from: {model_path}")
        
        # Check if it's a LoRA model
        if model_dir.exists() and (model_dir / "adapter_config.json").exists():
            self.is_lora_model = True
            logger.info("Detected LoRA adapter model")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=False
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        if self.is_lora_model:
            # Load base model first, then adapter
            config_path = model_dir / "adapter_config.json"
            with open(config_path) as f:
                adapter_config = json.load(f)
            
            base_model_name = adapter_config.get("base_model_name_or_path", "microsoft/DialoGPT-medium")
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map="auto" if self.device == 'cuda' else None,
                load_in_8bit=self.load_in_8bit
            )
            
            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(base_model, model_path)
        else:
            # Load full model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map="auto" if self.device == 'cuda' else None,
                load_in_8bit=self.load_in_8bit,
                trust_remote_code=False
            )
        
        # Move to device if not using device_map
        if self.device != 'cuda' or not hasattr(self.model, 'hf_device_map'):
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        logger.info(f"Model loaded successfully (LoRA: {self.is_lora_model})")
    
    def generate_content(
        self,
        prompt: str,
        content_type: ContentType = ContentType.STORY,
        age_group: AgeGroup = AgeGroup.PRIMARY,
        max_length: int = 200,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate educational content based on prompt and parameters.
        
        Args:
            prompt: Input prompt or topic
            content_type: Type of content to generate
            age_group: Target age group
            max_length: Maximum length of generated content
            temperature: Randomness in generation (0.0-1.0)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated content in Toaripi
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Create formatted prompt
        template = self.templates[content_type][age_group]
        formatted_prompt = template.format(topic=prompt, context=kwargs.get('context', 'daily life'))
        
        # Tokenize input
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate content
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                min_new_tokens=kwargs.get('min_length', 20),
                temperature=temperature,
                top_p=kwargs.get('top_p', 0.9),
                top_k=kwargs.get('top_k', 50),
                repetition_penalty=kwargs.get('repetition_penalty', 1.1),
                do_sample=kwargs.get('do_sample', True),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new content (remove prompt)
        if formatted_prompt in generated_text:
            generated_content = generated_text.replace(formatted_prompt, "").strip()
        else:
            generated_content = generated_text.strip()
        
        # Apply content filters
        if kwargs.get('filter_content', True):
            generated_content = self._filter_content(generated_content)
        
        return generated_content
    
    def generate_story(
        self,
        prompt: str,
        age_group: AgeGroup = AgeGroup.PRIMARY,
        max_length: int = 150,
        **kwargs
    ) -> str:
        """Generate a story in Toaripi."""
        return self.generate_content(
            prompt=prompt,
            content_type=ContentType.STORY,
            age_group=age_group,
            max_length=max_length,
            **kwargs
        )
    
    def generate_vocabulary(
        self,
        topic: str,
        age_group: AgeGroup = AgeGroup.PRIMARY,
        word_count: int = 10,
        **kwargs
    ) -> str:
        """Generate vocabulary list for a topic."""
        context = f"vocabulary list with {word_count} words"
        return self.generate_content(
            prompt=topic,
            content_type=ContentType.VOCABULARY,
            age_group=age_group,
            max_length=word_count * 30,  # Rough estimate per word
            context=context,
            **kwargs
        )
    
    def generate_qa(
        self,
        topic: str,
        age_group: AgeGroup = AgeGroup.PRIMARY,
        question_count: int = 5,
        **kwargs
    ) -> str:
        """Generate Q&A content for a topic."""
        context = f"{question_count} questions and answers"
        return self.generate_content(
            prompt=topic,
            content_type=ContentType.QA,
            age_group=age_group,
            max_length=question_count * 40,
            context=context,
            **kwargs
        )
    
    def generate_dialogue(
        self,
        scenario: str,
        age_group: AgeGroup = AgeGroup.PRIMARY,
        exchange_count: int = 6,
        **kwargs
    ) -> str:
        """Generate dialogue in Toaripi."""
        context = f"{exchange_count} conversational exchanges"
        return self.generate_content(
            prompt=scenario,
            content_type=ContentType.DIALOGUE,
            age_group=age_group,
            max_length=exchange_count * 25,
            context=context,
            **kwargs
        )
    
    def generate_comprehension(
        self,
        topic: str,
        age_group: AgeGroup = AgeGroup.PRIMARY,
        **kwargs
    ) -> str:
        """Generate reading comprehension exercise."""
        return self.generate_content(
            prompt=topic,
            content_type=ContentType.COMPREHENSION,
            age_group=age_group,
            max_length=200,
            **kwargs
        )
    
    def _filter_content(self, content: str) -> str:
        """Apply content appropriateness filters."""
        # Convert to lowercase for checking
        content_lower = content.lower()
        
        # Check for inappropriate keywords
        for keyword in self.inappropriate_keywords:
            if keyword in content_lower:
                logger.warning(f"Filtered inappropriate content containing: {keyword}")
                return "[Content filtered: inappropriate for educational use]"
        
        # Basic length validation
        if len(content.strip()) < 10:
            logger.warning("Generated content too short")
            return "[Content too short - please try again]"
        
        # Remove any remaining prompt artifacts
        content = re.sub(r'^(Create|Generate|Write|Develop).*?:', '', content, flags=re.IGNORECASE)
        content = content.strip()
        
        return content
    
    def batch_generate(
        self,
        prompts: List[str],
        content_type: ContentType = ContentType.STORY,
        age_group: AgeGroup = AgeGroup.PRIMARY,
        **kwargs
    ) -> List[str]:
        """
        Generate content for multiple prompts.
        
        Args:
            prompts: List of input prompts
            content_type: Type of content to generate
            age_group: Target age group
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated content strings
        """
        results = []
        for prompt in prompts:
            try:
                content = self.generate_content(
                    prompt=prompt,
                    content_type=content_type,
                    age_group=age_group,
                    **kwargs
                )
                results.append(content)
            except Exception as e:
                logger.error(f"Failed to generate content for prompt '{prompt}': {e}")
                results.append(f"[Generation failed: {str(e)}]")
        
        return results
    
    @classmethod
    def load(cls, model_path: str, **kwargs) -> "ToaripiGenerator":
        """
        Load a trained model and create generator instance.
        
        Args:
            model_path: Path to model directory
            **kwargs: Additional initialization parameters
            
        Returns:
            Loaded ToaripiGenerator instance
        """
        generator = cls(**kwargs)
        generator.load_model(model_path)
        return generator
    
    def save_config(self, config_path: str, config: GenerationConfig) -> None:
        """Save generation configuration to file."""
        config_dict = {
            'max_length': config.max_length,
            'temperature': config.temperature,
            'top_p': config.top_p,
            'top_k': config.top_k,
            'repetition_penalty': config.repetition_penalty,
            'content_type': config.content_type.value,
            'age_group': config.age_group.value
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def __repr__(self) -> str:
        model_info = f"model={self.model_path}" if self.model else "model=None"
        return f"ToaripiGenerator({model_info}, device={self.device})"