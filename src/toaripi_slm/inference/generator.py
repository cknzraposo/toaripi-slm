"""
Model inference and generation components for Toaripi SLM.
Handles educational content generation in Toaripi language.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from loguru import logger
import json
import re


class GenerationError(Exception):
    """Generation-related errors."""
    pass


class ToaripiGenerator:
    """
    Generate educational content in Toaripi language.
    """
    
    def __init__(self, model_path: Union[str, Path], 
                 device: Optional[str] = None):
        """
        Initialize generator with trained model.
        
        Args:
            model_path: Path to trained model directory
            device: Device to run inference on (auto-detect if None)
        """
        self.model_path = Path(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        
        # Content type templates
        self.content_templates = {
            "story": {
                "prompt": "Write a simple story in Toaripi about {topic}. The story should be suitable for primary school children.",
                "max_length": 200,
                "temperature": 0.8
            },
            "vocabulary": {
                "prompt": "Create a vocabulary list in Toaripi for the topic '{topic}'. Include English translations.",
                "max_length": 150,
                "temperature": 0.7
            },
            "dialogue": {
                "prompt": "Write a simple dialogue in Toaripi about {topic}. Make it educational and appropriate for children.",
                "max_length": 180,
                "temperature": 0.8
            },
            "qa": {
                "prompt": "Create questions and answers in Toaripi about {topic}. Make them educational for primary students.",
                "max_length": 120,
                "temperature": 0.7
            }
        }
        
    def load_model(self) -> None:
        """Load the trained model and tokenizer."""
        if not self.model_path.exists():
            raise GenerationError(f"Model directory not found: {self.model_path}")
        
        try:
            logger.info(f"Loading model from: {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Check if this is a LoRA adapter
            adapter_config_path = self.model_path / "adapter_config.json"
            
            if adapter_config_path.exists():
                # Load base model and LoRA adapter
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                
                base_model_name = adapter_config.get("base_model_name_or_path")
                if not base_model_name:
                    raise GenerationError("Base model name not found in adapter config")
                
                logger.info(f"Loading base model: {base_model_name}")
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
                
                # Load LoRA adapter
                self.model = PeftModel.from_pretrained(base_model, self.model_path)
            else:
                # Load full fine-tuned model
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
            
            # Set up generation config
            self.generation_config = GenerationConfig(
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            raise GenerationError(f"Error loading model: {e}")
    
    def generate_text(self, prompt: str, 
                     max_length: int = 150,
                     temperature: float = 0.7,
                     **kwargs) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise GenerationError("Model not loaded. Call load_model() first.")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Update generation config
            gen_config = GenerationConfig.from_dict({
                **self.generation_config.to_dict(),
                "max_new_tokens": max_length,
                "temperature": temperature,
                **kwargs
            })
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=gen_config
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            # Remove the original prompt from output
            generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            raise GenerationError(f"Error during generation: {e}")
    
    def generate_story(self, topic: str, 
                      age_group: str = "primary",
                      length: str = "short") -> str:
        """
        Generate an educational story in Toaripi.
        
        Args:
            topic: Story topic or theme
            age_group: Target age group ("primary" or "secondary")
            length: Story length ("short", "medium", "long")
            
        Returns:
            Generated story in Toaripi
        """
        # Adjust parameters based on age group and length
        if age_group == "primary":
            max_length = {"short": 100, "medium": 150, "long": 200}[length]
            complexity = "simple"
        else:
            max_length = {"short": 150, "medium": 200, "long": 300}[length]
            complexity = "more detailed"
        
        prompt = (
            f"Write a {complexity} story in Toaripi about {topic}. "
            f"The story should be suitable for {age_group} school children. "
            f"Make it educational and culturally appropriate.\n\n"
            f"Story in Toaripi:"
        )
        
        return self.generate_text(
            prompt=prompt,
            max_length=max_length,
            temperature=0.8
        )
    
    def generate_vocabulary(self, topic: str, 
                          count: int = 10,
                          include_examples: bool = True) -> List[Dict[str, str]]:
        """
        Generate vocabulary exercises for a topic.
        
        Args:
            topic: Vocabulary topic
            count: Number of vocabulary items
            include_examples: Whether to include example sentences
            
        Returns:
            List of vocabulary items with translations
        """
        examples_text = " with example sentences" if include_examples else ""
        
        prompt = (
            f"Create {count} vocabulary words in Toaripi related to {topic}. "
            f"For each word, provide the English translation{examples_text}. "
            f"Format as: Toaripi word - English meaning\n\n"
            f"Vocabulary list:"
        )
        
        generated = self.generate_text(
            prompt=prompt,
            max_length=150,
            temperature=0.7
        )
        
        # Parse the generated vocabulary into structured format
        vocab_items = []
        lines = generated.split('\n')
        
        for line in lines:
            line = line.strip()
            if '-' in line and line:
                parts = line.split('-', 1)
                if len(parts) == 2:
                    toaripi_word = parts[0].strip()
                    english_meaning = parts[1].strip()
                    vocab_items.append({
                        "toaripi": toaripi_word,
                        "english": english_meaning,
                        "topic": topic
                    })
        
        return vocab_items[:count]  # Limit to requested count
    
    def generate_dialogue(self, scenario: str, 
                         participants: List[str] = None,
                         age_group: str = "primary") -> str:
        """
        Generate an educational dialogue in Toaripi.
        
        Args:
            scenario: Dialogue scenario or context
            participants: List of participant names (default: generic)
            age_group: Target age group
            
        Returns:
            Generated dialogue
        """
        if participants is None:
            participants = ["Ama", "Apa"]  # Mother, Father in Toaripi context
        
        participants_text = ", ".join(participants)
        
        prompt = (
            f"Write a dialogue in Toaripi between {participants_text} "
            f"about {scenario}. Make it educational and appropriate for "
            f"{age_group} students. Show the conversation clearly.\n\n"
            f"Dialogue:"
        )
        
        return self.generate_text(
            prompt=prompt,
            max_length=180,
            temperature=0.8
        )
    
    def generate_comprehension_questions(self, text: str, 
                                       num_questions: int = 3) -> List[str]:
        """
        Generate reading comprehension questions for a Toaripi text.
        
        Args:
            text: Source text in Toaripi
            num_questions: Number of questions to generate
            
        Returns:
            List of comprehension questions
        """
        prompt = (
            f"Based on this Toaripi text, create {num_questions} simple "
            f"comprehension questions in Toaripi suitable for primary students:\n\n"
            f"Text: {text}\n\n"
            f"Questions:"
        )
        
        generated = self.generate_text(
            prompt=prompt,
            max_length=120,
            temperature=0.7
        )
        
        # Parse questions
        questions = []
        lines = generated.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line.endswith('?') or 'parai' in line.lower()):
                questions.append(line)
        
        return questions[:num_questions]
    
    def generate_content(self, content_type: str, 
                        topic: str,
                        **kwargs) -> Union[str, List[Dict], List[str]]:
        """
        Generate educational content of specified type.
        
        Args:
            content_type: Type of content ("story", "vocabulary", "dialogue", "qa")
            topic: Content topic
            **kwargs: Additional parameters for specific content types
            
        Returns:
            Generated content (format depends on content type)
        """
        if content_type == "story":
            return self.generate_story(topic, **kwargs)
        elif content_type == "vocabulary":
            return self.generate_vocabulary(topic, **kwargs)
        elif content_type == "dialogue":
            return self.generate_dialogue(topic, **kwargs)
        elif content_type == "qa":
            return self.generate_comprehension_questions(topic, **kwargs)
        else:
            raise GenerationError(f"Unsupported content type: {content_type}")
    
    def batch_generate(self, prompts: List[str], 
                      **generation_kwargs) -> List[str]:
        """
        Generate text for multiple prompts efficiently.
        
        Args:
            prompts: List of input prompts
            **generation_kwargs: Generation parameters
            
        Returns:
            List of generated texts
        """
        results = []
        
        for prompt in prompts:
            try:
                result = self.generate_text(prompt, **generation_kwargs)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to generate for prompt: {prompt[:50]}... Error: {e}")
                results.append("")
        
        return results
    
    @classmethod
    def load(cls, model_path: Union[str, Path], 
             device: Optional[str] = None) -> "ToaripiGenerator":
        """
        Load a trained model and return ready-to-use generator.
        
        Args:
            model_path: Path to model directory
            device: Device to use for inference
            
        Returns:
            Loaded ToaripiGenerator instance
        """
        generator = cls(model_path, device)
        generator.load_model()
        return generator