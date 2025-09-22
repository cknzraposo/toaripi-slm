"""
Model wrapper utilities for Toaripi SLM.

This module provides defensive model loading and management
utilities for educational content generation.
"""

from typing import Optional, Dict, Any, Union
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)


class ModelConfig:
    """Configuration for model loading with defensive validation."""
    
    def __init__(
        self,
        model_name: str,
        max_length: int = 512,
        use_fp16: bool = True,
        device_map: str = "auto",
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize model configuration.
        
        Args:
            model_name: HuggingFace model identifier
            max_length: Maximum sequence length
            use_fp16: Use mixed precision
            device_map: Device mapping strategy
            cache_dir: Optional cache directory
        """
        self.model_name = model_name
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.device_map = device_map
        self.cache_dir = cache_dir
        
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if not self.model_name:
            raise ValueError("Model name cannot be empty")
        
        if not 64 <= self.max_length <= 4096:
            raise ValueError(f"max_length must be between 64-4096, got: {self.max_length}")
        
        if self.device_map not in ["auto", "cpu", "cuda"]:
            raise ValueError(f"Invalid device_map: {self.device_map}")
        
        logger.info(f"Model config validated: {self.model_name} (max_len={self.max_length})")


class ToaripiModelWrapper:
    """
    Defensive wrapper for Toaripi language models.
    
    Provides safe loading, validation, and educational content focus
    for Toaripi language preservation models.
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize model wrapper with configuration."""
        self.config = config
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
        logger.info(f"ToaripiModelWrapper initialized for: {config.model_name}")
    
    def load_model(self):
        """Load model and tokenizer with defensive error handling."""
        try:
            # Import here to avoid issues when libraries aren't available
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            logger.info(f"Loading tokenizer: {self.config.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir,
                trust_remote_code=False
            )
            
            # Ensure pad token exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Set pad_token to eos_token")
            
            logger.info(f"Loading model: {self.config.model_name}")
            import torch
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir,
                torch_dtype=torch.float16 if self.config.use_fp16 else torch.float32,
                device_map=self.config.device_map,
                trust_remote_code=False,
                low_cpu_mem_usage=True
            )
            
            self.is_loaded = True
            logger.info("Model and tokenizer loaded successfully")
            
        except ImportError as e:
            logger.error(f"Required libraries not available: {e}")
            raise RuntimeError(f"Missing dependencies: {e}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def validate_educational_content(self, text: str) -> bool:
        """
        Validate that text is appropriate for educational use.
        
        Args:
            text: Text to validate
            
        Returns:
            True if content is educational and appropriate
        """
        if not text or not text.strip():
            return False
        
        # Check for inappropriate content
        inappropriate_terms = [
            "violence", "weapon", "kill", "death", "adult",
            "sexual", "drug", "alcohol", "gambling"
        ]
        
        text_lower = text.lower()
        for term in inappropriate_terms:
            if term in text_lower:
                logger.warning(f"Inappropriate content detected: {term}")
                return False
        
        # Check for minimum educational value
        if len(text.split()) < 3:
            logger.warning("Content too short for educational value")
            return False
        
        return True
    
    def generate_educational_content(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> Optional[str]:
        """
        Generate educational content with validation.
        
        Args:
            prompt: Input prompt for generation
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            Generated text if valid, None otherwise
        """
        if not self.is_loaded:
            raise RuntimeError("Model must be loaded before generation")
        
        # Validate prompt
        if not self.validate_educational_content(prompt):
            logger.error("Prompt failed educational content validation")
            return None
        
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            # Generate with defensive parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=min(max_new_tokens, 200),  # Cap generation length
                    temperature=max(0.1, min(temperature, 2.0)),  # Clamp temperature
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,  # Reduce repetition
                    length_penalty=1.0
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new content
            new_content = generated_text[len(prompt):].strip()
            
            # Validate generated content
            if self.validate_educational_content(new_content):
                logger.info(f"Generated {len(new_content)} characters of educational content")
                return new_content
            else:
                logger.warning("Generated content failed validation")
                return None
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return None
    
    def save_model(self, output_dir: Path):
        """Save model and tokenizer to directory."""
        if not self.is_loaded:
            raise RuntimeError("Model must be loaded before saving")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
        
        # Save configuration
        config_path = output_dir / "model_config.json"
        config_dict = {
            "model_name": self.config.model_name,
            "max_length": self.config.max_length,
            "use_fp16": self.config.use_fp16,
            "device_map": self.config.device_map
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Model saved to: {output_dir}")


def load_toaripi_model(model_path: Union[str, Path]) -> ToaripiModelWrapper:
    """
    Load a Toaripi model from a saved directory.
    
    Args:
        model_path: Path to model directory
        
    Returns:
        Loaded ToaripiModelWrapper
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    # Load configuration
    config_path = model_path / "model_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config = ModelConfig(
            model_name=str(model_path),  # Use local path
            max_length=config_dict.get("max_length", 512),
            use_fp16=config_dict.get("use_fp16", True),
            device_map=config_dict.get("device_map", "auto")
        )
    else:
        # Default configuration
        config = ModelConfig(model_name=str(model_path))
    
    wrapper = ToaripiModelWrapper(config)
    wrapper.load_model()
    
    return wrapper