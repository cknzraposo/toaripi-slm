"""
Toaripi language model training infrastructure.

This module provides defensive training capabilities for fine-tuning
language models on Toaripi educational content with LoRA/QLoRA support.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer,
    PreTrainedModel, PreTrainedTokenizer
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from pathlib import Path
import json
from dataclasses import dataclass
import os
from datetime import datetime

from ..data import ToaripiParallelDataset
from ..utils import ensure_dir

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class LoraTrainingConfig:
    """Configuration for LoRA training with defensive validation."""
    
    # LoRA parameters
    r: int = 16                      # Low rank dimension
    lora_alpha: int = 32             # LoRA alpha parameter
    lora_dropout: float = 0.1        # LoRA dropout
    target_modules: Optional[List[str]] = None # Target modules for LoRA
    
    # Training parameters
    learning_rate: float = 2e-5      # Learning rate
    batch_size: int = 4              # Training batch size
    gradient_accumulation_steps: int = 4  # Gradient accumulation
    num_epochs: int = 3              # Number of training epochs
    warmup_steps: int = 100          # Warmup steps
    max_grad_norm: float = 1.0       # Gradient clipping
    
    # Model parameters
    max_length: int = 512            # Maximum sequence length
    use_fp16: bool = True            # Use mixed precision
    use_gradient_checkpointing: bool = True  # Gradient checkpointing
    
    # Educational content parameters
    content_filtering: bool = True    # Enable content filtering
    cultural_validation: bool = True  # Enable cultural validation
    age_appropriate_only: bool = True # Only age-appropriate content
    
    def __post_init__(self):
        """Defensive validation of configuration."""
        if self.target_modules is None:
            # Default target modules for most causal LMs
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters with educational focus."""
        # LoRA validation
        if not 1 <= self.r <= 256:
            raise ValueError(f"LoRA rank r must be between 1-256, got: {self.r}")
        
        if not 1 <= self.lora_alpha <= 512:
            raise ValueError(f"LoRA alpha must be between 1-512, got: {self.lora_alpha}")
        
        if not 0.0 <= self.lora_dropout <= 0.9:
            raise ValueError(f"LoRA dropout must be between 0.0-0.9, got: {self.lora_dropout}")
        
        # Training validation
        if not 1e-6 <= self.learning_rate <= 1e-2:
            raise ValueError(f"Learning rate must be between 1e-6 and 1e-2, got: {self.learning_rate}")
        
        if not 1 <= self.batch_size <= 64:
            raise ValueError(f"Batch size must be between 1-64, got: {self.batch_size}")
        
        if not 1 <= self.num_epochs <= 20:
            raise ValueError(f"Number of epochs must be between 1-20, got: {self.num_epochs}")
        
        if not 64 <= self.max_length <= 2048:
            raise ValueError(f"Max length must be between 64-2048, got: {self.max_length}")
        
        # Educational content validation
        if not self.content_filtering:
            logger.warning("Content filtering disabled - educational appropriateness not guaranteed")
        
        if not self.cultural_validation:
            logger.warning("Cultural validation disabled - cultural sensitivity not guaranteed")
        
        logger.info(f"Training configuration validated: r={self.r}, alpha={self.lora_alpha}, lr={self.learning_rate}")


class ToaripiTrainer:
    """
    Defensive trainer for Toaripi educational content models.
    
    Implements LoRA fine-tuning with comprehensive validation and
    educational content focus for cultural preservation.
    """
    
    def __init__(
        self,
        model_name: str,
        config: LoraTrainingConfig,
        output_dir: Path,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize trainer with defensive validation.
        
        Args:
            model_name: HuggingFace model identifier
            config: LoRA training configuration
            output_dir: Directory for saving models and logs
            cache_dir: Optional cache directory for models
        """
        self.model_name = model_name
        self.config = config
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Initialize components
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.model: Optional[Union[PreTrainedModel, PeftModel]] = None
        self.trainer: Optional[Trainer] = None
        
        # Training state
        self.is_loaded = False
        self.is_lora_applied = False
        self.training_started = False
        
        self._validate_inputs()
        self._setup_directories()
        
        logger.info(f"ToaripiTrainer initialized for model: {model_name}")
    
    def _validate_inputs(self):
        """Validate trainer initialization inputs."""
        if not self.model_name or not isinstance(self.model_name, str):
            raise ValueError("model_name must be a non-empty string")
        
        if not isinstance(self.config, LoraTrainingConfig):
            raise TypeError("config must be a LoraTrainingConfig instance")
        
        if not self.output_dir:
            raise ValueError("output_dir cannot be empty")
        
        # Check model name format (basic validation)
        if '/' not in self.model_name and not self.model_name.startswith('gpt'):
            logger.warning(f"Model name '{self.model_name}' may not be a valid HuggingFace identifier")
    
    def _setup_directories(self):
        """Create necessary directories for training."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "final_model").mkdir(exist_ok=True)
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Training directories created: {self.output_dir}")
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer with defensive error handling."""
        try:
            logger.info(f"Loading tokenizer: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=False  # Security: don't execute remote code
            )
            
            # Ensure pad token exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Set pad_token to eos_token")
            
            logger.info(f"Loading model: {self.model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if self.config.use_fp16 else torch.float32,
                device_map="auto",
                trust_remote_code=False,  # Security: don't execute remote code
                low_cpu_mem_usage=True
            )
            
            # Enable gradient checkpointing if requested
            if self.config.use_gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
            
            self.is_loaded = True
            logger.info(f"Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model/tokenizer: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def apply_lora(self):
        """Apply LoRA configuration to the model."""
        if not self.is_loaded:
            raise RuntimeError("Model must be loaded before applying LoRA")
        
        if self.is_lora_applied:
            logger.warning("LoRA already applied, skipping")
            return
        
        try:
            # Create LoRA configuration
            lora_config = LoraConfig(
                r=self.config.r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            # Apply LoRA to model
            self.model = get_peft_model(self.model, lora_config)
            self.is_lora_applied = True
            
            # Print trainable parameters
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            
            logger.info(f"LoRA applied successfully")
            logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
            
        except Exception as e:
            logger.error(f"Failed to apply LoRA: {e}")
            raise RuntimeError(f"LoRA application failed: {e}")
    
    def prepare_training(
        self,
        train_dataset: ToaripiParallelDataset,
        val_dataset: Optional[ToaripiParallelDataset] = None
    ):
        """Prepare training with defensive validation."""
        if not self.is_loaded:
            raise RuntimeError("Model and tokenizer must be loaded first")
        
        if not self.is_lora_applied:
            raise RuntimeError("LoRA must be applied before training preparation")
        
        # Validate datasets
        if not isinstance(train_dataset, ToaripiParallelDataset):
            raise TypeError("train_dataset must be a ToaripiParallelDataset")
        
        if val_dataset and not isinstance(val_dataset, ToaripiParallelDataset):
            raise TypeError("val_dataset must be a ToaripiParallelDataset or None")
        
        if len(train_dataset) == 0:
            raise ValueError("Training dataset cannot be empty")
        
        logger.info(f"Preparing training with {len(train_dataset)} training samples")
        if val_dataset:
            logger.info(f"Validation dataset: {len(val_dataset)} samples")
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "checkpoints"),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            max_grad_norm=self.config.max_grad_norm,
            fp16=self.config.use_fp16,
            logging_steps=10,
            save_steps=100,
            eval_steps=100 if val_dataset else None,
            evaluation_strategy="steps" if val_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
            greater_is_better=False,
            remove_unused_columns=False,
            dataloader_drop_last=True,
            run_name=f"toaripi-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
        )
        
        logger.info("Training preparation completed")
    
    def train(self):
        """Execute training with comprehensive monitoring."""
        if not self.trainer:
            raise RuntimeError("Training must be prepared before execution")
        
        logger.info("Starting Toaripi language model training...")
        self.training_started = True
        
        try:
            # Execute training
            train_result = self.trainer.train()
            
            # Save final model
            final_model_path = self.output_dir / "final_model"
            self.trainer.save_model(str(final_model_path))
            self.tokenizer.save_pretrained(str(final_model_path))
            
            # Save training metrics
            metrics_path = self.output_dir / "training_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(train_result.metrics, f, indent=2)
            
            logger.info(f"Training completed successfully")
            logger.info(f"Final model saved to: {final_model_path}")
            logger.info(f"Training metrics saved to: {metrics_path}")
            
            return train_result
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise RuntimeError(f"Training execution failed: {e}")
    
    def save_config(self):
        """Save training configuration for reproducibility."""
        config_path = self.output_dir / "training_config.json"
        
        config_dict = {
            "model_name": self.model_name,
            "lora_config": {
                "r": self.config.r,
                "lora_alpha": self.config.lora_alpha,
                "lora_dropout": self.config.lora_dropout,
                "target_modules": self.config.target_modules
            },
            "training_config": {
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "num_epochs": self.config.num_epochs,
                "max_length": self.config.max_length,
                "use_fp16": self.config.use_fp16
            },
            "educational_config": {
                "content_filtering": self.config.content_filtering,
                "cultural_validation": self.config.cultural_validation,
                "age_appropriate_only": self.config.age_appropriate_only
            },
            "timestamp": datetime.now().isoformat()
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to: {config_path}")