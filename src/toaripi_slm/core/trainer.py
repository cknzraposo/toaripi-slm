"""
Core training components for Toaripi SLM.
Handles model fine-tuning with LoRA and educational content generation.
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from loguru import logger
import json


class TrainingError(Exception):
    """Training-related errors."""
    pass


class ToaripiTrainer:
    """
    Fine-tune language models for Toaripi educational content generation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        self.trainer = None
        
        # Validate config
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Validate training configuration."""
        required_keys = ["model", "training"]
        for key in required_keys:
            if key not in self.config:
                raise TrainingError(f"Missing required config section: {key}")
        
        model_config = self.config["model"]
        if "name" not in model_config:
            raise TrainingError("Model name not specified in config")
    
    def load_model(self, model_name: Optional[str] = None) -> None:
        """
        Load base model and tokenizer.
        
        Args:
            model_name: Optional model name override
        """
        model_name = model_name or self.config["model"]["name"]
        cache_dir = self.config["model"].get("cache_dir", "./models/cache")
        
        logger.info(f"Loading model: {model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=self.config["model"].get("trust_remote_code", False)
            )
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=self.config["model"].get("trust_remote_code", False),
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            logger.info(f"Model loaded successfully. Parameters: {self.model.num_parameters():,}")
            
        except Exception as e:
            raise TrainingError(f"Error loading model {model_name}: {e}")
    
    def setup_lora(self, lora_config: Optional[Dict] = None) -> None:
        """
        Set up LoRA (Low-Rank Adaptation) for efficient fine-tuning.
        
        Args:
            lora_config: Optional LoRA configuration override
        """
        if self.model is None:
            raise TrainingError("Model must be loaded before setting up LoRA")
        
        # Default LoRA config for educational content generation
        default_lora_config = {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        }
        
        # Use provided config or default
        lora_params = lora_config or self.config.get("lora", default_lora_config)
        
        logger.info(f"Setting up LoRA with config: {lora_params}")
        
        try:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_params["r"],
                lora_alpha=lora_params["lora_alpha"],
                target_modules=lora_params["target_modules"],
                lora_dropout=lora_params["lora_dropout"],
                bias=lora_params["bias"]
            )
            
            self.peft_model = get_peft_model(self.model, peft_config)
            self.peft_model.print_trainable_parameters()
            
        except Exception as e:
            raise TrainingError(f"Error setting up LoRA: {e}")
    
    def prepare_dataset(self, data: Union[pd.DataFrame, List[Dict], str, Path]) -> Dataset:
        """
        Prepare training dataset from parallel data.
        
        Args:
            data: Training data (DataFrame, list of dicts, or file path)
            
        Returns:
            HuggingFace Dataset object
        """
        if self.tokenizer is None:
            raise TrainingError("Tokenizer must be loaded before preparing dataset")
        
        # Load data if path provided
        if isinstance(data, (str, Path)):
            data = pd.read_csv(data)
        elif isinstance(data, list):
            data = pd.DataFrame(data)
        
        # Create educational prompts
        prompts = []
        for _, row in data.iterrows():
            # Create different types of educational content prompts
            content_types = [
                {
                    "type": "story",
                    "prompt": f"Write a simple story in Toaripi based on this English text.\nEnglish: {row['english']}\nToaripi reference: {row['toaripi']}\n\nStory in Toaripi:",
                    "completion": f"{row['toaripi']}"
                },
                {
                    "type": "vocabulary", 
                    "prompt": f"Translate this English text to Toaripi.\nEnglish: {row['english']}\nToaripi:",
                    "completion": f"{row['toaripi']}"
                }
            ]
            
            for content in content_types:
                full_text = content["prompt"] + " " + content["completion"]
                prompts.append({"text": full_text})
        
        logger.info(f"Created {len(prompts)} training prompts")
        
        # Create dataset
        dataset = Dataset.from_list(prompts)
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=self.config["training"].get("max_length", 512)
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset
    
    def train(self, train_dataset: Dataset, 
              val_dataset: Optional[Dataset] = None,
              output_dir: Union[str, Path] = "./models/checkpoints") -> None:
        """
        Train the model with educational content data.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            output_dir: Directory to save model checkpoints
        """
        if self.peft_model is None:
            raise TrainingError("LoRA model must be set up before training")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training arguments
        training_config = self.config["training"]
        args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=training_config.get("epochs", 3),
            per_device_train_batch_size=training_config.get("batch_size", 4),
            per_device_eval_batch_size=training_config.get("eval_batch_size", 4),
            gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
            learning_rate=training_config.get("learning_rate", 2e-5),
            weight_decay=training_config.get("weight_decay", 0.01),
            warmup_steps=training_config.get("warmup_steps", 100),
            logging_steps=training_config.get("logging_steps", 10),
            evaluation_strategy="steps" if val_dataset else "no",
            eval_steps=training_config.get("eval_steps", 100) if val_dataset else None,
            save_strategy="steps",
            save_steps=training_config.get("save_steps", 500),
            save_total_limit=training_config.get("save_total_limit", 3),
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
            greater_is_better=False,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.peft_model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        logger.info("Starting training...")
        
        try:
            # Train the model
            self.trainer.train()
            
            # Save the final model
            final_model_dir = output_dir / "final"
            self.trainer.save_model(str(final_model_dir))
            self.tokenizer.save_pretrained(str(final_model_dir))
            
            logger.info(f"Training completed. Final model saved to: {final_model_dir}")
            
        except Exception as e:
            raise TrainingError(f"Training failed: {e}")
    
    def save_model(self, output_dir: Union[str, Path]) -> None:
        """
        Save the trained model.
        
        Args:
            output_dir: Directory to save the model
        """
        if self.peft_model is None:
            raise TrainingError("No trained model to save")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA adapter
        self.peft_model.save_pretrained(str(output_dir))
        
        # Save tokenizer
        if self.tokenizer:
            self.tokenizer.save_pretrained(str(output_dir))
        
        # Save training config
        config_path = output_dir / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Model saved to: {output_dir}")
    
    def estimate_training_time(self, dataset_size: int) -> Dict[str, float]:
        """
        Estimate training time based on dataset size and configuration.
        
        Args:
            dataset_size: Number of training samples
            
        Returns:
            Dictionary with time estimates
        """
        # Rough estimates based on typical hardware
        config = self.config["training"]
        epochs = config.get("epochs", 3)
        batch_size = config.get("batch_size", 4)
        
        # Estimate steps
        steps_per_epoch = dataset_size // batch_size
        total_steps = steps_per_epoch * epochs
        
        # Time estimates (very rough)
        if torch.cuda.is_available():
            seconds_per_step = 2.0  # With GPU
        else:
            seconds_per_step = 10.0  # CPU only
        
        total_seconds = total_steps * seconds_per_step
        
        return {
            "total_steps": total_steps,
            "estimated_seconds": total_seconds,
            "estimated_minutes": total_seconds / 60,
            "estimated_hours": total_seconds / 3600,
            "device": "GPU" if torch.cuda.is_available() else "CPU"
        }
    
    @classmethod
    def from_config_file(cls, config_path: Union[str, Path]) -> "ToaripiTrainer":
        """
        Create trainer from configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Configured ToaripiTrainer instance
        """
        from ..utils import load_config
        config = load_config(config_path)
        return cls(config)