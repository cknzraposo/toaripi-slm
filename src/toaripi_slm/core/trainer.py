"""
Core training module for Toaripi SLM.

This module provides the ToaripiTrainer class for fine-tuning language models
on Toaripi educational content using LoRA (Low-Rank Adaptation) techniques.
"""

import os
import yaml
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from loguru import logger

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import wandb


@dataclass
class ToaripiTrainingConfig:
    """Configuration for Toaripi SLM training."""
    
    # Model settings
    model_name: str = "microsoft/DialoGPT-medium"
    device: str = "auto"
    torch_dtype: str = "auto"
    
    # Training hyperparameters
    learning_rate: float = 5e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # LoRA settings
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    
    # Data settings
    max_length: int = 512
    validation_split: float = 0.1
    
    # Output settings
    output_dir: str = "./checkpoints/toaripi-model"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 3
    
    # Monitoring
    use_wandb: bool = False
    wandb_project: str = "toaripi-slm"
    wandb_run_name: str = "training"
    
    def __post_init__(self):
        if self.target_modules is None:
            # Will be auto-detected based on model architecture
            self.target_modules = []


class ToaripiTrainer:
    """
    Trainer for fine-tuning language models on Toaripi educational content.
    
    This class handles:
    - Loading and preparing base models for fine-tuning
    - Applying LoRA (Low-Rank Adaptation) for efficient training
    - Processing parallel English-Toaripi training data
    - Training with educational content focus
    - Saving and exporting trained models
    
    Example:
        >>> config = ToaripiTrainingConfig(
        ...     model_name="mistralai/Mistral-7B-Instruct-v0.2",
        ...     epochs=3,
        ...     use_lora=True
        ... )
        >>> trainer = ToaripiTrainer(config)
        >>> trainer.load_model()
        >>> trainer.prepare_data("data/samples/parallel_data.csv")
        >>> trainer.train()
        >>> trainer.save_model("models/hf/toaripi-mistral")
    """
    
    def __init__(self, config: ToaripiTrainingConfig):
        """
        Initialize the trainer with configuration.
        
        Args:
            config: Training configuration object
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
        self.trainer = None
        
        logger.info("Initialized ToaripiTrainer")
        logger.info(f"Configuration: {self.config}")
    
    def _detect_target_modules(self) -> List[str]:
        """
        Auto-detect appropriate target modules for LoRA based on model architecture.
        
        Returns:
            List of module names suitable for LoRA targeting
        """
        if self.model is None:
            raise ValueError("Model must be loaded before detecting target modules")
        
        # Common patterns for different model architectures
        target_patterns = {
            # Transformer patterns (GPT, Mistral, Llama, etc.)
            'transformer': ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
            'mistral': ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
            'llama': ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
            
            # DialoGPT/GPT-2 patterns
            'gpt2': ['c_attn', 'c_proj'],
            'dialogpt': ['c_attn', 'c_proj'],
            
            # BERT patterns
            'bert': ['query', 'value', 'key', 'dense'],
            
            # T5 patterns
            't5': ['q', 'v', 'k', 'o'],
        }
        
        # Get all named modules
        named_modules = dict(self.model.named_modules())
        module_names = set(named_modules.keys())
        
        logger.info(f"Total modules in model: {len(module_names)}")
        
        # Try to detect model type and find matching modules
        model_name_lower = self.config.model_name.lower()
        
        detected_modules = []
        
        # Check for specific model patterns
        if 'dialogpt' in model_name_lower or 'gpt2' in model_name_lower:
            # DialoGPT/GPT-2 uses c_attn and c_proj
            for pattern in ['c_attn', 'c_proj']:
                matching_modules = [name for name in module_names if pattern in name]
                detected_modules.extend(matching_modules)
        
        elif 'mistral' in model_name_lower:
            # Mistral uses q_proj, k_proj, v_proj, o_proj
            for pattern in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                matching_modules = [name for name in module_names if pattern in name]
                detected_modules.extend(matching_modules)
        
        elif 'llama' in model_name_lower:
            # Llama uses q_proj, k_proj, v_proj, o_proj
            for pattern in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                matching_modules = [name for name in module_names if pattern in name]
                detected_modules.extend(matching_modules)
        
        else:
            # Generic detection - try common patterns
            common_patterns = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'c_attn', 'c_proj']
            for pattern in common_patterns:
                matching_modules = [name for name in module_names if pattern in name]
                if matching_modules:
                    detected_modules.extend(matching_modules)
                    break
        
        # If nothing detected, find Linear layers in attention
        if not detected_modules:
            logger.warning("No standard attention modules detected, searching for Linear layers...")
            for name, module in named_modules.items():
                if isinstance(module, torch.nn.Linear) and ('attn' in name or 'attention' in name):
                    detected_modules.append(name)
        
        # Remove duplicates and get unique base names
        unique_modules = []
        seen_patterns = set()
        
        for module in detected_modules:
            # Extract the base pattern (e.g., 'c_attn' from 'transformer.h.0.attn.c_attn')
            for pattern in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'c_attn', 'c_proj', 'query', 'key', 'value']:
                if pattern in module and pattern not in seen_patterns:
                    unique_modules.append(pattern)
                    seen_patterns.add(pattern)
        
        if not unique_modules:
            # Fallback to a safe default
            logger.warning("Could not auto-detect target modules, using fallback")
            unique_modules = ['c_attn']  # Safe for most GPT-style models
        
        logger.info(f"Detected target modules: {unique_modules}")
        
        # Show some example full module names for verification
        example_modules = [name for name in module_names if any(pattern in name for pattern in unique_modules)][:5]
        logger.info(f"Example matching modules: {example_modules}")
        
        return unique_modules
    
    def load_model(self, model_name: str = None) -> None:
        """
        Load the base model and tokenizer.
        
        Args:
            model_name: HuggingFace model identifier (overrides config if provided)
        """
        model_name = model_name or self.config.model_name
        logger.info(f"Loading model: {model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # Add pad token if missing (common for GPT models)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Added pad token (using eos_token)")
            
            # Load model
            device_map = None if self.config.device == "auto" else self.config.device
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype=torch.float16 if self.config.torch_dtype == "auto" else self.config.torch_dtype,
                trust_remote_code=True
            )
            
            # Auto-detect target modules for LoRA
            if self.config.use_lora and not self.config.target_modules:
                self.config.target_modules = self._detect_target_modules()
            
            logger.info(f"Model loaded successfully. Parameters: {self._count_parameters()}")
            
            # Apply LoRA if requested
            if self.config.use_lora:
                self._apply_lora()
                
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def _apply_lora(self) -> None:
        """Apply LoRA (Low-Rank Adaptation) to the model."""
        logger.info("Applying LoRA configuration...")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none"
        )
        
        logger.info(f"LoRA target modules: {self.config.target_modules}")
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"LoRA applied - Trainable params: {trainable_params:,} "
                   f"({100 * trainable_params / total_params:.2f}% of total)")
    
    def _count_parameters(self) -> str:
        """Count and format model parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        total_str = f"{total_params / 1e6:.1f}M" if total_params < 1e9 else f"{total_params / 1e9:.1f}B"
        trainable_str = f"{trainable_params / 1e6:.1f}M" if trainable_params < 1e9 else f"{trainable_params / 1e9:.1f}B"
        
        return f"{total_str} total, {trainable_str} trainable"
    
    def prepare_data(self, data_path: str, text_column: str = "text") -> None:
        """
        Prepare training data from CSV file.
        
        Args:
            data_path: Path to CSV file with training data
            text_column: Name of column containing text data
        """
        logger.info(f"Loading training data from: {data_path}")
        
        # Load data
        df = pd.read_csv(data_path)
        
        # Create training text from parallel data
        if 'english' in df.columns and 'toaripi' in df.columns:
            # Format as educational prompts
            training_texts = []
            for _, row in df.iterrows():
                prompt = self._create_educational_prompt(row['english'], row['toaripi'])
                training_texts.append(prompt)
        elif text_column in df.columns:
            training_texts = df[text_column].tolist()
        else:
            raise ValueError(f"Could not find text data. Available columns: {df.columns.tolist()}")
        
        logger.info(f"Prepared {len(training_texts)} training samples")
        
        # Tokenize the data
        logger.info("Tokenizing training data...")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.config.max_length,
                return_tensors="pt"
            )
        
        # Create dataset
        dataset = Dataset.from_dict({"text": training_texts})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Split into train/validation
        if self.config.validation_split > 0:
            split_dataset = tokenized_dataset.train_test_split(
                test_size=self.config.validation_split, 
                seed=42
            )
            self.train_dataset = split_dataset['train']
            self.eval_dataset = split_dataset['test']
            
            logger.info(f"Training samples: {len(self.train_dataset)}")
            logger.info(f"Validation samples: {len(self.eval_dataset)}")
        else:
            self.train_dataset = tokenized_dataset
            self.eval_dataset = None
            logger.info(f"Training samples: {len(self.train_dataset)}")
    
    def _create_educational_prompt(self, english_text: str, toaripi_text: str) -> str:
        """
        Create educational prompt from parallel text.
        
        Args:
            english_text: English reference text
            toaripi_text: Toaripi translation
            
        Returns:
            Formatted educational prompt
        """
        prompt = f"""Create educational content in Toaripi language for primary school students.

English reference: {english_text}
Toaripi translation: {toaripi_text}

Generate age-appropriate educational content based on this example."""
        
        return prompt
    
    def train(self) -> None:
        """Execute the training process."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if self.train_dataset is None:
            raise ValueError("Training data not prepared. Call prepare_data() first.")
        
        logger.info("Starting training...")
        
        # Initialize wandb if requested
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=self.config.__dict__
            )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if self.eval_dataset else None,
            save_total_limit=self.config.save_total_limit,
            evaluation_strategy="steps" if self.eval_dataset else "no",
            load_best_model_at_end=True if self.eval_dataset else False,
            metric_for_best_model="eval_loss" if self.eval_dataset else None,
            report_to="wandb" if self.config.use_wandb else None,
            remove_unused_columns=False,
            dataloader_pin_memory=False,  # Better for CPU training
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Start training
        try:
            self.trainer.train()
            logger.info("Training completed successfully!")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
        finally:
            if self.config.use_wandb:
                wandb.finish()
    
    def save_model(self, output_path: str = None) -> None:
        """
        Save the trained model and tokenizer.
        
        Args:
            output_path: Path to save the model (uses config.output_dir if None)
        """
        output_path = output_path or self.config.output_dir
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to: {output_path}")
        
        # Save model and tokenizer
        if self.config.use_lora:
            # Save LoRA adapters
            self.model.save_pretrained(output_path)
        else:
            # Save full model
            self.model.save_pretrained(output_path)
        
        self.tokenizer.save_pretrained(output_path)
        
        # Save configuration
        import json
        config_path = output_path / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        logger.info("Model saved successfully!")
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on validation data.
        
        Returns:
            Dictionary of evaluation metrics
        """
        if self.trainer is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if self.eval_dataset is None:
            logger.warning("No validation dataset available")
            return {}
        
        logger.info("Evaluating model...")
        results = self.trainer.evaluate()
        
        logger.info(f"Evaluation results: {results}")
        return results
    
    def generate_sample(self, prompt: str, max_length: int = 100) -> str:
        """
        Generate text from a prompt using the trained model.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            
        Returns:
            Generated text
        """
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text[len(prompt):].strip()