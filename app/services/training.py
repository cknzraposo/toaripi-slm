"""
Training service for managing model training sessions
"""

import asyncio
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Dict, List

from app.core.config import settings
from app.models.schemas import (
    TrainingSession, TrainingConfig, TrainingProgressUpdate, 
    TrainingStatus
)

logger = logging.getLogger(__name__)

class TrainingManager:
    """Manages model training sessions and integration with existing CLI tools"""
    
    def __init__(self):
        self.active_sessions: Dict[str, subprocess.Popen] = {}
        self.session_logs: Dict[str, List[str]] = {}
    
    async def validate_training_config(
        self, 
        upload_ids: List[str], 
        config: TrainingConfig,
        model_base: str
    ) -> Dict:
        """
        Validate training configuration and data requirements
        
        Args:
            upload_ids: List of validated upload IDs
            config: Training configuration
            model_base: Base model identifier
            
        Returns:
            Validation result dict
        """
        try:
            # TODO: Import upload validation data
            # For now, simulate validation
            
            # Check minimum data requirements
            estimated_pairs = 500  # Placeholder - would get from actual uploads
            
            if estimated_pairs < settings.MIN_TRAINING_PAIRS:
                return {
                    "valid": False,
                    "errors": [{
                        "field": "upload_ids",
                        "error_code": "INSUFFICIENT_DATA",
                        "message": f"Minimum {settings.MIN_TRAINING_PAIRS} training pairs required, found {estimated_pairs}"
                    }]
                }
            
            # Estimate resource requirements
            estimated_memory_gb = self._estimate_memory_usage(config, estimated_pairs)
            
            if estimated_memory_gb > settings.MAX_MEMORY_GB:
                return {
                    "valid": False,
                    "errors": [{
                        "field": "batch_size",
                        "error_code": "MEMORY_LIMIT_EXCEEDED",
                        "message": f"Configuration exceeds memory limit ({estimated_memory_gb:.1f}GB > {settings.MAX_MEMORY_GB}GB)",
                        "suggested_value": max(1, config.batch_size // 2)
                    }],
                    "estimated_memory_usage_gb": estimated_memory_gb
                }
            
            # Estimate training time and steps
            estimated_steps = self._estimate_training_steps(config, estimated_pairs)
            estimated_duration = self._estimate_training_duration(estimated_steps, config)
            
            return {
                "valid": True,
                "estimated_duration_minutes": estimated_duration,
                "estimated_memory_usage_gb": estimated_memory_gb,
                "total_training_pairs": estimated_pairs,
                "estimated_steps": estimated_steps
            }
            
        except Exception as e:
            logger.error(f"Training validation error: {e}")
            return {
                "valid": False,
                "errors": [{
                    "field": "system",
                    "error_code": "VALIDATION_ERROR",
                    "message": "Training validation failed"
                }]
            }
    
    async def run_training(self, session: TrainingSession) -> AsyncGenerator[TrainingProgressUpdate, None]:
        """
        Run training session and yield progress updates
        
        Args:
            session: Training session to run
            
        Yields:
            Training progress updates
        """
        try:
            session_id = session.session_id
            
            # Prepare training data
            yield TrainingProgressUpdate(
                session_id=session_id,
                progress_percentage=5.0,
                current_step=0,
                total_steps=session.total_steps,
                status=TrainingStatus.STARTING,
                message="Preparing training data"
            )
            
            # TODO: Integrate with existing training scripts
            # For now, simulate training process
            
            await asyncio.sleep(2)  # Simulate data preparation
            
            # Start training
            yield TrainingProgressUpdate(
                session_id=session_id,
                progress_percentage=10.0,
                current_step=0,
                total_steps=session.total_steps,
                status=TrainingStatus.TRAINING,
                message="Training started"
            )
            
            # Simulate training progress
            for step in range(0, session.total_steps, max(1, session.total_steps // 20)):
                progress = min(95.0, 10.0 + (step / session.total_steps) * 85.0)
                
                # Simulate loss decrease
                loss = 2.0 * (1.0 - step / session.total_steps) + 0.1
                
                # Estimate time remaining
                elapsed_steps = step + 1
                time_per_step = 2.0  # seconds per step (estimated)
                remaining_steps = session.total_steps - elapsed_steps
                estimated_time_remaining = int(remaining_steps * time_per_step)
                
                yield TrainingProgressUpdate(
                    session_id=session_id,
                    progress_percentage=progress,
                    current_step=step,
                    total_steps=session.total_steps,
                    current_loss=loss,
                    estimated_time_remaining=estimated_time_remaining,
                    status=TrainingStatus.TRAINING,
                    message=f"Training step {step}/{session.total_steps}"
                )
                
                await asyncio.sleep(1)  # Simulate training time
            
            # Evaluation phase
            yield TrainingProgressUpdate(
                session_id=session_id,
                progress_percentage=95.0,
                current_step=session.total_steps,
                total_steps=session.total_steps,
                status=TrainingStatus.EVALUATING,
                message="Evaluating model performance"
            )
            
            await asyncio.sleep(3)  # Simulate evaluation
            
            # Export phase
            yield TrainingProgressUpdate(
                session_id=session_id,
                progress_percentage=98.0,
                current_step=session.total_steps,
                total_steps=session.total_steps,
                status=TrainingStatus.EXPORTING,
                message="Exporting model files"
            )
            
            await asyncio.sleep(2)  # Simulate export
            
            # Set final model path
            session.output_model_path = str(settings.MODELS_DIR / "hf" / f"toaripi-{session_id[:8]}")
            session.evaluation_metrics = {
                "final_loss": 0.18,
                "perplexity": 3.2,
                "bleu_score": 0.75,
                "safety_score": 0.96
            }
            
            # Complete
            yield TrainingProgressUpdate(
                session_id=session_id,
                progress_percentage=100.0,
                current_step=session.total_steps,
                total_steps=session.total_steps,
                status=TrainingStatus.COMPLETED,
                message="Training completed successfully"
            )
            
        except Exception as e:
            logger.error(f"Training execution error: {e}")
            yield TrainingProgressUpdate(
                session_id=session_id,
                progress_percentage=0.0,
                current_step=0,
                total_steps=session.total_steps,
                status=TrainingStatus.FAILED,
                message=f"Training failed: {str(e)}"
            )
    
    async def cancel_session(self, session_id: str):
        """Cancel an active training session"""
        try:
            if session_id in self.active_sessions:
                process = self.active_sessions[session_id]
                if process.poll() is None:  # Process is still running
                    process.terminate()
                    await asyncio.sleep(2)
                    if process.poll() is None:
                        process.kill()  # Force kill if terminate didn't work
                
                del self.active_sessions[session_id]
                logger.info(f"Training session cancelled: {session_id}")
            
        except Exception as e:
            logger.error(f"Error cancelling session {session_id}: {e}")
            raise
    
    def _estimate_memory_usage(self, config: TrainingConfig, num_pairs: int) -> float:
        """Estimate memory usage for training configuration"""
        # Simple heuristic for memory estimation
        base_memory = 4.0  # Base model loading
        lora_memory = (config.lora_rank * config.lora_alpha) / 1000.0
        batch_memory = config.batch_size * 0.5
        sequence_memory = config.max_seq_length / 1000.0
        
        total_memory = base_memory + lora_memory + batch_memory + sequence_memory
        return min(total_memory, 16.0)  # Cap at reasonable maximum
    
    def _estimate_training_steps(self, config: TrainingConfig, num_pairs: int) -> int:
        """Estimate number of training steps"""
        steps_per_epoch = max(1, num_pairs // config.batch_size)
        total_steps = steps_per_epoch * config.max_epochs
        return total_steps
    
    def _estimate_training_duration(self, steps: int, config: TrainingConfig) -> int:
        """Estimate training duration in minutes"""
        # Simple heuristic: ~2 seconds per step
        seconds_per_step = 2.0
        if config.batch_size > 4:
            seconds_per_step *= 1.5  # Larger batches take longer
        
        total_seconds = steps * seconds_per_step
        return max(1, int(total_seconds / 60))
    
    async def run_actual_training(self, session: TrainingSession) -> AsyncGenerator[TrainingProgressUpdate, None]:
        """
        Run actual training using existing CLI scripts (for future implementation)
        
        This would integrate with the existing training infrastructure:
        - Use scripts/train.py or similar
        - Monitor log files for progress
        - Parse training metrics
        """
        # TODO: Implement integration with existing training scripts
        
        # Example implementation:
        # 1. Prepare training config file
        # 2. Run training script as subprocess
        # 3. Monitor log files for progress
        # 4. Parse metrics and yield updates
        # 5. Handle model export and validation
        
        pass