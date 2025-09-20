"""
Model management API endpoints for listing, activating, and downloading models
"""

import asyncio
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.core.startup import get_system_state, update_system_state
from app.models.schemas import (
    ModelInfo, ModelDownloadRequest, ModelDownloadStatus,
    ModelActivationRequest, ModelActivationStatus, ModelMetrics,
    AvailableModel
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Global state for download tracking
_download_tasks: Dict[str, Dict] = {}

@router.get("", response_model=List[ModelInfo])
async def list_models():
    """List all available models (local and downloadable)"""
    
    try:
        models = []
        
        # Get local models
        local_models = await _get_local_models()
        models.extend(local_models)
        
        # Get downloadable models
        downloadable_models = await _get_downloadable_models()
        models.extend(downloadable_models)
        
        # Sort by name
        models.sort(key=lambda m: m.name)
        
        return models
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail="Failed to list models")

@router.get("/available", response_model=List[AvailableModel])
async def list_available_models():
    """List models available for download from the model hub"""
    
    try:
        # Available models for download
        available_models = [
            {
                "name": "toaripi-base-v1",
                "description": "Base Toaripi language model fine-tuned on Bible data",
                "version": "1.0.0",
                "size_mb": 4200,
                "capabilities": ["story_generation", "vocabulary", "basic_qa"],
                "min_memory_gb": 8,
                "download_url": "https://huggingface.co/toaripi/toaripi-base-v1",
                "format": "hf_transformers",
                "license": "CC-BY-SA-4.0",
                "created_date": datetime(2024, 1, 15),
                "model_type": "educational",
                "language_code": "tqo",
                "training_data_size": "2.5M tokens"
            },
            {
                "name": "toaripi-base-v1-gguf",
                "description": "Quantized Toaripi model for edge deployment",
                "version": "1.0.0",
                "size_mb": 2800,
                "capabilities": ["story_generation", "vocabulary", "basic_qa"],
                "min_memory_gb": 4,
                "download_url": "https://huggingface.co/toaripi/toaripi-base-v1-gguf",
                "format": "gguf",
                "license": "CC-BY-SA-4.0",
                "created_date": datetime(2024, 1, 15),
                "model_type": "educational",
                "language_code": "tqo",
                "training_data_size": "2.5M tokens",
                "quantization": "q4_k_m"
            },
            {
                "name": "toaripi-educational-v2",
                "description": "Enhanced educational content generation model",
                "version": "2.0.0",
                "size_mb": 6800,
                "capabilities": ["story_generation", "vocabulary", "dialogue", "comprehension_qa"],
                "min_memory_gb": 12,
                "download_url": "https://huggingface.co/toaripi/toaripi-educational-v2",
                "format": "hf_transformers",
                "license": "CC-BY-SA-4.0",
                "created_date": datetime(2024, 2, 1),
                "model_type": "educational",
                "language_code": "tqo",
                "training_data_size": "8.2M tokens"
            }
        ]
        
        return [AvailableModel(**model) for model in available_models]
        
    except Exception as e:
        logger.error(f"Error listing available models: {e}")
        raise HTTPException(status_code=500, detail="Failed to list available models")

@router.get("/local", response_model=List[ModelInfo])
async def list_local_models():
    """List only locally installed models"""
    
    try:
        local_models = await _get_local_models()
        return local_models
        
    except Exception as e:
        logger.error(f"Error listing local models: {e}")
        raise HTTPException(status_code=500, detail="Failed to list local models")

@router.get("/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: str):
    """Get detailed information about a specific model"""
    
    try:
        # Check if model exists locally
        local_models = await _get_local_models()
        for model in local_models:
            if model.name == model_name:
                return model
        
        # Check if model is available for download
        available_models = await _get_downloadable_models()
        for model in available_models:
            if model.name == model_name:
                return model
        
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info for {model_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model information")

@router.post("/{model_name}/activate", response_model=ModelActivationStatus)
async def activate_model(model_name: str, request: ModelActivationRequest):
    """Activate a model for inference"""
    
    try:
        # Check if model exists locally
        local_models = await _get_local_models()
        model_to_activate = None
        
        for model in local_models:
            if model.name == model_name:
                model_to_activate = model
                break
        
        if not model_to_activate:
            raise HTTPException(
                status_code=404, 
                detail=f"Model '{model_name}' not found locally. Please download it first."
            )
        
        if not model_to_activate.is_local:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_name}' is not available locally"
            )
        
        # Check system resources
        import psutil
        memory = psutil.virtual_memory()
        available_memory_gb = memory.available / (1024**3)
        
        if available_memory_gb < model_to_activate.memory_requirements_gb:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient memory. Required: {model_to_activate.memory_requirements_gb}GB, Available: {available_memory_gb:.1f}GB"
            )
        
        # Simulate model activation (in production, would load the actual model)
        await asyncio.sleep(2)  # Simulate loading time
        
        # Update system state
        system_state = get_system_state()
        old_model = system_state.get("active_model")
        update_system_state({
            "active_model": model_name,
            "model_loaded": True,
            "model_load_time": datetime.utcnow(),
            "model_path": str(model_to_activate.path) if model_to_activate.path else None
        })
        
        logger.info(f"Activated model: {model_name}")
        
        return ModelActivationStatus(
            model_name=model_name,
            status="active",
            activation_time=datetime.utcnow(),
            previous_model=old_model,
            memory_usage_gb=model_to_activate.memory_requirements_gb,
            inference_ready=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error activating model {model_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to activate model")

@router.post("/{model_name}/deactivate")
async def deactivate_model(model_name: str):
    """Deactivate the currently active model"""
    
    try:
        system_state = get_system_state()
        active_model = system_state.get("active_model")
        
        if active_model != model_name:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_name}' is not currently active"
            )
        
        # Simulate model deactivation
        await asyncio.sleep(1)
        
        # Update system state
        update_system_state({
            "active_model": None,
            "model_loaded": False,
            "model_load_time": None,
            "model_path": None
        })
        
        logger.info(f"Deactivated model: {model_name}")
        
        return {
            "model_name": model_name,
            "status": "deactivated",
            "deactivation_time": datetime.utcnow(),
            "message": "Model successfully deactivated"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deactivating model {model_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to deactivate model")

@router.post("/{model_name}/download", response_model=ModelDownloadStatus)
async def start_model_download(
    model_name: str, 
    request: ModelDownloadRequest,
    background_tasks: BackgroundTasks
):
    """Start downloading a model from the model hub"""
    
    try:
        # Check if model is already being downloaded
        if model_name in _download_tasks:
            existing_task = _download_tasks[model_name]
            if existing_task["status"] in ["downloading", "pending"]:
                return ModelDownloadStatus(
                    model_name=model_name,
                    status=existing_task["status"],
                    progress_percent=existing_task.get("progress", 0),
                    download_speed_mbps=existing_task.get("speed", 0),
                    eta_minutes=existing_task.get("eta", 0),
                    started_at=existing_task["started_at"]
                )
        
        # Check if model is available for download
        available_models = await list_available_models()
        model_to_download = None
        
        for model in available_models:
            if model.name == model_name:
                model_to_download = model
                break
        
        if not model_to_download:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not available for download"
            )
        
        # Check disk space
        import psutil
        disk_usage = psutil.disk_usage(str(settings.MODEL_STORAGE_PATH))
        available_space_mb = disk_usage.free / (1024**2)
        
        if available_space_mb < model_to_download.size_mb * 1.2:  # 20% buffer
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient disk space. Required: {model_to_download.size_mb * 1.2:.0f}MB, Available: {available_space_mb:.0f}MB"
            )
        
        # Create download task
        download_id = f"download_{model_name}_{int(datetime.utcnow().timestamp())}"
        _download_tasks[model_name] = {
            "id": download_id,
            "status": "pending",
            "progress": 0,
            "speed": 0,
            "eta": 0,
            "started_at": datetime.utcnow(),
            "model_info": model_to_download
        }
        
        # Start background download
        background_tasks.add_task(_download_model_background, model_name, model_to_download, request)
        
        return ModelDownloadStatus(
            model_name=model_name,
            status="pending",
            progress_percent=0,
            download_speed_mbps=0,
            eta_minutes=0,
            started_at=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting download for {model_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to start model download")

@router.get("/{model_name}/download/status", response_model=ModelDownloadStatus)
async def get_download_status(model_name: str):
    """Get the current download status for a model"""
    
    try:
        if model_name not in _download_tasks:
            raise HTTPException(
                status_code=404,
                detail=f"No download task found for model '{model_name}'"
            )
        
        task = _download_tasks[model_name]
        
        return ModelDownloadStatus(
            model_name=model_name,
            status=task["status"],
            progress_percent=task.get("progress", 0),
            download_speed_mbps=task.get("speed", 0),
            eta_minutes=task.get("eta", 0),
            started_at=task["started_at"],
            completed_at=task.get("completed_at"),
            error_message=task.get("error")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting download status for {model_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get download status")

@router.delete("/{model_name}/download")
async def cancel_download(model_name: str):
    """Cancel an ongoing model download"""
    
    try:
        if model_name not in _download_tasks:
            raise HTTPException(
                status_code=404,
                detail=f"No download task found for model '{model_name}'"
            )
        
        task = _download_tasks[model_name]
        
        if task["status"] not in ["downloading", "pending"]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel download. Current status: {task['status']}"
            )
        
        # Cancel the download
        task["status"] = "cancelled"
        task["completed_at"] = datetime.utcnow()
        
        logger.info(f"Cancelled download for model: {model_name}")
        
        return {
            "model_name": model_name,
            "status": "cancelled",
            "cancelled_at": datetime.utcnow(),
            "message": "Download cancelled successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling download for {model_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to cancel download")

@router.delete("/{model_name}")
async def delete_model(model_name: str, force: bool = False):
    """Delete a locally installed model"""
    
    try:
        # Check if model exists locally
        local_models = await _get_local_models()
        model_to_delete = None
        
        for model in local_models:
            if model.name == model_name:
                model_to_delete = model
                break
        
        if not model_to_delete:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found locally"
            )
        
        # Check if model is currently active
        system_state = get_system_state()
        if system_state.get("active_model") == model_name and not force:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot delete active model '{model_name}'. Deactivate first or use force=true"
            )
        
        # Deactivate if active and force is true
        if system_state.get("active_model") == model_name and force:
            await deactivate_model(model_name)
        
        # Delete model files
        if model_to_delete.path and model_to_delete.path.exists():
            if model_to_delete.path.is_file():
                model_to_delete.path.unlink()
            else:
                shutil.rmtree(model_to_delete.path)
        
        logger.info(f"Deleted model: {model_name}")
        
        return {
            "model_name": model_name,
            "status": "deleted",
            "deleted_at": datetime.utcnow(),
            "message": "Model deleted successfully",
            "freed_space_mb": model_to_delete.size_mb
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model {model_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete model")

@router.get("/{model_name}/metrics", response_model=ModelMetrics)
async def get_model_metrics(model_name: str):
    """Get performance metrics for a model"""
    
    try:
        # Check if model is active
        system_state = get_system_state()
        if system_state.get("active_model") != model_name:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_name}' is not currently active"
            )
        
        # Return metrics (would be real metrics in production)
        metrics = ModelMetrics(
            model_name=model_name,
            inference_count=0,
            average_response_time_ms=1250,
            tokens_generated=0,
            memory_usage_mb=4200,
            cpu_usage_percent=35.2,
            cache_hit_rate_percent=87.5,
            error_rate_percent=0.1,
            uptime_minutes=145,
            last_inference=datetime.utcnow() if system_state.get("model_loaded") else None
        )
        
        return metrics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metrics for {model_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model metrics")

# Helper functions
async def _get_local_models() -> List[ModelInfo]:
    """Get list of locally installed models"""
    models = []
    
    # Check HuggingFace format models
    hf_path = settings.MODEL_STORAGE_PATH / "hf"
    if hf_path.exists():
        for model_dir in hf_path.iterdir():
            if model_dir.is_dir():
                models.append(await _create_model_info_from_path(model_dir, "hf_transformers"))
    
    # Check GGUF format models
    gguf_path = settings.MODEL_STORAGE_PATH / "gguf"
    if gguf_path.exists():
        for model_file in gguf_path.glob("*.gguf"):
            models.append(await _create_model_info_from_path(model_file, "gguf"))
    
    return models

async def _get_downloadable_models() -> List[ModelInfo]:
    """Get list of models available for download"""
    available_models = await list_available_models()
    
    # Convert to ModelInfo format
    models = []
    for model in available_models:
        model_info = ModelInfo(
            name=model.name,
            version=model.version,
            description=model.description,
            size_mb=model.size_mb,
            format=model.format,
            is_local=False,
            is_active=False,
            path=None,
            created_date=model.created_date,
            memory_requirements_gb=model.min_memory_gb,
            capabilities=model.capabilities,
            model_type=model.model_type,
            license=model.license
        )
        models.append(model_info)
    
    return models

async def _create_model_info_from_path(path: Path, format: str) -> ModelInfo:
    """Create ModelInfo from a local model path"""
    
    # Get basic info
    name = path.stem if path.is_file() else path.name
    
    # Get size
    if path.is_file():
        size_mb = path.stat().st_size / (1024**2)
    else:
        size_mb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024**2)
    
    # Estimate memory requirements (rough heuristic)
    memory_gb = max(4, size_mb / 1000)  # Minimum 4GB, roughly 1GB per 1000MB model
    
    # Check if active
    system_state = get_system_state()
    is_active = system_state.get("active_model") == name
    
    return ModelInfo(
        name=name,
        version="1.0.0",  # Would parse from model config
        description=f"Local {format} model",
        size_mb=int(size_mb),
        format=format,
        is_local=True,
        is_active=is_active,
        path=path,
        created_date=datetime.fromtimestamp(path.stat().st_ctime),
        memory_requirements_gb=memory_gb,
        capabilities=["story_generation", "vocabulary"],  # Would parse from config
        model_type="educational",
        license="Unknown"
    )

async def _download_model_background(model_name: str, model_info: AvailableModel, request: ModelDownloadRequest):
    """Background task to download a model"""
    
    try:
        task = _download_tasks[model_name]
        task["status"] = "downloading"
        
        # Simulate download progress
        total_size_mb = model_info.size_mb
        downloaded_mb = 0
        
        # Create target directory
        if model_info.format == "gguf":
            target_path = settings.MODEL_STORAGE_PATH / "gguf" / f"{model_name}.gguf"
        else:
            target_path = settings.MODEL_STORAGE_PATH / "hf" / model_name
        
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Simulate download in chunks
        chunk_size_mb = 50  # 50MB chunks
        download_speed_mbps = 25  # 25 MB/s simulation
        
        while downloaded_mb < total_size_mb and task["status"] == "downloading":
            await asyncio.sleep(chunk_size_mb / download_speed_mbps)  # Simulate download time
            
            downloaded_mb = min(downloaded_mb + chunk_size_mb, total_size_mb)
            progress = (downloaded_mb / total_size_mb) * 100
            
            remaining_mb = total_size_mb - downloaded_mb
            eta_minutes = remaining_mb / download_speed_mbps / 60
            
            task.update({
                "progress": progress,
                "speed": download_speed_mbps,
                "eta": eta_minutes
            })
        
        if task["status"] == "downloading":
            # Create placeholder file (in production, would be actual model)
            if model_info.format == "gguf":
                target_path.touch()
            else:
                target_path.mkdir(exist_ok=True)
                (target_path / "config.json").touch()
                (target_path / "pytorch_model.bin").touch()
            
            task.update({
                "status": "completed",
                "progress": 100,
                "completed_at": datetime.utcnow()
            })
            
            logger.info(f"Download completed for model: {model_name}")
        
    except Exception as e:
        logger.error(f"Download failed for {model_name}: {e}")
        if model_name in _download_tasks:
            _download_tasks[model_name].update({
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.utcnow()
            })