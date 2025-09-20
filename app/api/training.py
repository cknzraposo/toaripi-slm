"""
Training API from app.models.schemas import (
    TrainingSession, TrainingRequest, TrainingSessionResponse,
    TrainingStatus, TrainingProgressUpdate, ValidationError, TrainingConfig
)
from app.services.training import TrainingManager

router = APIRouter()
logger = logging.getLogger(__name__)

# Global training manager and session store
training_manager = TrainingManager()
_training_sessions: Dict[str, TrainingSession] = {} model training session management
"""

import asyncio
import json
import logging
import psutil
import time
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from app.core.config import settings
from app.models.schemas import (
    TrainingStartRequest, TrainingStartResponse, TrainingSession,
    TrainingStatus, TrainingProgressUpdate, ValidationError, TrainingConfig
)
from app.services.training import TrainingManager

router = APIRouter()
logger = logging.getLogger(__name__)

# Global training manager and session store
training_manager = TrainingManager()
_training_sessions: Dict[str, TrainingSession] = {}
_session_progress: Dict[str, TrainingProgressUpdate] = {}

@router.post("/start", response_model=TrainingStartResponse)
async def start_training_session(
    request: TrainingStartRequest,
    background_tasks: BackgroundTasks
):
    """Start a new model training session from validated CSV uploads"""
    
    try:
        # Check if training is already in progress
        active_sessions = [
            s for s in _training_sessions.values() 
            if s.status in [TrainingStatus.QUEUED, TrainingStatus.STARTING, TrainingStatus.TRAINING]
        ]
        
        if len(active_sessions) >= settings.MAX_CONCURRENT_TRAINING:
            raise HTTPException(
                status_code=409,
                detail=f"Training session already in progress: {active_sessions[0].session_id}"
            )
        
        # Validate training configuration
        validation_result = await training_manager.validate_training_config(
            request.upload_ids,
            request.training_config,
            request.model_base
        )
        
        if not validation_result["valid"]:
            return TrainingStartResponse(
                success=False,
                message="Invalid training configuration",
                validation_errors=[
                    ValidationError(**error) for error in validation_result["errors"]
                ]
            )
        
        # Create training session
        session = TrainingSession(
            upload_ids=request.upload_ids,
            config_version="v1.0",
            model_base=request.model_base,
            training_params=request.training_config,
            status=TrainingStatus.QUEUED,
            total_steps=validation_result["estimated_steps"]
        )
        
        # Store session
        _training_sessions[session.session_id] = session
        
        # Initialize progress tracking
        _session_progress[session.session_id] = TrainingProgressUpdate(
            session_id=session.session_id,
            progress_percentage=0.0,
            current_step=0,
            total_steps=session.total_steps,
            status=TrainingStatus.QUEUED,
            message="Training session queued"
        )
        
        # Start training in background
        background_tasks.add_task(
            run_training_session,
            session.session_id
        )
        
        logger.info(f"Training session started: {session.session_id}")
        
        return TrainingStartResponse(
            success=True,
            session_id=session.session_id,
            message="Training session started successfully",
            estimated_duration_minutes=validation_result["estimated_duration_minutes"],
            total_training_pairs=validation_result["total_training_pairs"],
            training_config=request.training_config.dict(),
            progress_stream_url=f"/api/training/{session.session_id}/progress"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training start error: {e}")
        raise HTTPException(status_code=500, detail="Failed to start training")


@router.get("/{session_id}/status")
async def get_training_status(session_id: str):
    """Get current status of training session"""
    
    if session_id not in _training_sessions:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    session = _training_sessions[session_id]
    progress = _session_progress.get(session_id)
    
    response_data = {
        "session_id": session_id,
        "status": session.status,
        "progress_percentage": session.progress_percentage,
        "current_step": session.current_step,
        "total_steps": session.total_steps,
        "start_timestamp": session.start_timestamp.isoformat() if session.start_timestamp else None,
        "current_loss": session.loss_history[-1] if session.loss_history else None,
        "best_loss": min(session.loss_history) if session.loss_history else None
    }
    
    # Add completion info if finished
    if session.status == TrainingStatus.COMPLETED:
        response_data.update({
            "end_timestamp": session.end_timestamp.isoformat() if session.end_timestamp else None,
            "final_loss": session.loss_history[-1] if session.loss_history else None,
            "model_output_path": session.output_model_path,
            "evaluation_metrics": session.evaluation_metrics
        })
    
    # Add resource usage if available
    try:
        memory_usage = psutil.virtual_memory().percent
        response_data["memory_usage_gb"] = psutil.virtual_memory().used / (1024**3)
        response_data["memory_usage_percent"] = memory_usage
    except Exception:
        pass
    
    return response_data


@router.get("/{session_id}/progress")
async def training_progress_stream(session_id: str):
    """Real-time training progress via Server-Sent Events"""
    
    if session_id not in _training_sessions:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    async def event_generator():
        """Generate Server-Sent Events for training progress"""
        try:
            last_update = None
            
            while True:
                # Get current progress
                if session_id in _session_progress:
                    current_progress = _session_progress[session_id]
                    
                    # Only send update if something changed
                    if current_progress != last_update:
                        data = {
                            "session_id": current_progress.session_id,
                            "status": current_progress.status,
                            "progress_percentage": current_progress.progress_percentage,
                            "current_step": current_progress.current_step,
                            "total_steps": current_progress.total_steps,
                            "message": current_progress.message,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        
                        # Add optional fields if available
                        if current_progress.current_loss is not None:
                            data["current_loss"] = current_progress.current_loss
                        
                        if current_progress.estimated_time_remaining is not None:
                            data["estimated_time_remaining"] = current_progress.estimated_time_remaining
                        
                        yield json.dumps(data)
                        last_update = current_progress
                    
                    # Stop streaming if training is complete or failed
                    if current_progress.status in [
                        TrainingStatus.COMPLETED, 
                        TrainingStatus.FAILED, 
                        TrainingStatus.CANCELLED
                    ]:
                        break
                
                # Wait before next update
                await asyncio.sleep(2)  # Update every 2 seconds
                
        except Exception as e:
            logger.error(f"Progress stream error: {e}")
            yield json.dumps({
                "error": "Progress stream failed",
                "session_id": session_id
            })
    
    return EventSourceResponse(event_generator())


@router.post("/{session_id}/cancel")
async def cancel_training_session(session_id: str):
    """Cancel an active training session"""
    
    if session_id not in _training_sessions:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    session = _training_sessions[session_id]
    
    # Check if session can be cancelled
    if session.status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.CANCELLED]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel training session with status: {session.status}"
        )
    
    try:
        # Request cancellation
        await training_manager.cancel_session(session_id)
        
        # Update session status
        session.status = TrainingStatus.CANCELLED
        session.end_timestamp = datetime.utcnow()
        
        # Update progress
        if session_id in _session_progress:
            _session_progress[session_id].status = TrainingStatus.CANCELLED
            _session_progress[session_id].message = "Training cancelled by user"
        
        logger.info(f"Training session cancelled: {session_id}")
        
        return {
            "success": True,
            "session_id": session_id,
            "message": "Training session cancelled successfully",
            "cancelled_at_step": session.current_step,
            "cancelled_at_progress": session.progress_percentage,
            "partial_model_saved": session.current_step > 0
        }
        
    except Exception as e:
        logger.error(f"Training cancellation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to cancel training")


@router.get("/history")
async def get_training_history(
    limit: int = 10,
    offset: int = 0,
    status: Optional[str] = None
):
    """Get list of all training sessions"""
    
    try:
        # Filter sessions by status if specified
        sessions = list(_training_sessions.values())
        
        if status:
            sessions = [s for s in sessions if s.status == status]
        
        # Sort by start time (newest first)
        sessions.sort(
            key=lambda s: s.start_timestamp or datetime.min, 
            reverse=True
        )
        
        # Paginate
        total_sessions = len(sessions)
        paginated_sessions = sessions[offset:offset + limit]
        
        # Format response
        session_data = []
        for session in paginated_sessions:
            session_info = {
                "session_id": session.session_id,
                "session_name": f"Training Session {session.session_id[:8]}",
                "status": session.status,
                "start_timestamp": session.start_timestamp.isoformat() if session.start_timestamp else None,
                "end_timestamp": session.end_timestamp.isoformat() if session.end_timestamp else None,
                "model_output_path": session.output_model_path
            }
            
            # Add duration if completed
            if session.start_timestamp and session.end_timestamp:
                duration = session.end_timestamp - session.start_timestamp
                session_info["duration_minutes"] = int(duration.total_seconds() / 60)
            
            # Add final metrics if available
            if session.loss_history:
                session_info["final_loss"] = session.loss_history[-1]
            
            session_data.append(session_info)
        
        return {
            "total_sessions": total_sessions,
            "sessions": session_data,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "has_next": offset + limit < total_sessions
            }
        }
        
    except Exception as e:
        logger.error(f"Training history error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get training history")


@router.post("/validate-config")
async def validate_training_config(request: TrainingStartRequest):
    """Validate training configuration before starting session"""
    
    try:
        validation_result = await training_manager.validate_training_config(
            request.upload_ids,
            request.training_config,
            request.model_base
        )
        
        if validation_result["valid"]:
            return {
                "valid": True,
                "estimated_duration_minutes": validation_result["estimated_duration_minutes"],
                "estimated_memory_usage_gb": validation_result["estimated_memory_usage_gb"],
                "total_training_pairs": validation_result["total_training_pairs"],
                "estimated_steps": validation_result["estimated_steps"],
                "hardware_compatibility": "COMPATIBLE",
                "constitutional_compliance": "PASSED"
            }
        else:
            return {
                "valid": False,
                "validation_errors": validation_result["errors"],
                "estimated_memory_usage_gb": validation_result.get("estimated_memory_usage_gb"),
                "memory_limit_gb": settings.MAX_MEMORY_GB
            }
            
    except Exception as e:
        logger.error(f"Config validation error: {e}")
        raise HTTPException(status_code=500, detail="Configuration validation failed")


async def run_training_session(session_id: str):
    """Background task to run training session"""
    
    try:
        session = _training_sessions[session_id]
        progress = _session_progress[session_id]
        
        logger.info(f"Starting training session: {session_id}")
        
        # Update session status
        session.status = TrainingStatus.STARTING
        session.start_timestamp = datetime.utcnow()
        progress.status = TrainingStatus.STARTING
        progress.message = "Initializing training environment"
        
        # Run training through training manager
        async for update in training_manager.run_training(session):
            # Update session state
            session.progress_percentage = update.progress_percentage
            session.current_step = update.current_step
            session.status = update.status
            
            if update.current_loss is not None:
                session.loss_history.append(update.current_loss)
            
            # Update progress for streaming
            _session_progress[session_id] = update
            
            logger.debug(f"Training progress: {update.progress_percentage:.1f}%")
        
        # Training completed
        session.status = TrainingStatus.COMPLETED
        session.end_timestamp = datetime.utcnow()
        progress.status = TrainingStatus.COMPLETED
        progress.message = "Training completed successfully"
        progress.progress_percentage = 100.0
        
        logger.info(f"Training session completed: {session_id}")
        
    except Exception as e:
        logger.error(f"Training session failed: {session_id}, error: {e}")
        
        # Update session with failure
        session = _training_sessions[session_id]
        session.status = TrainingStatus.FAILED
        session.end_timestamp = datetime.utcnow()
        
        progress = _session_progress[session_id]
        progress.status = TrainingStatus.FAILED
        progress.message = f"Training failed: {str(e)}"