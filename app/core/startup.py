"""
System startup and shutdown management
"""

import asyncio
import logging
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)

# Global state for system components
_system_state = {
    "initialized": False,
    "model_loaded": False,
    "active_model": None,
    "training_sessions": {},
    "upload_cache": {}
}

async def initialize_system():
    """Initialize all system components"""
    try:
        logger.info("Initializing Toaripi SLM system...")
        
        # Create necessary directories
        _ensure_directories()
        
        # Initialize model cache
        await _initialize_model_cache()
        
        # Initialize training session manager
        _initialize_training_manager()
        
        # Initialize upload manager
        _initialize_upload_manager()
        
        # Load active model if available
        await _load_active_model()
        
        _system_state["initialized"] = True
        logger.info("System initialization complete")
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        raise

async def cleanup_system():
    """Cleanup system resources on shutdown"""
    try:
        logger.info("Cleaning up system resources...")
        
        # Cancel any running training sessions
        await _cleanup_training_sessions()
        
        # Clear upload cache
        _cleanup_upload_cache()
        
        # Unload models to free memory
        await _unload_models()
        
        _system_state["initialized"] = False
        logger.info("System cleanup complete")
        
    except Exception as e:
        logger.error(f"System cleanup failed: {e}")

def _ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        settings.UPLOAD_DIR,
        settings.MODEL_OUTPUT_DIR,
        settings.DATA_DIR / "processed",
        settings.MODELS_DIR / "hf",
        settings.MODELS_DIR / "gguf",
        settings.MODELS_DIR / "cache"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")

async def _initialize_model_cache():
    """Initialize model caching system"""
    logger.info("Initializing model cache...")
    # Model cache will be implemented in model management
    pass

def _initialize_training_manager():
    """Initialize training session manager"""
    logger.info("Initializing training manager...")
    _system_state["training_sessions"] = {}

def _initialize_upload_manager():
    """Initialize upload manager"""
    logger.info("Initializing upload manager...")
    _system_state["upload_cache"] = {}

async def _load_active_model():
    """Load the active model for content generation"""
    try:
        # Check for existing models
        models_dir = settings.MODELS_DIR / "gguf"
        if models_dir.exists():
            model_files = list(models_dir.glob("*.gguf"))
            if model_files:
                # Load the most recent model as active
                latest_model = max(model_files, key=lambda f: f.stat().st_mtime)
                logger.info(f"Found model: {latest_model.name}")
                _system_state["active_model"] = latest_model.stem
                _system_state["model_loaded"] = True
            else:
                logger.info("No trained models found")
        else:
            logger.info("Models directory not found")
            
    except Exception as e:
        logger.warning(f"Error loading active model: {e}")

async def _cleanup_training_sessions():
    """Cancel any running training sessions"""
    training_sessions = _system_state.get("training_sessions", {})
    if training_sessions:
        logger.info(f"Cancelling {len(training_sessions)} active training sessions...")
        for session_id, session in training_sessions.items():
            try:
                if hasattr(session, 'cancel'):
                    await session.cancel()
                logger.info(f"Cancelled training session: {session_id}")
            except Exception as e:
                logger.warning(f"Error cancelling session {session_id}: {e}")

def _cleanup_upload_cache():
    """Clear upload cache"""
    upload_cache = _system_state.get("upload_cache", {})
    if upload_cache:
        logger.info(f"Clearing {len(upload_cache)} cached uploads...")
        upload_cache.clear()

async def _unload_models():
    """Unload models to free memory"""
    if _system_state.get("model_loaded"):
        logger.info("Unloading active model...")
        _system_state["model_loaded"] = False
        _system_state["active_model"] = None

def get_system_state():
    """Get current system state"""
    return _system_state.copy()

def update_system_state(updates: dict):
    """Update system state with new values"""
    _system_state.update(updates)

def is_system_ready():
    """Check if system is ready to handle requests"""
    return _system_state.get("initialized", False)