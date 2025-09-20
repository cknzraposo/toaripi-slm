"""
Main FastAPI application for Toaripi SLM Web Interface

This module provides a web interface for:
- CSV data upload and validation
- Model training with real-time progress
- Model management and deployment
- Educational content generation
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from app.api.upload import router as upload_router
from app.api.training import router as training_router
from app.api.models import router as models_router
from app.api.health import router as health_router
from app.api.generate import router as generate_router
from app.core.config import settings
from app.core.startup import initialize_system, cleanup_system

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    logger.info("Starting Toaripi SLM Web Interface...")
    
    # Initialize system components
    await initialize_system()
    logger.info("System initialization complete")
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down Toaripi SLM Web Interface...")
    await cleanup_system()
    logger.info("Cleanup complete")

# Create FastAPI application
app = FastAPI(
    title="Toaripi SLM Web Interface",
    description="Web interface for training and using Toaripi Small Language Models",
    version="2.1.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware for web browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# API routers
app.include_router(upload_router, prefix="/api/upload", tags=["Upload"])
app.include_router(training_router, prefix="/api/training", tags=["Training"])
app.include_router(models_router, prefix="/api/models", tags=["Models"])
app.include_router(generate_router, prefix="/api/generate", tags=["Generation"])
app.include_router(health_router, prefix="/api/health", tags=["Health"])

# Static files for web interface
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Web interface routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface"""
    try:
        html_file = Path(__file__).parent / "static" / "index.html"
        if html_file.exists():
            return html_file.read_text()
        else:
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Toaripi SLM Web Interface</title>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .container { max-width: 800px; margin: 0 auto; }
                    .status { padding: 20px; background: #e8f5e8; border-radius: 8px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🌺 Toaripi SLM Web Interface</h1>
                    <div class="status">
                        <h2>✅ Backend Running</h2>
                        <p>The Toaripi Small Language Model web interface backend is running successfully.</p>
                        <p><strong>Frontend under construction...</strong></p>
                        <h3>Available APIs:</h3>
                        <ul>
                            <li><a href="/api/docs">📚 API Documentation</a></li>
                            <li><a href="/api/health">💓 Health Check</a></li>
                            <li><a href="/api/models">🧠 Model Management</a></li>
                        </ul>
                    </div>
                </div>
            </body>
            </html>
            """
    except Exception as e:
        logger.error(f"Error serving root page: {e}")
        raise HTTPException(status_code=500, detail="Error loading web interface")

@app.get("/favicon.ico")
async def favicon():
    """Serve favicon"""
    favicon_path = Path(__file__).parent / "static" / "favicon.ico"
    if favicon_path.exists():
        return favicon_path.read_bytes()
    return HTMLResponse(content="", status_code=404)

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info"
    )