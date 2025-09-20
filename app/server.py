#!/usr/bin/env python3
"""
Web server for Toaripi SLM.
Provides REST API for educational content generation.
"""

import argparse
import sys
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Union
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from toaripi_slm.inference import ToaripiGenerator
from toaripi_slm.utils import setup_logging

# Pydantic models for API
class GenerationRequest(BaseModel):
    prompt: Optional[str] = None
    topic: Optional[str] = None
    content_type: str = "story"  # story, vocabulary, dialogue, qa
    age_group: str = "primary"   # primary, secondary
    length: str = "short"        # short, medium, long (for stories)
    count: int = 10              # for vocabulary
    max_length: int = 150
    temperature: float = 0.7

class GenerationResponse(BaseModel):
    content: Union[str, List[Dict], List[str]]
    content_type: str
    topic: Optional[str] = None
    age_group: str
    metadata: Dict

class BatchRequest(BaseModel):
    requests: List[GenerationRequest]

class BatchResponse(BaseModel):
    results: List[GenerationResponse]
    processed: int
    errors: int

# Global generator instance
generator: Optional[ToaripiGenerator] = None

# Create FastAPI app
app = FastAPI(
    title="Toaripi SLM API",
    description="Educational content generation API for Toaripi language",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global generator
    model_path = getattr(app.state, 'model_path', None)
    
    if model_path:
        try:
            print(f"🔄 Loading model from: {model_path}")
            generator = ToaripiGenerator.load(model_path)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise e

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "Toaripi SLM Educational Content Generation API",
        "version": "0.1.0",
        "model_loaded": generator is not None,
        "endpoints": {
            "/generate": "Generate educational content",
            "/batch": "Batch generation",
            "/health": "Health check",
            "/content-types": "Available content types"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if generator is not None else "model_not_loaded",
        "model_loaded": generator is not None
    }

@app.get("/content-types")
async def get_content_types():
    """Get available content types and their descriptions."""
    return {
        "content_types": {
            "story": {
                "description": "Generate educational stories in Toaripi",
                "parameters": ["topic", "age_group", "length"]
            },
            "vocabulary": {
                "description": "Generate vocabulary lists with translations",
                "parameters": ["topic", "count", "age_group"]
            },
            "dialogue": {
                "description": "Generate educational dialogues",
                "parameters": ["topic", "age_group"]
            },
            "qa": {
                "description": "Generate comprehension questions",
                "parameters": ["topic", "count"]
            }
        },
        "age_groups": ["primary", "secondary"],
        "lengths": ["short", "medium", "long"]
    }

@app.post("/generate", response_model=GenerationResponse)
async def generate_content(request: GenerationRequest):
    """Generate educational content."""
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Validate request
        if not request.prompt and not request.topic:
            raise HTTPException(status_code=400, detail="Either 'prompt' or 'topic' must be provided")
        
        # Generate content based on type
        if request.prompt:
            # Custom prompt generation
            content = generator.generate_text(
                prompt=request.prompt,
                max_length=request.max_length,
                temperature=request.temperature
            )
            content_type = "custom"
            topic = None
            
        else:
            # Structured content generation
            topic = request.topic
            content_type = request.content_type
            
            if content_type == "story":
                content = generator.generate_story(
                    topic=topic,
                    age_group=request.age_group,
                    length=request.length
                )
                
            elif content_type == "vocabulary":
                content = generator.generate_vocabulary(
                    topic=topic,
                    count=request.count,
                    include_examples=True
                )
                
            elif content_type == "dialogue":
                content = generator.generate_dialogue(
                    scenario=topic,
                    age_group=request.age_group
                )
                
            elif content_type == "qa":
                # Create a context for Q&A generation
                context = f"Educational content about {topic} for {request.age_group} students."
                content = generator.generate_comprehension_questions(
                    text=context,
                    num_questions=request.count
                )
                
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported content type: {content_type}")
        
        # Create response
        response = GenerationResponse(
            content=content,
            content_type=content_type,
            topic=topic,
            age_group=request.age_group,
            metadata={
                "length": request.length if content_type == "story" else None,
                "count": request.count if content_type in ["vocabulary", "qa"] else None,
                "temperature": request.temperature,
                "max_length": request.max_length
            }
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/batch", response_model=BatchResponse)
async def batch_generate(request: BatchRequest):
    """Batch generation endpoint."""
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    errors = 0
    
    for req in request.requests:
        try:
            # Generate content for each request
            gen_request = GenerationRequest(**req.dict())
            result = await generate_content(gen_request)
            results.append(result)
        except Exception as e:
            errors += 1
            # Add error result
            error_result = GenerationResponse(
                content=f"Error: {str(e)}",
                content_type="error",
                age_group=req.age_group,
                metadata={"error": True}
            )
            results.append(error_result)
    
    return BatchResponse(
        results=results,
        processed=len(results),
        errors=errors
    )

@app.get("/examples")
async def get_examples():
    """Get example requests for different content types."""
    return {
        "examples": {
            "story": {
                "topic": "children helping with fishing",
                "content_type": "story",
                "age_group": "primary",
                "length": "short"
            },
            "vocabulary": {
                "topic": "classroom objects",
                "content_type": "vocabulary",
                "count": 5,
                "age_group": "primary"
            },
            "dialogue": {
                "topic": "greeting family members",
                "content_type": "dialogue",
                "age_group": "primary"
            },
            "qa": {
                "topic": "traditional Toaripi fishing",
                "content_type": "qa",
                "count": 3
            },
            "custom": {
                "prompt": "Write a simple sentence in Toaripi about the weather.",
                "max_length": 50,
                "temperature": 0.7
            }
        }
    }

def main():
    """Main entry point for the web server."""
    parser = argparse.ArgumentParser(description="Toaripi SLM Web Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model", help="Path to model directory")
    parser.add_argument("--config", help="Configuration file")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    print(f"🌐 Toaripi SLM Web Server")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    
    # Setup logging
    setup_logging(level=args.log_level.upper())
    
    # Set model path for startup
    if args.model:
        app.state.model_path = args.model
        print(f"Model: {args.model}")
    else:
        print("⚠️ No model specified - API will be available but generation will fail")
        print("   Use --model /path/to/model to load a trained model")
    
    try:
        # Start server
        uvicorn.run(
            "app.server:app" if __name__ != "__main__" else app,
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,
            log_level=args.log_level
        )
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
        return 0
    except Exception as e:
        print(f"❌ Server error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())