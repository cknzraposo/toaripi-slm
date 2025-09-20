"""
CSV Upload API endpoints for training data validation and processing
"""

import asyncio
import csv
import hashlib
import io
import logging
from pathlib import Path
from typing import Dict, List

from fastapi import APIRouter, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.models.schemas import (
    UploadResponse, ValidationError, UploadStatus, 
    TrainingDataUpload, ParallelTrainingData
)
from app.services.validation import CSVValidator
from app.services.safety import SafetyChecker

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory store for upload tracking (would use database in production)
_upload_store: Dict[str, TrainingDataUpload] = {}
_validation_cache: Dict[str, List[ParallelTrainingData]] = {}

@router.post("/csv", response_model=UploadResponse)
async def upload_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload CSV file containing parallel English↔Toaripi training data
    
    The file must have columns: english, toaripi
    Optional columns: category, difficulty
    """
    try:
        # Validate file type and size
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
            
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=400, 
                detail="Only CSV files are allowed"
            )
        
        # Read file content
        content = await file.read()
        file_size = len(content)
        
        if file_size > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE // 1024 // 1024}MB"
            )
        
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Calculate checksum
        checksum = hashlib.sha256(content).hexdigest()
        
        # Check for duplicate uploads
        for upload in _upload_store.values():
            if upload.checksum == checksum:
                raise HTTPException(
                    status_code=409,
                    detail="File already uploaded recently"
                )
        
        # Create upload record
        upload_record = TrainingDataUpload(
            filename=file.filename,
            file_size=file_size,
            mime_type=file.content_type or "text/csv",
            checksum=checksum,
            status=UploadStatus.PENDING
        )
        
        # Store upload record
        _upload_store[upload_record.upload_id] = upload_record
        
        # Save file to disk
        upload_path = settings.UPLOAD_DIR / f"{upload_record.upload_id}.csv"
        with open(upload_path, "wb") as f:
            f.write(content)
        
        logger.info(f"File uploaded: {file.filename} ({file_size} bytes)")
        
        # Start background validation
        background_tasks.add_task(
            validate_csv_background,
            upload_record.upload_id,
            upload_path
        )
        
        return UploadResponse(
            success=True,
            upload_id=upload_record.upload_id,
            message="File uploaded successfully. Validation in progress...",
            total_pairs=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail="Upload failed")


@router.get("/{upload_id}/status")
async def get_upload_status(upload_id: str):
    """Get processing status of uploaded CSV file"""
    
    if upload_id not in _upload_store:
        raise HTTPException(status_code=404, detail="Upload not found")
    
    upload = _upload_store[upload_id]
    
    response_data = {
        "upload_id": upload_id,
        "status": upload.status,
        "filename": upload.filename,
        "upload_timestamp": upload.upload_timestamp.isoformat(),
        "total_pairs": upload.row_count,
        "processing_progress": 100.0 if upload.status in [UploadStatus.VALID, UploadStatus.INVALID] else 50.0,
        "ready_for_training": upload.status == UploadStatus.VALID
    }
    
    if upload.status == UploadStatus.VALID and upload_id in _validation_cache:
        valid_pairs = len([p for p in _validation_cache[upload_id] 
                          if p.content_safety_score >= settings.SAFETY_THRESHOLD])
        response_data["valid_pairs"] = valid_pairs
    
    return response_data


@router.get("/{upload_id}/validation")
async def get_validation_details(upload_id: str):
    """Get detailed validation results for uploaded CSV"""
    
    if upload_id not in _upload_store:
        raise HTTPException(status_code=404, detail="Upload not found")
    
    upload = _upload_store[upload_id]
    
    if upload.status not in [UploadStatus.VALID, UploadStatus.INVALID]:
        raise HTTPException(status_code=400, detail="Validation not complete")
    
    validation_data = _validation_cache.get(upload_id, [])
    
    # Calculate summary statistics
    total_rows = len(validation_data)
    valid_rows = len([p for p in validation_data 
                     if p.content_safety_score >= settings.SAFETY_THRESHOLD])
    safety_warnings = len([p for p in validation_data 
                          if p.safety_flags])
    
    response = {
        "upload_id": upload_id,
        "validation_summary": {
            "total_rows": total_rows,
            "valid_rows": valid_rows,
            "invalid_rows": total_rows - valid_rows,
            "safety_warnings": safety_warnings
        },
        "validation_errors": upload.error_details or [],
        "safety_issues": [],
        "character_issues": []
    }
    
    # Add safety and character issues
    for pair in validation_data:
        if pair.safety_flags:
            response["safety_issues"].append({
                "row_number": pair.row_number,
                "concerns": pair.safety_flags,
                "score": pair.content_safety_score,
                "action": "EXCLUDED" if pair.content_safety_score < settings.SAFETY_THRESHOLD else "INCLUDED"
            })
        
        if not pair.character_validation:
            response["character_issues"].append({
                "row_number": pair.row_number,
                "field": "toaripi",
                "suggestion": "Use standard Toaripi orthography"
            })
    
    return response


@router.get("/{upload_id}/preview")
async def get_upload_preview(
    upload_id: str,
    limit: int = 10,
    offset: int = 0
):
    """Get sample of validated training data for user review"""
    
    if upload_id not in _upload_store:
        raise HTTPException(status_code=404, detail="Upload not found")
    
    upload = _upload_store[upload_id]
    
    if upload.status != UploadStatus.VALID:
        raise HTTPException(status_code=400, detail="Upload not validated")
    
    validation_data = _validation_cache.get(upload_id, [])
    
    # Filter valid pairs only
    valid_pairs = [
        p for p in validation_data 
        if p.content_safety_score >= settings.SAFETY_THRESHOLD
    ]
    
    # Paginate results
    total_valid = len(valid_pairs)
    start_idx = offset
    end_idx = min(offset + limit, total_valid)
    
    preview_pairs = valid_pairs[start_idx:end_idx]
    
    preview_data = []
    for pair in preview_pairs:
        preview_data.append({
            "pair_id": pair.pair_id,
            "row_number": pair.row_number,
            "english": pair.english_text,
            "toaripi": pair.toaripi_text,
            "safety_score": pair.content_safety_score
        })
    
    return {
        "upload_id": upload_id,
        "total_valid_pairs": total_valid,
        "preview_data": preview_data,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "has_next": end_idx < total_valid
        }
    }


async def validate_csv_background(upload_id: str, file_path: Path):
    """Background task to validate CSV content"""
    try:
        upload = _upload_store[upload_id]
        upload.status = UploadStatus.VALIDATING
        
        logger.info(f"Starting validation for upload {upload_id}")
        
        # Initialize validators
        csv_validator = CSVValidator()
        safety_checker = SafetyChecker()
        
        # Read and validate CSV
        validation_results = []
        errors = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Parse CSV
        try:
            reader = csv.DictReader(io.StringIO(content))
            
            # Check required columns
            if 'english' not in reader.fieldnames or 'toaripi' not in reader.fieldnames:
                errors.append(ValidationError(
                    field="columns",
                    error_code="MISSING_COLUMNS",
                    message="Required columns 'english' and 'toaripi' not found",
                    suggested_fix="Add columns with headers: english,toaripi"
                ))
                upload.status = UploadStatus.INVALID
                upload.error_details = [error.dict() for error in errors]
                return
            
            # Validate each row
            for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
                try:
                    english_text = row.get('english', '').strip()
                    toaripi_text = row.get('toaripi', '').strip()
                    
                    if not english_text or not toaripi_text:
                        errors.append(ValidationError(
                            field="text",
                            row_number=row_num,
                            error_code="EMPTY_TEXT",
                            message="English or Toaripi text is empty",
                            suggested_fix="Provide text for both columns"
                        ))
                        continue
                    
                    # Validate text length
                    if len(english_text) < settings.MIN_TEXT_LENGTH or len(english_text) > settings.MAX_TEXT_LENGTH:
                        errors.append(ValidationError(
                            field="english",
                            row_number=row_num,
                            error_code="TEXT_LENGTH_INVALID",
                            message=f"Text length must be {settings.MIN_TEXT_LENGTH}-{settings.MAX_TEXT_LENGTH} characters",
                            suggested_fix="Adjust text length"
                        ))
                        continue
                    
                    if len(toaripi_text) < settings.MIN_TEXT_LENGTH or len(toaripi_text) > settings.MAX_TEXT_LENGTH:
                        errors.append(ValidationError(
                            field="toaripi",
                            row_number=row_num,
                            error_code="TEXT_LENGTH_INVALID",
                            message=f"Text length must be {settings.MIN_TEXT_LENGTH}-{settings.MAX_TEXT_LENGTH} characters",
                            suggested_fix="Adjust text length"
                        ))
                        continue
                    
                    # Validate character sets
                    char_validation = csv_validator.validate_toaripi_characters(toaripi_text)
                    
                    # Check content safety
                    safety_score = await safety_checker.check_content_safety(english_text, toaripi_text)
                    safety_flags = safety_checker.get_safety_flags(english_text, toaripi_text)
                    
                    # Create validation result
                    pair_data = ParallelTrainingData(
                        upload_id=upload_id,
                        english_text=english_text,
                        toaripi_text=toaripi_text,
                        row_number=row_num,
                        content_safety_score=safety_score,
                        safety_flags=safety_flags,
                        character_validation=char_validation,
                        length_validation=True
                    )
                    
                    validation_results.append(pair_data)
                    
                except Exception as e:
                    logger.warning(f"Error validating row {row_num}: {e}")
                    errors.append(ValidationError(
                        field="row",
                        row_number=row_num,
                        error_code="PARSING_ERROR",
                        message=f"Error parsing row: {str(e)}",
                        suggested_fix="Check CSV format and encoding"
                    ))
        
        except Exception as e:
            logger.error(f"CSV parsing error: {e}")
            errors.append(ValidationError(
                field="file",
                error_code="CSV_PARSE_ERROR",
                message="Unable to parse CSV file",
                suggested_fix="Check CSV format and encoding"
            ))
            upload.status = UploadStatus.INVALID
            upload.error_details = [error.dict() for error in errors]
            return
        
        # Check if we have enough valid data
        valid_pairs = [
            p for p in validation_results 
            if p.content_safety_score >= settings.SAFETY_THRESHOLD
        ]
        
        if len(valid_pairs) < settings.MIN_TRAINING_PAIRS:
            errors.append(ValidationError(
                field="data",
                error_code="INSUFFICIENT_DATA",
                message=f"Minimum {settings.MIN_TRAINING_PAIRS} valid pairs required, found {len(valid_pairs)}",
                suggested_fix="Add more high-quality training data"
            ))
            upload.status = UploadStatus.INVALID
        else:
            upload.status = UploadStatus.VALID
        
        # Update upload record
        upload.row_count = len(validation_results)
        upload.error_details = [error.dict() for error in errors] if errors else None
        
        # Cache validation results
        _validation_cache[upload_id] = validation_results
        
        logger.info(f"Validation complete for upload {upload_id}: {len(valid_pairs)} valid pairs")
        
    except Exception as e:
        logger.error(f"Validation failed for upload {upload_id}: {e}")
        upload.status = UploadStatus.INVALID
        upload.error_details = [{
            "field": "system",
            "error_code": "VALIDATION_FAILED",
            "message": "Internal validation error",
            "suggested_fix": "Contact support"
        }]