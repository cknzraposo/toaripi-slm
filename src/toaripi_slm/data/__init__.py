"""
Data processing and management for Toaripi SLM.

This module provides comprehensive data handling capabilities including:
- Custom dataset classes for parallel text data
- Educational prompt templates for content generation
- Defensive data validation and preprocessing utilities
"""

# Only import the preprocessing utilities by default (no torch dependency)
from .preprocessing import TextCleaner, DataValidator, preprocess_dataset

# Lazy imports for torch-dependent components
def get_dataset_classes():
    """Lazy import of dataset classes that require torch."""
    from .dataset import ToaripiParallelDataset, DataSample, create_dataloaders
    return ToaripiParallelDataset, DataSample, create_dataloaders

def get_prompt_classes():
    """Lazy import of prompt template classes."""
    from .prompts import (
        ContentType, 
        AgeGroup, 
        PromptTemplate, 
        ToaripiPromptTemplates,
        create_educational_prompt
    )
    return ContentType, AgeGroup, PromptTemplate, ToaripiPromptTemplates, create_educational_prompt

__all__ = [
    # Preprocessing utilities (always available)
    "TextCleaner",
    "DataValidator", 
    "preprocess_dataset",
    
    # Lazy loading functions
    "get_dataset_classes",
    "get_prompt_classes"
]