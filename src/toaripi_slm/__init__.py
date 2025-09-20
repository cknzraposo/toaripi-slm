"""
Toaripi Small Language Model (SLM)

A specialized language model for generating educational content in Toaripi,
an endangered language from Papua New Guinea's Gulf Province.

This package provides tools for:
- Fine-tuning language models on Toaripi data
- Generating educational content (stories, vocabulary, Q&A)
- Deploying models for online and offline use
- Supporting language preservation and education efforts

Example usage:
    >>> from toaripi_slm import ToaripiGenerator
    >>> generator = ToaripiGenerator.load("models/toaripi-mistral")
    >>> story = generator.generate_story("children fishing", age_group="primary")
"""

__version__ = "0.1.0"
__author__ = "Toaripi SLM Contributors"
__email__ = "cknzraposo@gmail.com"
__license__ = "MIT"

# Main imports
try:
    from .core import ToaripiTrainer, ToaripiTrainingConfig
    from .inference import ToaripiGenerator, ContentType, AgeGroup
    from .data import DataProcessor
    from .utils import load_config, setup_logging
    
    __all__ = [
        "ToaripiTrainer",
        "ToaripiTrainingConfig",
        "ToaripiGenerator", 
        "ContentType",
        "AgeGroup",
        "DataProcessor",
        "load_config",
        "setup_logging",
    ]
except ImportError:
    # Handle case where dependencies aren't installed yet
    __all__ = []

# Package metadata
PACKAGE_INFO = {
    "name": "toaripi-slm",
    "version": __version__,
    "description": "Small Language Model for Toaripi Educational Content",
    "language": "Toaripi (tqo)",
    "region": "Gulf Province, Papua New Guinea",
    "purpose": "Educational content generation and language preservation",
    "deployment": ["online", "offline", "edge"],
    "model_sizes": ["1B", "3B", "7B"],
}