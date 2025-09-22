"""
Toaripi SLM Trainer Launcher Package

This package provides a user-friendly launcher system for the Toaripi SLM
educational content trainer. It includes system validation, dependency
checking, and guided setup for users of all technical levels.

The launcher maintains focus on educational content generation while
providing robust technical validation and clear user guidance.
"""

__version__ = "1.0.0"
__author__ = "Toaripi SLM Project"

from .launcher import ToaripiLauncher
from .validator import SystemValidator, ValidationResult
from .guidance import UserGuidance

__all__ = [
    "ToaripiLauncher",
    "SystemValidator", 
    "ValidationResult",
    "UserGuidance"
]