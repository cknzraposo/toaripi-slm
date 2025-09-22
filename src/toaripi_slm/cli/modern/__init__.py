"""
Modern CLI Framework for Toaripi SLM

Provides a sleek, user-friendly command-line interface with:
- Rich terminal formatting and animations
- Intelligent user guidance and suggestions
- Adaptive interface based on user experience level
- Cultural sensitivity and educational focus
"""

from .framework import ModernCLI, CLIContext
from .user_profiles import UserProfile
from .guidance_system import SmartGuidance, GuidanceEngine
from .progress_display import ModernProgress, ProgressManager
from .error_handling import ErrorHandler, SmartErrorRecovery

__all__ = [
    'ModernCLI',
    'CLIContext', 
    'UserProfile',
    'SmartGuidance',
    'GuidanceEngine',
    'ModernProgress',
    'ProgressManager',
    'ErrorHandler',
    'SmartErrorRecovery'
]