"""
Unified CLI context and configuration management system.
"""

import os
import json
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List

from rich.console import Console

# CLI Configuration
CLI_VERSION = "0.2.0"  # Updated with new smart help features

import os
import sys
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
import json
import yaml

from rich.console import Console

# Default directories
DEFAULT_CONFIG_DIR = Path("./configs")
DEFAULT_DATA_DIR = Path("./data") 
DEFAULT_MODELS_DIR = Path("./models")
DEFAULT_CACHE_DIR = Path("./cache")
DEFAULT_SESSIONS_DIR = Path("./chat_sessions")

@dataclass
class ToaripiCLIContext:
    """Unified context for all CLI operations."""
    
    # Directory paths
    config_dir: Path = field(default_factory=lambda: DEFAULT_CONFIG_DIR)
    data_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR)
    models_dir: Path = field(default_factory=lambda: DEFAULT_MODELS_DIR)
    cache_dir: Path = field(default_factory=lambda: DEFAULT_CACHE_DIR)
    sessions_dir: Path = field(default_factory=lambda: DEFAULT_SESSIONS_DIR)
    
    # CLI behavior
    verbose: bool = False
    debug: bool = False
    quiet: bool = False
    profile: str = "default"  # beginner, intermediate, advanced, expert
    
    # System info
    platform: str = field(default_factory=lambda: platform.system().lower())
    python_version: tuple = field(default_factory=lambda: sys.version_info[:2])
    
    # Console for rich output
    console: Console = field(default_factory=Console)
    
    # User preferences
    preferences: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize context after creation."""
        self._ensure_directories()
        self._load_preferences()
    
    def _ensure_directories(self):
        """Ensure all required directories exist."""
        directories = [
            self.config_dir,
            self.data_dir,
            self.models_dir,
            self.cache_dir,
            self.sessions_dir,
            self.config_dir / "training",
            self.config_dir / "data",
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.data_dir / "samples",
            self.models_dir / "hf",
            self.models_dir / "gguf",
            self.models_dir / "cache",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_preferences(self):
        """Load user preferences from config file."""
        prefs_file = self.config_dir / "preferences.json"
        if prefs_file.exists():
            try:
                with open(prefs_file, 'r') as f:
                    self.preferences = json.load(f)
            except Exception as e:
                if self.debug:
                    self.log(f"Failed to load preferences: {e}", "debug")
    
    def save_preferences(self):
        """Save current preferences to config file."""
        prefs_file = self.config_dir / "preferences.json"
        try:
            with open(prefs_file, 'w') as f:
                json.dump(self.preferences, f, indent=2)
        except Exception as e:
            self.log(f"Failed to save preferences: {e}", "warning")
    
    def log(self, message: str, level: str = "info"):
        """Log messages with appropriate styling."""
        if self.quiet and level in ["info", "debug"]:
            return
            
        if level == "error":
            self.console.print(f"❌ {message}", style="red")
        elif level == "warning":
            self.console.print(f"⚠️  {message}", style="yellow")
        elif level == "success":
            self.console.print(f"✅ {message}", style="green")
        elif level == "info":
            if self.verbose or not self.quiet:
                self.console.print(f"ℹ️  {message}", style="blue")
        elif level == "debug" and self.debug:
            self.console.print(f"🐛 {message}", style="dim")
    
    def get_profile_config(self) -> Dict[str, Any]:
        """Get configuration based on current profile."""
        profiles = {
            "beginner": {
                "show_progress": True,
                "auto_fix_issues": True,
                "detailed_help": True,
                "interactive_mode": True,
                "safety_checks": "strict",
                "default_training_epochs": 1,
            },
            "intermediate": {
                "show_progress": True,
                "auto_fix_issues": False,
                "detailed_help": True,
                "interactive_mode": "optional",
                "safety_checks": "normal",
                "default_training_epochs": 2,
            },
            "advanced": {
                "show_progress": True,
                "auto_fix_issues": False,
                "detailed_help": False,
                "interactive_mode": False,
                "safety_checks": "minimal",
                "default_training_epochs": 3,
            },
            "expert": {
                "show_progress": False,
                "auto_fix_issues": False,
                "detailed_help": False,
                "interactive_mode": False,
                "safety_checks": "off",
                "default_training_epochs": 5,
            }
        }
        
        return profiles.get(self.profile, profiles["beginner"])
    
    def check_environment(self) -> Dict[str, Any]:
        """Check the current environment and dependencies."""
        status = {
            "python_version": self.python_version,
            "platform": self.platform,
            "working_directory": Path.cwd(),
            "directories": {
                "config": self.config_dir.exists(),
                "data": self.data_dir.exists(),
                "models": self.models_dir.exists(),
            },
            "dependencies": {}
        }
        
        # Check for required dependencies
        dependencies = [
            "torch", "transformers", "datasets", "accelerate", 
            "peft", "yaml", "pandas", "numpy", "rich", "click"
        ]
        
        for dep in dependencies:
            try:
                __import__(dep)
                status["dependencies"][dep] = "available"
            except ImportError:
                status["dependencies"][dep] = "missing"
        
        return status
    
    def suggest_next_action(self, current_command: str) -> Optional[str]:
        """Suggest next logical action based on current command."""
        suggestions = {
            "model train": [
                "toaripi model test",
                "toaripi chat interactive", 
                "toaripi model export --format gguf"
            ],
            "model test": [
                "toaripi chat interactive",
                "toaripi model export --format gguf"
            ],
            "model export": [
                "toaripi chat interactive --version latest"
            ],
            "data validate": [
                "toaripi model train --interactive"
            ],
            "system doctor": [
                "toaripi workflow quickstart"
            ]
        }
        
        return suggestions.get(current_command, [])

# Global context instance
cli_context = ToaripiCLIContext()

def get_context() -> ToaripiCLIContext:
    """Get the global CLI context."""
    return cli_context

def update_context(**kwargs) -> ToaripiCLIContext:
    """Update the global CLI context."""
    global cli_context
    for key, value in kwargs.items():
        if hasattr(cli_context, key):
            setattr(cli_context, key, value)
    return cli_context