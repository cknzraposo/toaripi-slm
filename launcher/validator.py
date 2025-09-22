"""System validation for Toaripi SLM launcher."""

import sys
import subprocess
import importlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from rich.console import Console

@dataclass
class ValidationIssue:
    """Represents a system validation issue."""
    component: str
    issue: str
    severity: str  # 'error', 'warning', 'info'
    fix_suggestion: str
    auto_fixable: bool = False

@dataclass
class ValidationResult:
    """Results of system validation."""
    is_valid: bool
    issues: List[ValidationIssue]
    python_version: Optional[str] = None
    venv_exists: bool = False
    dependencies_installed: bool = False
    toaripi_data_available: bool = False
    educational_config_valid: bool = False
    system_info: Dict[str, Any] = None

    def __post_init__(self):
        if self.system_info is None:
            self.system_info = {}

class SystemValidator:
    """Validates system requirements for Toaripi SLM."""
    
    def __init__(self, console: Console):
        self.console = console
        self.project_root = Path.cwd()
        
        # Required Python packages for basic functionality
        self.required_packages = [
            'click',
            'rich', 
            'pydantic',
            'pandas',
            'psutil'
        ]
        
        # Educational-specific requirements
        self.educational_requirements = [
            'transformers',
            'torch',
            'datasets'
        ]
        
    def validate_all(self) -> ValidationResult:
        """Run all validation checks."""
        self.console.print("[dim]Running comprehensive system validation...[/dim]")
        issues = []
        
        # Basic system checks
        python_issue = self._check_python_version()
        if python_issue:
            issues.append(python_issue)
            
        venv_issue = self._check_virtual_environment()  
        if venv_issue:
            issues.append(venv_issue)
            
        deps_issue = self._check_dependencies()
        if deps_issue:
            issues.extend(deps_issue)
            
        # Toaripi-specific checks
        structure_issue = self._check_project_structure()
        if structure_issue:
            issues.append(structure_issue)
            
        data_issue = self._check_educational_data()
        if data_issue:
            issues.append(data_issue)
            
        config_issue = self._check_educational_config()
        if config_issue:
            issues.append(config_issue)
            
        # Determine overall system health
        error_count = len([i for i in issues if i.severity == 'error'])
        is_valid = error_count == 0
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            venv_exists=self._venv_exists(),
            dependencies_installed=self._dependencies_installed(),
            toaripi_data_available=self._data_available(),
            educational_config_valid=self._config_valid(),
            system_info=self._get_system_info()
        )
        
    def _check_python_version(self) -> Optional[ValidationIssue]:
        """Check Python version compatibility."""
        version = sys.version_info
        
        if version < (3, 10):
            return ValidationIssue(
                component="Python",
                issue=f"Python {version.major}.{version.minor} is too old for Toaripi SLM",
                severity="error",
                fix_suggestion="Install Python 3.10 or newer from https://python.org/downloads/",
                auto_fixable=False
            )
        elif version >= (3, 13):
            return ValidationIssue(
                component="Python",
                issue=f"Python {version.major}.{version.minor} may have compatibility issues",
                severity="warning", 
                fix_suggestion="Python 3.10-3.12 are recommended for best compatibility",
                auto_fixable=False
            )
        return None
        
    def _check_virtual_environment(self) -> Optional[ValidationIssue]:
        """Check virtual environment setup."""
        venv_path = self.project_root / ".venv"
        
        if not venv_path.exists():
            return ValidationIssue(
                component="Virtual Environment",
                issue="No virtual environment found - required for educational content training",
                severity="error",
                fix_suggestion="Run 'python -m venv .venv' to create virtual environment",
                auto_fixable=True
            )
            
        # Check if virtual environment has Python
        if sys.platform == "win32":
            python_exe = venv_path / "Scripts" / "python.exe"
        else:
            python_exe = venv_path / "bin" / "python"
            
        if not python_exe.exists():
            return ValidationIssue(
                component="Virtual Environment",
                issue="Virtual environment appears corrupted",
                severity="error",
                fix_suggestion="Delete .venv folder and run 'python -m venv .venv' to recreate",
                auto_fixable=True
            )
            
        return None
        
    def _check_dependencies(self) -> List[ValidationIssue]:
        """Check required Python dependencies."""
        issues = []
        missing_basic = []
        missing_educational = []
        
        # Check basic requirements
        for package in self.required_packages:
            try:
                importlib.import_module(package)
            except ImportError:
                missing_basic.append(package)
                
        # Check educational requirements  
        for package in self.educational_requirements:
            try:
                importlib.import_module(package)
            except ImportError:
                missing_educational.append(package)
                
        if missing_basic:
            issues.append(ValidationIssue(
                component="Basic Dependencies",
                issue=f"Missing required packages: {', '.join(missing_basic)}",
                severity="error",
                fix_suggestion="Run 'pip install -e .' to install basic requirements",
                auto_fixable=True
            ))
            
        if missing_educational:
            issues.append(ValidationIssue(
                component="Educational Dependencies", 
                issue=f"Missing educational packages: {', '.join(missing_educational)}",
                severity="warning",
                fix_suggestion="Run 'pip install -r requirements.txt' for full educational features",
                auto_fixable=True
            ))
            
        return issues
        
    def _check_project_structure(self) -> Optional[ValidationIssue]:
        """Check Toaripi SLM project structure."""
        required_paths = [
            "src/toaripi_slm",
            "configs",
            "data", 
            "docs"
        ]
        
        missing_paths = []
        for path in required_paths:
            if not (self.project_root / path).exists():
                missing_paths.append(path)
                
        if missing_paths:
            return ValidationIssue(
                component="Project Structure",
                issue=f"Missing required directories: {', '.join(missing_paths)}",
                severity="error",
                fix_suggestion="Ensure you're running launcher from the toaripi-slm project root",
                auto_fixable=False
            )
            
        return None
        
    def _check_educational_data(self) -> Optional[ValidationIssue]:
        """Check availability of educational training data."""
        data_files = [
            "data/raw/Full_bible_english_toaripi.csv",
            "data/samples/sample_parallel.csv"
        ]
        
        available_files = []
        for file_path in data_files:
            if (self.project_root / file_path).exists():
                available_files.append(file_path)
                
        if not available_files:
            return ValidationIssue(
                component="Educational Data",
                issue="No Toaripi-English parallel training data found",
                severity="warning",
                fix_suggestion="Place Toaripi-English parallel text files in data/raw/ directory",
                auto_fixable=False
            )
            
        return None
        
    def _check_educational_config(self) -> Optional[ValidationIssue]:
        """Check educational configuration files."""
        config_files = [
            "configs/training/toaripi_educational_config.yaml",
            "configs/data/preprocessing_config.yaml"
        ]
        
        missing_configs = []
        for config_path in config_files:
            if not (self.project_root / config_path).exists():
                missing_configs.append(config_path)
                
        if missing_configs:
            return ValidationIssue(
                component="Educational Configuration",
                issue=f"Missing educational config files: {', '.join(missing_configs)}",
                severity="warning", 
                fix_suggestion="Educational configurations will use defaults for age-appropriate content",
                auto_fixable=False
            )
            
        return None
        
    def _venv_exists(self) -> bool:
        """Check if virtual environment exists."""
        return (self.project_root / ".venv").exists()
        
    def _dependencies_installed(self) -> bool:
        """Check if basic dependencies are installed."""
        for package in self.required_packages:
            try:
                importlib.import_module(package)
            except ImportError:
                return False
        return True
        
    def _data_available(self) -> bool:
        """Check if educational data is available."""
        data_files = [
            "data/raw/Full_bible_english_toaripi.csv",
            "data/samples/sample_parallel.csv"
        ]
        
        return any((self.project_root / f).exists() for f in data_files)
        
    def _config_valid(self) -> bool:
        """Check if educational configurations are valid."""
        config_files = [
            "configs/training/toaripi_educational_config.yaml",
            "configs/data/preprocessing_config.yaml"
        ]
        
        return any((self.project_root / f).exists() for f in config_files)
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for diagnostics."""
        try:
            import psutil
            memory_gb = round(psutil.virtual_memory().total / (1024**3), 1)
            cpu_count = psutil.cpu_count()
        except ImportError:
            memory_gb = "unknown"
            cpu_count = "unknown"
            
        return {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": sys.platform,
            "memory_gb": memory_gb,
            "cpu_count": cpu_count,
            "project_root": str(self.project_root)
        }