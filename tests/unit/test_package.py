"""Unit tests for core package functionality."""

import pytest
from pathlib import Path


def test_package_metadata():
    """Test package metadata is accessible."""
    import src.toaripi_slm as pkg
    
    assert hasattr(pkg, '__version__')
    assert hasattr(pkg, '__author__')
    assert hasattr(pkg, 'PACKAGE_INFO')
    
    # Check package info structure
    info = pkg.PACKAGE_INFO
    required_keys = ['name', 'version', 'description', 'language', 'purpose']
    for key in required_keys:
        assert key in info, f"Missing key {key} in PACKAGE_INFO"


def test_import_structure():
    """Test that expected modules can be imported."""
    # Test that modules exist (even if classes aren't implemented yet)
    module_paths = [
        'src.toaripi_slm.core',
        'src.toaripi_slm.data', 
        'src.toaripi_slm.inference',
        'src.toaripi_slm.utils'
    ]
    
    for module_path in module_paths:
        try:
            __import__(module_path)
        except ImportError as e:
            pytest.fail(f"Could not import {module_path}: {e}")