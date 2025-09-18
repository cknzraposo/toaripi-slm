#!/usr/bin/env python3
"""
Verification script for Toaripi SLM project setup.
Checks that all dependencies are installed and basic functionality works.
"""

import sys
import subprocess
import importlib
import os
from pathlib import Path


def check_python_version():
    """Check Python version is 3.10+"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.10+)")
        return False


def check_package_imports():
    """Check that core packages can be imported"""
    print("\n📦 Checking package imports...")
    
    required_packages = [
        "torch",
        "transformers", 
        "datasets",
        "accelerate",
        "peft",
        "fastapi",
        "pandas",
        "numpy",
        "yaml"
    ]
    
    success = True
    for package in required_packages:
        try:
            module = importlib.import_module(package)
            # Get version if available
            version = getattr(module, "__version__", "unknown")
            print(f"  ✓ {package} ({version})")
        except ImportError:
            print(f"  ✗ {package} - not installed")
            success = False
    
    return success


def check_cuda_availability():
    """Check CUDA availability for GPU training"""
    print("\n🚀 Checking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  ✓ CUDA available with {gpu_count} GPU(s)")
            print(f"  ✓ Primary GPU: {gpu_name}")
            return True
        else:
            print("  ⚠️  CUDA not available - will use CPU")
            return False
    except ImportError:
        print("  ✗ PyTorch not installed")
        return False


def check_project_structure():
    """Check that project directories exist"""
    print("\n📁 Checking project structure...")
    
    required_dirs = [
        "src/toaripi_slm",
        "src/toaripi_slm/core",
        "src/toaripi_slm/data", 
        "src/toaripi_slm/inference",
        "src/toaripi_slm/utils",
        "app/api",
        "app/config",
        "app/ui",
        "configs/data",
        "configs/training",
        "data/raw",
        "data/processed",
        "data/samples",
        "models/hf",
        "models/gguf",
        "tests/unit",
        "tests/integration",
        "scripts",
        "notebooks"
    ]
    
    success = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} - missing")
            success = False
    
    return success


def check_config_files():
    """Check that configuration files exist"""
    print("\n⚙️  Checking configuration files...")
    
    required_files = [
        "requirements.txt",
        "requirements-dev.txt", 
        "setup.py",
        "configs/data/preprocessing_config.yaml",
        "configs/training/base_config.yaml",
        "configs/training/lora_config.yaml",
        ".pre-commit-config.yaml",
        ".env.template"
    ]
    
    success = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - missing")
            success = False
    
    return success


def check_sample_data():
    """Check that sample data files exist and are valid"""
    print("\n📊 Checking sample data...")
    
    try:
        import pandas as pd
        
        # Check parallel data CSV
        csv_path = Path("data/samples/parallel_data.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if "english" in df.columns and "toaripi" in df.columns:
                print(f"  ✓ parallel_data.csv ({len(df)} rows)")
            else:
                print("  ✗ parallel_data.csv - missing required columns")
                return False
        else:
            print("  ✗ parallel_data.csv - missing")
            return False
        
        # Check educational prompts JSON
        import json
        json_path = Path("data/samples/educational_prompts.json")
        if json_path.exists():
            with open(json_path) as f:
                prompts = json.load(f)
            print(f"  ✓ educational_prompts.json ({len(prompts)} prompts)")
        else:
            print("  ✗ educational_prompts.json - missing")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error checking sample data: {e}")
        return False


def check_toaripi_package():
    """Check that the toaripi_slm package can be imported"""
    print("\n🔧 Checking toaripi_slm package...")
    
    try:
        # Add src to path for development install
        sys.path.insert(0, str(Path("src").absolute()))
        
        import toaripi_slm
        print(f"  ✓ toaripi_slm package imported")
        print(f"  ✓ Version: {getattr(toaripi_slm, '__version__', 'dev')}")
        
        # Try importing submodules
        submodules = ['core', 'data', 'inference', 'utils']
        for module in submodules:
            try:
                importlib.import_module(f'toaripi_slm.{module}')
                print(f"  ✓ toaripi_slm.{module}")
            except ImportError as e:
                print(f"  ⚠️  toaripi_slm.{module} - {e}")
        
        return True
        
    except ImportError as e:
        print(f"  ✗ toaripi_slm package - {e}")
        return False


def run_basic_test():
    """Run a basic functionality test"""
    print("\n🧪 Running basic functionality test...")
    
    try:
        # Test PyTorch tensor creation
        import torch
        x = torch.tensor([1.0, 2.0, 3.0])
        print(f"  ✓ PyTorch tensor creation: {x}")
        
        # Test Transformers tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokens = tokenizer("Hello Toaripi!")
        print(f"  ✓ Transformers tokenization: {len(tokens['input_ids'])} tokens")
        
        # Test FastAPI import
        from fastapi import FastAPI
        app = FastAPI()
        print("  ✓ FastAPI application creation")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Basic test failed: {e}")
        return False


def main():
    """Run all verification checks"""
    print("🔍 Toaripi SLM Project Verification")
    print("=" * 40)
    
    checks = [
        check_python_version,
        check_project_structure,
        check_config_files,
        check_sample_data,
        check_package_imports,
        check_toaripi_package,
        check_cuda_availability,
        run_basic_test
    ]
    
    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"  ✗ Check failed with error: {e}")
            results.append(False)
    
    print("\n" + "=" * 40)
    print("📋 Verification Summary")
    print("=" * 40)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 All {total} checks passed! Project is ready for development.")
        return 0
    else:
        print(f"⚠️  {passed}/{total} checks passed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())