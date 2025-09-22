#!/usr/bin/env python3
"""
Integration test for Toaripi SLM Phase 1 implementation.

This test validates that all core data handling infrastructure
works properly together.
"""

import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_phase1_integration():
    """Test all Phase 1 components work together."""
    print("Testing Toaripi SLM Phase 1 Integration...")
    print("=" * 50)
    
    try:
        # Test data module imports
        from toaripi_slm.data import (
            ToaripiParallelDataset, 
            ContentType, 
            AgeGroup, 
            create_educational_prompt
        )
        print("✓ Data module imports successful")
        
        # Test core module imports  
        from toaripi_slm.core import ModelConfig, ToaripiModelWrapper
        print("✓ Core module imports successful")
        
        # Test utils module imports
        from toaripi_slm.utils import (
            get_device_info, 
            setup_logger,
            ensure_dir,
            safe_json_save
        )
        print("✓ Utils module imports successful")
        
        # Test device info
        device_info = get_device_info()
        print(f"✓ Device info: {device_info['platform']}")
        print(f"  - PyTorch available: {device_info.get('has_torch', False)}")
        if device_info.get('has_torch'):
            print(f"  - PyTorch version: {device_info.get('torch_version', 'unknown')}")
            print(f"  - CUDA available: {device_info.get('has_cuda', False)}")
        
        # Test prompt creation
        prompt = create_educational_prompt(
            english_text="Children learn to fish in the village.",
            toaripi_text="Araro hagane mai manu hua hau ahi.",
            content_type=ContentType.STORY,
            age_group=AgeGroup.PRIMARY_MIDDLE
        )
        print(f"✓ Educational prompt created: {len(prompt)} characters")
        
        # Test model configuration
        config = ModelConfig(
            model_name='microsoft/DialoGPT-small',
            max_length=256,
            use_fp16=False,  # CPU-friendly
            device_map="cpu"
        )
        print(f"✓ Model config created: {config.model_name}")
        
        # Test logger setup
        logger = setup_logger("test_logger")
        logger.info("Logger test successful")
        print("✓ Logger setup successful")
        
        # Test directory creation
        test_dir = Path("test_output")
        ensure_dir(test_dir)
        print(f"✓ Directory creation: {test_dir}")
        
        # Test JSON saving
        test_data = {
            "test": "data",
            "phase": 1,
            "status": "complete"
        }
        json_path = test_dir / "test.json"
        safe_json_save(test_data, json_path)
        print(f"✓ JSON saving: {json_path}")
        
        # Cleanup
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)
        print("✓ Cleanup completed")
        
        print("\n" + "=" * 50)
        print("🎉 Phase 1 Integration Test PASSED!")
        print("All core data handling infrastructure is working properly.")
        print("\nReady to proceed to Phase 2: Core Training Infrastructure")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_phase1_integration()
    sys.exit(0 if success else 1)