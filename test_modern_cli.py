"""
Test Modern CLI Components

Simple test to verify the modern CLI framework is working correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_framework_import():
    """Test that framework components can be imported."""
    try:
        from src.toaripi_slm.cli.modern.framework import CLIContext, ModernCLI, create_modern_cli_context
        print("✅ Framework import: OK")
        return True
    except Exception as e:
        print(f"❌ Framework import failed: {e}")
        return False

def test_user_profiles():
    """Test user profile system."""
    try:
        from src.toaripi_slm.cli.modern.user_profiles import UserProfile, UserProfileManager
        print("✅ User profiles import: OK")
        
        # Test creating a profile
        profile = UserProfile(
            display_name="Test User",
            user_type="teacher",
            experience_level="beginner"
        )
        print(f"✅ Profile creation: {profile.display_name}")
        return True
    except Exception as e:
        print(f"❌ User profiles failed: {e}")
        return False

def test_guidance_system():
    """Test guidance system."""
    try:
        from src.toaripi_slm.cli.modern.guidance_system import SmartGuidance, GuidanceEngine
        print("✅ Guidance system import: OK")
        return True
    except Exception as e:
        print(f"❌ Guidance system failed: {e}")
        return False

def test_progress_display():
    """Test progress display system."""
    try:
        from src.toaripi_slm.cli.modern.progress_display import ModernProgress, ProgressManager
        print("✅ Progress display import: OK")
        return True
    except Exception as e:
        print(f"❌ Progress display failed: {e}")
        return False

def test_error_handling():
    """Test error handling system."""
    try:
        from src.toaripi_slm.cli.modern.error_handling import ErrorHandler, SmartErrorRecovery
        print("✅ Error handling import: OK")
        return True
    except Exception as e:
        print(f"❌ Error handling failed: {e}")
        return False

def test_workflows():
    """Test workflow system."""
    try:
        from src.toaripi_slm.cli.modern.workflows import SmartWelcome
        print("✅ Workflows import: OK")
        return True
    except Exception as e:
        print(f"❌ Workflows failed: {e}")
        return False

def test_context_creation():
    """Test creating CLI context."""
    try:
        from src.toaripi_slm.cli.modern.framework import create_modern_cli_context
        
        context = create_modern_cli_context(
            verbose=True,
            working_directory=Path.cwd()
        )
        
        print(f"✅ Context creation: {context.session_id}")
        print(f"   Working dir: {context.working_directory}")
        print(f"   Educational mode: {context.educational_mode}")
        return True
    except Exception as e:
        print(f"❌ Context creation failed: {e}")
        return False

def test_welcome_system():
    """Test smart welcome system."""
    try:
        from src.toaripi_slm.cli.modern.framework import create_modern_cli_context
        from src.toaripi_slm.cli.modern.workflows import SmartWelcome
        
        context = create_modern_cli_context()
        welcome = SmartWelcome(context)
        
        print("✅ Smart welcome system: OK")
        print("   Testing welcome display...")
        
        # Test without actually showing full welcome
        # Just verify the object was created correctly
        assert hasattr(welcome, 'context')
        assert hasattr(welcome, 'show_welcome')
        
        return True
    except Exception as e:
        print(f"❌ Welcome system failed: {e}")
        return False

def run_all_tests():
    """Run all component tests."""
    print("🧪 Testing Modern CLI Framework Components")
    print("=" * 50)
    
    tests = [
        ("Framework Core", test_framework_import),
        ("User Profiles", test_user_profiles),
        ("Guidance System", test_guidance_system),
        ("Progress Display", test_progress_display),
        ("Error Handling", test_error_handling),
        ("Workflows", test_workflows),
        ("Context Creation", test_context_creation),
        ("Welcome System", test_welcome_system),
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n🔍 Testing {name}...")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {name} test crashed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Modern CLI framework is ready.")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed. Check the issues above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)