"""
Simple Modern CLI Demo

A demonstration of the modern CLI framework components without
external dependencies that may not be available.
"""

import sys
import os
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.toaripi_slm.cli.modern.framework import CLIContext, ModernCLI, create_modern_cli_context
from src.toaripi_slm.cli.modern.workflows import SmartWelcome


def demo_modern_cli():
    """
    Demonstrate the modern CLI framework capabilities.
    
    This shows how the various components work together to provide
    an enhanced user experience.
    """
    
    print("🌟 Toaripi SLM Modern CLI Demo")
    print("=" * 50)
    
    # Create modern CLI context
    print("\n📋 Creating CLI context...")
    context = create_modern_cli_context(
        verbose=True,
        working_directory=Path.cwd()
    )
    
    # Initialize modern CLI
    print("✨ Initializing modern CLI...")
    modern_cli = ModernCLI(context)
    
    # Show welcome experience
    print("\n🎉 Smart Welcome Experience:")
    print("-" * 30)
    smart_welcome = SmartWelcome(context)
    smart_welcome.show_welcome()
    
    # Demonstrate project analysis
    print("\n🔍 Project Status Analysis:")
    print("-" * 27)
    
    # Check for key project directories and files
    project_root = Path.cwd()
    
    checks = [
        ("Training data", project_root / "data" / "processed" / "train.csv"),
        ("Configuration", project_root / "configs" / "training"),
        ("Models directory", project_root / "models"),
        ("Scripts", project_root / "scripts"),
    ]
    
    for name, path in checks:
        status = "✅ Found" if path.exists() else "❌ Missing"
        print(f"  {status}: {name}")
    
    # Show guidance suggestions
    print("\n💡 Next Steps Guidance:")
    print("-" * 23)
    
    if context.guidance_engine:
        try:
            suggestions = context.guidance_engine.suggest_next_actions(max_suggestions=3)
            for i, suggestion in enumerate(suggestions, 1):
                print(f"  {i}. {suggestion.title}")
                print(f"     {suggestion.description}")
                if suggestion.command:
                    print(f"     Command: toaripi {suggestion.command}")
                print()
        except Exception as e:
            print(f"  Guidance system demo: {e}")
    else:
        print("  📚 Prepare your training data")
        print("  🎓 Configure training parameters")
        print("  🚀 Start model training")
    
    print("\n🎯 Educational Focus Areas:")
    print("-" * 28)
    print("  📖 Story generation for primary students")
    print("  📝 Vocabulary exercises with cultural context")
    print("  🗣️ Interactive dialogues for classroom practice")
    print("  🧠 Reading comprehension materials")
    print("  🌍 Cultural preservation through education")
    
    print("\n✨ Modern CLI Features Demonstrated:")
    print("-" * 38)
    print("  🎨 Rich formatting with fallbacks")
    print("  👤 User profile management")
    print("  🧭 Intelligent guidance system")
    print("  📊 Project status analysis")
    print("  🎓 Educational content focus")
    print("  🌏 Cultural sensitivity built-in")
    print("  📱 Beginner-friendly interface")
    
    print(f"\n🏁 Demo complete! Modern CLI framework ready.")
    print(f"📁 Working directory: {context.working_directory}")
    print(f"🔧 Configuration: {context.config_file or 'Default settings'}")


def interactive_demo():
    """Interactive demonstration of modern CLI features."""
    
    print("🌟 Interactive Modern CLI Demo")
    print("=" * 50)
    
    context = create_modern_cli_context()
    smart_welcome = SmartWelcome(context)
    
    while True:
        print("\n🎯 Choose a demo:")
        print("1. Smart Welcome Experience")
        print("2. Project Status Analysis")
        print("3. Natural Language Input")
        print("4. Exit")
        
        try:
            choice = input("\n👉 Your choice (1-4): ").strip()
            
            if choice == "1":
                print("\n🎉 Smart Welcome:")
                smart_welcome.show_welcome()
                
            elif choice == "2":
                print("\n Project Analysis:")
                # This would show project status
                print("Analyzing project structure...")
                print("✅ Modern CLI framework: Active")
                print("🎓 Educational mode: Enabled")
                print("🌏 Cultural validation: Active")
                
            elif choice == "3":
                query = input("\n💬 Ask in natural language: ")
                if query:
                    print(f"\n🤖 Processing: '{query}'")
                    # This would use natural language processing
                    print("💡 Suggested action: Check available commands")
                    
            elif choice == "4":
                print("\n👋 Thanks for trying the modern CLI demo!")
                break
                
            else:
                print("❌ Invalid choice. Please try again.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Demo interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Demo error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_demo()
    else:
        demo_modern_cli()