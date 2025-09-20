#!/usr/bin/env python3
"""
Interactive command for Toaripi SLM CLI.

Provides real-time interaction with trained models for content generation.
"""

import json
import readline
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import click
from loguru import logger


class ToaripiInteractiveSession:
    """Interactive session manager for Toaripi SLM"""
    
    def __init__(self, model_path: Path, content_type: str = "story"):
        self.model_path = model_path
        self.content_type = content_type
        self.generator = None
        self.history = []
        self.session_stats = {
            "generations": 0,
            "total_time": 0,
            "avg_time": 0
        }
    
    def load_model(self):
        """Load the model for interaction"""
        try:
            from ....inference import ToaripiGenerator
            click.echo(f"🔄 Loading model from {self.model_path}...")
            start_time = time.time()
            self.generator = ToaripiGenerator.load(str(self.model_path))
            load_time = time.time() - start_time
            click.echo(f"✅ Model loaded in {load_time:.2f}s")
            return True
        except ImportError:
            click.echo("❌ Could not import ToaripiGenerator")
            click.echo("💡 Make sure the package is properly installed")
            return False
        except Exception as e:
            click.echo(f"❌ Failed to load model: {e}")
            return False
    
    def generate_content(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate content based on prompt and current settings"""
        if not self.generator:
            return None
        
        start_time = time.time()
        try:
            if self.content_type == "story":
                result = self.generator.generate_story(prompt, **kwargs)
            elif self.content_type == "vocabulary":
                result = self.generator.generate_vocabulary(prompt, **kwargs)
            elif self.content_type == "qa":
                result = self.generator.generate_qa(prompt, **kwargs)
            elif self.content_type == "dialogue":
                result = self.generator.generate_dialogue(prompt, **kwargs)
            else:
                result = self.generator.generate_story(prompt, **kwargs)  # Default
            
            generation_time = time.time() - start_time
            
            # Update stats
            self.session_stats["generations"] += 1
            self.session_stats["total_time"] += generation_time
            self.session_stats["avg_time"] = self.session_stats["total_time"] / self.session_stats["generations"]
            
            # Add to history
            self.history.append({
                "prompt": prompt,
                "generated": result,
                "content_type": self.content_type,
                "time": generation_time,
                "timestamp": time.time()
            })
            
            return result
            
        except Exception as e:
            click.echo(f"❌ Generation failed: {e}")
            return None
    
    def print_help(self):
        """Print help for interactive commands"""
        help_text = """
🌴 Toaripi SLM Interactive Commands:

Content Generation:
  <prompt>              Generate content based on your prompt
  /story <prompt>       Generate a story
  /vocab <topic>        Generate vocabulary for a topic
  /qa <text>           Generate questions about text
  /dialogue <scenario> Generate a dialogue

Settings:
  /type <content_type>  Change content type (story/vocab/qa/dialogue)
  /age <age_group>     Change age group (primary/secondary/adult)
  /length <max_length> Change max generation length

Session Management:
  /history             Show generation history
  /stats               Show session statistics
  /save [filename]     Save session to file
  /clear               Clear history

System:
  /help                Show this help
  /quit or /exit       Exit interactive mode

Examples:
  Children playing by the river
  /story A family gathering for dinner
  /vocab Animals in the village
  /qa What is the importance of fishing?
        """
        click.echo(help_text)
    
    def print_stats(self):
        """Print session statistics"""
        stats = self.session_stats
        click.echo(f"\n📊 Session Statistics:")
        click.echo(f"   Generations: {stats['generations']}")
        click.echo(f"   Total time: {stats['total_time']:.2f}s")
        click.echo(f"   Average time: {stats['avg_time']:.2f}s per generation")
        click.echo(f"   Content type: {self.content_type}")
    
    def save_session(self, filename: Optional[str] = None):
        """Save session history to file"""
        if not filename:
            filename = f"toaripi_session_{int(time.time())}.json"
        
        session_data = {
            "model_path": str(self.model_path),
            "content_type": self.content_type,
            "stats": self.session_stats,
            "history": self.history
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(session_data, f, indent=2)
            click.echo(f"💾 Session saved to {filename}")
        except Exception as e:
            click.echo(f"❌ Failed to save session: {e}")
    
    def run(self, age_group: str = "primary", max_length: int = 200):
        """Run the interactive session"""
        if not self.load_model():
            return
        
        click.echo("\n🌴 Welcome to Toaripi SLM Interactive Mode!")
        click.echo("Type your prompts to generate content, or /help for commands")
        click.echo("Press Ctrl+C or type /quit to exit\n")
        
        # Generation settings
        settings = {
            "age_group": age_group,
            "max_length": max_length
        }
        
        try:
            while True:
                try:
                    # Get user input
                    prompt = input(f"[{self.content_type}] > ").strip()
                    
                    if not prompt:
                        continue
                    
                    # Handle commands
                    if prompt.startswith('/'):
                        parts = prompt.split(' ', 1)
                        command = parts[0].lower()
                        args = parts[1] if len(parts) > 1 else ""
                        
                        if command in ['/quit', '/exit']:
                            break
                        elif command == '/help':
                            self.print_help()
                        elif command == '/stats':
                            self.print_stats()
                        elif command == '/history':
                            if self.history:
                                click.echo("\n📚 Generation History:")
                                for i, entry in enumerate(self.history[-10:], 1):  # Last 10
                                    click.echo(f"{i}. [{entry['content_type']}] {entry['prompt'][:50]}...")
                                    click.echo(f"   → {entry['generated'][:100]}...")
                                    click.echo(f"   ⏱️  {entry['time']:.2f}s\n")
                            else:
                                click.echo("📚 No history yet")
                        elif command == '/clear':
                            self.history.clear()
                            self.session_stats = {"generations": 0, "total_time": 0, "avg_time": 0}
                            click.echo("🗑️  History cleared")
                        elif command == '/save':
                            filename = args if args else None
                            self.save_session(filename)
                        elif command == '/type':
                            if args in ['story', 'vocab', 'qa', 'dialogue']:
                                self.content_type = args
                                click.echo(f"🎯 Content type changed to: {self.content_type}")
                            else:
                                click.echo("❌ Invalid content type. Use: story, vocab, qa, dialogue")
                        elif command == '/age':
                            if args in ['primary', 'secondary', 'adult']:
                                settings['age_group'] = args
                                click.echo(f"👥 Age group changed to: {args}")
                            else:
                                click.echo("❌ Invalid age group. Use: primary, secondary, adult")
                        elif command == '/length':
                            try:
                                length = int(args)
                                if 10 <= length <= 1000:
                                    settings['max_length'] = length
                                    click.echo(f"📏 Max length changed to: {length}")
                                else:
                                    click.echo("❌ Length must be between 10 and 1000")
                            except ValueError:
                                click.echo("❌ Invalid length. Enter a number")
                        elif command.startswith('/story'):
                            old_type = self.content_type
                            self.content_type = "story"
                            if args:
                                result = self.generate_content(args, **settings)
                                if result:
                                    click.echo(f"\n📖 Generated Story:\n{result}\n")
                                    click.echo(f"⏱️  Generated in {self.history[-1]['time']:.2f}s")
                            self.content_type = old_type
                        elif command.startswith('/vocab'):
                            old_type = self.content_type
                            self.content_type = "vocabulary"
                            if args:
                                result = self.generate_content(args, **settings)
                                if result:
                                    click.echo(f"\n📚 Generated Vocabulary:\n{result}\n")
                                    click.echo(f"⏱️  Generated in {self.history[-1]['time']:.2f}s")
                            self.content_type = old_type
                        elif command.startswith('/qa'):
                            old_type = self.content_type
                            self.content_type = "qa"
                            if args:
                                result = self.generate_content(args, **settings)
                                if result:
                                    click.echo(f"\n❓ Generated Q&A:\n{result}\n")
                                    click.echo(f"⏱️  Generated in {self.history[-1]['time']:.2f}s")
                            self.content_type = old_type
                        elif command.startswith('/dialogue'):
                            old_type = self.content_type
                            self.content_type = "dialogue"
                            if args:
                                result = self.generate_content(args, **settings)
                                if result:
                                    click.echo(f"\n💬 Generated Dialogue:\n{result}\n")
                                    click.echo(f"⏱️  Generated in {self.history[-1]['time']:.2f}s")
                            self.content_type = old_type
                        else:
                            click.echo(f"❌ Unknown command: {command}")
                            click.echo("💡 Type /help for available commands")
                    
                    else:
                        # Regular generation
                        click.echo("🔄 Generating...")
                        result = self.generate_content(prompt, **settings)
                        
                        if result:
                            click.echo(f"\n✨ Generated {self.content_type.title()}:")
                            click.echo(f"{result}\n")
                            click.echo(f"⏱️  Generated in {self.history[-1]['time']:.2f}s")
                            click.echo(f"📊 Total generations: {self.session_stats['generations']}")
                        else:
                            click.echo("❌ Generation failed")
                
                except EOFError:
                    break
                except KeyboardInterrupt:
                    break
        
        except Exception as e:
            click.echo(f"❌ Session error: {e}")
        
        finally:
            click.echo("\n👋 Thanks for using Toaripi SLM!")
            if self.history:
                save_session = click.confirm("Save this session?", default=True)
                if save_session:
                    self.save_session()


def find_available_models() -> List[Path]:
    """Find available models for interaction"""
    models = []
    
    # Check checkpoints
    checkpoints_dir = Path("checkpoints")
    if checkpoints_dir.exists():
        for model_dir in checkpoints_dir.glob("*/"):
            if (model_dir / "config.json").exists():
                models.append(model_dir)
    
    # Check models directory
    models_dir = Path("models")
    if models_dir.exists():
        for model_dir in models_dir.glob("*/"):
            if (model_dir / "config.json").exists():
                models.append(model_dir)
    
    return models


@click.command()
@click.option(
    '--model-path',
    type=click.Path(exists=True, path_type=Path),
    help='Path to trained model for interaction'
)
@click.option(
    '--content-type',
    type=click.Choice(['story', 'vocabulary', 'qa', 'dialogue']),
    default='story',
    help='Default content type for generation'
)
@click.option(
    '--age-group',
    type=click.Choice(['primary', 'secondary', 'adult']),
    default='primary',
    help='Target age group for content'
)
@click.option(
    '--max-length',
    type=int,
    default=200,
    help='Maximum length for generated content'
)
@click.option(
    '--guided',
    is_flag=True,
    help='Run in guided mode with model selection'
)
def interact(
    model_path: Optional[Path],
    content_type: str,
    age_group: str,
    max_length: int,
    guided: bool
):
    """
    Start an interactive session with a trained Toaripi model.
    
    This command provides a conversational interface for generating
    educational content in real-time. You can change settings during
    the session and save your interaction history.
    
    \b
    Examples:
        toaripi interact --guided           # Select model interactively
        toaripi interact --model-path models/my-model
        toaripi interact --content-type vocabulary
        toaripi interact --age-group secondary
    """
    
    if guided or not model_path:
        click.echo("💬 Welcome to Toaripi SLM Interactive Mode!\n")
        
        if not model_path:
            click.echo("🔍 Searching for available models...")
            available_models = find_available_models()
            
            if not available_models:
                click.echo("❌ No trained models found!")
                click.echo("💡 Tips:")
                click.echo("   • Train a model first: toaripi train")
                click.echo("   • Download a pre-trained model")
                click.echo("   • Check that models have config.json files")
                return
            
            click.echo("📚 Available models:")
            for i, model in enumerate(available_models, 1):
                click.echo(f"  {i}. {model}")
            
            choice = click.prompt(
                f"\nSelect model for interaction (1-{len(available_models)})",
                type=click.IntRange(1, len(available_models))
            )
            
            model_path = available_models[choice - 1]
        
        # Content type selection
        if guided:
            click.echo(f"\n🎯 What type of content would you like to generate?")
            content_options = [
                ("story", "Educational stories and narratives"),
                ("vocabulary", "Vocabulary lists and word exercises"),
                ("qa", "Questions and answers for comprehension"),
                ("dialogue", "Conversations and dialogues")
            ]
            
            for i, (name, desc) in enumerate(content_options, 1):
                click.echo(f"  {i}. {name.title()}: {desc}")
            
            choice = click.prompt(
                f"\nSelect content type (1-{len(content_options)})",
                type=click.IntRange(1, len(content_options)),
                default=1
            )
            
            content_type = content_options[choice - 1][0]
        
        # Age group selection
        if guided:
            click.echo(f"\n👥 What age group is your target audience?")
            age_options = [
                ("primary", "Primary school students (6-12 years)"),
                ("secondary", "Secondary school students (13-18 years)"),
                ("adult", "Adult learners (18+ years)")
            ]
            
            for i, (name, desc) in enumerate(age_options, 1):
                click.echo(f"  {i}. {name.title()}: {desc}")
            
            choice = click.prompt(
                f"\nSelect age group (1-{len(age_options)})",
                type=click.IntRange(1, len(age_options)),
                default=1
            )
            
            age_group = age_options[choice - 1][0]
    
    # Validate model path
    if not model_path or not model_path.exists():
        click.echo(f"❌ Model not found: {model_path}")
        return
    
    # Check for config.json
    if not (model_path / "config.json").exists():
        click.echo(f"❌ Invalid model: {model_path} (missing config.json)")
        return
    
    # Start interactive session
    session = ToaripiInteractiveSession(model_path, content_type)
    session.run(age_group, max_length)


if __name__ == '__main__':
    interact()