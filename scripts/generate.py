#!/usr/bin/env python3
"""
Content generation script for Toaripi SLM.
Generates educational content in Toaripi language.
"""

import argparse
import sys
from pathlib import Path
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from toaripi_slm.inference import ToaripiGenerator
from toaripi_slm.utils import setup_logging

def main():
    """Main entry point for content generation."""
    parser = argparse.ArgumentParser(description="Generate Toaripi educational content")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--prompt", help="Generation prompt (for custom generation)")
    parser.add_argument("--topic", help="Topic for structured content generation")
    parser.add_argument("--type", choices=["story", "vocabulary", "dialogue", "qa"], 
                       default="story", help="Content type")
    parser.add_argument("--age-group", choices=["primary", "secondary"], 
                       default="primary", help="Target age group")
    parser.add_argument("--length", choices=["short", "medium", "long"],
                       default="short", help="Content length")
    parser.add_argument("--count", type=int, default=10, help="Number of vocabulary items (for vocabulary type)")
    parser.add_argument("--output", help="Output file (JSON format)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--batch", help="Batch generation from prompts file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    
    print(f"✨ Toaripi Content Generator")
    print(f"Model: {args.model}")
    
    try:
        # Load model
        print("\n📦 Loading trained model...")
        generator = ToaripiGenerator.load(args.model)
        print("✅ Model loaded successfully")
        
        if args.interactive:
            return interactive_mode(generator, args)
        elif args.batch:
            return batch_mode(generator, args)
        else:
            return single_generation(generator, args)
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

def single_generation(generator, args):
    """Generate single piece of content."""
    result = {}
    
    if args.prompt:
        # Custom prompt generation
        print(f"\n🔤 Custom prompt generation:")
        print(f"Prompt: {args.prompt}")
        
        generated = generator.generate_text(
            prompt=args.prompt,
            max_length=150,
            temperature=0.7
        )
        
        result = {
            "type": "custom",
            "prompt": args.prompt,
            "generated": generated
        }
        
    elif args.topic:
        # Structured content generation
        print(f"\n📝 Generating {args.type} content:")
        print(f"Topic: {args.topic}")
        print(f"Age group: {args.age_group}")
        print(f"Length: {args.length}")
        
        if args.type == "story":
            generated = generator.generate_story(
                topic=args.topic,
                age_group=args.age_group,
                length=args.length
            )
            
        elif args.type == "vocabulary":
            generated = generator.generate_vocabulary(
                topic=args.topic,
                count=args.count,
                include_examples=True
            )
            
        elif args.type == "dialogue":
            generated = generator.generate_dialogue(
                scenario=args.topic,
                age_group=args.age_group
            )
            
        elif args.type == "qa":
            # For Q&A, we need some context text
            context = f"Educational content about {args.topic} for {args.age_group} students."
            generated = generator.generate_comprehension_questions(
                text=context,
                num_questions=5
            )
        
        result = {
            "type": args.type,
            "topic": args.topic,
            "age_group": args.age_group,
            "length": args.length if args.type == "story" else None,
            "count": args.count if args.type == "vocabulary" else None,
            "generated": generated
        }
    
    else:
        print("❌ Either --prompt or --topic must be specified")
        return 1
    
    # Display result
    print(f"\n📄 Generated content:")
    print("=" * 50)
    
    if isinstance(result["generated"], str):
        print(result["generated"])
    elif isinstance(result["generated"], list):
        for i, item in enumerate(result["generated"], 1):
            if isinstance(item, dict):
                print(f"{i}. {item.get('toaripi', item)}")
                if 'english' in item:
                    print(f"   ({item['english']})")
            else:
                print(f"{i}. {item}")
    
    print("=" * 50)
    
    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved to: {args.output}")
    
    return 0

def interactive_mode(generator, args):
    """Interactive generation mode."""
    print(f"\n🔄 Interactive mode - type 'quit' to exit")
    print("Commands:")
    print("  story <topic>      - Generate a story")
    print("  vocab <topic>      - Generate vocabulary")
    print("  dialogue <topic>   - Generate dialogue")
    print("  qa <topic>         - Generate Q&A")
    print("  custom <prompt>    - Custom generation")
    print("  quit              - Exit")
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            parts = user_input.split(' ', 1)
            if len(parts) < 2:
                print("❌ Please provide a command and topic/prompt")
                continue
            
            command, content = parts
            command = command.lower()
            
            print(f"\n⚡ Generating {command} content...")
            
            if command == "story":
                result = generator.generate_story(content, age_group=args.age_group)
            elif command == "vocab":
                result = generator.generate_vocabulary(content, count=5)
            elif command == "dialogue":
                result = generator.generate_dialogue(content, age_group=args.age_group)
            elif command == "qa":
                result = generator.generate_comprehension_questions(content, num_questions=3)
            elif command == "custom":
                result = generator.generate_text(content, max_length=150)
            else:
                print(f"❌ Unknown command: {command}")
                continue
            
            # Display result
            print("\n📄 Generated:")
            print("-" * 30)
            if isinstance(result, str):
                print(result)
            elif isinstance(result, list):
                for i, item in enumerate(result, 1):
                    if isinstance(item, dict):
                        print(f"{i}. {item.get('toaripi', item)}")
                        if 'english' in item:
                            print(f"   ({item['english']})")
                    else:
                        print(f"{i}. {item}")
            print("-" * 30)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return 0

def batch_mode(generator, args):
    """Batch generation from file."""
    batch_file = Path(args.batch)
    if not batch_file.exists():
        print(f"❌ Batch file not found: {batch_file}")
        return 1
    
    print(f"\n📦 Batch generation from: {batch_file}")
    
    try:
        with open(batch_file, 'r', encoding='utf-8') as f:
            prompts_data = json.load(f)
        
        if not isinstance(prompts_data, list):
            print("❌ Batch file should contain a list of prompts")
            return 1
        
        results = []
        
        for i, prompt_item in enumerate(prompts_data, 1):
            print(f"\n🔄 Processing {i}/{len(prompts_data)}...")
            
            if isinstance(prompt_item, str):
                # Simple prompt string
                generated = generator.generate_text(prompt_item, max_length=150)
                result = {
                    "prompt": prompt_item,
                    "generated": generated
                }
            elif isinstance(prompt_item, dict):
                # Structured prompt
                prompt_type = prompt_item.get("type", "story")
                topic = prompt_item.get("topic", "")
                
                if prompt_type == "story":
                    generated = generator.generate_story(topic)
                elif prompt_type == "vocabulary":
                    generated = generator.generate_vocabulary(topic)
                elif prompt_type == "dialogue":
                    generated = generator.generate_dialogue(topic)
                elif prompt_type == "qa":
                    generated = generator.generate_comprehension_questions(topic)
                else:
                    generated = generator.generate_text(topic)
                
                result = {**prompt_item, "generated": generated}
            
            results.append(result)
        
        # Save results
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = batch_file.parent / f"{batch_file.stem}_results.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Batch generation completed!")
        print(f"   Processed: {len(results)} items")
        print(f"   Results saved to: {output_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error in batch mode: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())