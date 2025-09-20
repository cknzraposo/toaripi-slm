#!/usr/bin/env python3
"""
Model export script for Toaripi SLM.
Exports trained models to GGUF format for edge deployment.
"""

import argparse
import sys
from pathlib import Path
import subprocess
import shutil
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from toaripi_slm.utils import setup_logging

def main():
    """Main entry point for model export."""
    parser = argparse.ArgumentParser(description="Export Toaripi model to GGUF")
    parser.add_argument("--model", required=True, help="Path to trained model directory")
    parser.add_argument("--output", required=True, help="Output GGUF file path")
    parser.add_argument("--quantization", choices=["f16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q4_k_m", "q5_k_m"], 
                       default="q4_k_m", help="Quantization level")
    parser.add_argument("--vocab-type", choices=["spm", "bpe"], 
                       default="spm", help="Tokenizer type")
    parser.add_argument("--context-length", type=int, default=2048, help="Context length")
    parser.add_argument("--dry-run", action="store_true", help="Show commands without executing")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    
    print(f"📦 Toaripi Model Export Tool")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Quantization: {args.quantization}")
    
    try:
        model_path = Path(args.model)
        output_path = Path(args.output)
        
        # Validate input model
        if not model_path.exists():
            print(f"❌ Model directory not found: {model_path}")
            return 1
        
        # Check if this is a LoRA adapter or full model
        adapter_config = model_path / "adapter_config.json"
        is_lora = adapter_config.exists()
        
        if is_lora:
            print("🔄 Detected LoRA adapter - will merge with base model first")
            return export_lora_model(model_path, output_path, args)
        else:
            print("🔄 Detected full model - converting directly")
            return export_full_model(model_path, output_path, args)
            
    except Exception as e:
        print(f"❌ Error during export: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

def export_lora_model(model_path, output_path, args):
    """Export LoRA adapter by merging with base model."""
    try:
        # Check if we have required dependencies
        if not check_dependencies():
            return 1
        
        print("\n🔀 Merging LoRA adapter with base model...")
        
        # Create temporary directory for merged model
        temp_dir = Path("/tmp") / "toaripi_merged_model"
        temp_dir.mkdir(exist_ok=True)
        
        # Merge LoRA adapter with base model
        merge_command = [
            "python", "-c",
            f"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys

try:
    model_path = '{model_path}'
    output_path = '{temp_dir}'
    
    # Load adapter config to get base model
    import json
    with open('{model_path}/adapter_config.json', 'r') as f:
        adapter_config = json.load(f)
    
    base_model_name = adapter_config['base_model_name_or_path']
    print(f'Loading base model: {{base_model_name}}')
    
    # Load base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        torch_dtype=torch.float16,
        device_map='cpu'
    )
    
    # Load and merge LoRA
    print('Loading and merging LoRA adapter...')
    model = PeftModel.from_pretrained(base_model, model_path)
    merged_model = model.merge_and_unload()
    
    # Save merged model
    print(f'Saving merged model to {{output_path}}')
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print('✅ Model merger completed successfully')
    
except Exception as e:
    print(f'❌ Error during merge: {{e}}')
    sys.exit(1)
"""
        ]
        
        if args.dry_run:
            print(f"Would run: {' '.join(merge_command[:2])} [python merge script]")
        else:
            result = subprocess.run(merge_command, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Merge failed: {result.stderr}")
                return 1
            print(result.stdout)
        
        # Now export the merged model
        return export_full_model(temp_dir, output_path, args)
        
    except Exception as e:
        print(f"❌ Error exporting LoRA model: {e}")
        return 1

def export_full_model(model_path, output_path, args):
    """Export full model to GGUF format."""
    try:
        print(f"\n🔧 Converting to GGUF format...")
        
        # Check for llama.cpp
        llama_cpp_path = find_llama_cpp()
        if not llama_cpp_path and not args.dry_run:
            print("❌ llama.cpp not found. Install with: pip install llama-cpp-python")
            print("   Or clone and build: https://github.com/ggerganov/llama.cpp")
            return 1
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to GGUF
        if llama_cpp_path:
            convert_script = llama_cpp_path / "convert-hf-to-gguf.py"
        else:
            # Fallback to python module
            convert_script = "llama_cpp.convert"
        
        # Base conversion command
        convert_command = [
            "python", str(convert_script) if isinstance(convert_script, Path) else "-m",
            str(model_path),
            "--outfile", str(output_path.with_suffix('.gguf')),
            "--vocab-type", args.vocab_type,
            "--ctx", str(args.context_length)
        ]
        
        if isinstance(convert_script, str):
            convert_command.insert(2, convert_script)
        
        print(f"Conversion command: {' '.join(convert_command)}")
        
        if args.dry_run:
            print(f"Would run: {' '.join(convert_command)}")
        else:
            result = subprocess.run(convert_command, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Conversion failed: {result.stderr}")
                return 1
            print("✅ Model converted to GGUF successfully")
        
        # Quantize if requested
        if args.quantization != "f16":
            return quantize_model(output_path.with_suffix('.gguf'), args)
        
        print(f"\n🎉 Export completed successfully!")
        print(f"   Output: {output_path}")
        print(f"   Ready for edge deployment")
        
        # Create deployment info
        create_deployment_info(model_path, output_path, args)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error exporting model: {e}")
        return 1

def quantize_model(gguf_path, args):
    """Quantize GGUF model."""
    try:
        print(f"\n⚡ Quantizing to {args.quantization}...")
        
        llama_cpp_path = find_llama_cpp()
        if llama_cpp_path:
            quantize_binary = llama_cpp_path / "llama-quantize"
        else:
            print("❌ llama.cpp quantize binary not found")
            return 1
        
        quantized_path = gguf_path.with_suffix(f'.{args.quantization}.gguf')
        
        quantize_command = [
            str(quantize_binary),
            str(gguf_path),
            str(quantized_path),
            args.quantization
        ]
        
        print(f"Quantization command: {' '.join(quantize_command)}")
        
        if args.dry_run:
            print(f"Would run: {' '.join(quantize_command)}")
        else:
            result = subprocess.run(quantize_command, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Quantization failed: {result.stderr}")
                return 1
            
            # Remove unquantized file to save space
            gguf_path.unlink()
            print(f"✅ Quantized to {quantized_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during quantization: {e}")
        return 1

def find_llama_cpp():
    """Find llama.cpp installation."""
    # Check common locations
    common_paths = [
        Path.home() / "llama.cpp",
        Path("/usr/local/llama.cpp"),
        Path("/opt/llama.cpp")
    ]
    
    for path in common_paths:
        if path.exists() and (path / "convert-hf-to-gguf.py").exists():
            return path
    
    return None

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import torch
        import transformers
        import peft
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install torch transformers peft")
        return False

def create_deployment_info(model_path, output_path, args):
    """Create deployment information file."""
    try:
        info = {
            "model_type": "toaripi-slm",
            "source_model": str(model_path),
            "quantization": args.quantization,
            "vocab_type": args.vocab_type,
            "context_length": args.context_length,
            "deployment_instructions": {
                "raspberry_pi": "Use llama.cpp with CPU inference",
                "cpu_only": f"Load with: llama-cpp-python {output_path}",
                "memory_requirement": "4-8GB RAM depending on quantization"
            }
        }
        
        info_file = output_path.with_suffix('.json')
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"📋 Deployment info saved: {info_file}")
        
    except Exception as e:
        print(f"⚠️ Could not create deployment info: {e}")

if __name__ == "__main__":
    sys.exit(main())