"""Lightweight model generation wrapper for CLI usage.

This intentionally avoids coupling to training internals; it simply
loads a HF-format directory (with config.json) and performs greedy /
sampling generation according to passed parameters.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional, Any, Dict
import json

from rich.console import Console


class ToaripiGenerator:
    def __init__(self, model_path: Path, console: Console | None = None):
        self.model_path = Path(model_path)
        self.console = console or Console()
        self._model = None
        self._tokenizer = None
        self._loaded = False

    # ------------------------------------------------------------------
    def load(self, *, allow_fallback: bool = True) -> bool:
        """Load a materialized HF directory; fallback to base model if needed.

        Normal path: directory contains config.json (fully materialized export or
        training output). Fallback path: directory only has model_info.json with
        metadata referencing a base model + checkpoint / adapter weights.
        """
        try:
            if not self.model_path.exists():
                self.console.print(f"❌ Model path does not exist: {self.model_path}")
                return False

            cfg = self.model_path / "config.json"
            if cfg.exists():
                return self._load_local()

            # Attempt fallback using model_info.json
            if allow_fallback:
                info_file = self.model_path / "model_info.json"
                if info_file.exists():
                    try:
                        with open(info_file, "r") as f:
                            info: Dict[str, Any] = json.load(f)
                    except Exception as e:  # pragma: no cover - unlikely
                        self.console.print(f"❌ Failed reading model_info.json: {e}")
                        return False
                    return self._load_from_metadata(info)
                else:
                    self.console.print("❌ Neither config.json nor model_info.json present; cannot load.")
            else:
                self.console.print("❌ config.json missing and fallback disabled.")
            return False
        except Exception as e:  # pragma: no cover - safety
            self.console.print(f"❌ Failed to load model: {e}")
            return False

    # ------------------------------------------------------------------
    def _load_local(self) -> bool:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                trust_remote_code=True,
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._loaded = True
            self.console.print("✅ Model loaded successfully (materialized directory).")
            return True
        except Exception as e:  # pragma: no cover
            self.console.print(f"❌ Local load failed: {e}")
            return False

    # ------------------------------------------------------------------
    def _load_from_metadata(self, info: Dict[str, Any]) -> bool:
        """Fallback: load base model + (optional) adapter from checkpoint_dir.

        This enables using a lightweight version directory that only stores
        metadata. If LoRA / adapter weights exist in checkpoint_dir, attempt to
        merge or load them; otherwise return base model.
        """
        base = info.get("base_model")
        checkpoint_dir = info.get("checkpoint_dir")
        if not base:
            self.console.print("❌ base_model missing in model_info.json")
            return False
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            from peft import PeftModel  # type: ignore
        except ImportError:
            # peft optional; still allow base model load
            PeftModel = None  # type: ignore
        try:
            self.console.print(f"ℹ️  Fallback loading base model: {base}")
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            self._tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                base,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                trust_remote_code=True,
            )
            # Attempt adapter merge
            if checkpoint_dir and PeftModel is not None:
                ckpt_path = Path(checkpoint_dir)
                adapter_files = list(ckpt_path.glob("*adapter_model.bin"))
                if adapter_files:
                    try:
                        from peft import PeftModel
                        self.console.print(f"🔗 Applying adapter from {ckpt_path}")
                        self._model = PeftModel.from_pretrained(self._model, str(ckpt_path))  # type: ignore
                    except Exception as e:  # pragma: no cover
                        self.console.print(f"⚠️  Adapter load failed, continuing with base model: {e}")
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._loaded = True
            self.console.print("✅ Fallback model loaded (base + optional adapter).")
            return True
        except Exception as e:  # pragma: no cover
            self.console.print(f"❌ Fallback load failed: {e}")
            return False

    # ------------------------------------------------------------------
    def bilingual(self, prompt: str, content_type: str, *, max_length: int, temperature: float) -> Tuple[str, str]:
        if not self._loaded:
            return ("Model not loaded", "Model not loaded")
        if content_type == "chat":
            return self._chat(prompt)
        return self._generate(prompt, content_type, max_length, temperature)

    def _generate(self, prompt: str, content_type: str, max_length: int, temperature: float) -> Tuple[str, str]:
        try:
            if not self._loaded:
                return ("❌ Model not loaded", "❌ Toaripi model not available. Please train a model first using: toaripi train")
                
            # Check if this is a base model without Toaripi training
            if hasattr(self._model, 'config') and not hasattr(self._model, 'peft_config'):
                self.console.print("⚠️  [yellow]Warning: Using base model without Toaripi fine-tuning.[/yellow]")
                self.console.print("   For proper Toaripi generation, please train a model first:")
                self.console.print("   [cyan]toaripi train --interactive[/cyan]")
            
            import torch
            
            # Enhanced prompts for better base model performance
            if content_type == "story":
                model_prompt = f"""Create a simple educational story in Toaripi language about: {prompt}

English: {prompt}
Toaripi: """
            elif content_type == "vocabulary":
                model_prompt = f"""Translate these English words to Toaripi:
English: {prompt}
Toaripi: """
            elif content_type == "translation":
                model_prompt = f"""Translate to Toaripi language:
English: {prompt}
Toaripi: """
            else:
                model_prompt = f"""Generate content in Toaripi language about: {prompt}
Toaripi: """

            inputs = self._tokenizer(
                model_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self._tokenizer.eos_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
            
            generated = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract Toaripi content
            if "Toaripi:" in generated:
                tqo = generated.split("Toaripi:")[-1].strip()
            else:
                tqo = generated[len(model_prompt):].strip()
            
            tqo = tqo.split("\n")[0].strip()
            
            # If output doesn't look like Toaripi, provide helpful message
            if not tqo or len(tqo.split()) < 2:
                tqo = "⚠️  Model needs Toaripi training. Use sample words: 'Narau apu poroporosi' (child playing)"
            
            eng = f"Generated {content_type} for: {prompt}"
            return eng, tqo
            
        except Exception as e:
            self.console.print(f"⚠️  Generation failed: {e}")
            return (f"Generation failed for: {prompt}", "❌ Please train a Toaripi model first: toaripi train")

    def _chat(self, question: str) -> Tuple[str, str]:
        try:
            import torch
            template = (
                "You are a Toaripi language teacher. When asked about a word or concept, respond in this exact format:\n"
                "toaripi_word/english_word - cultural_description\n\nQuestion: {q}\nAnswer:"
            )
            prompt = template.format(q=question)
            inputs = self._tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self._tokenizer.eos_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
            generated = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "Answer:" in generated:
                ans = generated.split("Answer:")[-1].strip()
            else:
                ans = generated[len(prompt):].strip()
            ans = ans.split("\n")[0].strip()
            if "/" in ans and "-" in ans:
                tqo = ans
            else:
                tqo = f"Model response: {ans}"
            eng = f"Question: {question}"
            return eng, tqo
        except Exception as e:  # pragma: no cover
            self.console.print(f"⚠️  Chat generation failed: {e}")
            return (f"Question: {question}", "Generation error")
