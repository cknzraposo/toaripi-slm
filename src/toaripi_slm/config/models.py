"""Unified configuration models for Toaripi SLM.

These pydantic models provide a single validated source of truth for
training, inference and application/server configuration.

Environment variable override convention:
  TOARIPI__SECTION__FIELD=value (case-insensitive)
Example:
  TOARIPI__TRAINING__LEARNING_RATE=3e-5

Only scalar & simple list overrides are supported for env substitution initially.
"""
from __future__ import annotations

from typing import Optional, List, Literal, Dict, Any, Callable
from pydantic import BaseModel, Field, field_validator
from pydantic import FieldValidationInfo


class ModelSettings(BaseModel):
    name: str = Field(..., description="Base model name or path")
    cache_dir: Optional[str] = Field(None, description="Cache directory for model weights")
    trust_remote_code: bool = False
    device_map: str = Field("auto", description="Device placement strategy")
    max_length: int = Field(512, ge=64, le=4096)


class LoraSettings(BaseModel):
    enabled: bool = False
    rank: int = Field(16, ge=1)
    alpha: int = 32
    dropout: float = Field(0.1, ge=0.0, le=0.5)
    target_modules: List[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


class TrainingSettings(BaseModel):
    epochs: int = Field(3, ge=1, le=50)
    learning_rate: float = Field(2e-5, gt=0, lt=1)
    batch_size: int = Field(4, ge=1, le=512)
    gradient_accumulation_steps: int = Field(1, ge=1, le=1024)
    warmup_steps: Optional[int] = Field(None, ge=0)
    warmup_ratio: Optional[float] = Field(None, ge=0, le=1)
    weight_decay: float = Field(0.0, ge=0, le=1)
    eval_strategy: str = Field("steps")
    eval_steps: int = Field(500, ge=1)
    save_strategy: str = Field("steps")
    save_steps: int = Field(500, ge=1)
    logging_steps: int = Field(100, ge=1)
    early_stopping_patience: Optional[int] = Field(None, ge=1)
    early_stopping_threshold: Optional[float] = Field(None, ge=0, le=1)
    gradient_checkpointing: bool = False
    validation_split: float = Field(0.2, ge=0.0, le=0.9)
    use_fp16: Optional[bool] = None
    fp16: Optional[bool] = None  # Accept from external configs
    bf16: Optional[bool] = None
    tf32: Optional[bool] = None

    @field_validator("use_fp16", mode="before")
    @classmethod
    def alias_fp16(cls, v: Optional[bool], info: FieldValidationInfo):  # noqa: D401
        """Align fp16/use_fp16 synonyms; prefer explicit `use_fp16` if provided.

        If `use_fp16` is absent but legacy `fp16` provided in raw data, map it.
        """
        if v is not None:
            return v
        # raw values accessible via context on root model; fallback False
        data: Dict[str, Any] = info.data or {}
        legacy = data.get("fp16")
        if legacy is not None:
            try:
                return bool(legacy)
            except Exception:
                return False
        return False


class DataSettings(BaseModel):
    max_length: int = Field(512, ge=64, le=4096)
    padding: bool = True
    truncation: bool = True
    return_tensors: str = "pt"
    min_length: Optional[int] = None
    remove_duplicates: bool = False


class OptimizationSettings(BaseModel):
    optimizer: str = "adamw"
    lr_scheduler_type: str = "linear"
    fp16: Optional[bool] = None
    bf16: Optional[bool] = None
    dataloader_pin_memory: Optional[bool] = None
    remove_unused_columns: Optional[bool] = None
    tf32: Optional[bool] = None


class OutputSettings(BaseModel):
    checkpoint_dir: str = "./checkpoints"
    save_total_limit: int = 3  # validated via validator below if needed
    push_to_hub: bool = False
    hub_model_id: str = ""

    @field_validator("save_total_limit")
    @classmethod
    def _valid_save_limit(cls, v: int) -> int:
        if not (1 <= v <= 50):
            raise ValueError("save_total_limit must be between 1 and 50")
        return v


class LoggingSettings(BaseModel):
    use_wandb: bool = False
    project_name: str = "toaripi-slm"
    run_name: str = "run"
    log_level: str = "INFO"


class EducationalSettings(BaseModel):
    content_types: List[str] = Field(default_factory=lambda: ["story", "vocabulary", "dialogue", "question_answer"])
    age_groups: List[str] = Field(default_factory=lambda: ["primary_early", "primary_middle", "primary_late"])
    cultural_validation: bool = True
    age_appropriate_only: bool = True


class TrainingConfig(BaseModel):
    model: ModelSettings
    training: TrainingSettings
    data: DataSettings
    optimization: Optional[OptimizationSettings] = None
    # Direct default instances (simpler for type checkers)
    output: OutputSettings = OutputSettings()
    logging: LoggingSettings = LoggingSettings()
    lora: Optional[LoraSettings] = None
    educational: Optional[EducationalSettings] = None


class InferenceConfig(BaseModel):
    model_path: str = Field(..., description="Path or name for inference model (HF or local)")
    max_new_tokens: int = Field(128, ge=1, le=1024)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.1, le=1.0)
    top_k: int = Field(50, ge=0, le=1000)
    repetition_penalty: float = Field(1.1, ge=0.8, le=2.5)
    device_map: str = Field("auto")
    use_fp16: bool = True
    context_window_tokens: int = Field(2048, ge=256, le=8192)


class AppConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    chat_ui_enabled: bool = False
    metrics_enabled: bool = True
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])


ConfigType = Literal["training", "inference", "app"]


def _merge_overrides(raw: Dict[str, Any], env: Dict[str, str]) -> Dict[str, Any]:
    """Apply environment variable overrides using TOARIPI__SECTION__FIELD convention."""
    prefix = "TOARIPI__"
    for key, value in env.items():
        if not key.upper().startswith(prefix):
            continue
        parts = key[len(prefix):].split("__")
        if len(parts) != 2:
            continue
        section, field = parts
        section_l = section.lower()
        field_l = field.lower()
        if section_l in raw and isinstance(raw[section_l], dict) and field_l in raw[section_l]:
            current = raw[section_l][field_l]
            if isinstance(current, bool):
                raw[section_l][field_l] = value.lower() in {"1", "true", "yes", "on"}
            elif isinstance(current, int):
                try:
                    raw[section_l][field_l] = int(value)
                except ValueError:
                    pass
            elif isinstance(current, float):
                try:
                    raw[section_l][field_l] = float(value)
                except ValueError:
                    pass
            elif isinstance(current, list):
                raw[section_l][field_l] = [v.strip() for v in value.split(",") if v.strip()]
            else:
                raw[section_l][field_l] = value
    return raw


def parse_training_config(raw: Dict[str, Any]) -> TrainingConfig:
    return TrainingConfig(**raw)


def parse_inference_config(raw: Dict[str, Any]) -> InferenceConfig:
    return InferenceConfig(**raw)


def parse_app_config(raw: Dict[str, Any]) -> AppConfig:
    return AppConfig(**raw)


PARSERS: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    "training": parse_training_config,
    "inference": parse_inference_config,
    "app": parse_app_config,
}


def load_config_dict(path: str) -> Dict[str, Any]:
    import yaml
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    from typing import cast
    with p.open("r", encoding="utf-8") as f:
        raw_loaded = yaml.safe_load(f) or {}  # type: ignore[assignment]
    if not isinstance(raw_loaded, dict):  # defensive
        raise TypeError("Top-level YAML structure must be a mapping")
    data = cast(Dict[str, Any], raw_loaded)
    norm: Dict[str, Any] = {str(k).lower(): v for k, v in data.items()}
    return norm  # type: ignore[return-value]


def load_config(path: str, kind: ConfigType) -> Any:
    raw: Dict[str, Any] = load_config_dict(path)
    raw = _merge_overrides(raw, dict(**__import__("os").environ))
    parser: Callable[[Dict[str, Any]], Any] = PARSERS[kind]
    return parser(raw)
