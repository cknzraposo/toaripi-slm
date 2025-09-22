# Configuration Guide

This guide covers all configuration options for the Toaripi SLM system, from data processing to model training and deployment.

## Configuration Overview

The Toaripi SLM uses YAML configuration files organized by functionality:

```
configs/
├── data/
│   └── preprocessing_config.yaml    # Data processing settings
├── training/
│   ├── base_config.yaml            # Basic training configuration
│   ├── lora_config.yaml            # LoRA fine-tuning settings
│   └── toaripi_educational_config.yaml  # Educational-specific training
└── deployment/
    ├── edge_config.yaml            # Raspberry Pi deployment
    └── api_config.yaml             # Web API settings
```

## Data Processing Configuration

### preprocessing_config.yaml

```yaml
# Data source configuration
data:
  source_file: "data/raw/Full_bible_english_toaripi.csv"
  output_dir: "data/processed"
  backup_original: true
  
  # Required columns in source data
  required_columns:
    english: "english"
    toaripi: "toaripi"
    reference: "reference"  # Optional: verse references

# Educational content settings
educational:
  # Target age groups for content generation
  target_age_groups:
    - "primary_lower"     # 6-8 years
    - "primary_upper"     # 9-11 years
  
  # Content types to support
  content_types:
    - "story"
    - "vocabulary"
    - "dialogue"
    - "comprehension"
  
  # Validation strictness
  validation_level: "educational"  # basic, educational, strict
  
  # Age-appropriate vocabulary limits
  vocabulary_limits:
    primary_lower:
      max_words_per_sentence: 10
      max_sentences_per_story: 8
      complexity_score_max: 0.6
    primary_upper:
      max_words_per_sentence: 15
      max_sentences_per_story: 12
      complexity_score_max: 0.8

# Cultural validation settings
cultural:
  enable_validation: true
  sensitivity_threshold: 0.8
  
  # Cultural context preservation
  preserve_traditional_elements: true
  validate_cultural_appropriateness: true
  
  # Content filtering
  exclude_inappropriate_content: true
  maintain_cultural_authenticity: true

# Text processing options
processing:
  # Text cleaning
  normalize_unicode: true
  remove_extra_whitespace: true
  preserve_punctuation: true
  
  # Language detection
  verify_language_pairs: true
  english_confidence_threshold: 0.9
  toaripi_validation: true
  
  # Data quality filters
  min_text_length: 10
  max_text_length: 500
  filter_incomplete_pairs: true
  remove_duplicates: true

# Output configuration
output:
  # Training data splits
  train_split: 0.8
  validation_split: 0.15
  test_split: 0.05
  
  # File formats
  save_csv: true
  save_parquet: true
  save_json: false
  
  # Educational metadata
  include_age_labels: true
  include_content_types: true
  include_difficulty_scores: true
```

## Training Configuration

### base_config.yaml

```yaml
# Model selection
model:
  # Base model from HuggingFace
  base_model: "microsoft/DialoGPT-small"
  
  # Model configuration
  use_auth_token: false
  trust_remote_code: false
  
  # Hardware optimization
  device_map: "auto"
  torch_dtype: "float16"
  load_in_8bit: false
  load_in_4bit: true

# Training hyperparameters
training:
  # Basic settings
  num_epochs: 3
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 4
  
  # Optimization
  learning_rate: 0.00002
  weight_decay: 0.01
  warmup_steps: 100
  max_grad_norm: 1.0
  
  # Scheduler
  lr_scheduler_type: "cosine"
  num_warmup_steps: 100
  
  # Evaluation
  evaluation_strategy: "steps"
  eval_steps: 500
  save_steps: 500
  logging_steps: 50
  
  # Memory optimization
  gradient_accumulation_steps: 1
  dataloader_num_workers: 2
  remove_unused_columns: false

# Output configuration
output:
  output_dir: "models/training_runs"
  run_name: "toaripi_educational"
  
  # Checkpointing
  save_total_limit: 3
  save_strategy: "steps"
  load_best_model_at_end: true
  metric_for_best_model: "eval_educational_score"
  
  # Logging
  logging_dir: "logs"
  report_to: ["tensorboard"]
  
  # Final model export
  push_to_hub: false
  hub_model_id: null

# Data configuration
data:
  # Dataset files
  train_file: "data/processed/train.csv"
  validation_file: "data/processed/validation.csv"
  test_file: "data/processed/test.csv"
  
  # Text processing
  max_length: 512
  padding: "max_length"
  truncation: true
  
  # Educational prompt formatting
  use_educational_prompts: true
  prompt_template: "educational_content"

# Educational-specific settings
educational:
  # Validation during training
  enable_educational_validation: true
  validation_frequency: 100  # steps
  
  # Content quality metrics
  track_age_appropriateness: true
  track_cultural_sensitivity: true
  track_vocabulary_complexity: true
  
  # Early stopping on educational metrics
  early_stopping_patience: 3
  early_stopping_threshold: 0.01
  
  # Educational objectives
  focus_areas:
    - "language_preservation"
    - "cultural_education"
    - "primary_literacy"
```

### lora_config.yaml

```yaml
# LoRA (Low-Rank Adaptation) Configuration
lora:
  # Enable LoRA fine-tuning
  use_lora: true
  
  # LoRA hyperparameters
  r: 16                    # Rank of adaptation
  lora_alpha: 32          # LoRA scaling parameter
  lora_dropout: 0.1       # Dropout probability
  
  # Target modules for LoRA
  target_modules:
    - "q_proj"
    - "v_proj"
    - "k_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
  
  # LoRA configuration
  bias: "none"            # Bias handling: none, all, lora_only
  task_type: "CAUSAL_LM"  # Task type for PEFT
  
  # Memory optimization
  use_gradient_checkpointing: true
  use_cpu_offload: false

# Model loading for LoRA
model:
  base_model: "microsoft/DialoGPT-small"
  
  # Quantization for memory efficiency
  load_in_4bit: true
  bnb_4bit_compute_dtype: "float16"
  bnb_4bit_use_double_quant: true
  bnb_4bit_quant_type: "nf4"

# Training with LoRA
training:
  # Adjusted batch sizes for memory efficiency
  per_device_train_batch_size: 2
  per_device_eval_batch_size: 2
  gradient_accumulation_steps: 2
  
  # LoRA-specific learning rates
  learning_rate: 0.0002   # Higher LR for LoRA
  
  # Training length
  num_epochs: 5           # More epochs for LoRA
  max_steps: -1
  
  # Memory management
  fp16: false
  bf16: true              # Better for modern hardware
  tf32: true
  
  # Optimizer
  optim: "adamw_torch"
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_epsilon: 1e-8

# Educational focus with LoRA
educational:
  # Enhanced validation for efficient training
  validation_level: "strict"
  
  # LoRA-specific educational metrics
  track_adaptation_quality: true
  monitor_educational_drift: true
  
  # Regularization for educational content
  educational_regularization_weight: 0.1
  cultural_preservation_weight: 0.2
```

### toaripi_educational_config.yaml

```yaml
# Toaripi-specific educational configuration
project:
  name: "toaripi_educational_model"
  description: "Educational content generation for Toaripi language preservation"
  version: "1.0.0"
  
  # Educational objectives
  objectives:
    - "Preserve Toaripi language through engaging stories"
    - "Create age-appropriate educational content"
    - "Support primary school literacy development"
    - "Maintain cultural authenticity and sensitivity"

# Language-specific settings
language:
  # Primary language pair
  source_language: "english"
  target_language: "toaripi"
  
  # Language characteristics
  toaripi:
    iso_639_3: "tqo"
    writing_system: "latin"
    cultural_region: "papua_new_guinea"
    
    # Language features
    tonal: false
    agglutinative: true
    verb_final: true
    
    # Educational considerations
    literacy_level: "developing"
    primary_education_language: true
    cultural_transmission_language: true

# Educational content configuration
content:
  # Age group specifications
  age_groups:
    early_childhood:
      age_range: "3-5"
      vocabulary_size: 100
      sentence_length_max: 5
      story_length_max: 50
      cultural_complexity: "basic"
      
    primary_lower:
      age_range: "6-8"
      vocabulary_size: 300
      sentence_length_max: 10
      story_length_max: 150
      cultural_complexity: "simple"
      
    primary_upper:
      age_range: "9-11"
      vocabulary_size: 800
      sentence_length_max: 15
      story_length_max: 300
      cultural_complexity: "moderate"
  
  # Content types and specifications
  content_types:
    story:
      structure_requirements:
        - "clear_beginning"
        - "simple_plot"
        - "cultural_context"
        - "moral_lesson"
      
      themes:
        - "traditional_fishing"
        - "family_cooperation"
        - "respect_for_elders"
        - "environmental_stewardship"
        - "community_values"
      
    vocabulary:
      categories:
        - "family_relations"
        - "daily_activities"
        - "natural_environment"
        - "traditional_tools"
        - "cultural_practices"
      
      presentation_format:
        include_english_translation: true
        include_example_sentence: true
        include_cultural_context: true
        include_pronunciation_guide: false
        
    dialogue:
      scenarios:
        - "child_elder_conversation"
        - "classroom_interaction"
        - "community_gathering"
        - "traditional_activity"
      
      requirements:
        natural_speech_patterns: true
        age_appropriate_topics: true
        cultural_communication_styles: true

# Cultural validation configuration
cultural:
  # Cultural authenticity checks
  authenticity:
    validate_traditional_knowledge: true
    check_cultural_appropriateness: true
    preserve_cultural_nuances: true
    avoid_stereotypes: true
  
  # Cultural elements to preserve
  preserve_elements:
    - "traditional_fishing_methods"
    - "family_hierarchies"
    - "seasonal_activities"
    - "ceremonial_practices"
    - "oral_traditions"
  
  # Sensitivity guidelines
  sensitivity:
    respect_sacred_knowledge: true
    age_appropriate_cultural_content: true
    avoid_controversial_topics: true
    maintain_positive_representation: true

# Quality assurance
quality:
  # Content validation metrics
  metrics:
    educational_value_score: 0.8      # Minimum educational value
    cultural_appropriateness: 0.9     # Minimum cultural score
    age_appropriateness: 0.85         # Minimum age appropriateness
    language_quality: 0.8             # Minimum language quality
  
  # Review process
  review:
    require_cultural_review: true
    require_educational_review: true
    require_linguistic_review: true
    
  # Continuous improvement
  feedback:
    collect_teacher_feedback: true
    track_learning_outcomes: true
    monitor_cultural_reception: true

# Deployment configuration
deployment:
  # Target environments
  environments:
    - "classroom_tablets"
    - "raspberry_pi_stations"
    - "offline_laptops"
    - "community_centers"
  
  # Technical requirements
  requirements:
    max_model_size: "2GB"
    min_ram: "4GB"
    cpu_only: true
    offline_capable: true
    
  # Performance targets
  performance:
    generation_time_max: "5s"        # Maximum generation time
    startup_time_max: "30s"          # Maximum startup time
    memory_usage_max: "2GB"          # Maximum memory usage
```

## Deployment Configuration

### edge_config.yaml

```yaml
# Edge deployment configuration for Raspberry Pi and low-resource devices
deployment:
  name: "toaripi_edge_deployment"
  target_device: "raspberry_pi"
  
  # Device specifications
  device:
    cpu_architecture: "arm64"
    ram_limit: "4GB"
    storage_limit: "32GB"
    gpu_available: false
    internet_connection: false

# Model optimization for edge
model:
  # Quantization settings
  quantization:
    method: "gguf"
    precision: "q4_k_m"     # 4-bit quantization
    optimize_for_cpu: true
    
  # Model selection
  base_model: "models/gguf/toaripi-primary-q4.gguf"
  fallback_model: "models/gguf/toaripi-basic-q8.gguf"
  
  # Loading configuration
  context_length: 512
  threads: 4              # CPU threads to use
  batch_size: 1           # Single request processing
  
  # Memory management
  mlock: true             # Lock model in memory
  mmap: true              # Memory map model file
  numa: false             # NUMA optimization

# Educational content constraints for edge
educational:
  # Content generation limits
  generation:
    max_tokens: 200
    max_concurrent_requests: 1
    timeout_seconds: 30
    
  # Age group filtering
  age_groups:
    - "primary_lower"
    - "primary_upper"
    
  # Content validation
  validation:
    enable_offline_validation: true
    cache_validation_results: true
    strict_age_filtering: true

# Performance optimization
performance:
  # CPU optimization
  cpu:
    use_avx: true
    use_avx2: true
    use_fma: true
    optimize_for_arm: true
    
  # Memory optimization
  memory:
    garbage_collection_frequency: 100
    cache_size_mb: 512
    preload_vocabulary: true
    
  # Response time optimization
  response:
    warm_up_model: true
    cache_common_prompts: true
    optimize_batch_processing: false

# Offline configuration
offline:
  # Pre-generated content cache
  cache:
    enable_content_cache: true
    cache_size_mb: 1024
    cache_stories: 50
    cache_vocabulary_sets: 20
    
  # Fallback strategies
  fallback:
    use_template_responses: true
    template_library_size: 100
    enable_basic_generation: true
```

### api_config.yaml

```yaml
# Web API configuration
api:
  # Server settings
  server:
    host: "0.0.0.0"
    port: 8000
    workers: 2
    
  # API configuration
  title: "Toaripi Educational Content API"
  description: "API for generating educational content in Toaripi language"
  version: "1.0.0"
  
  # CORS settings
  cors:
    allow_origins: ["*"]
    allow_methods: ["GET", "POST"]
    allow_headers: ["*"]

# Authentication and security
security:
  # API key authentication
  api_key:
    enabled: false
    header_name: "X-API-Key"
    
  # Rate limiting
  rate_limiting:
    enabled: true
    requests_per_minute: 60
    requests_per_hour: 1000
    
  # Content filtering
  content_filter:
    enable_input_validation: true
    max_prompt_length: 500
    block_inappropriate_requests: true

# Model serving configuration
serving:
  # Model loading
  model:
    path: "models/gguf/toaripi-api.gguf"
    load_on_startup: true
    
  # Generation settings
  generation:
    default_max_tokens: 200
    default_temperature: 0.7
    default_top_p: 0.9
    timeout_seconds: 30
    
  # Batch processing
  batch:
    max_batch_size: 4
    batch_timeout_ms: 1000

# Educational API features
educational:
  # Content validation
  validation:
    enable_response_validation: true
    validate_age_appropriateness: true
    validate_cultural_sensitivity: true
    
  # Age group handling
  age_groups:
    default: "primary_lower"
    available:
      - "early_childhood"
      - "primary_lower"
      - "primary_upper"
      
  # Content types
  content_types:
    available:
      - "story"
      - "vocabulary"
      - "dialogue"
      - "comprehension"
    default: "story"

# Monitoring and logging
monitoring:
  # Logging configuration
  logging:
    level: "INFO"
    format: "json"
    file: "logs/api.log"
    
  # Metrics collection
  metrics:
    enable_prometheus: false
    track_generation_time: true
    track_content_quality: true
    track_user_satisfaction: true
    
  # Health checks
  health:
    endpoint: "/health"
    include_model_status: true
    include_educational_metrics: true
```

## Environment Configuration

### .env Example

```bash
# Model paths
TOARIPI_MODEL_PATH=models/gguf/toaripi-primary.gguf
TOARIPI_FALLBACK_MODEL=models/gguf/toaripi-basic.gguf

# Data paths
TOARIPI_DATA_DIR=data
TOARIPI_CONFIG_DIR=configs
TOARIPI_CACHE_DIR=cache

# Educational settings
TOARIPI_DEFAULT_AGE_GROUP=primary_lower
TOARIPI_VALIDATION_LEVEL=educational
TOARIPI_CULTURAL_VALIDATION=true

# Performance settings
TOARIPI_CPU_THREADS=4
TOARIPI_MEMORY_LIMIT=4GB
TOARIPI_BATCH_SIZE=2

# API settings (if using web API)
TOARIPI_API_HOST=0.0.0.0
TOARIPI_API_PORT=8000
TOARIPI_API_WORKERS=2

# Logging
TOARIPI_LOG_LEVEL=INFO
TOARIPI_LOG_FILE=logs/toaripi.log

# Development settings
TOARIPI_DEBUG=false
TOARIPI_PROFILE=false
```

## Configuration Validation

The system automatically validates all configuration files to ensure:

- Required fields are present
- Value ranges are appropriate for educational content
- Age group specifications are consistent
- Cultural validation settings are properly configured
- Hardware requirements are realistic for target devices

### Validation Command

```bash
# Validate all configurations
toaripi-slm config validate

# Validate specific configuration
toaripi-slm config validate --config configs/training/base_config.yaml

# Check configuration compatibility
toaripi-slm config check-compatibility --training configs/training/base_config.yaml --data configs/data/preprocessing_config.yaml
```

This configuration system ensures that all aspects of the Toaripi SLM maintain focus on educational effectiveness, cultural sensitivity, and technical feasibility for classroom deployment.