# CLI Command Contracts

## Overview

This document defines the command-line interface contracts for the Toaripi SLM CLI tool. Each command specifies its inputs, outputs, exit codes, and error conditions.

## Command Structure

### Global Options

```bash
toaripi [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS] [ARGS]

Global Options:
--config PATH     Configuration file path (default: ~/.toaripi/config.yaml)
--verbose, -v     Verbose output (can be repeated for more verbosity)
--quiet, -q       Suppress non-error output
--help, -h        Show help message
--version         Show version information
```

## Data Commands

### data prepare

**Purpose**: Validate and prepare English-Toaripi parallel data for training

**Signature**:
```bash
toaripi data prepare SOURCE_FILE [OPTIONS]
```

**Parameters**:
- `SOURCE_FILE` (required): Path to CSV file with English-Toaripi parallel text
- `--output, -o PATH`: Output directory for processed data (default: ~/.toaripi/datasets/)
- `--name NAME`: Human-friendly dataset name (default: derived from filename)
- `--validate-only`: Only validate data without processing
- `--cultural-review`: Enable cultural appropriateness checks
- `--age-group {primary,secondary}`: Target age group (default: primary)

**Success Output** (exit code 0):
```
✅ Dataset prepared successfully
📊 Statistics:
   - Total pairs: 1,234
   - English words: 12,345  
   - Toaripi words: 11,234
   - Language balance: 0.91
📁 Saved to: ~/.toaripi/datasets/bible_2025_09_22/
🆔 Dataset ID: bible_2025_09_22_abc123
```

**Error Conditions**:
- Exit code 1: File not found or not readable
- Exit code 2: Invalid CSV format or missing required columns
- Exit code 3: Cultural appropriateness check failed
- Exit code 4: Insufficient disk space
- Exit code 5: Data quality validation failed

**Error Output Example**:
```
❌ Data Error: Missing Toaripi translations
🔍 Cause: 15 rows have empty 'toaripi' column
✅ Solution: Add translations or remove incomplete rows
```

### data validate

**Purpose**: Validate existing dataset without reprocessing

**Signature**:
```bash
toaripi data validate DATASET_ID [OPTIONS]
```

**Parameters**:
- `DATASET_ID` (required): Dataset identifier to validate
- `--cultural-review`: Re-run cultural appropriateness checks
- `--detailed`: Show detailed validation report

**Success Output** (exit code 0):
```
✅ Dataset validation passed
📊 Dataset: bible_2025_09_22_abc123
   - Status: Valid
   - Cultural review: Passed
   - Age appropriateness: Primary school suitable
```

### data list

**Purpose**: List all prepared datasets

**Signature**:
```bash
toaripi data list [OPTIONS]
```

**Parameters**:
- `--status {valid,invalid,pending}`: Filter by validation status
- `--age-group {primary,secondary}`: Filter by target age group
- `--format {table,json}`: Output format (default: table)

**Success Output** (exit code 0):
```
📚 Available Datasets:

ID                    Name               Status  Created     Pairs
bible_2025_09_22_abc123  Bible Parallel    Valid   2025-09-22  1,234
stories_2025_09_20_def456 Folk Stories     Valid   2025-09-20    456
```

## Training Commands

### train start

**Purpose**: Start model training session

**Signature**:
```bash
toaripi train start DATASET_ID [OPTIONS]
```

**Parameters**:
- `DATASET_ID` (required): Dataset to use for training
- `--config CONFIG_ID`: Training configuration (default: default)
- `--name NAME`: Session name (default: auto-generated)
- `--resume SESSION_ID`: Resume existing session
- `--background, -b`: Run training in background

**Success Output** (exit code 0):
```
🚀 Training started successfully
📊 Session: bible_training_abc123
   - Dataset: bible_2025_09_22_abc123
   - Configuration: educational_default
   - Estimated duration: 2.5 hours
   - Progress: 0% (0/3 epochs)
🔄 Use 'toaripi train status bible_training_abc123' to monitor progress
```

**Progress Output** (during training):
```
🔄 Training Progress: bible_training_abc123
Epoch 1/3 ████████████████████████████████████ 100% | Loss: 2.34 | ETA: 1.2h
Step 245/738 | Learning Rate: 2.0e-5 | GPU Memory: 6.2GB/8GB
```

### train stop

**Purpose**: Stop running training session

**Signature**:
```bash
toaripi train stop SESSION_ID [OPTIONS]
```

**Parameters**:
- `SESSION_ID` (required): Training session to stop
- `--force, -f`: Force stop without checkpoint save
- `--save-checkpoint`: Explicitly save checkpoint before stopping

**Success Output** (exit code 0):
```
⏹️  Training stopped successfully
💾 Checkpoint saved: ~/.toaripi/sessions/bible_training_abc123/checkpoints/epoch_1_step_245/
🔄 Use 'toaripi train resume bible_training_abc123' to continue later
```

### train status

**Purpose**: Show training session status and progress

**Signature**:
```bash
toaripi train status [SESSION_ID] [OPTIONS]
```

**Parameters**:
- `SESSION_ID` (optional): Specific session (default: current active session)
- `--follow, -f`: Follow real-time updates
- `--logs`: Include recent log entries

**Success Output** (exit code 0):
```
📊 Training Status: bible_training_abc123
   - Status: Running
   - Progress: 33% (1/3 epochs, 245/738 steps)
   - Current Loss: 2.34
   - Learning Rate: 2.0e-5
   - Started: 2025-09-22 14:30:15
   - ETA: 1.2 hours remaining
   - Checkpoint: epoch_1_step_245 (saved 5 minutes ago)
```

### train list

**Purpose**: List all training sessions

**Signature**:
```bash
toaripi train list [OPTIONS]
```

**Parameters**:
- `--status {running,completed,failed,paused}`: Filter by status
- `--recent N`: Show only N most recent sessions
- `--format {table,json}`: Output format (default: table)

## Model Commands

### model list

**Purpose**: List all trained models and checkpoints

**Signature**:
```bash
toaripi model list [OPTIONS]
```

**Parameters**:
- `--status {training,completed,exported}`: Filter by model status
- `--format {table,json}`: Output format (default: table)

**Success Output** (exit code 0):
```
🤖 Available Models:

Session ID           Name              Status     Created     Size   Final Loss
bible_training_abc123  Bible Model v1    Completed  2025-09-22  3.2GB  1.85
stories_training_def456 Stories Model     Training   2025-09-21  -      2.34
```

### model test

**Purpose**: Test content generation with trained model

**Signature**:
```bash
toaripi model test MODEL_ID [OPTIONS]
```

**Parameters**:
- `MODEL_ID` (required): Model checkpoint or session ID
- `--prompt TEXT`: Custom generation prompt
- `--content-type {story,vocabulary,qa}`: Type of content to generate
- `--age-group {primary,secondary}`: Target age group
- `--length N`: Maximum generation length (default: 200)

**Success Output** (exit code 0):
```
🎯 Content Generation Test: bible_model_v1

Prompt: "Create a simple story about fishing"
Content Type: story
Age Group: primary

Generated Content:
---
Tau kopi teo kea. Bava tauhu kea ravi. Kopi teu ravi tau bava kea.
(The children went fishing. Father caught many fish. The children were happy with father's catch.)
---

✅ Cultural Review: Passed
✅ Age Appropriateness: Primary school suitable
✅ Educational Value: High (teaches family cooperation)
```

### model export

**Purpose**: Export model for edge deployment

**Signature**:
```bash
toaripi model export MODEL_ID [OPTIONS]
```

**Parameters**:
- `MODEL_ID` (required): Model to export
- `--format {gguf,onnx,hf}`: Export format (default: gguf)
- `--quantization {q4_k_m,q5_k_m,q8_0}`: Quantization level (default: q4_k_m)
- `--output PATH`: Output directory (default: ~/.toaripi/models/exports/)

**Success Output** (exit code 0):
```
📦 Model export completed
🎯 Original size: 3.2GB → Exported size: 1.8GB (44% compression)
📁 Saved to: ~/.toaripi/models/exports/bible_model_v1_q4.gguf
🚀 Ready for Raspberry Pi deployment
```

## Interactive Commands

### interactive

**Purpose**: Launch interactive menu-driven mode

**Signature**:
```bash
toaripi interactive
```

**Interactive Interface**:
```
🌟 Toaripi SLM Management Tool

Welcome! Please choose an option:

1. 📊 Data Management
   a. Prepare new training data
   b. Validate existing data
   c. List all datasets

2. 🚀 Training Control
   a. Start new training
   b. Check training status
   c. Stop/pause training

3. 🤖 Model Operations
   a. List trained models
   b. Test content generation
   c. Export for deployment

4. ⚙️  Configuration
   a. View current settings
   b. Create training config
   c. System requirements check

q. Quit

Enter your choice (1a, 2b, q, etc.): 
```

## System Commands

### config

**Purpose**: Manage configuration settings

**Signature**:
```bash
toaripi config SUBCOMMAND [OPTIONS]
```

**Subcommands**:
- `show`: Display current configuration
- `set KEY VALUE`: Update configuration value  
- `init`: Initialize default configuration
- `check`: Validate configuration and system requirements

### version

**Purpose**: Show version and system information

**Signature**:
```bash
toaripi version [OPTIONS]
```

**Parameters**:
- `--system`: Include system information
- `--dependencies`: Show dependency versions

**Success Output** (exit code 0):
```
Toaripi SLM CLI Tool v1.0.0

System Information:
- Python: 3.10.12
- Platform: Linux-5.15.0-78-generic-x86_64
- Available Memory: 8.0GB
- Available Disk: 45.2GB
- CUDA Available: No (CPU-only mode)

Dependencies:
- transformers: 4.34.0
- click: 8.1.7
- rich: 13.5.2
- pydantic: 2.4.2
```

## Error Handling

### Global Error Codes
- 0: Success
- 1: File/resource not found
- 2: Invalid input/configuration
- 3: Permission/access denied
- 4: Insufficient resources (disk/memory)
- 5: Validation failed
- 6: Training/operation failed
- 7: Internal system error

### Error Message Format
All error messages follow the constitutional educational format:
```
❌ [Error Type]: [What went wrong]
🔍 Cause: [Why it happened]
✅ Solution: [How to fix it]
[Additional context if helpful]
```

## Output Formats

### Table Format (default)
Human-readable tables with rich formatting for terminal display

### JSON Format
Machine-readable JSON for scripting:
```json
{
  "status": "success",
  "data": {
    "datasets": [
      {
        "id": "bible_2025_09_22_abc123",
        "name": "Bible Parallel",
        "status": "valid",
        "created_at": "2025-09-22T14:30:15Z",
        "total_pairs": 1234
      }
    ]
  }
}
```

### Progress Format
Real-time progress with rich terminal UI including progress bars, live metrics, and status updates.