# Toaripi SLM CLI - Quick Start Guide

Welcome to the Toaripi Small Language Model CLI! This tool provides a sleek command-line interface for training, testing, and interacting with language models designed for Toaripi educational content generation.

## 🚀 Installation

### Option 1: Using the Installation Script (Recommended)

**Linux/macOS:**
```bash
./scripts/install_cli.sh
```

**Windows:**
```cmd
scripts\install_cli.bat
```

### Option 2: Manual Installation

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv toaripi_env
   
   # Linux/macOS
   source toaripi_env/bin/activate
   
   # Windows
   toaripi_env\Scripts\activate.bat
   ```

2. **Install the package:**
   ```bash
   pip install -e .
   ```

3. **Verify installation:**
   ```bash
   toaripi --help
   ```

## 🔍 First Steps

### 1. Check System Status
```bash
toaripi status --detailed
```

This shows your Python version, dependencies, and project structure.

### 2. Run System Diagnostics
```bash
toaripi doctor
```

Get a comprehensive health check of your system, including:
- Dependency validation
- Hardware compatibility
- Project structure verification
- Performance recommendations

### 3. Check Available Commands
```bash
toaripi --help
```

See all available commands and options.

## 📚 Main Commands

### `toaripi train` - Train Your Model

**Interactive training (recommended for beginners):**
```bash
toaripi train --interactive
```

**Quick training with dry run:**
```bash
toaripi train --dry-run
```

**Train with specific configuration:**
```bash
toaripi train --config configs/training/custom_config.yaml
```

### `toaripi test` - Evaluate Your Model

**Basic model testing:**
```bash
toaripi test --model ./models/hf
```

**Comprehensive testing with benchmarks:**
```bash
toaripi test --benchmark --interactive
```

### `toaripi interact` - Chat with Your Model

**Start interactive session:**
```bash
toaripi interact --model ./models/hf
```

**Generate specific content types:**
```bash
toaripi interact --content-type vocabulary --temperature 0.5
```

### `toaripi status` - System Information

**Quick status check:**
```bash
toaripi status
```

**Detailed system information:**
```bash
toaripi status --detailed
```

### `toaripi doctor` - System Diagnostics

**Basic health check:**
```bash
toaripi doctor
```

**Detailed diagnostics with report:**
```bash
toaripi doctor --detailed --export-report health_report.json
```

## 🎯 Content Types

The CLI supports generating different types of educational content:

1. **📖 Stories** - Educational stories for primary students
2. **📝 Vocabulary** - Word lists with translations
3. **💬 Dialogue** - Conversations between characters
4. **❓ Questions** - Reading comprehension questions
5. **🔄 Translation** - English to Toaripi translation

## 🛠️ Interactive Commands

When using `toaripi interact`, you can use these special commands:

- `/help` - Show help
- `/type story` - Change to story generation
- `/settings` - Adjust generation parameters
- `/history` - View conversation history
- `/save` - Save session to file
- `/quit` - Exit interactive mode

## 🐛 Troubleshooting

### Common Issues

**"toaripi: command not found"**
```bash
# Reinstall the package
pip install -e .

# Or use direct Python execution
python -m toaripi_slm.cli
```

**"Missing required dependencies"**
```bash
# Install dependencies
pip install -r requirements.txt

# Check what's missing
toaripi doctor
```

**"Model not found"**
```bash
# Check available models
ls models/

# Train a model first
toaripi train --interactive
```

**"Data validation failed"**
```bash
# Check data format
toaripi status --detailed

# Prepare training data
toaripi-prepare-data
```

### Debug Mode

Enable detailed error messages:
```bash
toaripi --debug --verbose command
```

## 📈 Workflow Example

Here's a typical workflow for training and using a Toaripi model:

```bash
# 1. Check system health
toaripi doctor

# 2. Prepare your data (if needed)
toaripi-prepare-data

# 3. Train your model interactively
toaripi train --interactive

# 4. Test the trained model
toaripi test --model ./models/hf --benchmark

# 5. Interactive chat with the model
toaripi interact --model ./models/hf

# 6. Generate specific content
# (Use /type vocabulary in interactive mode)
```

## 🔧 Configuration

The CLI uses YAML configuration files in `configs/`:

- `configs/training/base_config.yaml` - Training parameters
- `configs/training/lora_config.yaml` - LoRA fine-tuning settings
- `configs/data/preprocessing_config.yaml` - Data processing options

## 🌟 Tips for Best Results

1. **Start with interactive mode** - The CLI guides you through the process
2. **Use dry-run first** - Validate your setup before training
3. **Check system health** - Run `toaripi doctor` to identify issues early
4. **Save your sessions** - Use `/save` in interactive mode
5. **Adjust parameters** - Use `/settings` to fine-tune generation

## 📖 Further Reading

- **Full CLI Guide:** `docs/CLI_GUIDE.md`
- **Training Documentation:** `docs/contributing/Training Toaripi Language Model.pdf`
- **Project Specifications:** `specs/001-offline-capable-toaripi/`

## 🤝 Getting Help

- **Built-in help:** `toaripi --help`
- **Command help:** `toaripi COMMAND --help`
- **System diagnostics:** `toaripi doctor --detailed`
- **Debug mode:** `toaripi --debug COMMAND`

Happy training! 🚀

---

*The Toaripi SLM CLI is designed to make language model training accessible and intuitive for educators, linguists, and developers working on language preservation.*