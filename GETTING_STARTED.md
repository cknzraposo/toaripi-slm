# 🚀 Getting Started with Toaripi SLM

**Welcome!** This guide will get you from zero to training Toaripi language models in under 5 minutes.

Looking for a minimal copy/paste cross‑platform (Linux / WSL / Windows) cheat sheet? See `docs/QUICK_PLATFORM_START.md`.

## 🎯 Quick Start (Choose Your Method)

### Method 1: Automated Setup (Recommended)
```bash
# Navigate to the project
cd /path/to/toaripi-slm

# Option A: Universal launcher (works everywhere)
python3 start_toaripi.py

# Option B: Shell script (Linux/Mac/WSL - faster)
./start_toaripi.sh

# Option C: Windows batch file
start_toaripi.bat
```

### Method 2: Manual Setup
```bash
# 1. Create virtual environment
python3 -m venv toaripi_env

# 2. Activate it
source toaripi_env/bin/activate  # Linux/Mac
# OR
toaripi_env\Scripts\activate.bat  # Windows

# 3. Install Toaripi SLM
pip install -e .

# 4. Verify installation
toaripi --help
```

### Method 3: Convenient Aliases
```bash
# Setup aliases (only need to do this once)
source setup_aliases.sh

# Then use convenient commands
toaripi-status      # Check system
toaripi-train       # Start training
toaripi-chat        # Interactive chat
```

## 🎮 Your First Training Session

Once setup is complete, start your first training:

```bash
# 1. Check everything is working
toaripi status

# 2. Run system diagnostics (optional but recommended)
toaripi doctor

# 3. Start interactive training
toaripi train
```

The training command will guide you through:
- ✅ **Data preparation** - Processing English↔Toaripi parallel texts
- ✅ **Model selection** - Choosing optimal base models for Toaripi
- ✅ **Training configuration** - LoRA fine-tuning settings
- ✅ **Progress monitoring** - Real-time training metrics
- ✅ **Model evaluation** - Educational content quality checks

## 🧪 Testing Your Model

After training, evaluate your model:

```bash
# Run comprehensive evaluation
toaripi test

# Test specific capabilities
toaripi test --focus stories
toaripi test --focus vocabulary
```

## 💬 Interacting with Your Model

Chat with your trained model:

```bash
# Start interactive session
toaripi interact

# Or specific content types
toaripi interact --mode story
toaripi interact --mode vocabulary
```

## 🛠️ Available Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `toaripi status` | System overview | Check installation, data, models |
| `toaripi doctor` | Diagnostics | Hardware, dependencies, troubleshooting |
| `toaripi train` | Model training | Interactive fine-tuning workflow |
| `toaripi test` | Model evaluation | Educational content assessment |
| `toaripi interact` | Chat interface | Generate stories, vocabulary, Q&A |

## 📁 Project Structure Overview

```
toaripi-slm/
├── 🚀 start_toaripi.py     # Universal launcher
├── 🐧 start_toaripi.sh     # Linux/Mac launcher  
├── 🪟 start_toaripi.bat    # Windows launcher
├── ⚙️ setup_aliases.sh     # Convenience aliases
├── 📊 data/                # Training data
│   ├── raw/               # Original Bible texts
│   ├── processed/         # Aligned parallel data
│   └── samples/           # Example data
├── 🧠 models/             # Trained models
│   ├── hf/               # HuggingFace format
│   └── gguf/             # Quantized for edge
├── ⚙️ configs/            # Training configurations
└── 📖 docs/              # Documentation
```

## 🎯 Educational Content Focus

Toaripi SLM is specifically designed for **educational content generation**:

### 📚 **Stories**
- Age-appropriate narratives
- Cultural context from Toaripi community
- Reading comprehension exercises

### 📝 **Vocabulary**
- Topic-based word lists
- Example sentences
- Interactive exercises

### ❓ **Q&A Generation**
- Reading comprehension questions
- Cultural knowledge assessment
- Language learning prompts

### 💬 **Dialogues**
- Conversational practice
- Real-world scenarios
- Cultural exchanges

## 🔧 Troubleshooting

### Python Not Found
```bash
# Ubuntu/Debian
sudo apt install python3 python3-venv

# macOS
brew install python3

# Windows
# Download from python.org
```

### Permission Denied (Linux/Mac)
```bash
chmod +x start_toaripi.sh
chmod +x setup_aliases.sh
```

### Virtual Environment Issues
```bash
# Remove and recreate
rm -rf toaripi_env
python3 start_toaripi.py
```

### CLI Not Found
```bash
# Check if in virtual environment
which python
pip list | grep toaripi

# Reactivate environment
source toaripi_env/bin/activate
```

### Dependencies Failed
```bash
# Update pip and try again
pip install --upgrade pip
pip install -e . --verbose
```

## 🎮 Next Steps

1. **🔍 Explore**: Run `toaripi status` to see what's available
2. **🧪 Diagnose**: Run `toaripi doctor` for health check
3. **📊 Prepare Data**: Use sample data or add your own parallel texts
4. **🏋️ Train**: Start with `toaripi train` for guided experience
5. **🎯 Test**: Evaluate with `toaripi test` 
6. **💬 Chat**: Try `toaripi interact` to generate content

## 📚 Learning Resources

- 📖 **Documentation**: `/docs/` directory
- 🔬 **Specifications**: `/specs/` for technical details
- 💾 **Sample Data**: `/data/samples/` for examples
- ⚙️ **Configurations**: `/configs/` for training settings

## 🎯 Goals & Philosophy

**Educational First**: Every feature prioritizes learning outcomes for Toaripi students and cultural preservation.

**Offline Capable**: Run on Raspberry Pi, laptops, or servers without internet.

**Teacher Friendly**: Intuitive interfaces designed for educators, not just developers.

**Culturally Respectful**: Content generation respects Toaripi cultural context and community values.

---

## 🚀 Ready to Start?

Choose your launcher and begin your Toaripi SLM journey:

```bash
# Universal (recommended)
python3 start_toaripi.py

# Quick shell script
./start_toaripi.sh

# With aliases
source setup_aliases.sh && toaripi-status
```

**🎓 Goal**: Get your first Toaripi educational content model trained and generating stories for primary school students!

**⏱️ Time**: 5 minutes setup + 30 minutes first training session

**📧 Support**: Check `toaripi doctor` for diagnostics or review `/docs/` for detailed guides.

---

*Happy training! 🌟*