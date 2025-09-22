# Quickstart Guide: Toaripi SLM CLI Tool

## Overview

This guide walks you through the essential workflow of preparing data, training a model, and generating educational content using the Toaripi SLM CLI tool.

## Prerequisites

- Python 3.10 or higher installed
- At least 8GB of available RAM
- 10GB of free disk space
- CSV file with English-Toaripi parallel text

## Installation

```bash
# Install the Toaripi SLM CLI tool
pip install toaripi-slm

# Verify installation
toaripi version --system
```

Expected output:
```
Toaripi SLM CLI Tool v1.0.0
System Requirements: ✅ PASSED
- Available Memory: 8.2GB (sufficient)
- Available Disk: 45.2GB (sufficient)  
- Python Version: 3.10.12 (compatible)
```

## Quick Start Workflow

### Step 1: Prepare Training Data

Start with a CSV file containing English-Toaripi parallel text:

```csv
english,toaripi
"The children went fishing","Tau kopi teo kea"
"Father caught many fish","Bava tauhu kea ravi"
"The family was happy","Kopi ravi kea tau"
```

Prepare the data for training:

```bash
# Prepare your parallel text data
toaripi data prepare my_bible_data.csv --name "Bible Training Data" --cultural-review

# Verify the prepared data
toaripi data list
```

Expected output:
```
✅ Dataset prepared successfully
📊 Statistics:
   - Total pairs: 1,234
   - English words: 12,345  
   - Toaripi words: 11,234
   - Language balance: 0.91
   - Cultural review: ✅ PASSED
📁 Saved to: ~/.toaripi/datasets/bible_training_data_abc123/
```

### Step 2: Start Model Training

Begin training your Toaripi language model:

```bash
# Start training with the prepared dataset
toaripi train start bible_training_data_abc123 --name "My First Toaripi Model"

# Monitor training progress
toaripi train status --follow
```

Expected output:
```
🚀 Training started successfully
📊 Session: my_first_toaripi_model_def456
   - Dataset: bible_training_data_abc123 (1,234 pairs)
   - Configuration: educational_default
   - Estimated duration: 2.5 hours
   - Memory usage: 6.2GB/8GB

🔄 Training Progress:
Epoch 1/3 ████████████████████████████████████ 100% | Loss: 2.34 | ETA: 1.8h
Step 245/738 | Learning Rate: 2.0e-5 | Checkpoints: Auto-saved every 50 steps
```

### Step 3: Test Your Trained Model

Once training completes, test content generation:

```bash
# Test story generation
toaripi model test my_first_toaripi_model_def456 \
  --content-type story \
  --prompt "Children learning about fishing"

# Test vocabulary generation  
toaripi model test my_first_toaripi_model_def456 \
  --content-type vocabulary \
  --prompt "Ocean and fishing words"
```

Expected output:
```
🎯 Content Generation Test: my_first_toaripi_model_def456

Generated Story:
---
Tau kopi teo kea bava. Bava amai tau kopi rava kea. 
Kopi haro amai teo ravi oma tau.

(The children went with father. Father taught the children about fishing.
The children learned well about catching fish.)
---

✅ Cultural Review: PASSED
✅ Age Appropriateness: Primary school suitable  
✅ Educational Value: High (teaches family cooperation and learning)
```

### Step 4: Export for Deployment

Export your model for use on resource-constrained devices:

```bash
# Export model for Raspberry Pi deployment
toaripi model export my_first_toaripi_model_def456 \
  --format gguf \
  --quantization q4_k_m \
  --output ./my_toaripi_model/
```

Expected output:
```
📦 Model export completed
🎯 Original size: 3.2GB → Exported size: 1.8GB (44% compression)
📁 Exported to: ./my_toaripi_model/toaripi_educational_q4.gguf
🚀 Ready for offline deployment on Raspberry Pi
```

## Interactive Mode

For users who prefer guided menu navigation:

```bash
# Launch interactive mode
toaripi interactive
```

The interactive mode provides a menu-driven interface:

```
🌟 Toaripi SLM Management Tool - Interactive Mode

Welcome! What would you like to do?

1. 📊 Data Management
   a. Prepare new training data
   b. Check existing datasets
   c. Validate data quality

2. 🚀 Training Operations  
   a. Start new training session
   b. Check training progress
   c. Manage training sessions

3. 🤖 Model Management
   a. List available models
   b. Test content generation
   c. Export models

4. ❓ Help & Documentation
   a. View this quickstart guide
   b. Check system requirements
   c. Educational guidelines

Enter your choice (1a, 2b, etc.) or 'q' to quit: 1a
```

## Common Workflows

### Workflow 1: Creating Educational Stories

```bash
# 1. Prepare story-focused training data
toaripi data prepare story_data.csv \
  --name "Toaripi Stories" \
  --age-group primary \
  --cultural-review

# 2. Train with story-optimized configuration
toaripi train start story_data_abc123 \
  --config story_generation \
  --name "Story Generator Model"

# 3. Test story generation
toaripi model test story_generator_def456 \
  --content-type story \
  --prompt "A story about children helping their community"
```

### Workflow 2: Building Vocabulary Tools

```bash
# 1. Prepare vocabulary-rich data
toaripi data prepare vocabulary_data.csv \
  --name "Toaripi Vocabulary" \
  --cultural-review

# 2. Train vocabulary model  
toaripi train start vocabulary_data_ghi789 \
  --config vocabulary_focused \
  --name "Vocabulary Helper"

# 3. Generate vocabulary exercises
toaripi model test vocabulary_helper_jkl012 \
  --content-type vocabulary \
  --prompt "Ocean and fishing vocabulary for primary students"
```

### Workflow 3: Question-Answer Generation

```bash
# 1. Use comprehension-focused data
toaripi data prepare comprehension_data.csv \
  --name "Reading Comprehension"

# 2. Train Q&A model
toaripi train start comprehension_data_mno345 \
  --config qa_generation \
  --name "Comprehension Assistant"

# 3. Generate questions
toaripi model test comprehension_assistant_pqr678 \
  --content-type qa \
  --prompt "Questions about traditional Toaripi fishing methods"
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: Training runs out of memory

```
❌ System Error: Insufficient memory for training
🔍 Cause: Training batch size (8) requires 12GB RAM, only 8GB available
✅ Solution: Reduce batch size to 4 in configuration
```

**Fix**:
```bash
# Create custom config with smaller batch size
toaripi config create low_memory \
  --batch-size 4 \
  --gradient-accumulation-steps 2
  
# Use the low-memory configuration
toaripi train start dataset_id --config low_memory
```

#### Issue: Cultural review failures

```
❌ Content Error: Cultural appropriateness check failed
🔍 Cause: Training data contains content not suitable for educational use
✅ Solution: Review and filter training data before preparation
```

**Fix**:
```bash
# Re-run data preparation with stricter cultural checks
toaripi data prepare data.csv \
  --cultural-review \
  --age-group primary \
  --validate-only  # Check without processing first
```

#### Issue: Model generation quality issues

```
❌ Generation Error: Output quality below educational standards
🔍 Cause: Insufficient training data or too few training epochs
✅ Solution: Add more training data or increase training duration
```

**Fix**:
```bash
# Check training statistics
toaripi data validate dataset_id --detailed

# If needed, train longer
toaripi train start dataset_id \
  --config extended_training \
  --name "Extended Quality Training"
```

## Next Steps

After completing this quickstart:

1. **Explore Advanced Features**: Learn about custom training configurations and advanced content generation
2. **Deploy Your Model**: Set up your trained model on Raspberry Pi or other edge devices
3. **Educational Integration**: Integrate generated content into classroom materials
4. **Community Contribution**: Share high-quality datasets and models with the Toaripi language community

## Educational Best Practices

### Data Quality Guidelines

- **Parallel Text Quality**: Ensure accurate English-Toaripi translations
- **Cultural Sensitivity**: Review all content for cultural appropriateness
- **Age Appropriateness**: Match vocabulary complexity to target age group
- **Educational Value**: Focus on content that teaches language and culture

### Training Tips

- **Start Small**: Begin with a small, high-quality dataset (500-1000 pairs)
- **Monitor Progress**: Regularly check training metrics and generated samples
- **Cultural Review**: Always enable cultural appropriateness checks
- **Resource Planning**: Ensure sufficient disk space and memory before training

### Content Generation Guidelines

- **Clear Prompts**: Use specific, educational prompts for better results
- **Iterative Testing**: Test multiple prompts to find what works best
- **Educational Review**: Always review generated content before classroom use
- **Community Feedback**: Share and get feedback on generated educational materials

## Support and Resources

- **Documentation**: Full documentation at `toaripi --help`
- **Configuration**: View all settings with `toaripi config show`
- **System Check**: Verify requirements with `toaripi config check`
- **Community**: Join the Toaripi language preservation community discussions

Remember: This tool exists to support Toaripi language preservation and education. Always prioritize cultural sensitivity and educational value in your work.