# Web Interface Quickstart Guide

**Feature**: Web Interface for CSV Data Upload and Model Training  
**Target Users**: Teachers, Toaripi speakers, educational content developers  
**Date**: 2025-09-20

## 🚀 Quick Start (5 Minutes)

### 1. Access the Web Interface

Open your web browser and navigate to:
```
http://localhost:8000
```

You'll see the Toaripi SLM web interface with three main sections:
- **📂 Data Upload** - Upload CSV training data
- **🧠 Model Training** - Train new models
- **📝 Content Generation** - Generate educational content

### 2. Upload Your First Training Data

**Prepare your CSV file:**
```csv
english,toaripi
"The children are playing","Teme havere umupama"
"Water is flowing","Vai umuava"
"Birds are singing","Marubu umukanakana"
"The sun is shining","Laira umupokopoko"
"Fish swim in the river","Ika vai emekoro umuava"
```

**Upload steps:**
1. Click **"Upload Training Data"** 
2. Select your CSV file (max 50MB)
3. Wait for validation (usually 10-30 seconds)
4. Review the preview of validated data
5. Click **"Accept for Training"** if everything looks good

**What happens during validation:**
- ✅ File format checking
- ✅ Required columns verification (`english`, `toaripi`)
- ✅ Text length validation (5-300 characters)
- ✅ Content safety screening
- ✅ Duplicate removal
- ✅ Character set validation for Toaripi

### 3. Start Your First Training Session

Once your data is validated:

1. Click **"Start Training"** 
2. Choose your training settings:
   - **Model Name**: Give your model a descriptive name
   - **Base Model**: Keep default "Mistral-7B-Instruct" 
   - **Training Intensity**: Start with "Balanced" (recommended)
3. Click **"Begin Training"**

**Training progress:**
- Real-time progress bar and time estimates
- Live loss metrics and performance indicators  
- Training typically takes 30-60 minutes for 1000+ text pairs
- You can safely close the browser - training continues in background

### 4. Generate Educational Content

After training completes:

1. Go to **"Content Generation"** tab
2. Choose content type:
   - **📖 Stories** - Short educational stories
   - **📚 Vocabulary** - Word lists with examples
   - **💬 Dialogues** - Conversational scenarios
   - **❓ Q&A** - Reading comprehension questions
3. Enter a topic (e.g., "fishing in the village")
4. Select age group: **Primary** (ages 6-12) or **Secondary** (ages 13+)
5. Click **"Generate Content"**

**Example output:**
```
Topic: Fishing in the village
Age Group: Primary
Content Type: Story

Toaripi Story:
Teme havere vai emekoro umukevakeva. Papa-la ika umuhevaheva. 
Mama-la net umupokapoka. Havere umukuikui ika umupokoia.
Aio havere umuheai papa mama umudaia.

English Translation:
The children go to the river. Father catches fish.
Mother repairs the net. The children watch and learn to catch fish.
Now the children help father and mother with work.
```

---

## 📋 Detailed Workflows

### Data Upload Workflow

**Step 1: Prepare Quality Training Data**

Your CSV should follow this format:
```csv
english,toaripi,category,difficulty
"Good morning","Aiodia kore",greetings,beginner
"How are you today?","Koepuka aruaia?",conversation,beginner
"The weather is nice","Havaia kaiakaia",daily_life,intermediate
```

**Required columns:**
- `english` - English text (5-300 characters)
- `toaripi` - Toaripi translation (5-300 characters)

**Optional columns:**
- `category` - Content category (greetings, daily_life, nature, etc.)
- `difficulty` - Difficulty level (beginner, intermediate, advanced)

**Best practices:**
- ✅ Use simple, clear sentences appropriate for students
- ✅ Include diverse topics (daily life, nature, culture)
- ✅ Ensure accurate translations
- ✅ Aim for 200+ text pairs minimum for good results
- ❌ Avoid complex technical terms
- ❌ Don't include adult or inappropriate content
- ❌ Avoid religious or controversial topics

**Step 2: Upload and Validation**

1. Click **"Choose File"** and select your CSV
2. The system validates your data:
   ```
   ⏳ Uploading file... (5-10 seconds)
   ⏳ Validating format... (5-15 seconds)  
   ⏳ Checking content safety... (10-30 seconds)
   ⏳ Processing text pairs... (5-20 seconds)
   ✅ Upload complete!
   ```

3. Review validation results:
   - **Green**: All data valid, ready for training
   - **Yellow**: Some issues found, partial data usable
   - **Red**: Significant problems, needs fixing

**Step 3: Handle Validation Issues**

Common issues and fixes:

| Issue | Description | How to Fix |
|-------|-------------|------------|
| Text too long | Sentences exceed 300 characters | Break into shorter sentences |
| Invalid characters | Non-Toaripi characters found | Use standard Toaripi spelling |
| Missing translations | Empty cells in required columns | Fill in all English↔Toaripi pairs |
| Safety concerns | Content flagged as inappropriate | Remove or rephrase flagged content |
| Duplicates | Same text pairs found multiple times | Remove duplicate rows |

### Model Training Workflow

**Step 1: Configure Training Settings**

**Basic Settings (Recommended for beginners):**
- **Model Name**: "My Toaripi Model v1.0"
- **Training Intensity**: "Balanced"
- **Base Model**: "Mistral-7B-Instruct" (default)

**Advanced Settings (For experienced users):**
```yaml
Learning Rate: 2e-4      # How fast the model learns
LoRA Rank: 16           # Model adaptation complexity  
Batch Size: 4           # Training efficiency
Max Epochs: 3           # Training rounds
```

**Hardware compatibility check:**
- System automatically validates settings against available resources
- Adjusts parameters if needed to fit within 8GB RAM limit
- Estimates training time and memory usage

**Step 2: Monitor Training Progress**

Real-time training dashboard shows:

```
🧠 Training: Toaripi Educational v2.1
📊 Progress: ████████░░ 67% (1,441 / 2,135 steps)
⏱️ Time: 28 min elapsed, ~15 min remaining
📈 Loss: 0.245 (decreasing ✓)
💾 Memory: 6.8 GB / 8.0 GB available
🔥 Status: Training epoch 2 of 3
```

**Training phases:**
1. **Starting** (1-2 min) - Loading model and data
2. **Training** (30-90 min) - Learning from your data
3. **Evaluating** (2-5 min) - Testing model quality
4. **Exporting** (3-8 min) - Creating optimized model files
5. **Complete** - Ready for content generation!

**Step 3: Training Results**

Upon completion, you'll see:
```
✅ Training Complete!

📊 Final Metrics:
   Loss: 0.187 (excellent)
   Safety Score: 96% (passed)
   Quality Score: 78% (good)

📁 Model Files:
   - Raspberry Pi: toaripi-v2.1-q4.gguf (3.9 GB)
   - Full Model: toaripi-v2.1-hf (13.4 GB)

🎯 Ready for content generation!
```

### Content Generation Workflow

**Step 1: Choose Content Type**

**📖 Stories** - Educational narratives
- **Best for**: Teaching cultural concepts, moral lessons
- **Length**: 3-8 sentences
- **Example topic**: "Helping family with fishing"

**📚 Vocabulary** - Word lists with context
- **Best for**: Building language skills, topical learning
- **Length**: 5-20 words with examples  
- **Example topic**: "Kitchen items and cooking"

**💬 Dialogues** - Conversational practice
- **Best for**: Speaking practice, social scenarios
- **Length**: 4-10 exchanges
- **Example topic**: "Meeting a friend at the market"

**❓ Q&A** - Comprehension questions
- **Best for**: Reading comprehension, assessment
- **Length**: 2-5 questions with answers
- **Example topic**: "Understanding weather patterns"

**Step 2: Set Generation Parameters**

```
Topic: [Enter 2-80 characters]
Examples: "village fishing", "preparing traditional food", 
         "children playing games", "family garden work"

Age Group: 
🧒 Primary (6-12) - Simple language, short sentences
🧑 Secondary (13+) - More complex concepts and vocabulary

Content Length:
📏 Short (3-5 items) - Quick exercises
📏 Medium (5-8 items) - Standard lessons
📏 Long (8-12 items) - Comprehensive content
```

**Step 3: Review and Use Generated Content**

Generated content appears with:
- **Toaripi text** - Target language content
- **English translation** - For teacher reference
- **Age appropriateness** - Confirmed suitable for selected age group
- **Cultural sensitivity** - Validated for Toaripi cultural context

**Download options:**
- 📄 **PDF** - Formatted for printing and classroom use
- 📝 **Word Document** - Editable for customization
- 💾 **Text File** - Plain text for digital distribution
- 🌐 **Web Page** - Shareable online content

---

## 🛠️ Troubleshooting

### Common Upload Issues

**❌ "File too large" error**
- **Cause**: CSV file exceeds 50MB limit
- **Solution**: Split large files into smaller batches or compress

**❌ "Invalid CSV format" error**  
- **Cause**: File is not properly formatted CSV
- **Solution**: Save as CSV from Excel, check for special characters

**❌ "Missing required columns" error**
- **Cause**: CSV doesn't have 'english' and 'toaripi' columns
- **Solution**: Add column headers exactly as: `english,toaripi`

**❌ "Insufficient training data" error**
- **Cause**: Less than 150 valid text pairs after validation
- **Solution**: Add more text pairs or fix validation errors

### Training Problems

**❌ Training stuck at "Starting"**
- **Cause**: System may be busy or insufficient resources
- **Solution**: Wait 5 minutes, refresh page, or contact support

**❌ "Out of memory" error**
- **Cause**: Training parameters exceed available RAM
- **Solution**: Reduce batch size to 2 or use fewer training examples

**❌ Training fails with high loss**
- **Cause**: Poor quality training data or mismatched translations
- **Solution**: Review and improve training data quality

### Generation Issues

**❌ Generated content is repetitive**
- **Cause**: Insufficient training data variety
- **Solution**: Add more diverse examples to training data

**❌ Content not age-appropriate**
- **Cause**: Model needs more examples for target age group
- **Solution**: Include more age-specific examples in training

**❌ Poor translation quality**
- **Cause**: Limited training data or low-quality translations
- **Solution**: Verify translation accuracy in training data

### System Requirements

**Minimum Requirements:**
- **RAM**: 8GB available (12GB recommended)
- **Storage**: 20GB free space for model files
- **Browser**: Chrome 90+, Firefox 88+, Safari 14+
- **Connection**: Stable internet for initial setup

**Optimal Performance:**
- **RAM**: 16GB+ for faster training
- **Storage**: SSD with 50GB+ free space
- **CPU**: 8+ cores for faster processing
- **GPU**: CUDA-compatible GPU (optional, improves training speed)

---

## 📚 Learning Resources

### Video Tutorials
- **Getting Started** (5 min) - Basic upload and training
- **Advanced Training** (12 min) - Custom configuration options
- **Content Creation** (8 min) - Generating different content types
- **Troubleshooting** (6 min) - Common issues and solutions

### Example Training Data
Download sample CSV files to get started:
- **Basic Conversations** (500 pairs) - Greetings, daily interactions
- **Educational Stories** (300 pairs) - Simple narratives for children
- **Vocabulary Building** (800 pairs) - Topical word lists with context
- **Cultural Content** (400 pairs) - Traditional stories and practices

### Best Practices Guide
- **Data Quality Guidelines** - Creating effective training data
- **Training Optimization** - Getting best results from your models
- **Content Creation Tips** - Generating engaging educational materials
- **Classroom Integration** - Using generated content for teaching

### Community Support
- **User Forum** - Connect with other Toaripi language educators
- **Weekly Q&A** - Live sessions with development team
- **Feature Requests** - Suggest improvements and new capabilities
- **Bug Reports** - Report issues and get technical support

---

## 🎯 Next Steps

### Immediate Actions (Today)
1. ✅ **Upload your first training data** - Start with 100-200 text pairs
2. ✅ **Run a training session** - Use default settings for first attempt  
3. ✅ **Generate sample content** - Try each content type (story, vocabulary, etc.)
4. ✅ **Download and review** - Export content for classroom testing

### This Week
1. **Expand training data** - Add 500+ more high-quality text pairs
2. **Test with students** - Use generated content in actual lessons
3. **Gather feedback** - Note what works and what needs improvement
4. **Iterate and improve** - Upload better data, retrain models

### This Month  
1. **Build content library** - Generate 20+ pieces of educational content
2. **Train specialized models** - Create models for specific topics or age groups
3. **Share with community** - Contribute to Toaripi language preservation
4. **Advanced features** - Explore custom training configurations

### Long-term Goals
1. **Comprehensive curriculum** - Full educational content suite in Toaripi
2. **Offline deployment** - Set up Raspberry Pi for classroom use
3. **Teacher training** - Help other educators use these tools
4. **Language preservation** - Contribute to digital Toaripi language resources

---

**Need Help?** 
- 📧 Email: support@toaripi-slm.org
- 💬 Community Forum: https://forum.toaripi-slm.org  
- 📚 Documentation: https://docs.toaripi-slm.org
- 🐛 Bug Reports: https://github.com/toaripi-slm/issues

## 6. Fine-Tune
Invoke training script with config + data path; outputs HF adapter + merged model under `models/hf/` with metadata JSON.

## 7. Quantize
Convert model to GGUF (Q4_K_M) into `models/gguf/` for offline inference.

## 8. Launch Inference API
Start FastAPI app (to be implemented) exposing /generate, /health, /evaluation-pack.

## 9. Generate Content
Send POST /generate with JSON: topic, content_type, target_length, mode, include_english_support.

## 10. Evaluation Pack
POST /evaluation-pack to produce 12-sample pack; review manually.

## 11. Logs & Metadata
Inspect `logs/generation.log` for JSON lines containing generation metadata.

## 12. Stable Mode
Repeat a request with mode=stable to verify ≥90% token overlap.

## 13. Safety Verification
Submit prompts near restricted boundaries to ensure block behavior (no theological, violent, adult content allowed).

## 14. Cleanup
Purge old logs (>30 days) and stale cache entries automatically handled by maintenance job (to be implemented).
