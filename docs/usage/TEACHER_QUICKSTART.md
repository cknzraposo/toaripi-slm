# Quick Start Guide for Teachers

## What is Toaripi SLM?

Toaripi SLM is an educational tool that helps generate learning materials in the Toaripi language. It creates stories, vocabulary exercises, and other educational content appropriate for different age groups.

## Getting Started

### Step 1: Check Your System

First, make sure everything is working:

```bash
toaripi-slm validate
```

This will check that your computer has everything needed.

### Step 2: See What Data You Have

Check what educational materials are available:

```bash
toaripi-slm data list
```

### Step 3: Generate Your First Content

Start the educational content server:

```bash
toaripi-slm serve start --model models/toaripi-primary.gguf --age-filter primary_lower
```

## Quick Commands for Daily Use

### For Primary Lower Students (Ages 6-8)

Generate a simple story:
```bash
toaripi-slm model test models/toaripi-primary.gguf \
  --age-group primary_lower \
  --content-type story \
  --prompt "Children playing by the river"
```

Create vocabulary list:
```bash
toaripi-slm model test models/toaripi-primary.gguf \
  --age-group primary_lower \
  --content-type vocabulary \
  --prompt "Animals in the forest"
```

### For Primary Upper Students (Ages 9-11)

Generate reading comprehension:
```bash
toaripi-slm model test models/toaripi-primary.gguf \
  --age-group primary_upper \
  --content-type comprehension \
  --prompt "Traditional fishing methods"
```

Create dialogue practice:
```bash
toaripi-slm model test models/toaripi-primary.gguf \
  --age-group primary_upper \
  --content-type dialogue \
  --prompt "Visiting the market"
```

## Classroom Setup

### Setting Up for Multiple Students

1. **Start the server for the whole class:**
   ```bash
   toaripi-slm serve start \
     --model models/toaripi-primary.gguf \
     --host 0.0.0.0 \
     --port 8080 \
     --age-filter primary_lower \
     --cultural-validation
   ```

2. **Students can access from tablets/computers at:**
   ```
   http://teacher-computer:8080
   ```

### Creating Lesson-Specific Content

1. **Prepare content for today's lesson:**
   ```bash
   toaripi-slm data prepare \
     --input lesson-materials/marine-life.csv \
     --age-groups primary_lower \
     --content-types story vocabulary
   ```

2. **Validate content is appropriate:**
   ```bash
   toaripi-slm data validate \
     --file lesson-materials/marine-life.csv \
     --check-educational \
     --check-cultural
   ```

## Safety and Cultural Guidelines

### Always Check Content

Before using generated content with students:

```bash
toaripi-slm model test your-model \
  --age-group primary_lower \
  --cultural-check
```

### Age-Appropriate Filters

- **Early Childhood (3-5)**: Very simple words and concepts
- **Primary Lower (6-8)**: Basic stories and vocabulary
- **Primary Upper (9-11)**: More complex narratives
- **Secondary (12-18)**: Advanced content and cultural studies

### Cultural Sensitivity

The system automatically checks for:
- Traditional Toaripi values
- Appropriate cultural content
- Community-respectful language
- Age-suitable concepts

## Troubleshooting

### "No server running"
Start the server first:
```bash
toaripi-slm serve start --model models/toaripi-primary.gguf
```

### "Model not found"
Check available models:
```bash
toaripi-slm model list
```

### "Content inappropriate"
Use stricter validation:
```bash
toaripi-slm model test your-model --cultural-check --age-group primary_lower
```

### Getting Help
```bash
toaripi-slm --help
toaripi-slm serve --help
toaripi-slm model --help
```

## Sample Lesson Plan

### Topic: Ocean Animals (Primary Lower)

1. **Prepare vocabulary:**
   ```bash
   toaripi-slm model test models/toaripi-primary.gguf \
     --age-group primary_lower \
     --content-type vocabulary \
     --prompt "Ocean animals - fish, whale, dolphin, turtle"
   ```

2. **Create a story:**
   ```bash
   toaripi-slm model test models/toaripi-primary.gguf \
     --age-group primary_lower \
     --content-type story \
     --prompt "A little fish explores the coral reef"
   ```

3. **Make comprehension questions:**
   ```bash
   toaripi-slm model test models/toaripi-primary.gguf \
     --age-group primary_lower \
     --content-type comprehension \
     --prompt "Questions about ocean animals story"
   ```

4. **Practice dialogue:**
   ```bash
   toaripi-slm model test models/toaripi-primary.gguf \
     --age-group primary_lower \
     --content-type dialogue \
     --prompt "Two children talking about their favorite ocean animals"
   ```

## Daily Workflow

### Morning Setup (5 minutes)
```bash
# Check system
toaripi-slm status

# Start server for class
toaripi-slm serve start \
  --model models/toaripi-primary.gguf \
  --age-filter primary_lower \
  --daemon
```

### During Lessons
- Generate content as needed using the model test commands
- All content is automatically checked for age and cultural appropriateness
- Students can access the educational API from their devices

### End of Day
```bash
# Stop the server
toaripi-slm serve stop
```

## Tips for Best Results

1. **Be Specific in Prompts**: Instead of "animals", use "forest animals that children see near the village"

2. **Check Cultural Context**: Always use the `--cultural-check` flag for new content

3. **Age-Appropriate Language**: The system automatically adjusts language complexity based on the age group

4. **Save Good Content**: Keep successful prompts and content for future lessons

5. **Student Safety**: Always preview generated content before sharing with students

## Need More Help?

- Check the full CLI reference: `docs/usage/CLI_REFERENCE.md`
- Validate your setup: `toaripi-slm validate`
- Contact technical support with specific error messages

Remember: This tool is designed to assist teachers, not replace the important work of education and cultural transmission that you do every day.