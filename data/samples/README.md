# Sample Data for Toaripi SLM

This directory contains sample datasets for development and testing.

## Files

- `sample_parallel.csv`: Basic parallel English-Toaripi sentences
- `educational_prompts.json`: Sample prompts for educational content generation
- `vocabulary_topics.yaml`: Topics for vocabulary generation

## Data Format

### Parallel Data (CSV)

```csv
english,toaripi,verse_id,book,chapter
"The child is playing in the garden.","Narau apu poroporosi hoi-ia.",sample_001,Samples,1
```

### Educational Prompts (JSON)

```json
{
  "story_prompts": [
    {
      "topic": "daily_activities",
      "prompt": "Write a story about children helping with daily chores",
      "age_group": "primary",
      "length": "short"
    }
  ]
}
```

## Usage

Use these samples for:

- Testing data processing pipelines
- Validating model training workflows
- Developing content generation features
- Creating unit tests

**Note:** These are synthetic samples for development only. Real training data should contain authentic Toaripi language content validated by native speakers.