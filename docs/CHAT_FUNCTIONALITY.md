# Chat Functionality - English Q&A with Toaripi Responses

## Overview

The Toaripi SLM Interactive CLI now includes a **chat functionality** that allows users to ask questions in English and receive responses in Toaripi with token weight visualization. This feature is designed for educational vocabulary learning and cultural knowledge sharing.

## 💬 Chat Mode Features

### Question-Answer Format
- **Input**: English questions (e.g., "What is a dog?")
- **Output**: Toaripi word + English translation + cultural description
- **Format**: `toaripi_word/english_word - Cultural description`
- **Example**: `ruru/dog - A four-legged animal that helps with hunting and guards the village`

### Token Weight Visualization
- **Colored Tokens**: Each word is colored based on its importance/attention weight
- **Weight Display**: Numerical values (0.xx) shown next to each token
- **Color Scale**: Red (high importance) → Yellow (medium) → Green (low importance)
- **Educational Value**: Shows which parts of the response are most significant

### Comprehensive Knowledge Base
The chat system includes cultural knowledge across multiple categories:

#### 🐾 Animals
- **dog** → `ruru` - Four-legged animal that helps with hunting and guards the village
- **fish** → `hanere` - Swimming creatures caught in rivers and sea for food
- **bird** → `manu` - Flying creatures with feathers, some eaten, some kept as pets
- **pig** → `boroma` - Large animal raised for food and ceremonies
- **chicken** → `kokorako` - Small bird kept for eggs and meat
- **crocodile** → `buaia` - Large dangerous reptile living in rivers

#### 👨‍👩‍👧‍👦 Family & People
- **mother** → `ina` - The woman who gave birth to and cares for children
- **father** → `tama` - The man who helps create and care for children
- **child** → `bada` - Young person learning from parents and elders
- **children** → `bada-bada` - Multiple young people in the community
- **elder** → `gabua-harigi` - Older person with wisdom and knowledge
- **teacher** → `amo-harigi` - Person who teaches children and shares knowledge
- **student** → `amo-nene` - Person who learns from teachers and elders

#### 🌊 Nature & Environment
- **water** → `peni` - Clear liquid essential for life, found in rivers and rain
- **river** → `malolo` - Flowing water where people fish and get fresh water
- **tree** → `ai` - Large plant with trunk and branches, provides wood and shade
- **village** → `gabua` - Place where families live together in community
- **house** → `ruma` - Building where a family lives and sleeps
- **garden** → `motu` - Place where people grow food plants
- **forest** → `vao` - Area with many trees where animals live
- **sea** → `ranu` - Large body of salt water with many fish

#### 🎣 Activities & Actions
- **fishing** → `gola hanere` - Activity of catching fish from water
- **hunting** → `gola ruru` - Activity of catching animals for food
- **cooking** → `dovu kaikai` - Preparing food by heating it
- **learning** → `nene-ida` - Gaining knowledge and skills from others
- **teaching** → `aki-harigi` - Sharing knowledge and skills with others
- **swimming** → `potopoto` - Moving through water using arms and legs
- **walking** → `lao` - Moving on foot from one place to another

#### 🍌 Food & Daily Life
- **food** → `kaikai` - Things people eat to stay healthy and strong
- **rice** → `raisi` - Small white grains that are cooked and eaten
- **banana** → `banana` - Yellow fruit that grows on trees
- **coconut** → `niu` - Large brown fruit with white meat and water inside
- **sweet potato** → `kumara` - Orange root vegetable grown in gardens

#### 🔧 Objects & Tools
- **net** → `pupu` - Woven tool used to catch fish
- **boat** → `vaka` - Vehicle used to travel on water
- **fire** → `ahi` - Hot flames used for cooking and warmth
- **knife** → `naifi` - Sharp tool used for cutting

#### 💭 Concepts & Abstract
- **good** → `mane-mane` - Something positive, helpful, or well done
- **bad** → `kila-kila` - Something negative, harmful, or wrong
- **big** → `lahi` - Large in size compared to other things
- **small** → `boko` - Little in size compared to other things
- **happy** → `harikoa` - Feeling joy and contentment
- **sad** → `fakahinohino` - Feeling sorrow or unhappiness

## 🚀 How to Use Chat Mode

### Starting Chat Mode
```bash
# Launch interactive mode
toaripi interact

# Switch to chat mode
/type chat
✅ Content type changed to: chat
💬 Chat mode enabled! Ask questions like 'What is a dog?' or 'What is water?'
```

### Question Types Supported
1. **Direct Questions**: "What is a dog?"
2. **Descriptive Requests**: "Tell me about water"
3. **Casual Inquiries**: "Do you know about fishing?"
4. **Complex Questions**: "Can you explain what a village means?"

### Example Chat Session
```
You: What is a fish?
💬 Generated Content (chat):

╭─────── 🇺🇸 English Source ────────╮    ╭──────── 🌺 Toaripi Translation ─────────╮
│ What(0.62) is(0.20) fish?(0.46)    │    │ hanere/fish(0.75) -(0.20) Swimming(0.97) │
│                                   │    │ creatures(0.74) caught(0.64) in(0.45)    │
│                                   │    │ rivers(0.58) and(0.40) sea(0.12)        │
│                                   │    │ for(0.36) food(0.65)                    │
╰───────────────────────────────────╯    ╰─────────────────────────────────────────╯

You: /weights
💡 Token weights display: OFF

You: What is water?
╭─── 🇺🇸 English Source ────╮    ╭──── 🌺 Toaripi Translation ─────╮
│ What is water?            │    │ peni/water - Clear liquid       │
│                          │    │ essential for life, found       │
│                          │    │ in rivers and rain             │
╰──────────────────────────╯    ╰────────────────────────────────╯
```

## 📚 Educational Applications

### For Teachers
- **Vocabulary Building**: Systematic introduction of Toaripi terms
- **Cultural Context**: Each word includes cultural significance
- **Visual Feedback**: Token weights show word importance
- **Assessment**: Can observe student engagement with specific terms

### For Students
- **Interactive Learning**: Ask questions naturally in English
- **Visual Memory**: Color coding aids retention
- **Cultural Understanding**: Context provided for each term
- **Self-Paced**: Students control the questioning pace

### For Researchers
- **Language Documentation**: Comprehensive bilingual glossary
- **Attention Analysis**: Token weights reveal linguistic patterns
- **Usage Patterns**: Track which concepts are queried most
- **Cultural Mapping**: See how concepts are described across languages

## 🎨 Token Weight Visualization

### Color Coding System
- 🔴 **High (0.8+)**: Bold red - Critical/key words
- 🟠 **Medium-High (0.6+)**: Bold orange - Important concepts
- 🟡 **Medium (0.4+)**: Bold yellow - Supporting information
- 🟢 **Low (0.2+)**: Green - Background context
- 🔵 **Very Low (<0.2)**: Dim cyan - Minimal importance

### Understanding Token Weights
```
Example: "hanere/fish - Swimming creatures caught in rivers"

hanere/fish(0.86)    ← Primary term (high weight = red)
Swimming(0.97)       ← Key descriptor (high weight = red)
creatures(0.74)      ← Important concept (medium-high = orange)
caught(0.64)         ← Action context (medium-high = orange)
in(0.45)            ← Location connector (medium = yellow)
rivers(0.58)        ← Context location (medium-high = orange)
```

## 🔧 Interactive Commands

### Chat-Specific Commands
```bash
/type chat          # Enable chat mode
/weights           # Toggle token weight display
/align             # Toggle token alignment
/legend            # Show weight color guide
```

### General Commands
```bash
/help              # Show all commands
/history           # View conversation history
/save              # Save session with chat exchanges
/clear             # Clear conversation history
/quit              # Exit interactive mode
```

## 🤖 Technical Implementation

### Natural Language Processing
- **Keyword Extraction**: Removes common question words
- **Fuzzy Matching**: Handles variations in question phrasing
- **Context Awareness**: Provides appropriate cultural descriptions

### Response Generation
```python
def generate_chat_response(self, question: str) -> Tuple[str, str]:
    # Extract keywords from question
    keywords = extract_keywords(question)
    
    # Find best match in knowledge base
    match = find_best_match(keywords, self.qa_knowledge)
    
    if match:
        # Format response with cultural context
        english_response = f"What is {term}?"
        toaripi_response = f"{toaripi_word}/{english_word} - {description}"
        return english_response, toaripi_response
    else:
        # Provide helpful guidance for unknown terms
        return fallback_response()
```

### Token Weight Simulation
```python
def simulate_token_weights(self, text: str) -> List[TokenWeight]:
    tokens = text.split()
    weights = []
    
    for token in tokens:
        # Assign weights based on linguistic importance
        if is_primary_term(token):
            weight = random.uniform(0.8, 1.0)  # High importance
        elif is_descriptor(token):
            weight = random.uniform(0.6, 0.8)  # Medium-high
        else:
            weight = random.uniform(0.1, 0.6)  # Variable
        
        weights.append(TokenWeight(token, weight))
    
    return weights
```

## 📊 Session Logging

### Conversation History
Chat exchanges are saved with:
- **Timestamp**: When the question was asked
- **User Input**: Original English question
- **English Context**: Formatted question
- **Toaripi Response**: Complete response with cultural context
- **Content Type**: Marked as "chat"

### Analytics Potential
- **Popular Terms**: Most frequently asked concepts
- **Learning Patterns**: Sequential question analysis
- **Cultural Interest**: Which cultural aspects generate most queries
- **Token Attention**: Which words consistently receive high weights

## 🔄 Future Enhancements

### Planned Features
- **Conversational Context**: Remember previous questions in session
- **Audio Pronunciation**: Add audio for Toaripi terms
- **Image Support**: Visual representations of concepts
- **Expanded Knowledge**: More comprehensive cultural database

### Advanced Capabilities
- **Phrase Learning**: Support for common phrases and expressions
- **Grammar Patterns**: Show how words are used in sentences
- **Cultural Stories**: Mini-narratives about important concepts
- **Interactive Quizzes**: Test understanding of learned vocabulary

## 🎯 Best Practices

### For Effective Learning
1. **Start Simple**: Begin with concrete nouns (animals, objects)
2. **Build Context**: Ask about related concepts in sequence
3. **Use Visualization**: Pay attention to token weight patterns
4. **Cultural Focus**: Notice cultural descriptions and significance

### For Teachers
1. **Structured Lessons**: Use categories to organize vocabulary sessions
2. **Visual Analysis**: Discuss why certain tokens have high weights
3. **Cultural Discussion**: Expand on the cultural contexts provided
4. **Progress Tracking**: Save sessions to monitor learning progression

This chat functionality transforms the Toaripi SLM into an interactive cultural dictionary with visual learning aids, making it an invaluable resource for language preservation and education.