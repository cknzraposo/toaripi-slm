# Enhanced Interactive CLI - Side-by-Side Display with Token Weights

## Overview

The Toaripi SLM Interactive CLI has been enhanced with advanced visualization features that display English text and Toaripi translations side by side, with color-coded token weights showing model attention patterns.

## 🌟 New Features

### 1. Side-by-Side Bilingual Display
- **English Source**: Left panel showing the original English text
- **Toaripi Translation**: Right panel showing the generated Toaripi content
- **Synchronized Layout**: Both panels maintain visual alignment for easy comparison

### 2. Token Weight Visualization
- **Color-Coded Weights**: Each token is colored based on its attention weight
- **Weight Scale**: 
  - 🔴 **High (0.8+)**: Bold red - Critical tokens
  - 🟠 **Medium-High (0.6+)**: Bold orange - Important tokens
  - 🟡 **Medium (0.4+)**: Bold yellow - Relevant tokens
  - 🟢 **Low (0.2+)**: Green - Supporting tokens
  - 🔵 **Very Low (<0.2)**: Dim cyan - Minimal importance
- **Numerical Display**: Weight values shown as (0.xx) next to each token

### 3. Token Alignment System
- **Cross-Language Alignment**: Visual correspondence between English and Toaripi tokens
- **Padding Strategy**: Shorter sequences are padded to match longer ones
- **Toggle Support**: Can be enabled/disabled via `/align` command

### 4. Interactive Visualization Controls
- **Real-time Toggles**: Adjust display settings during conversation
- **Command Integration**: Seamless switching between visualization modes
- **Session Persistence**: Settings are saved with conversation history

## 🎮 Interactive Commands

### New Visualization Commands
- `/weights` - Toggle token weight display on/off
- `/align` - Toggle token alignment between languages
- `/legend` - Show color legend for token weights

### Enhanced Existing Commands
- `/type <content_type>` - Switch between story, vocabulary, dialogue, questions, translation
- `/settings` - Adjust generation parameters including visualization settings
- `/history` - View conversation history with bilingual display
- `/save` - Save session including display settings
- `/help` - Updated help with visualization features

## 📚 Content Type Support

### Story Generation
```
English: The children learned traditional fishing methods from their elders
Toaripi: Bada-bada na'a nene-ida hanere taumate mina gabua-harigi
```

### Vocabulary Training
```
English: Fish children water river traditional knowledge
Toaripi: Hanere bada-bada peni malolo taumate nene-ida
```

### Dialogue Practice
```
English: Teacher: What did you learn? Student: I learned about fishing.
Toaripi: Amo-harigi: Ami na'a nene? Amo-nene: Mina na'a nene hanere.
```

### Translation Practice
```
English: The fish are swimming in the clear water
Toaripi: Hanere potopoto malolo peni kura
```

### Comprehension Questions
```
English: Where did the children go fishing? What did they catch?
Toaripi: Sena na'a gola hanere bada-bada? Ami na'a gete hanere?
```

## 🚀 Usage Examples

### Basic Interactive Session
```bash
$ toaripi interact
💬 Toaripi SLM Interactive Mode

🌟 Enhanced Interactive Session Started
Current model: ./models/hf
Content type: story

You: Create a story about children learning to fish
💬 Generated Content (story):

[Side-by-side display with colored token weights]

You: /weights
💡 Token weights display: OFF

You: /type vocabulary
✅ Content type changed to: vocabulary

You: fishing vocabulary for children
[Vocabulary display with translations]
```

### Advanced Settings
```bash
$ toaripi interact --model ./models/custom --content-type dialogue --temperature 0.8
```

## 🎓 Educational Benefits

### For Teachers
- **Visual Feedback**: Token weights show model confidence and attention patterns
- **Translation Teaching**: Side-by-side display aids in teaching translation techniques
- **Content Variety**: Multiple content types support diverse lesson plans
- **Quality Assessment**: Color coding helps evaluate generation quality

### For Language Learners
- **Visual Connection**: Clear correspondence between English and Toaripi
- **Key Vocabulary**: High-weight tokens highlight important words
- **Pattern Recognition**: Repeated exposure to aligned structures
- **Interactive Exploration**: Controls allow student-driven discovery

### For Linguists & Researchers
- **Attention Analysis**: Token weights reveal model behavior patterns
- **Cross-linguistic Study**: Alignment shows correspondences between languages
- **Session Logging**: Complete interaction history for research analysis
- **Customizable Display**: Flexible settings for research needs

## 🔧 Technical Implementation

### BilingualDisplay Class
```python
class BilingualDisplay:
    def __init__(self, console: Console):
        self.console = console
        self.show_weights = True
        self.align_tokens = True
    
    def display_bilingual_content(self, english_text: str, toaripi_text: str, content_type: str):
        # Create side-by-side panels with token weights
        # Apply color coding based on attention weights
        # Show legend and alignment indicators
```

### TokenWeight Class
```python
class TokenWeight:
    def __init__(self, token: str, weight: float):
        self.token = token
        self.weight = weight
    
    def get_color_style(self) -> str:
        # Return Rich color style based on weight
```

### Enhanced Generator
```python
class ToaripiGenerator:
    def generate_bilingual_content(self, prompt: str, content_type: str) -> Tuple[str, str]:
        # Generate both English context and Toaripi translation
        # Extract attention weights from model
        # Return aligned content pairs
```

## 📊 Display Examples

### With Token Weights (Default)
```
╭─────────────── 🇺🇸 English Source ───────────────╮    ╭────────────── 🌺 Toaripi Translation ──────────────╮
│ The(0.42) children(0.87) went(0.66) fishing(0.98) │    │ Bada-bada(0.91) na'a(0.77) gola(0.47) hanere(0.49) │
╰───────────────────────────────────────────────────╯    ╰─────────────────────────────────────────────────────╯
```

### Without Token Weights
```
╭─────── 🇺🇸 English Source ────────╮    ╭───── 🌺 Toaripi Translation ─────╮
│ The children went fishing          │    │ Bada-bada na'a gola hanere       │
╰────────────────────────────────────╯    ╰───────────────────────────────────╯
```

### Color Legend
```
╭──────────────────── 🎨 Weight Color Guide ─────────────────────╮
│ Token Weight Legend: High (0.8+) | Medium-High (0.6+) | ...   │
╰─────────────────────────────────────────────────────────────────╯
```

## 🔍 Testing & Validation

### Test Script
```bash
python3 test_interactive_cli.py
```

### Demo Script
```bash
python3 demo_enhanced_cli.py
```

### Manual Testing
1. Run `toaripi interact`
2. Try different content types
3. Toggle visualization features
4. Verify color coding and alignment
5. Test session saving/loading

## 🛠️ Configuration

### Default Settings
```python
DEFAULT_SETTINGS = {
    "show_weights": True,
    "align_tokens": True,
    "content_type": "story",
    "temperature": 0.7,
    "max_length": 200
}
```

### Session Settings
Settings are automatically saved with conversation history and can be restored in subsequent sessions.

## 🎯 Future Enhancements

### Planned Features
- **Real Model Integration**: Connect to actual fine-tuned Toaripi models
- **Advanced Alignment**: Implement proper cross-linguistic alignment algorithms
- **Export Options**: Save visualizations as images or HTML
- **Batch Processing**: Process multiple texts with visualization
- **Custom Color Schemes**: User-defined color palettes for token weights

### Integration Points
- **Web Interface**: Extend features to web-based interface
- **API Endpoints**: Provide REST API for bilingual display
- **Educational Platform**: Integration with learning management systems
- **Research Tools**: Export data for linguistic analysis

This enhanced interactive CLI transforms the Toaripi SLM into a powerful educational tool with visual feedback that helps users understand both the content and the model's behavior patterns.