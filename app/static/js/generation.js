/**
 * Generation Manager for educational content generation
 */

class GenerationManager {
    constructor() {
        this.currentContentType = 'story';
        this.generationHistory = [];
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.loadCapabilities();
    }
    
    setupEventListeners() {
        // Content type selection
        document.querySelectorAll('.content-type-card').forEach(card => {
            card.addEventListener('click', (e) => {
                const contentType = e.currentTarget.dataset.type;
                this.selectContentType(contentType);
            });
        });
        
        // Generation button
        const generateButton = document.getElementById('generate-content');
        if (generateButton) {
            generateButton.addEventListener('click', () => this.generateContent());
        }
        
        // Result actions
        const downloadButton = document.getElementById('download-content');
        const copyButton = document.getElementById('copy-content');
        const regenerateButton = document.getElementById('regenerate-content');
        
        if (downloadButton) {
            downloadButton.addEventListener('click', () => this.downloadContent());
        }
        
        if (copyButton) {
            copyButton.addEventListener('click', () => this.copyContent());
        }
        
        if (regenerateButton) {
            regenerateButton.addEventListener('click', () => this.regenerateContent());
        }
        
        // Model management buttons
        const loadModelButton = document.getElementById('load-model-btn');
        const modelManagerButton = document.getElementById('model-manager-btn');
        
        if (loadModelButton) {
            loadModelButton.addEventListener('click', () => this.showModelSelector());
        }
        
        if (modelManagerButton) {
            modelManagerButton.addEventListener('click', () => {
                window.app.showModal('model-manager-modal');
                if (window.modelManager) {
                    window.modelManager.refreshModels();
                }
            });
        }
    }
    
    selectContentType(contentType) {
        this.currentContentType = contentType;
        
        // Update active card
        document.querySelectorAll('.content-type-card').forEach(card => {
            card.classList.remove('active');
        });
        document.querySelector(`[data-type="${contentType}"]`).classList.add('active');
        
        // Show corresponding options
        document.querySelectorAll('.generation-options').forEach(options => {
            options.classList.remove('active');
        });
        document.getElementById(`${contentType}-options`).classList.add('active');
    }
    
    async generateContent() {
        try {
            const request = this.buildGenerationRequest();
            
            window.app.showLoading('Generating content...');
            
            const endpoint = this.getGenerationEndpoint();
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(request)
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Generation failed');
            }
            
            const result = await response.json();
            
            window.app.hideLoading();
            this.displayResults(result);
            
            // Add to history
            this.generationHistory.unshift({
                type: this.currentContentType,
                request: request,
                result: result,
                timestamp: new Date()
            });
            
            window.app.showNotification('Content generated successfully!', 'success');
            
        } catch (error) {
            window.app.hideLoading();
            window.app.showNotification(`Generation failed: ${error.message}`, 'error');
        }
    }
    
    buildGenerationRequest() {
        switch (this.currentContentType) {
            case 'story':
                return {
                    prompt: document.getElementById('story-prompt')?.value || '',
                    age_group: document.getElementById('story-age-group')?.value || 'primary',
                    length: document.getElementById('story-length')?.value || 'medium',
                    cultural_elements: this.getSelectedCulturalElements(),
                    learning_objectives: ['vocabulary_building', 'cultural_awareness']
                };
                
            case 'vocabulary':
                return {
                    topic: document.getElementById('vocab-topic')?.value || '',
                    age_group: document.getElementById('vocab-age-group')?.value || 'primary',
                    word_count: parseInt(document.getElementById('vocab-count')?.value || '10'),
                    include_examples: document.getElementById('include-examples')?.checked || true,
                    difficulty_level: 'beginner'
                };
                
            case 'dialogue':
                return {
                    scenario: document.getElementById('dialogue-scenario')?.value || '',
                    age_group: document.getElementById('dialogue-age-group')?.value || 'primary',
                    participant_count: parseInt(document.getElementById('dialogue-participants')?.value || '2'),
                    turn_count: 6,
                    vocabulary_focus: []
                };
                
            case 'qa':
                return {
                    text: document.getElementById('qa-text')?.value || '',
                    age_group: document.getElementById('qa-age-group')?.value || 'primary',
                    question_count: parseInt(document.getElementById('qa-count')?.value || '5'),
                    question_types: ['comprehension', 'analysis'],
                    include_answers: true
                };
                
            default:
                throw new Error('Unknown content type');
        }
    }
    
    getSelectedCulturalElements() {
        const checkboxes = document.querySelectorAll('.checkbox-group input[type="checkbox"]:checked');
        return Array.from(checkboxes).map(cb => cb.value);
    }
    
    getGenerationEndpoint() {
        return `/api/generate/${this.currentContentType}`;
    }
    
    displayResults(result) {
        const resultsContainer = document.getElementById('generation-results');
        const resultsContent = document.getElementById('results-content');
        
        if (!resultsContainer || !resultsContent) return;
        
        resultsContent.innerHTML = this.formatResults(result);
        resultsContainer.style.display = 'block';
        resultsContainer.scrollIntoView({ behavior: 'smooth' });
        
        this.currentResult = result;
    }
    
    formatResults(result) {
        switch (this.currentContentType) {
            case 'story':
                return this.formatStoryResults(result);
            case 'vocabulary':
                return this.formatVocabularyResults(result);
            case 'dialogue':
                return this.formatDialogueResults(result);
            case 'qa':
                return this.formatQAResults(result);
            default:
                return '<p>Unknown content type</p>';
        }
    }
    
    formatStoryResults(story) {
        return `
            <div class="generated-story">
                <div class="story-text">
                    <div class="toaripi-text">${story.toaripi_text}</div>
                    <div class="english-text">${story.english_translation}</div>
                </div>
                
                ${story.vocabulary_words && story.vocabulary_words.length > 0 ? `
                    <div class="story-vocabulary">
                        <h4>Key Vocabulary</h4>
                        <div class="vocabulary-list">
                            ${story.vocabulary_words.map(word => `
                                <div class="vocabulary-word">
                                    <div class="word-toaripi">${word}</div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}
                
                ${story.comprehension_questions && story.comprehension_questions.length > 0 ? `
                    <div class="story-questions">
                        <h4>Comprehension Questions</h4>
                        ${story.comprehension_questions.map((question, i) => `
                            <div class="qa-question">
                                <div class="question-text">${i + 1}. ${question}</div>
                            </div>
                        `).join('')}
                    </div>
                ` : ''}
                
                <div class="generation-meta">
                    <p><strong>Reading Time:</strong> ${story.estimated_reading_time_minutes || 'N/A'} minutes</p>
                    <p><strong>Word Count:</strong> ${story.word_count || 'N/A'} words</p>
                    <p><strong>Safety Score:</strong> ${story.safety_score ? story.safety_score.toFixed(2) : 'N/A'}</p>
                </div>
            </div>
        `;
    }
    
    formatVocabularyResults(vocab) {
        return `
            <div class="generated-vocabulary">
                <div class="vocabulary-list">
                    ${vocab.words.map(word => `
                        <div class="vocabulary-word">
                            <div class="word-toaripi">${word.toaripi}</div>
                            <div class="word-english">${word.english}</div>
                            ${word.definition ? `<div class="word-definition">${word.definition}</div>` : ''}
                        </div>
                    `).join('')}
                </div>
                
                ${vocab.example_sentences && vocab.example_sentences.length > 0 ? `
                    <div class="vocabulary-examples">
                        <h4>Example Sentences</h4>
                        ${vocab.example_sentences.map(sentence => `
                            <div class="example-sentence">
                                <div class="toaripi-text">${sentence}</div>
                            </div>
                        `).join('')}
                    </div>
                ` : ''}
                
                <div class="generation-meta">
                    <p><strong>Topic:</strong> ${vocab.topic}</p>
                    <p><strong>Difficulty:</strong> ${vocab.difficulty_level || 'N/A'}</p>
                    <p><strong>Safety Score:</strong> ${vocab.safety_score ? vocab.safety_score.toFixed(2) : 'N/A'}</p>
                </div>
            </div>
        `;
    }
    
    formatDialogueResults(dialogue) {
        return `
            <div class="generated-dialogue">
                <div class="dialogue-turns">
                    ${dialogue.dialogue.map(turn => `
                        <div class="dialogue-turn">
                            <div class="speaker-name">${turn.speaker}</div>
                            <div class="toaripi-text">${turn.toaripi_text}</div>
                            <div class="english-text">${turn.english_translation}</div>
                        </div>
                    `).join('')}
                </div>
                
                ${dialogue.vocabulary_focus && dialogue.vocabulary_focus.length > 0 ? `
                    <div class="dialogue-vocabulary">
                        <h4>Vocabulary Focus</h4>
                        <div class="vocabulary-tags">
                            ${dialogue.vocabulary_focus.map(word => `
                                <span class="vocabulary-tag">${word}</span>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}
                
                <div class="generation-meta">
                    <p><strong>Participants:</strong> ${dialogue.participants.join(', ')}</p>
                    <p><strong>Estimated Duration:</strong> ${dialogue.estimated_duration_minutes || 'N/A'} minutes</p>
                    <p><strong>Safety Score:</strong> ${dialogue.safety_score ? dialogue.safety_score.toFixed(2) : 'N/A'}</p>
                </div>
            </div>
        `;
    }
    
    formatQAResults(qa) {
        return `
            <div class="generated-qa">
                <div class="qa-source">
                    <h4>Source Text</h4>
                    <div class="source-text">${qa.source_text}</div>
                </div>
                
                <div class="qa-questions">
                    <h4>Questions</h4>
                    ${qa.questions.map((q, i) => `
                        <div class="qa-question">
                            <div class="question-text">${i + 1}. ${q.question}</div>
                            ${q.answer ? `<div class="answer-text">Answer: ${q.answer}</div>` : ''}
                        </div>
                    `).join('')}
                </div>
                
                <div class="generation-meta">
                    <p><strong>Question Count:</strong> ${qa.question_count || 'N/A'}</p>
                    <p><strong>Completion Time:</strong> ${qa.estimated_completion_time_minutes || 'N/A'} minutes</p>
                    <p><strong>Safety Score:</strong> ${qa.safety_score ? qa.safety_score.toFixed(2) : 'N/A'}</p>
                </div>
            </div>
        `;
    }
    
    downloadContent() {
        if (!this.currentResult) return;
        
        const content = this.formatForDownload(this.currentResult);
        const filename = `toaripi-${this.currentContentType}-${Date.now()}.txt`;
        
        const blob = new Blob([content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        
        URL.revokeObjectURL(url);
        
        window.app.showNotification('Content downloaded successfully', 'success');
    }
    
    formatForDownload(result) {
        const timestamp = new Date().toLocaleString();
        let content = `Toaripi Educational Content\nGenerated: ${timestamp}\nType: ${this.currentContentType}\n\n`;
        
        switch (this.currentContentType) {
            case 'story':
                content += `STORY\n`;
                content += `Toaripi: ${result.toaripi_text}\n`;
                content += `English: ${result.english_translation}\n\n`;
                if (result.vocabulary_words) {
                    content += `VOCABULARY: ${result.vocabulary_words.join(', ')}\n\n`;
                }
                if (result.comprehension_questions) {
                    content += `QUESTIONS:\n${result.comprehension_questions.map((q, i) => `${i + 1}. ${q}`).join('\n')}\n`;
                }
                break;
                
            case 'vocabulary':
                content += `VOCABULARY - ${result.topic}\n\n`;
                result.words.forEach(word => {
                    content += `${word.toaripi} - ${word.english}\n`;
                    if (word.definition) content += `  ${word.definition}\n`;
                });
                break;
                
            case 'dialogue':
                content += `DIALOGUE\n\n`;
                result.dialogue.forEach(turn => {
                    content += `${turn.speaker}:\n`;
                    content += `  Toaripi: ${turn.toaripi_text}\n`;
                    content += `  English: ${turn.english_translation}\n\n`;
                });
                break;
                
            case 'qa':
                content += `READING COMPREHENSION\n\n`;
                content += `Source Text:\n${result.source_text}\n\n`;
                content += `Questions:\n`;
                result.questions.forEach((q, i) => {
                    content += `${i + 1}. ${q.question}\n`;
                    if (q.answer) content += `   Answer: ${q.answer}\n`;
                });
                break;
        }
        
        return content;
    }
    
    copyContent() {
        if (!this.currentResult) return;
        
        const content = this.formatForDownload(this.currentResult);
        
        navigator.clipboard.writeText(content).then(() => {
            window.app.showNotification('Content copied to clipboard', 'success');
        }).catch(() => {
            window.app.showNotification('Failed to copy content', 'error');
        });
    }
    
    regenerateContent() {
        this.generateContent();
    }
    
    async showModelSelector() {
        try {
            const response = await fetch('/api/models/local');
            const models = await response.json();
            
            if (models.length === 0) {
                window.app.showNotification('No models available. Please train or download a model first.', 'warning');
                return;
            }
            
            // Simple model selection (could be enhanced with a proper modal)
            const modelNames = models.map(m => m.name);
            const selectedModel = prompt(`Select a model:\n${modelNames.map((name, i) => `${i + 1}. ${name}`).join('\n')}`);
            
            if (selectedModel) {
                const modelIndex = parseInt(selectedModel) - 1;
                if (modelIndex >= 0 && modelIndex < models.length) {
                    await this.activateModel(models[modelIndex].name);
                }
            }
            
        } catch (error) {
            window.app.showNotification(`Failed to load models: ${error.message}`, 'error');
        }
    }
    
    async activateModel(modelName) {
        try {
            window.app.showLoading('Activating model...');
            
            const response = await fetch(`/api/models/${modelName}/activate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({})
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to activate model');
            }
            
            window.app.hideLoading();
            window.app.activeModel = modelName;
            window.app.updateModelDisplay();
            
            window.app.showNotification(`Model "${modelName}" activated successfully`, 'success');
            
        } catch (error) {
            window.app.hideLoading();
            window.app.showNotification(`Failed to activate model: ${error.message}`, 'error');
        }
    }
    
    async loadCapabilities() {
        try {
            const response = await fetch('/api/generate/capabilities');
            const capabilities = await response.json();
            
            // Update UI based on capabilities (if needed)
            
        } catch (error) {
            console.error('Failed to load generation capabilities:', error);
        }
    }
    
    refreshStatus() {
        this.loadCapabilities();
    }
}

window.GenerationManager = GenerationManager;