/**
 * Training Manager for model training sessions
 */

class TrainingManager {
    constructor() {
        this.currentSession = null;
        this.eventSource = null;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.loadTrainingHistory();
    }
    
    setupEventListeners() {
        const startButton = document.getElementById('start-training');
        const stopButton = document.getElementById('stop-training');
        
        if (startButton) {
            startButton.addEventListener('click', () => this.startTraining());
        }
        
        if (stopButton) {
            stopButton.addEventListener('click', () => this.stopTraining());
        }
        
        // Update start button based on data availability
        this.updateStartButtonState();
    }
    
    async updateStartButtonState() {
        const startButton = document.getElementById('start-training');
        if (!startButton) return;
        
        try {
            // Check if we have valid training data
            const response = await fetch('/api/upload/status');
            const uploads = await response.json();
            
            const hasValidData = uploads.some(upload => upload.status === 'completed');
            const baseModelSelected = document.getElementById('base-model-select')?.value;
            
            startButton.disabled = !hasValidData || !baseModelSelected;
            
            if (!hasValidData) {
                startButton.title = 'Please upload and validate training data first';
            } else if (!baseModelSelected) {
                startButton.title = 'Please select a base model';
            } else {
                startButton.title = 'Start training session';
            }
            
        } catch (error) {
            console.error('Failed to check training readiness:', error);
        }
    }
    
    async startTraining() {
        try {
            const config = this.getTrainingConfig();
            
            window.app.showLoading('Starting training session...');
            
            const response = await fetch('/api/training/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to start training');
            }
            
            const session = await response.json();
            this.currentSession = session;
            
            window.app.hideLoading();
            this.showTrainingProgress();
            this.startProgressMonitoring(session.session_id);
            
            window.app.showNotification('Training session started successfully', 'success');
            
        } catch (error) {
            window.app.hideLoading();
            window.app.showNotification(`Failed to start training: ${error.message}`, 'error');
        }
    }
    
    getTrainingConfig() {
        return {
            base_model: document.getElementById('base-model-select')?.value || 'mistralai/Mistral-7B-Instruct-v0.2',
            learning_rate: parseFloat(document.getElementById('learning-rate')?.value || '2e-5'),
            batch_size: parseInt(document.getElementById('batch-size')?.value || '4'),
            epochs: parseInt(document.getElementById('epochs')?.value || '3'),
            use_lora: document.getElementById('use-lora')?.checked || true,
            model_name: `toaripi-custom-${Date.now()}`
        };
    }
    
    showTrainingProgress() {
        const progressContainer = document.getElementById('training-progress');
        const startButton = document.getElementById('start-training');
        const stopButton = document.getElementById('stop-training');
        
        if (progressContainer) progressContainer.style.display = 'block';
        if (startButton) startButton.style.display = 'none';
        if (stopButton) stopButton.style.display = 'inline-flex';
    }
    
    hideTrainingProgress() {
        const progressContainer = document.getElementById('training-progress');
        const startButton = document.getElementById('start-training');
        const stopButton = document.getElementById('stop-training');
        
        if (progressContainer) progressContainer.style.display = 'none';
        if (startButton) startButton.style.display = 'inline-flex';
        if (stopButton) stopButton.style.display = 'none';
    }
    
    startProgressMonitoring(sessionId) {
        if (this.eventSource) {
            this.eventSource.close();
        }
        
        this.eventSource = new EventSource(`/api/training/${sessionId}/progress`);
        
        this.eventSource.onmessage = (event) => {
            try {
                const progress = JSON.parse(event.data);
                this.updateProgress(progress);
                
                if (progress.status === 'completed' || progress.status === 'failed') {
                    this.eventSource.close();
                    this.eventSource = null;
                    this.onTrainingComplete(progress);
                }
                
            } catch (error) {
                console.error('Error parsing progress data:', error);
            }
        };
        
        this.eventSource.onerror = (error) => {
            console.error('Progress monitoring error:', error);
            this.eventSource.close();
            this.eventSource = null;
            
            window.app.showNotification('Lost connection to training progress', 'warning');
        };
    }
    
    updateProgress(progress) {
        // Update progress bar
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');
        
        if (progressFill) {
            progressFill.style.width = `${progress.progress || 0}%`;
        }
        
        if (progressText) {
            progressText.textContent = `${Math.round(progress.progress || 0)}%`;
        }
        
        // Update training stats
        const currentEpoch = document.getElementById('current-epoch');
        const currentLoss = document.getElementById('current-loss');
        const eta = document.getElementById('eta');
        const trainingSpeed = document.getElementById('training-speed');
        
        if (currentEpoch && progress.current_epoch !== undefined) {
            currentEpoch.textContent = `${progress.current_epoch}/${progress.total_epochs}`;
        }
        
        if (currentLoss && progress.loss !== undefined) {
            currentLoss.textContent = progress.loss.toFixed(4);
        }
        
        if (eta && progress.eta_minutes !== undefined) {
            eta.textContent = window.app.formatDuration(progress.eta_minutes * 60);
        }
        
        if (trainingSpeed && progress.steps_per_second !== undefined) {
            trainingSpeed.textContent = `${progress.steps_per_second.toFixed(1)} steps/s`;
        }
        
        // Add log entry
        this.addLogEntry(progress);
    }
    
    addLogEntry(progress) {
        const logContent = document.getElementById('log-content');
        if (!logContent) return;
        
        const timestamp = new Date().toLocaleTimeString();
        const message = this.formatLogMessage(progress);
        
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry ${progress.log_level || 'info'}`;
        logEntry.textContent = `[${timestamp}] ${message}`;
        
        logContent.appendChild(logEntry);
        logContent.scrollTop = logContent.scrollHeight;
        
        // Keep only last 100 log entries
        while (logContent.children.length > 100) {
            logContent.removeChild(logContent.firstChild);
        }
    }
    
    formatLogMessage(progress) {
        if (progress.message) {
            return progress.message;
        }
        
        if (progress.status === 'training') {
            return `Epoch ${progress.current_epoch}/${progress.total_epochs}, Loss: ${progress.loss?.toFixed(4) || 'N/A'}`;
        }
        
        return `Status: ${progress.status}`;
    }
    
    async stopTraining() {
        if (!this.currentSession) return;
        
        try {
            const response = await fetch(`/api/training/${this.currentSession.session_id}/stop`, {
                method: 'POST'
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to stop training');
            }
            
            window.app.showNotification('Training session stopped', 'warning');
            this.onTrainingComplete({ status: 'stopped' });
            
        } catch (error) {
            window.app.showNotification(`Failed to stop training: ${error.message}`, 'error');
        }
    }
    
    onTrainingComplete(finalProgress) {
        this.hideTrainingProgress();
        this.currentSession = null;
        
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
        
        if (finalProgress.status === 'completed') {
            window.app.showNotification('Training completed successfully!', 'success');
            
            // Refresh model list
            if (window.modelManager) {
                window.modelManager.refreshModels();
            }
        } else if (finalProgress.status === 'failed') {
            window.app.showNotification(`Training failed: ${finalProgress.error || 'Unknown error'}`, 'error');
        }
        
        // Re-enable start button
        this.updateStartButtonState();
    }
    
    async loadTrainingHistory() {
        try {
            const response = await fetch('/api/training/history');
            const history = await response.json();
            
            // Display recent training sessions (implement if needed)
            
        } catch (error) {
            console.error('Failed to load training history:', error);
        }
    }
    
    refreshStatus() {
        this.updateStartButtonState();
    }
}

window.TrainingManager = TrainingManager;