/**
 * Upload Manager for CSV file uploads and validation
 */

class UploadManager {
    constructor() {
        this.uploadQueue = [];
        this.currentUploads = new Map();
        this.validationResults = new Map();
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.setupDropZone();
    }
    
    setupEventListeners() {
        // File input
        const fileInput = document.getElementById('file-input');
        const browseButton = document.getElementById('browse-button');
        
        if (fileInput && browseButton) {
            browseButton.addEventListener('click', () => fileInput.click());
            fileInput.addEventListener('change', (e) => this.handleFileSelect(e.target.files));
        }
    }
    
    setupDropZone() {
        const uploadArea = document.getElementById('upload-area');
        if (!uploadArea) return;
        
        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, this.preventDefaults, false);
            document.body.addEventListener(eventName, this.preventDefaults, false);
        });
        
        // Highlight drop area when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => {
                uploadArea.classList.add('dragover');
            }, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => {
                uploadArea.classList.remove('dragover');
            }, false);
        });
        
        // Handle dropped files
        uploadArea.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            this.handleFileSelect(files);
        }, false);
    }
    
    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    handleFileSelect(files) {
        const validFiles = Array.from(files).filter(file => this.validateFile(file));
        
        if (validFiles.length === 0) {
            window.app.showNotification('No valid CSV files selected', 'warning');
            return;
        }
        
        // Add files to upload queue
        validFiles.forEach(file => {
            const uploadItem = {
                id: this.generateUploadId(),
                file: file,
                status: 'pending',
                progress: 0,
                validationId: null
            };
            
            this.uploadQueue.push(uploadItem);
        });
        
        this.updateUploadDisplay();
        this.startUploads();
    }
    
    validateFile(file) {
        // Check file type
        if (!file.name.toLowerCase().endsWith('.csv')) {
            window.app.showNotification(`"${file.name}" is not a CSV file`, 'error');
            return false;
        }
        
        // Check file size (50MB limit)
        const maxSize = 50 * 1024 * 1024; // 50MB
        if (file.size > maxSize) {
            window.app.showNotification(`"${file.name}" is too large (max 50MB)`, 'error');
            return false;
        }
        
        // Check if file is already in queue
        const existingFile = this.uploadQueue.find(item => 
            item.file.name === file.name && item.file.size === file.size
        );
        
        if (existingFile) {
            window.app.showNotification(`"${file.name}" is already in upload queue`, 'warning');
            return false;
        }
        
        return true;
    }
    
    generateUploadId() {
        return 'upload_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    async startUploads() {
        // Process pending uploads
        const pendingUploads = this.uploadQueue.filter(item => item.status === 'pending');
        
        for (const uploadItem of pendingUploads) {
            try {
                await this.uploadFile(uploadItem);
            } catch (error) {
                console.error('Upload failed:', error);
                uploadItem.status = 'error';
                uploadItem.error = error.message;
                this.updateUploadDisplay();
            }
        }
    }
    
    async uploadFile(uploadItem) {
        uploadItem.status = 'uploading';
        this.updateUploadDisplay();
        
        const formData = new FormData();
        formData.append('file', uploadItem.file);
        
        try {
            // Upload file
            const response = await fetch('/api/upload/csv', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Upload failed' }));
                throw new Error(errorData.detail || 'Upload failed');
            }
            
            const result = await response.json();
            uploadItem.validationId = result.validation_id;
            uploadItem.status = 'validating';
            uploadItem.progress = 100;
            
            this.updateUploadDisplay();
            
            // Start monitoring validation
            this.monitorValidation(uploadItem);
            
            window.app.showNotification(`"${uploadItem.file.name}" uploaded successfully`, 'success');
            
        } catch (error) {
            uploadItem.status = 'error';
            uploadItem.error = error.message;
            this.updateUploadDisplay();
            
            window.app.showNotification(`Failed to upload "${uploadItem.file.name}": ${error.message}`, 'error');
            throw error;
        }
    }
    
    async monitorValidation(uploadItem) {
        if (!uploadItem.validationId) return;
        
        const pollValidation = async () => {
            try {
                const response = await fetch(`/api/upload/csv/${uploadItem.validationId}/status`);
                const status = await response.json();
                
                uploadItem.validationStatus = status.status;
                
                if (status.status === 'completed') {
                    uploadItem.status = 'completed';
                    this.validationResults.set(uploadItem.validationId, status);
                    this.updateUploadDisplay();
                    this.displayValidationResults(uploadItem.validationId);
                } else if (status.status === 'failed') {
                    uploadItem.status = 'validation_failed';
                    uploadItem.error = status.error || 'Validation failed';
                    this.updateUploadDisplay();
                } else {
                    // Still processing, check again
                    setTimeout(pollValidation, 2000);
                }
                
            } catch (error) {
                console.error('Validation monitoring error:', error);
                uploadItem.status = 'validation_error';
                uploadItem.error = 'Failed to check validation status';
                this.updateUploadDisplay();
            }
        };
        
        // Start polling
        setTimeout(pollValidation, 1000);
    }
    
    updateUploadDisplay() {
        const uploadQueue = document.getElementById('upload-queue');
        const uploadItems = document.getElementById('upload-items');
        
        if (!uploadQueue || !uploadItems) return;
        
        if (this.uploadQueue.length === 0) {
            uploadQueue.style.display = 'none';
            return;
        }
        
        uploadQueue.style.display = 'block';
        
        uploadItems.innerHTML = this.uploadQueue.map(item => `
            <div class="upload-item" data-id="${item.id}">
                <div class="upload-item-info">
                    <div class="upload-item-icon">📄</div>
                    <div class="upload-item-details">
                        <h4>${item.file.name}</h4>
                        <p>${window.app.formatFileSize(item.file.size)} • ${this.getStatusText(item)}</p>
                    </div>
                </div>
                <div class="upload-item-progress">
                    ${this.renderProgress(item)}
                </div>
                <div class="upload-item-actions">
                    ${this.renderActions(item)}
                </div>
            </div>
        `).join('');
        
        // Attach event listeners
        this.attachItemEventListeners();
    }
    
    getStatusText(item) {
        switch (item.status) {
            case 'pending': return 'Waiting to upload';
            case 'uploading': return 'Uploading...';
            case 'validating': return 'Validating...';
            case 'completed': return 'Ready for training';
            case 'error': return `Error: ${item.error || 'Unknown error'}`;
            case 'validation_failed': return `Validation failed: ${item.error || 'Unknown error'}`;
            case 'validation_error': return 'Validation error';
            default: return 'Unknown status';
        }
    }
    
    renderProgress(item) {
        if (item.status === 'pending') {
            return '<div class="progress-bar"><div class="progress-fill" style="width: 0%"></div></div>';
        } else if (item.status === 'uploading') {
            return `
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${item.progress}%"></div>
                </div>
                <div class="progress-text">${item.progress}%</div>
            `;
        } else if (item.status === 'validating') {
            return `
                <div class="progress-bar">
                    <div class="progress-fill" style="width: 100%; background: var(--warning-color);"></div>
                </div>
                <div class="progress-text">Validating...</div>
            `;
        } else if (item.status === 'completed') {
            return `
                <div class="progress-bar">
                    <div class="progress-fill" style="width: 100%; background: var(--success-color);"></div>
                </div>
                <div class="progress-text">✓ Complete</div>
            `;
        } else {
            return `
                <div class="progress-bar">
                    <div class="progress-fill" style="width: 100%; background: var(--error-color);"></div>
                </div>
                <div class="progress-text">✗ Failed</div>
            `;
        }
    }
    
    renderActions(item) {
        switch (item.status) {
            case 'pending':
                return `<button class="btn btn-secondary" onclick="uploadManager.removeItem('${item.id}')">Remove</button>`;
            case 'uploading':
            case 'validating':
                return `<button class="btn btn-secondary" onclick="uploadManager.cancelItem('${item.id}')">Cancel</button>`;
            case 'completed':
                return `
                    <button class="btn btn-secondary" onclick="uploadManager.viewValidation('${item.validationId}')">View Results</button>
                    <button class="btn btn-secondary" onclick="uploadManager.removeItem('${item.id}')">Remove</button>
                `;
            case 'error':
            case 'validation_failed':
            case 'validation_error':
                return `
                    <button class="btn btn-secondary" onclick="uploadManager.retryItem('${item.id}')">Retry</button>
                    <button class="btn btn-secondary" onclick="uploadManager.removeItem('${item.id}')">Remove</button>
                `;
            default:
                return '';
        }
    }
    
    attachItemEventListeners() {
        // Event listeners are attached via onclick attributes in the HTML
        // This is simpler than managing event delegation for dynamic content
    }
    
    removeItem(itemId) {
        const index = this.uploadQueue.findIndex(item => item.id === itemId);
        if (index !== -1) {
            const item = this.uploadQueue[index];
            
            // Cancel any ongoing uploads/validations
            if (item.status === 'uploading' || item.status === 'validating') {
                this.cancelItem(itemId);
            }
            
            this.uploadQueue.splice(index, 1);
            this.updateUploadDisplay();
            
            window.app.showNotification(`"${item.file.name}" removed from queue`, 'info');
        }
    }
    
    cancelItem(itemId) {
        const item = this.uploadQueue.find(item => item.id === itemId);
        if (item) {
            item.status = 'cancelled';
            this.updateUploadDisplay();
            
            window.app.showNotification(`"${item.file.name}" upload cancelled`, 'warning');
        }
    }
    
    retryItem(itemId) {
        const item = this.uploadQueue.find(item => item.id === itemId);
        if (item) {
            item.status = 'pending';
            item.progress = 0;
            item.error = null;
            item.validationId = null;
            
            this.updateUploadDisplay();
            this.startUploads();
        }
    }
    
    viewValidation(validationId) {
        this.displayValidationResults(validationId);
    }
    
    async displayValidationResults(validationId) {
        const validationResults = document.getElementById('validation-results');
        const validationSummary = document.getElementById('validation-summary');
        const validationDetails = document.getElementById('validation-details');
        
        if (!validationResults) return;
        
        try {
            // Get validation results if not cached
            if (!this.validationResults.has(validationId)) {
                const response = await fetch(`/api/upload/csv/${validationId}/status`);
                const status = await response.json();
                this.validationResults.set(validationId, status);
            }
            
            const results = this.validationResults.get(validationId);
            
            if (!results || !results.validation_results) {
                validationResults.style.display = 'none';
                return;
            }
            
            const validation = results.validation_results;
            
            // Display summary metrics
            validationSummary.innerHTML = `
                <div class="validation-metric">
                    <div class="validation-metric-value">${validation.total_rows || 0}</div>
                    <div class="validation-metric-label">Total Rows</div>
                </div>
                <div class="validation-metric">
                    <div class="validation-metric-value">${validation.valid_rows || 0}</div>
                    <div class="validation-metric-label">Valid Rows</div>
                </div>
                <div class="validation-metric">
                    <div class="validation-metric-value">${validation.invalid_rows || 0}</div>
                    <div class="validation-metric-label">Invalid Rows</div>
                </div>
                <div class="validation-metric">
                    <div class="validation-metric-value">${validation.safety_score ? validation.safety_score.toFixed(2) : 'N/A'}</div>
                    <div class="validation-metric-label">Safety Score</div>
                </div>
            `;
            
            // Display detailed issues
            const issues = validation.issues || [];
            if (issues.length > 0) {
                validationDetails.innerHTML = `
                    <h4>Validation Issues</h4>
                    ${issues.map(issue => `
                        <div class="validation-issue ${issue.severity || 'info'}">
                            <strong>${issue.type || 'Issue'}:</strong> ${issue.message || 'Unknown issue'}
                            ${issue.row ? `(Row ${issue.row})` : ''}
                        </div>
                    `).join('')}
                `;
            } else {
                validationDetails.innerHTML = `
                    <div class="validation-issue info">
                        <strong>✓ All checks passed:</strong> No validation issues found.
                    </div>
                `;
            }
            
            validationResults.style.display = 'block';
            validationResults.scrollIntoView({ behavior: 'smooth' });
            
        } catch (error) {
            console.error('Failed to display validation results:', error);
            window.app.showNotification('Failed to load validation results', 'error');
        }
    }
    
    refreshStatus() {
        // Refresh validation status for completed uploads
        this.uploadQueue
            .filter(item => item.status === 'completed' && item.validationId)
            .forEach(item => this.displayValidationResults(item.validationId));
    }
    
    // Public methods for external access
    getUploadQueue() {
        return [...this.uploadQueue];
    }
    
    getValidationResults(validationId) {
        return this.validationResults.get(validationId);
    }
    
    clearCompleted() {
        const completedItems = this.uploadQueue.filter(item => item.status === 'completed');
        completedItems.forEach(item => this.removeItem(item.id));
        
        if (completedItems.length > 0) {
            window.app.showNotification(`Cleared ${completedItems.length} completed uploads`, 'info');
        }
    }
}

// Export for global access
window.UploadManager = UploadManager;