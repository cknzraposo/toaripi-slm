/**
 * Model Manager for model operations
 */

class ModelManager {
    constructor() {
        this.models = [];
        this.downloadTasks = new Map();
        
        this.init();
    }
    
    init() {
        this.refreshModels();
    }
    
    async refreshModels() {
        try {
            const response = await fetch('/api/models');
            this.models = await response.json();
            
            this.updateModelDisplay();
            
        } catch (error) {
            console.error('Failed to refresh models:', error);
            window.app.showNotification('Failed to load models', 'error');
        }
    }
    
    updateModelDisplay() {
        const modelList = document.getElementById('model-list');
        if (!modelList) return;
        
        if (this.models.length === 0) {
            modelList.innerHTML = `
                <div class="model-item">
                    <div class="model-item-info">
                        <div class="model-item-name">No models available</div>
                        <div class="model-item-details">Download a model to get started</div>
                    </div>
                </div>
            `;
            return;
        }
        
        modelList.innerHTML = this.models.map(model => `
            <div class="model-item ${model.is_active ? 'active' : ''}" data-model="${model.name}">
                <div class="model-item-info">
                    <div class="model-item-name">${model.name}</div>
                    <div class="model-item-details">
                        ${model.description} • ${window.app.formatFileSize(model.size_mb * 1024 * 1024)} • ${model.format}
                        ${model.is_local ? '' : ' (Available for download)'}
                    </div>
                </div>
                <div class="model-item-status">
                    <span class="status-badge ${this.getStatusClass(model)}">
                        ${this.getStatusText(model)}
                    </span>
                </div>
                <div class="model-item-actions">
                    ${this.renderModelActions(model)}
                </div>
            </div>
        `).join('');
        
        this.attachModelEventListeners();
    }
    
    getStatusClass(model) {
        if (model.is_active) return 'active';
        if (model.is_local) return 'available';
        return 'downloadable';
    }
    
    getStatusText(model) {
        if (model.is_active) return 'Active';
        if (model.is_local) return 'Local';
        return 'Download';
    }
    
    renderModelActions(model) {
        if (model.is_active) {
            return `
                <button class="btn btn-secondary" onclick="modelManager.deactivateModel('${model.name}')">
                    Deactivate
                </button>
                <button class="btn btn-error" onclick="modelManager.deleteModel('${model.name}')">
                    Delete
                </button>
            `;
        } else if (model.is_local) {
            return `
                <button class="btn btn-primary" onclick="modelManager.activateModel('${model.name}')">
                    Activate
                </button>
                <button class="btn btn-error" onclick="modelManager.deleteModel('${model.name}')">
                    Delete
                </button>
            `;
        } else {
            return `
                <button class="btn btn-primary" onclick="modelManager.downloadModel('${model.name}')">
                    Download
                </button>
            `;
        }
    }
    
    attachModelEventListeners() {
        // Event listeners are attached via onclick attributes for simplicity
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
            
            await this.refreshModels();
            
            window.app.showNotification(`Model "${modelName}" activated successfully`, 'success');
            
        } catch (error) {
            window.app.hideLoading();
            window.app.showNotification(`Failed to activate model: ${error.message}`, 'error');
        }
    }
    
    async deactivateModel(modelName) {
        try {
            window.app.showLoading('Deactivating model...');
            
            const response = await fetch(`/api/models/${modelName}/deactivate`, {
                method: 'POST'
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to deactivate model');
            }
            
            window.app.hideLoading();
            window.app.activeModel = null;
            window.app.updateModelDisplay();
            
            await this.refreshModels();
            
            window.app.showNotification(`Model "${modelName}" deactivated`, 'warning');
            
        } catch (error) {
            window.app.hideLoading();
            window.app.showNotification(`Failed to deactivate model: ${error.message}`, 'error');
        }
    }
    
    async downloadModel(modelName) {
        try {
            const response = await fetch(`/api/models/${modelName}/download`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    destination: 'local',
                    quantization: 'auto'
                })
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to start download');
            }
            
            const downloadStatus = await response.json();
            
            // Start monitoring download progress
            this.monitorDownload(modelName);
            
            window.app.showNotification(`Download started for "${modelName}"`, 'info');
            
        } catch (error) {
            window.app.showNotification(`Failed to start download: ${error.message}`, 'error');
        }
    }
    
    async monitorDownload(modelName) {
        const pollDownload = async () => {
            try {
                const response = await fetch(`/api/models/${modelName}/download/status`);
                const status = await response.json();
                
                this.updateDownloadProgress(modelName, status);
                
                if (status.status === 'completed') {
                    window.app.showNotification(`Download completed for "${modelName}"`, 'success');
                    await this.refreshModels();
                } else if (status.status === 'failed') {
                    window.app.showNotification(`Download failed for "${modelName}": ${status.error_message || 'Unknown error'}`, 'error');
                } else {
                    // Still downloading, check again
                    setTimeout(pollDownload, 2000);
                }
                
            } catch (error) {
                console.error('Download monitoring error:', error);
            }
        };
        
        setTimeout(pollDownload, 1000);
    }
    
    updateDownloadProgress(modelName, status) {
        const modelItem = document.querySelector(`[data-model="${modelName}"]`);
        if (!modelItem) return;
        
        const statusBadge = modelItem.querySelector('.status-badge');
        const actionsDiv = modelItem.querySelector('.model-item-actions');
        
        if (status.status === 'downloading') {
            statusBadge.textContent = `${Math.round(status.progress_percent)}%`;
            statusBadge.className = 'status-badge downloading';
            
            actionsDiv.innerHTML = `
                <div class="download-progress">
                    <div class="download-progress-bar">
                        <div class="download-progress-fill" style="width: ${status.progress_percent}%"></div>
                    </div>
                    <div class="download-stats">
                        <span>${status.download_speed_mbps}MB/s</span>
                        <span>ETA: ${Math.round(status.eta_minutes)}m</span>
                    </div>
                </div>
                <button class="btn btn-secondary" onclick="modelManager.cancelDownload('${modelName}')">
                    Cancel
                </button>
            `;
        }
    }
    
    async cancelDownload(modelName) {
        try {
            const response = await fetch(`/api/models/${modelName}/download`, {
                method: 'DELETE'
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to cancel download');
            }
            
            window.app.showNotification(`Download cancelled for "${modelName}"`, 'warning');
            await this.refreshModels();
            
        } catch (error) {
            window.app.showNotification(`Failed to cancel download: ${error.message}`, 'error');
        }
    }
    
    async deleteModel(modelName) {
        const confirmed = confirm(`Are you sure you want to delete the model "${modelName}"? This cannot be undone.`);
        if (!confirmed) return;
        
        try {
            window.app.showLoading('Deleting model...');
            
            const response = await fetch(`/api/models/${modelName}`, {
                method: 'DELETE'
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to delete model');
            }
            
            window.app.hideLoading();
            
            // Update active model if this was the active one
            if (window.app.activeModel === modelName) {
                window.app.activeModel = null;
                window.app.updateModelDisplay();
            }
            
            await this.refreshModels();
            
            window.app.showNotification(`Model "${modelName}" deleted successfully`, 'success');
            
        } catch (error) {
            window.app.hideLoading();
            window.app.showNotification(`Failed to delete model: ${error.message}`, 'error');
        }
    }
    
    async getModelMetrics(modelName) {
        try {
            const response = await fetch(`/api/models/${modelName}/metrics`);
            if (!response.ok) return null;
            
            return await response.json();
            
        } catch (error) {
            console.error('Failed to get model metrics:', error);
            return null;
        }
    }
    
    // Public methods
    getModels() {
        return [...this.models];
    }
    
    getActiveModel() {
        return this.models.find(model => model.is_active) || null;
    }
    
    getLocalModels() {
        return this.models.filter(model => model.is_local);
    }
    
    getDownloadableModels() {
        return this.models.filter(model => !model.is_local);
    }
}

window.ModelManager = ModelManager;