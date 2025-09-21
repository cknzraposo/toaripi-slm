/**
 * Main JavaScript for Toaripi SLM Web Interface
 * Handles navigation, system status, and common functionality
 */

class ToaripiApp {
    constructor() {
        this.currentTab = 'upload';
        this.systemStatus = 'checking';
        this.activeModel = null;
        this.eventSource = null;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.checkSystemHealth();
        this.loadActiveModel();
        
        // Check health every 30 seconds
        setInterval(() => this.checkSystemHealth(), 30000);
    }
    
    setupEventListeners() {
        // Tab navigation
        document.querySelectorAll('.tab-button').forEach(button => {
            button.addEventListener('click', (e) => {
                const tabName = e.target.dataset.tab;
                this.switchTab(tabName);
            });
        });
        
        // Modal close handlers
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal')) {
                this.closeModal(e.target.id);
            }
        });
        
        // Escape key to close modals
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeAllModals();
            }
        });
        
        // Global error handler
        window.addEventListener('error', (e) => {
            console.error('Global error:', e.error);
            this.showNotification('An unexpected error occurred', 'error');
        });
        
        // Unhandled promise rejections
        window.addEventListener('unhandledrejection', (e) => {
            console.error('Unhandled promise rejection:', e.reason);
            this.showNotification('A network error occurred', 'error');
        });
    }
    
    switchTab(tabName) {
        // Update active tab button
        document.querySelectorAll('.tab-button').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
        
        // Update active tab pane
        document.querySelectorAll('.tab-pane').forEach(pane => {
            pane.classList.remove('active');
        });
        document.getElementById(`${tabName}-tab`).classList.add('active');
        
        this.currentTab = tabName;
        
        // Tab-specific initialization
        switch (tabName) {
            case 'upload':
                if (window.uploadManager) {
                    window.uploadManager.refreshStatus();
                }
                break;
            case 'training':
                if (window.trainingManager) {
                    window.trainingManager.refreshStatus();
                }
                break;
            case 'generate':
                if (window.generationManager) {
                    window.generationManager.refreshStatus();
                }
                break;
        }
    }
    
    async checkSystemHealth() {
        try {
            const response = await fetch('/api/health');
            const health = await response.json();
            
            this.updateSystemStatus(health.status, health);
            
        } catch (error) {
            console.error('Health check failed:', error);
            this.updateSystemStatus('error', { error: 'Health check failed' });
        }
    }
    
    updateSystemStatus(status, details) {
        this.systemStatus = status;
        
        const statusIndicator = document.getElementById('system-status');
        const statusDot = statusIndicator.querySelector('.status-dot');
        const statusText = statusIndicator.querySelector('.status-text');
        
        // Remove all status classes
        statusDot.classList.remove('healthy', 'error');
        
        switch (status) {
            case 'healthy':
                statusDot.classList.add('healthy');
                statusText.textContent = 'System Healthy';
                break;
            case 'degraded':
                statusText.textContent = 'System Degraded';
                break;
            case 'unhealthy':
            case 'error':
                statusDot.classList.add('error');
                statusText.textContent = 'System Error';
                break;
            default:
                statusText.textContent = 'Checking...';
        }
        
        // Update model info if available
        if (details && details.active_model) {
            this.activeModel = details.active_model;
            this.updateModelDisplay();
        }
    }
    
    async loadActiveModel() {
        try {
            const response = await fetch('/api/models');
            const models = await response.json();
            
            const activeModel = models.find(model => model.is_active);
            if (activeModel) {
                this.activeModel = activeModel.name;
                this.updateModelDisplay();
            }
            
        } catch (error) {
            console.error('Failed to load active model:', error);
        }
    }
    
    updateModelDisplay() {
        const modelInfoElement = document.getElementById('active-model');
        if (modelInfoElement) {
            modelInfoElement.textContent = this.activeModel || 'No model active';
        }
        
        // Update generation tab model display
        const generationModelElement = document.getElementById('generation-active-model');
        if (generationModelElement) {
            generationModelElement.textContent = this.activeModel || 'No model loaded';
        }
        
        // Enable/disable generation based on model availability
        const generateButton = document.getElementById('generate-content');
        if (generateButton) {
            generateButton.disabled = !this.activeModel;
        }
    }
    
    showLoading(message = 'Processing...') {
        const overlay = document.getElementById('loading-overlay');
        const messageElement = document.getElementById('loading-message');
        
        if (messageElement) {
            messageElement.textContent = message;
        }
        
        if (overlay) {
            overlay.style.display = 'flex';
        }
    }
    
    hideLoading() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
    }
    
    showModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.style.display = 'flex';
            document.body.style.overflow = 'hidden';
            
            // Focus first focusable element
            const focusable = modal.querySelector('button, input, select, textarea');
            if (focusable) {
                focusable.focus();
            }
        }
    }
    
    closeModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.style.display = 'none';
            document.body.style.overflow = '';
        }
    }
    
    closeAllModals() {
        document.querySelectorAll('.modal').forEach(modal => {
            modal.style.display = 'none';
        });
        document.body.style.overflow = '';
    }
    
    showNotification(message, type = 'info', duration = 5000) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <span class="notification-icon">${this.getNotificationIcon(type)}</span>
                <span class="notification-message">${message}</span>
                <button class="notification-close" onclick="this.parentElement.parentElement.remove()">&times;</button>
            </div>
        `;
        
        // Add notification styles if not already present
        if (!document.getElementById('notification-styles')) {
            const styles = document.createElement('style');
            styles.id = 'notification-styles';
            styles.textContent = `
                .notification {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    z-index: 10001;
                    background: var(--surface);
                    border-radius: var(--border-radius);
                    box-shadow: var(--shadow-lg);
                    border-left: 4px solid;
                    animation: slideInRight 0.3s ease;
                    max-width: 400px;
                }
                .notification-info { border-left-color: var(--info-color); }
                .notification-success { border-left-color: var(--success-color); }
                .notification-warning { border-left-color: var(--warning-color); }
                .notification-error { border-left-color: var(--error-color); }
                .notification-content {
                    display: flex;
                    align-items: center;
                    gap: var(--spacing-sm);
                    padding: var(--spacing-md);
                }
                .notification-icon {
                    font-size: var(--font-size-lg);
                }
                .notification-message {
                    flex: 1;
                    color: var(--text-primary);
                }
                .notification-close {
                    background: none;
                    border: none;
                    font-size: var(--font-size-lg);
                    cursor: pointer;
                    color: var(--text-secondary);
                    padding: var(--spacing-xs);
                }
                @keyframes slideInRight {
                    from { transform: translateX(100%); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }
            `;
            document.head.appendChild(styles);
        }
        
        // Add to page
        document.body.appendChild(notification);
        
        // Auto remove after duration
        if (duration > 0) {
            setTimeout(() => {
                if (notification.parentElement) {
                    notification.remove();
                }
            }, duration);
        }
    }
    
    getNotificationIcon(type) {
        const icons = {
            info: 'ℹ️',
            success: '✅',
            warning: '⚠️',
            error: '❌'
        };
        return icons[type] || icons.info;
    }
    
    async apiRequest(url, options = {}) {
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
        };
        
        const config = { ...defaultOptions, ...options };
        
        try {
            const response = await fetch(url, config);
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
                throw new Error(errorData.detail || `HTTP ${response.status}`);
            }
            
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                return await response.json();
            } else {
                return await response.text();
            }
            
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 B';
        
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }
    
    formatDuration(seconds) {
        if (seconds < 60) {
            return `${Math.round(seconds)}s`;
        } else if (seconds < 3600) {
            const minutes = Math.floor(seconds / 60);
            const remainingSeconds = Math.round(seconds % 60);
            return `${minutes}m ${remainingSeconds}s`;
        } else {
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            return `${hours}h ${minutes}m`;
        }
    }
    
    formatTimestamp(timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleString();
    }
    
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
    
    throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }
}

// Global utility functions
window.closeModal = function(modalId) {
    if (window.app) {
        window.app.closeModal(modalId);
    }
};

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new ToaripiApp();
    
    // Initialize tab-specific managers
    if (typeof UploadManager !== 'undefined') {
        window.uploadManager = new UploadManager();
    }
    
    if (typeof TrainingManager !== 'undefined') {
        window.trainingManager = new TrainingManager();
    }
    
    if (typeof GenerationManager !== 'undefined') {
        window.generationManager = new GenerationManager();
    }
    
    if (typeof ModelManager !== 'undefined') {
        window.modelManager = new ModelManager();
    }
    
    console.log('🌺 Toaripi SLM Web Interface initialized');
});

// Export for use in other modules
window.ToaripiApp = ToaripiApp;