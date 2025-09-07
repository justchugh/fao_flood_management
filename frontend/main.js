// Global application state
const AppState = {
    currentTab: 'upload',
    beforeImage: null,
    afterImage: null,
    analysisResults: null,
    selectedParcels: new Set(),
    apiBase: 'http://localhost:8000'
};

// Utility functions
const utils = {
    formatNumber: (num) => {
        if (num === null || num === undefined) return '--';
        return new Intl.NumberFormat('en-US', { 
            maximumFractionDigits: 2 
        }).format(num);
    },

    formatCurrency: (amount) => {
        if (amount === null || amount === undefined) return '0 NPR';
        return new Intl.NumberFormat('en-US', { 
            maximumFractionDigits: 2 
        }).format(amount) + ' NPR';
    },

    fileToBase64: (file) => {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => resolve(reader.result);
            reader.onerror = error => reject(error);
        });
    },

    showLoading: (elementId, show = true) => {
        const element = document.getElementById(elementId);
        if (element) {
            element.style.display = show ? 'block' : 'none';
        }
    },

    showError: (message) => {
        alert(`Error: ${message}`);
    },

    showSuccess: (message) => {
        alert(`Success: ${message}`);
    }
};

// Tab management
const TabManager = {
    init() {
        const tabButtons = document.querySelectorAll('.tab-btn');
        const tabContents = document.querySelectorAll('.tab-content');

        tabButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const targetTab = e.target.getAttribute('data-tab');
                this.switchTab(targetTab);
            });
        });
    },

    switchTab(tabName) {
        // Update active states
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });

        // Activate target tab
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
        document.getElementById(tabName).classList.add('active');
        
        AppState.currentTab = tabName;
    }
};

// Image upload management
const ImageUploader = {
    init() {
        this.setupImageUpload('before');
        this.setupImageUpload('after');
        this.setupProcessButton();
    },

    setupImageUpload(type) {
        const input = document.getElementById(`${type}-input`);
        const button = document.getElementById(`${type}-btn`);
        const dropArea = document.getElementById(`${type}-drop`);
        const preview = document.getElementById(`${type}-preview`);
        const img = document.getElementById(`${type}-img`);
        const removeBtn = document.getElementById(`${type}-remove`);
        const info = document.getElementById(`${type}-info`);

        // Click to upload
        button.addEventListener('click', () => input.click());

        // File input change
        input.addEventListener('change', (e) => {
            if (e.target.files[0]) {
                this.handleFile(e.target.files[0], type);
            }
        });

        // Drag and drop
        dropArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropArea.classList.add('drag-over');
        });

        dropArea.addEventListener('dragleave', () => {
            dropArea.classList.remove('drag-over');
        });

        dropArea.addEventListener('drop', (e) => {
            e.preventDefault();
            dropArea.classList.remove('drag-over');
            const files = e.dataTransfer.files;
            if (files[0] && files[0].type.startsWith('image/')) {
                this.handleFile(files[0], type);
            }
        });

        // Remove button
        removeBtn.addEventListener('click', () => {
            this.removeImage(type);
        });
    },

    async handleFile(file, type) {
        try {
            const base64 = await utils.fileToBase64(file);
            
            // Store in app state
            AppState[`${type}Image`] = base64;

            // Update UI
            const img = document.getElementById(`${type}-img`);
            const preview = document.getElementById(`${type}-preview`);
            const dropArea = document.getElementById(`${type}-drop`);
            const info = document.getElementById(`${type}-info`);

            img.src = base64;
            preview.style.display = 'block';
            dropArea.style.display = 'none';
            info.textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;

            this.updateProcessButton();
        } catch (error) {
            utils.showError('Failed to load image: ' + error.message);
        }
    },

    removeImage(type) {
        AppState[`${type}Image`] = null;
        
        const preview = document.getElementById(`${type}-preview`);
        const dropArea = document.getElementById(`${type}-drop`);
        
        preview.style.display = 'none';
        dropArea.style.display = 'flex';
        
        this.updateProcessButton();
    },

    updateProcessButton() {
        const processBtn = document.getElementById('process-btn');
        processBtn.disabled = !(AppState.beforeImage && AppState.afterImage);
    },

    setupProcessButton() {
        const processBtn = document.getElementById('process-btn');
        processBtn.addEventListener('click', () => {
            this.processImages();
        });
    },

    async processImages() {
        if (!AppState.beforeImage || !AppState.afterImage) {
            utils.showError('Please upload both before and after images');
            return;
        }

        try {
            utils.showLoading('processing-loader', true);
            document.getElementById('process-btn').disabled = true;

            const response = await fetch(`${AppState.apiBase}/segment`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image_before: AppState.beforeImage.split(',')[1],
                    image_after: AppState.afterImage.split(',')[1]
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const results = await response.json();
            AppState.analysisResults = results;

            // Update UI
            this.updateAnalysisTab();
            utils.showSuccess('Images processed successfully!');
            
            // Switch to analysis tab
            TabManager.switchTab('analysis');

        } catch (error) {
            utils.showError('Processing failed: ' + error.message);
        } finally {
            utils.showLoading('processing-loader', false);
            document.getElementById('process-btn').disabled = false;
        }
    },

    updateAnalysisTab() {
        const results = AppState.analysisResults;
        if (!results) return;

        // Update stats
        document.getElementById('parcels-count').textContent = results.masks_before_count;
        const lostCount = results.matches.filter(m => m.status === 'Lost').length;
        document.getElementById('parcels-lost').textContent = lostCount;
        
        const totalArea = results.matches.reduce((sum, m) => sum + m.area_before_m2, 0);
        document.getElementById('total-area').textContent = utils.formatNumber(totalArea);

        // Update visualizations
        this.updateVisualizations();

        // Update results table
        this.updateResultsTable();
        
        // Show and setup area calibration
        this.setupAreaCalibration();
    },

    updateVisualizations() {
        const results = AppState.analysisResults;
        if (!results) return;

        // Update canvases with visualization images
        const beforeCanvas = document.getElementById('before-canvas');
        const afterCanvas = document.getElementById('after-canvas');
        
        if (results.before_visualization) {
            this.loadImageToCanvas(beforeCanvas, results.before_visualization);
        }
        
        if (results.after_visualization) {
            this.loadImageToCanvas(afterCanvas, results.after_visualization);
        }
    },

    loadImageToCanvas(canvas, base64Image) {
        const ctx = canvas.getContext('2d');
        const img = new Image();
        
        img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
        };
        
        img.src = base64Image;
    },

    updateResultsTable() {
        const results = AppState.analysisResults;
        if (!results) return;

        const tbody = document.getElementById('results-tbody');
        tbody.innerHTML = '';

        results.matches.forEach(match => {
            const row = document.createElement('tr');
            row.className = match.status === 'Lost' ? 'status-lost' : 'status-present';
            
            const statusIcon = match.status === 'Lost' ? '🔴' : 
                             match.loss_percentage > 20 ? '🟡' : '🟢';
            
            row.innerHTML = `
                <td>
                    <input type="checkbox" class="parcel-checkbox" value="${match.parcel_id}">
                    ${match.parcel_id}
                </td>
                <td>${utils.formatNumber(match.area_before_m2)}</td>
                <td>${utils.formatNumber(match.area_after_m2)}</td>
                <td>${utils.formatNumber(match.loss_percentage)}%</td>
                <td>${statusIcon} ${match.status}</td>
            `;
            
            tbody.appendChild(row);
        });

        // Setup parcel selection for financial analysis
        this.setupParcelSelection();
    },

    setupParcelSelection() {
        const checkboxes = document.querySelectorAll('.parcel-checkbox');
        checkboxes.forEach(checkbox => {
            checkbox.addEventListener('change', (e) => {
                const parcelId = parseInt(e.target.value);
                if (e.target.checked) {
                    AppState.selectedParcels.add(parcelId);
                } else {
                    AppState.selectedParcels.delete(parcelId);
                }
                FinancialCalculator.updateParcelSelection();
            });
        });
    },

    setupAreaCalibration() {
        const results = AppState.analysisResults;
        if (!results || !results.matches) return;

        // Show calibration section
        const calibrationSection = document.querySelector('.area-calibration-section');
        calibrationSection.style.display = 'block';

        // Update current conversion factor display
        const currentConversion = document.getElementById('current-conversion');
        currentConversion.textContent = `${results.conversion_factor || 0.0771} m²/pixel`;

        // Populate parcel dropdown with first 10 parcels
        const parcelSelect = document.getElementById('calibration-parcel');
        parcelSelect.innerHTML = '<option value="">Select a parcel...</option>';
        
        results.matches.slice(0, 10).forEach(match => {
            const option = document.createElement('option');
            option.value = match.parcel_id;
            option.textContent = `Parcel ${match.parcel_id} (${match.area_before_m2} m²)`;
            parcelSelect.appendChild(option);
        });

        // Setup form validation
        const actualAreaInput = document.getElementById('actual-area');
        const calibrateBtn = document.getElementById('calibrate-btn');

        const validateForm = () => {
            const parcelSelected = parcelSelect.value !== '';
            const areaEntered = actualAreaInput.value !== '' && parseFloat(actualAreaInput.value) > 0;
            calibrateBtn.disabled = !(parcelSelected && areaEntered);
        };

        parcelSelect.addEventListener('change', validateForm);
        actualAreaInput.addEventListener('input', validateForm);

        // Setup calibration button
        calibrateBtn.replaceWith(calibrateBtn.cloneNode(true)); // Remove existing listeners
        document.getElementById('calibrate-btn').addEventListener('click', () => {
            this.performCalibration();
        });
    },

    async performCalibration() {
        const parcelId = parseInt(document.getElementById('calibration-parcel').value);
        const actualArea = parseFloat(document.getElementById('actual-area').value);

        try {
            utils.showLoading('calibrating', true);

            const response = await fetch(`${AppState.apiBase}/calibrate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    parcel_id: parcelId,
                    actual_area_m2: actualArea,
                    analysis_results: AppState.analysisResults
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const recalibratedResults = await response.json();
            
            // Update app state with recalibrated results
            AppState.analysisResults = recalibratedResults;
            
            // Update UI with new results
            this.updateAnalysisTab();
            
            // Show calibration success
            const calibrationResults = document.getElementById('calibration-results');
            const newConversionSpan = document.getElementById('new-conversion');
            newConversionSpan.textContent = recalibratedResults.conversion_factor.toFixed(6);
            calibrationResults.style.display = 'block';

            utils.showSuccess('Areas successfully recalibrated!');

        } catch (error) {
            utils.showError('Calibration failed: ' + error.message);
        } finally {
            utils.showLoading('calibrating', false);
        }
    }
};

// Financial calculator
const FinancialCalculator = {
    init() {
        this.setupCalculateButton();
        this.setupFormValidation();
    },

    setupCalculateButton() {
        const calculateBtn = document.getElementById('calculate-btn');
        calculateBtn.addEventListener('click', () => {
            this.calculateFinancialImpact();
        });
    },

    setupFormValidation() {
        const cropSelect = document.getElementById('crop-select');
        const landSelect = document.getElementById('land-select');
        
        [cropSelect, landSelect].forEach(select => {
            select.addEventListener('change', () => {
                this.updateCalculateButton();
            });
        });
    },

    updateParcelSelection() {
        const parcelSelection = document.getElementById('parcel-selection');
        
        if (AppState.selectedParcels.size === 0) {
            parcelSelection.innerHTML = '<p class="no-parcels">No parcels selected for assessment.</p>';
        } else {
            const selectedList = Array.from(AppState.selectedParcels).map(id => {
                const match = AppState.analysisResults.matches.find(m => m.parcel_id === id);
                return `
                    <div class="selected-parcel">
                        <span>Parcel ${id}</span>
                        <span>Before: ${utils.formatNumber(match.area_before_m2)} m²</span>
                        <span>After: ${utils.formatNumber(match.area_after_m2)} m²</span>
                    </div>
                `;
            }).join('');
            
            parcelSelection.innerHTML = selectedList;
        }
        
        this.updateCalculateButton();
    },

    updateCalculateButton() {
        const calculateBtn = document.getElementById('calculate-btn');
        const cropSelected = document.getElementById('crop-select').value;
        const landSelected = document.getElementById('land-select').value;
        const parcelsSelected = AppState.selectedParcels.size > 0;
        
        calculateBtn.disabled = !(cropSelected && landSelected && parcelsSelected);
    },

    async calculateFinancialImpact() {
        const cropType = document.getElementById('crop-select').value;
        const landType = document.getElementById('land-select').value;
        const customCropRevenue = parseFloat(document.getElementById('custom-crop-revenue').value) || null;
        const customLandValue = parseFloat(document.getElementById('custom-land-value').value) || null;

        if (AppState.selectedParcels.size === 0) {
            utils.showError('Please select at least one parcel for assessment');
            return;
        }

        try {
            // Calculate total areas for selected parcels
            let totalPreFloodArea = 0;
            let totalPostFloodArea = 0;

            AppState.selectedParcels.forEach(parcelId => {
                const match = AppState.analysisResults.matches.find(m => m.parcel_id === parcelId);
                if (match) {
                    totalPreFloodArea += match.area_before_m2;
                    totalPostFloodArea += match.area_after_m2;
                }
            });

            const response = await fetch(`${AppState.apiBase}/calculate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    pre_flood_area: totalPreFloodArea,
                    post_flood_area: totalPostFloodArea,
                    crop_type: cropType,
                    land_type: landType,
                    custom_crop_revenue: customCropRevenue,
                    custom_land_value: customLandValue
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const results = await response.json();
            this.displayFinancialResults(results);

        } catch (error) {
            utils.showError('Financial calculation failed: ' + error.message);
        }
    },

    displayFinancialResults(results) {
        // Show results section
        const resultsSection = document.getElementById('financial-results');
        resultsSection.style.display = 'block';

        // Update summary values
        document.getElementById('total-loss-value').textContent = utils.formatCurrency(results.total_loss);
        document.getElementById('crop-loss-value').textContent = utils.formatCurrency(results.crop_revenue_loss);
        document.getElementById('land-loss-value').textContent = utils.formatCurrency(results.land_value_loss);

        // Update detail values
        document.getElementById('pre-crop-revenue').textContent = utils.formatCurrency(results.pre_flood_crop_revenue);
        document.getElementById('post-crop-revenue').textContent = utils.formatCurrency(results.post_flood_crop_revenue);
        document.getElementById('pre-land-value').textContent = utils.formatCurrency(results.pre_flood_land_value);
        document.getElementById('post-land-value').textContent = utils.formatCurrency(results.post_flood_land_value);

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
};

// Export functionality
const ExportManager = {
    init() {
        const exportBtn = document.getElementById('export-results');
        const reportBtn = document.getElementById('generate-report');
        
        if (exportBtn) {
            exportBtn.addEventListener('click', () => this.exportResults());
        }
        
        if (reportBtn) {
            reportBtn.addEventListener('click', () => this.generateReport());
        }
    },

    exportResults() {
        if (!AppState.analysisResults) {
            utils.showError('No analysis results to export');
            return;
        }

        const data = {
            timestamp: new Date().toISOString(),
            summary: {
                total_parcels: AppState.analysisResults.masks_before_count,
                parcels_lost: AppState.analysisResults.matches.filter(m => m.status === 'Lost').length,
                total_area_before: AppState.analysisResults.matches.reduce((sum, m) => sum + m.area_before_m2, 0),
                total_area_after: AppState.analysisResults.matches.reduce((sum, m) => sum + m.area_after_m2, 0)
            },
            parcels: AppState.analysisResults.matches
        };

        this.downloadJSON(data, 'flood-analysis-results.json');
    },

    generateReport() {
        // Simple HTML report generation
        const reportHTML = this.generateReportHTML();
        const blob = new Blob([reportHTML], { type: 'text/html' });
        const url = URL.createObjectURL(blob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = 'flood-damage-report.html';
        link.click();
        
        URL.revokeObjectURL(url);
    },

    generateReportHTML() {
        const results = AppState.analysisResults;
        const timestamp = new Date().toLocaleDateString();
        
        return `
            <!DOCTYPE html>
            <html>
            <head>
                <title>Flood Damage Assessment Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .header { text-align: center; margin-bottom: 30px; }
                    .summary { background: #f5f5f5; padding: 20px; margin: 20px 0; }
                    table { width: 100%; border-collapse: collapse; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🌊 Flood Damage Assessment Report</h1>
                    <p>Generated on: ${timestamp}</p>
                </div>
                
                <div class="summary">
                    <h2>Summary</h2>
                    <p><strong>Total Parcels Analyzed:</strong> ${results.masks_before_count}</p>
                    <p><strong>Parcels Lost:</strong> ${results.matches.filter(m => m.status === 'Lost').length}</p>
                    <p><strong>Total Area Before:</strong> ${utils.formatNumber(results.matches.reduce((sum, m) => sum + m.area_before_m2, 0))} m²</p>
                    <p><strong>Total Area After:</strong> ${utils.formatNumber(results.matches.reduce((sum, m) => sum + m.area_after_m2, 0))} m²</p>
                </div>
                
                <h2>Parcel Details</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Parcel ID</th>
                            <th>Before (m²)</th>
                            <th>After (m²)</th>
                            <th>Loss %</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${results.matches.map(match => `
                            <tr>
                                <td>${match.parcel_id}</td>
                                <td>${utils.formatNumber(match.area_before_m2)}</td>
                                <td>${utils.formatNumber(match.area_after_m2)}</td>
                                <td>${utils.formatNumber(match.loss_percentage)}%</td>
                                <td>${match.status}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </body>
            </html>
        `;
    },

    downloadJSON(data, filename) {
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        link.click();
        
        URL.revokeObjectURL(url);
    }
};

// Application initialization
document.addEventListener('DOMContentLoaded', () => {
    console.log('Flood Damage Assessment Tool - Initializing...');
    
    // Initialize all modules
    TabManager.init();
    ImageUploader.init();
    FinancialCalculator.init();
    ExportManager.init();
    
    console.log('Application initialized successfully');
    
    // Check API health
    fetch(`${AppState.apiBase}/health`)
        .then(response => response.json())
        .then(data => {
            if (data.sam_loaded) {
                console.log('SAM model loaded and ready');
            } else {
                console.warn('SAM model not loaded - check backend');
            }
        })
        .catch(error => {
            console.error('Backend connection failed:', error);
            utils.showError('Cannot connect to backend server. Please ensure it is running on port 8000.');
        });
});