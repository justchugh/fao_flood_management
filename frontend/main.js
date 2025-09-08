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
        document.getElementById('parcels-count').textContent = results.parcels_detected;
        document.getElementById('water-coverage').textContent = results.water_coverage_percentage + '%';
        
        const totalArea = results.total_areas.original_m2;
        document.getElementById('total-area').textContent = utils.formatNumber(totalArea);
        
        const damagedCount = results.damage_results.filter(d => d.damage_status !== 'Undamaged').length;
        document.getElementById('damaged-parcels').textContent = damagedCount;

        // Update 4 visualizations
        this.updateVisualizations();

        // Update results table
        this.updateResultsTable();
        
        // Show and setup area calibration if needed
        this.setupAreaCalibration();
    },

    updateVisualizations() {
        const results = AppState.analysisResults;
        if (!results || !results.visualizations) return;

        // Update all 4 canvases with new visualization images
        const beforeSamCanvas = document.getElementById('before-sam-canvas');
        const damageCanvas = document.getElementById('damage-canvas');
        const waterCanvas = document.getElementById('water-canvas');
        const landCanvas = document.getElementById('land-canvas');
        
        if (results.visualizations.before_sam_results) {
            this.loadImageToCanvas(beforeSamCanvas, results.visualizations.before_sam_results);
        }
        
        if (results.visualizations.flood_damage_assessment) {
            this.loadImageToCanvas(damageCanvas, results.visualizations.flood_damage_assessment);
        }
        
        if (results.visualizations.water_detection) {
            this.loadImageToCanvas(waterCanvas, results.visualizations.water_detection);
        }
        
        if (results.visualizations.land_only) {
            this.loadImageToCanvas(landCanvas, results.visualizations.land_only);
        }
    },

    loadImageToCanvas(canvas, base64Image) {
        console.log('Loading image to canvas:', canvas?.id, 'Image data length:', base64Image?.length);
        if (!canvas || !base64Image) {
            console.error('Missing canvas or image data:', canvas?.id, !!base64Image);
            return;
        }
        
        const ctx = canvas.getContext('2d');
        const img = new Image();
        
        img.onload = () => {
            console.log('Image loaded successfully:', canvas.id, img.width, img.height);
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
        };
        
        img.onerror = (e) => {
            console.error('Failed to load image for canvas:', canvas.id, e);
        };
        
        img.src = base64Image;
    },

    updateResultsTable() {
        const results = AppState.analysisResults;
        if (!results || !results.damage_results) return;

        const tbody = document.getElementById('results-tbody');
        tbody.innerHTML = '';

        results.damage_results.forEach(damage => {
            const row = document.createElement('tr');
            
            // Set row class based on damage status
            let rowClass = 'status-undamaged';
            let statusIcon = '[OK]';
            
            switch (damage.damage_status) {
                case 'Undamaged':
                    rowClass = 'status-undamaged';
                    statusIcon = '[OK]';
                    break;
                case 'Lightly Damaged':
                    rowClass = 'status-light';
                    statusIcon = '[LIGHT]';
                    break;
                case 'Moderately Damaged':
                    rowClass = 'status-moderate';
                    statusIcon = '[MOD]';
                    break;
                case 'Heavily Damaged':
                    rowClass = 'status-heavy';
                    statusIcon = '[HEAVY]';
                    break;
                case 'Severely Damaged':
                    rowClass = 'status-severe';
                    statusIcon = '[SEVERE]';
                    break;
            }
            
            row.className = rowClass;
            
            row.innerHTML = `
                <td>
                    <input type="checkbox" class="parcel-checkbox" value="${damage.parcel_id}">
                    ${damage.parcel_id}
                </td>
                <td>${utils.formatNumber(damage.original_area_m2)}</td>
                <td>${utils.formatNumber(damage.flooded_area_m2)}</td>
                <td>${utils.formatNumber(damage.remaining_area_m2)}</td>
                <td>${utils.formatNumber(damage.flood_percentage)}%</td>
                <td>${statusIcon} ${damage.damage_status}</td>
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
        if (!results || !results.damage_results) return;

        // Show calibration section
        const calibrationSection = document.querySelector('.area-calibration-section');
        calibrationSection.style.display = 'block';

        // Update current conversion factor display
        const currentConversion = document.getElementById('current-conversion');
        currentConversion.textContent = `${results.conversion_factor || 0.0771} m²/pixel`;

        // Calculate and display minimum area
        const minArea = Math.min(...results.damage_results.map(d => d.original_area_m2));
        const minAreaDisplay = document.getElementById('min-area');
        minAreaDisplay.textContent = `${utils.formatNumber(minArea)} m²`;

        // Populate parcel dropdown with all parcels (sorted by area)
        const parcelSelect = document.getElementById('calibration-parcel');
        parcelSelect.innerHTML = '<option value="">Select a parcel...</option>';
        
        // Sort parcels by area for easier selection
        const sortedParcels = [...results.damage_results].sort((a, b) => a.original_area_m2 - b.original_area_m2);
        
        sortedParcels.forEach(damage => {
            const option = document.createElement('option');
            option.value = damage.parcel_id;
            option.textContent = `Parcel ${damage.parcel_id} (${utils.formatNumber(damage.original_area_m2)} m²)`;
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
                const damage = AppState.analysisResults.damage_results.find(d => d.parcel_id === id);
                return `
                    <div class="selected-parcel">
                        <span>Parcel ${id}</span>
                        <span>Original: ${utils.formatNumber(damage.original_area_m2)} m²</span>
                        <span>Remaining: ${utils.formatNumber(damage.remaining_area_m2)} m²</span>
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
                const damage = AppState.analysisResults.damage_results.find(d => d.parcel_id === parcelId);
                if (damage) {
                    totalPreFloodArea += damage.original_area_m2;
                    totalPostFloodArea += damage.remaining_area_m2;
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
                total_parcels: AppState.analysisResults.total_parcels || AppState.analysisResults.damage_results.length,
                parcels_damaged: AppState.analysisResults.damage_results.filter(d => d.damage_status !== 'Undamaged').length,
                total_area_before: AppState.analysisResults.total_areas?.original_m2 || AppState.analysisResults.damage_results.reduce((sum, d) => sum + d.original_area_m2, 0),
                total_area_after: AppState.analysisResults.total_areas?.remaining_m2 || AppState.analysisResults.damage_results.reduce((sum, d) => sum + d.remaining_area_m2, 0)
            },
            parcels: AppState.analysisResults.damage_results
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
                    <p><strong>Total Parcels Analyzed:</strong> ${results.total_parcels || results.damage_results.length}</p>
                    <p><strong>Damaged Parcels:</strong> ${results.damage_results.filter(d => d.damage_status !== 'Undamaged').length}</p>
                    <p><strong>Water Coverage:</strong> ${results.water_coverage_percentage || 0}%</p>
                    <p><strong>Total Original Area:</strong> ${utils.formatNumber(results.total_areas?.original_m2 || results.damage_results.reduce((sum, d) => sum + d.original_area_m2, 0))} m²</p>
                    <p><strong>Total Remaining Area:</strong> ${utils.formatNumber(results.total_areas?.remaining_m2 || results.damage_results.reduce((sum, d) => sum + d.remaining_area_m2, 0))} m²</p>
                </div>
                
                <h2>Flood Damage Details</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Parcel ID</th>
                            <th>Original (m²)</th>
                            <th>Flooded (m²)</th>
                            <th>Remaining (m²)</th>
                            <th>Flood %</th>
                            <th>Damage Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${results.damage_results.map(damage => `
                            <tr>
                                <td>${damage.parcel_id}</td>
                                <td>${utils.formatNumber(damage.original_area_m2)}</td>
                                <td>${utils.formatNumber(damage.flooded_area_m2)}</td>
                                <td>${utils.formatNumber(damage.remaining_area_m2)}</td>
                                <td>${utils.formatNumber(damage.flood_percentage)}%</td>
                                <td>${damage.damage_status}</td>
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