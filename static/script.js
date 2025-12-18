// Configuration - Auto-detect API URL
const API_BASE_URL = window.location.origin + '/api/v1';
const POLL_INTERVAL = 3000; // 3 seconds
const THEME_KEY = 'pss-theme';

// Log for debugging
console.log('API Base URL:', API_BASE_URL);

// State
let uploadedPhotos = [];
let selectedTemplate = null;
let currentJobId = null;
let pollInterval = null;
let currentTheme = 'light';
let currentTemplateCategory = 'all';
const templateCache = new Map();

function applyTheme(theme) {
    const safeTheme = theme === 'dark' ? 'dark' : 'light';
    currentTheme = safeTheme;
    document.documentElement.classList.toggle('dark', safeTheme === 'dark');
    const toggleBtn = document.getElementById('themeToggle');
    if (toggleBtn) {
        toggleBtn.textContent = safeTheme === 'dark' ? '☀️' : '🌙';
        toggleBtn.setAttribute(
            'aria-label',
            safeTheme === 'dark' ? 'Activate light mode' : 'Activate dark mode'
        );
    }
}

function initTheme() {
    try {
        const storedTheme = localStorage.getItem(THEME_KEY);
        const prefersDark = window.matchMedia &&
            window.matchMedia('(prefers-color-scheme: dark)').matches;
        applyTheme(storedTheme || (prefersDark ? 'dark' : 'light'));
        const toggleBtn = document.getElementById('themeToggle');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => {
                const nextTheme = currentTheme === 'dark' ? 'light' : 'dark';
                applyTheme(nextTheme);
                localStorage.setItem(THEME_KEY, nextTheme);
            });
        }
    } catch (error) {
        console.warn('Theme initialization failed:', error);
    }
}

// Step Navigation - with error handling
function showStep(stepNumber) {
    try {
        const steps = document.querySelectorAll('.step');
        if (steps.length === 0) {
            console.error('No .step elements found');
            return;
        }
        
        steps.forEach(step => {
            step.classList.remove('active');
        });
        
        const targetStep = document.getElementById(`step${stepNumber}`);
        if (!targetStep) {
            console.error(`Step ${stepNumber} element not found`);
            return;
        }
        
        targetStep.classList.add('active');
        console.log(`Switched to step ${stepNumber}`);
    } catch (error) {
        console.error('Error in showStep:', error);
    }
}

function goToStep1() {
    showStep(1);
}

function goToStep2() {
    if (uploadedPhotos.length > 0) {
        showStep(2);
        loadTemplates();
    }
}

function goToStep3() {
    if (selectedTemplate && uploadedPhotos.length > 0) {
        showStep(3);
        processSwap();
    }
}

function resetStep1() {
    try {
        uploadedPhotos = [];
        const uploadedContainer = document.getElementById('uploadedPhotos');
        const customerPhotosInput = document.getElementById('customerPhotos');
        
        if (uploadedContainer) {
            uploadedContainer.innerHTML = '';
        }
        if (customerPhotosInput) {
            customerPhotosInput.value = '';
        }
        updateStep1Button();
    } catch (error) {
        console.error('Error in resetStep1:', error);
    }
}

// File Upload - initialize after DOM ready
function initializeFileUpload() {
    try {
        const customerPhotosInput = document.getElementById('customerPhotos');
        const uploadArea = document.getElementById('uploadArea');
        
        if (!customerPhotosInput) {
            console.error('customerPhotos input not found');
            return;
        }
        if (!uploadArea) {
            console.error('uploadArea not found');
            return;
        }
        const chooseButton = document.getElementById('choosePhotosBtn');
        
        customerPhotosInput.addEventListener('change', handleFileSelect);
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            handleFiles(e.dataTransfer.files);
        });

        if (chooseButton) {
            chooseButton.addEventListener('click', (event) => {
                event.stopPropagation();
                customerPhotosInput.click();
            });
        }
        
        uploadArea.addEventListener('click', () => {
            customerPhotosInput.click();
        });
        
        console.log('✓ File upload initialized');
    } catch (error) {
        console.error('Error initializing file upload:', error);
    }
}

function handleFileSelect(e) {
    handleFiles(e.target.files);
}

function handleFiles(files) {
    const validFiles = Array.from(files).filter(file => {
        return file.type.startsWith('image/') && file.size <= 10 * 1024 * 1024;
    });

    if (validFiles.length === 0) {
        alert('Please select valid image files (max 10MB each)');
        return;
    }

    if (validFiles.length > 2) {
        alert('Maximum 2 photos allowed');
        return;
    }

    uploadedPhotos = validFiles;
    displayUploadedPhotos();
    updateStep1Button();
}

function displayUploadedPhotos() {
    try {
        const container = document.getElementById('uploadedPhotos');
        if (!container) {
            console.error('uploadedPhotos container not found');
            return;
        }
        
        container.innerHTML = '';

        uploadedPhotos.forEach((file, index) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                const div = document.createElement('div');
                div.className = 'photo-preview';
                div.innerHTML = `
                    <img src="${e.target.result}" alt="Photo ${index + 1}">
                    <button class="remove-btn" onclick="removePhoto(${index})">×</button>
                `;
                container.appendChild(div);
            };
            reader.onerror = () => {
                console.error('Error reading file:', file.name);
            };
            reader.readAsDataURL(file);
        });
    } catch (error) {
        console.error('Error in displayUploadedPhotos:', error);
    }
}

function removePhoto(index) {
    uploadedPhotos.splice(index, 1);
    displayUploadedPhotos();
    updateStep1Button();
}

function updateStep1Button() {
    const btn = document.getElementById('nextToStep2');
    btn.disabled = uploadedPhotos.length === 0;
}

// Template Loading
async function loadTemplates(category = 'all') {
    const gallery = document.getElementById('templateGallery');
    if (!gallery) return;

    const normalizedCategory = category || 'all';
    currentTemplateCategory = normalizedCategory;

    const cached = templateCache.get(normalizedCategory);
    if (cached) {
        displayTemplates(cached);
        return;
    }

    gallery.innerHTML = '<div class="loading">Loading templates...</div>';

    try {
        const params = new URLSearchParams();
        if (normalizedCategory !== 'all') {
            params.append('category', normalizedCategory);
        }
        const query = params.toString();
        const response = await fetch(`/api/v1/templates${query ? `?${query}` : ''}`);
        if (!response.ok) {
            throw new Error('Failed to load templates');
        }
        const data = await response.json();
        const templates = data?.templates || [];
        templateCache.set(normalizedCategory, templates);
        displayTemplates(templates);
    } catch (error) {
        console.error('Template load error:', error);
        gallery.innerHTML = `<div class="loading">Failed to load templates. ${error.message}</div>`;
    }
}

function displayTemplates(templates) {
    const gallery = document.getElementById('templateGallery');
    gallery.innerHTML = '';

    if (!templates.length) {
        gallery.innerHTML = '<div class="loading">No templates found for this category.</div>';
        return;
    }

    templates.forEach(template => {
        const card = document.createElement('div');
        card.className = 'template-card';
        card.dataset.category = template.category;
        card.addEventListener('click', () => selectTemplate(template, card));

        card.innerHTML = `
            <img src="${template.preview_url}" alt="${template.name}">
            <div class="template-info">
                <h3>${template.name}</h3>
                <div class="template-tags">
                    ${template.tags.map(tag => `<span class="template-tag">${tag}</span>`).join('')}
                </div>
            </div>
        `;

        gallery.appendChild(card);
    });
}

function filterTemplates(category, button) {
    document.querySelectorAll('.category-btn').forEach(btn => btn.classList.remove('active'));
    if (button) {
        button.classList.add('active');
    }
    loadTemplates(category);
}

function selectTemplate(template, cardElement) {
    selectedTemplate = template;
    document.querySelectorAll('.template-card').forEach(card => {
        card.classList.remove('selected');
    });
    if (cardElement) {
        cardElement.classList.add('selected');
    }
    document.getElementById('nextToStep3').disabled = false;
}

// Processing
async function processSwap() {
    if (!selectedTemplate || uploadedPhotos.length === 0) {
        alert('Please select photos and a template');
        return;
    }

    const formData = new FormData();
    uploadedPhotos.forEach((photo, idx) => {
        formData.append('customer_photos', photo, `customer_${idx + 1}.jpg`);
    });

    formData.append('template_id', selectedTemplate.id);

    try {
        updateProcessingStatus('Uploading images...', 15);
        
        const response = await fetch(`${API_BASE_URL}/swap`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Failed to create swap job');
        }

        const data = await response.json();
        currentJobId = data.job_id;
        
        updateProcessingStatus('Processing started...', 20);
        
        // Start polling for status
        startPolling();
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to start processing. Please try again.');
        goToStep1();
    }
}

function startPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
    }

    pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/jobs/${currentJobId}`);
            const job = await response.json();

            updateProcessingStatus(job.current_stage || 'Processing...', job.progress * 100);
            document.getElementById('currentStage').textContent = job.current_stage || '';

            if (job.status === 'completed') {
                clearInterval(pollInterval);
                showResult(job);
            } else if (job.status === 'failed') {
                clearInterval(pollInterval);
                alert(`Processing failed: ${job.error || 'Unknown error'}`);
                goToStep1();
            }
        } catch (error) {
            console.error('Polling error:', error);
        }
    }, POLL_INTERVAL);
}

function updateProcessingStatus(status, progress) {
    document.getElementById('processingStatus').textContent = status;
    document.getElementById('progressFill').style.width = `${progress}%`;
    
    const estimatedMinutes = Math.max(1, Math.ceil((100 - progress) / 20));
    document.getElementById('estimatedTime').textContent = `${estimatedMinutes}-${estimatedMinutes + 2} minutes`;
}

async function showResult(job) {
    showStep(4);
    
    try {
        // Fetch result image
        const response = await fetch(`${API_BASE_URL}/jobs/${currentJobId}/result`);
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        document.getElementById('resultImage').src = url;

        displayQualityMetrics(job.quality_metrics);
        displayBodySummary(job.body_summary);
        displayFitReport(job.fit_report);
    } catch (error) {
        console.error('Error loading result:', error);
        alert('Result loaded but image may not display. Try downloading.');
    }
}

function displayQualityMetrics(metrics) {
    const container = document.getElementById('qualityMetrics');
    if (!container) return;
    
    if (!metrics || Object.keys(metrics).length === 0) {
        container.innerHTML = '<p class="text-sm text-slate-500 dark:text-slate-400">Quality metrics will appear here once available.</p>';
        return;
    }
    
    container.innerHTML = '';

    const metricNames = {
        overall_score: 'Overall Quality',
        face_similarity: 'Face Similarity',
        pose_accuracy: 'Pose Accuracy',
        clothing_fit: 'Clothing Fit',
        seamless_blending: 'Seamless Blending',
        sharpness: 'Sharpness'
    };

    Object.entries(metrics).forEach(([key, value]) => {
        if (typeof value === 'number' && metricNames[key]) {
            const div = document.createElement('div');
            div.className = 'quality-metric';
            div.innerHTML = `
                <div>
                    <div class="metric-label">${metricNames[key]}</div>
                    <div class="metric-bar">
                        <div class="metric-bar-fill" style="width: ${value * 100}%"></div>
                    </div>
                </div>
                <div class="metric-value">${(value * 100).toFixed(0)}%</div>
            `;
            container.appendChild(div);
        }
    });
}

function displayBodySummary(summary) {
    const container = document.getElementById('bodySummary');
    if (!container) return;
    
    if (!summary || Object.keys(summary).length === 0) {
        container.innerHTML = '<p class="text-sm text-slate-500 dark:text-slate-400">Body analysis data will appear here after processing.</p>';
        return;
    }
    
    const bodyType = summary.body_type ? summary.body_type.replace('_', ' ') : 'Unknown';
    const confidence = typeof summary.confidence === 'number'
        ? `${Math.round(Math.max(0, Math.min(1, summary.confidence)) * 100)}%`
        : '—';
    
    const measurementLabels = {
        shoulder_width: 'Shoulders',
        hip_width: 'Hips',
        waist_width: 'Waist',
        torso_height: 'Torso Height',
        leg_length: 'Leg Length',
        shoulder_hip_ratio: 'Shoulder/Hip Ratio'
    };
    
    const measurements = summary.measurements || {};
    const measurementEntries = Object.entries(measurementLabels)
        .filter(([key]) => typeof measurements[key] === 'number')
        .slice(0, 4)
        .map(([key, label]) => {
            const value = measurements[key];
            const formatted = key.includes('ratio')
                ? value.toFixed(2)
                : `${Math.round(value)}`;
            return `
                <div class="rounded-2xl bg-white/60 dark:bg-white/5 border border-white/60 dark:border-white/10 p-3">
                    <div class="text-xs uppercase tracking-[0.25em] text-slate-500 dark:text-slate-400">${label}</div>
                    <div class="text-lg font-semibold text-slate-900 dark:text-white">${formatted}</div>
                </div>
            `;
        }).join('');
    
    container.innerHTML = `
        <div class="flex flex-wrap items-center gap-3">
            <span class="inline-flex items-center rounded-full bg-brandAccent/10 text-brandAccent dark:text-white dark:bg-white/10 px-4 py-1 text-sm font-semibold capitalize">${bodyType}</span>
            <span class="text-sm text-slate-500 dark:text-slate-300">Confidence: <strong>${confidence}</strong></span>
        </div>
        ${measurementEntries
            ? `<div class="grid grid-cols-2 gap-3">${measurementEntries}</div>`
            : '<p class="text-sm text-slate-500 dark:text-slate-400">Insufficient measurements.</p>'}
    `;
}

function displayFitReport(report) {
    const container = document.getElementById('fitReport');
    if (!container) return;
    
    if (!report || !report.items) {
        container.innerHTML = '<p class="text-sm text-slate-500 dark:text-slate-400">Fit adjustments will be listed here after processing.</p>';
        return;
    }
    
    const scaleMap = report.scale_map || {};
    const chips = Object.entries(scaleMap).slice(0, 6).map(([key, value]) => {
        const percent = typeof value === 'number' ? `${(value * 100).toFixed(0)}%` : value;
        return `<span class="inline-flex items-center rounded-full bg-white/60 dark:bg-white/10 border border-white/80 dark:border-white/10 px-3 py-1 text-xs font-semibold text-slate-700 dark:text-slate-200">${key.replace('_', ' ')} · ${percent}</span>`;
    }).join('');
    
    const itemEntries = Object.entries(report.items);
    const itemCards = itemEntries.map(([item, details]) => {
        const status = details.status || 'scaled';
        const scaleX = typeof details.scale_x === 'number' ? `${(details.scale_x * 100).toFixed(0)}%` : '—';
        const scaleY = typeof details.scale_y === 'number' ? `${(details.scale_y * 100).toFixed(0)}%` : '—';
        return `
            <div class="rounded-2xl bg-white/60 dark:bg-white/5 border border-white/60 dark:border-white/10 p-4">
                <div class="text-xs uppercase tracking-[0.3em] text-slate-500 dark:text-slate-400">${item.replace('_', ' ')}</div>
                <div class="text-lg font-semibold text-slate-900 dark:text-white">${status === 'scaled' ? `${scaleX} width · ${scaleY} height` : 'No change'}</div>
            </div>
        `;
    }).join('');
    const itemsHtml = itemCards
        ? `<div class="grid grid-cols-1 sm:grid-cols-2 gap-3">${itemCards}</div>`
        : '<p class="text-sm text-slate-500 dark:text-slate-400">No clothing adaptations were necessary for this job.</p>';
    
    const skinNote = report.skin_synthesis_applied
        ? '<div class="text-xs text-amber-600 dark:text-amber-300 font-semibold">Open-chest region stabilized with skin synthesis.</div>'
        : '';
    
    container.innerHTML = `
        <div class="space-y-3">
            ${chips ? `<div class="flex flex-wrap gap-2">${chips}</div>` : ''}
            ${itemsHtml}
            ${skinNote}
        </div>
    `;
}

function downloadResult() {
    const img = document.getElementById('resultImage');
    const link = document.createElement('a');
    link.href = img.src;
    link.download = `swap_result_${currentJobId}.png`;
    link.click();
}

async function downloadBundle() {
    if (!currentJobId) {
        alert('No job available to download.');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/jobs/${currentJobId}/bundle`);
        if (!response.ok) {
            throw new Error('Bundle not ready yet. Please wait until processing completes.');
        }
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `swap_bundle_${currentJobId}.zip`;
        link.click();
        URL.revokeObjectURL(url);
    } catch (error) {
        console.error('Bundle download failed:', error);
        alert(error.message || 'Unable to download bundle right now.');
    }
}

function shareResult() {
    if (navigator.share) {
        navigator.share({
            title: 'Check out my photo swap!',
            text: 'I transformed my photo using Photo Swap Studio',
            url: window.location.href
        });
    } else {
        // Fallback: copy to clipboard
        navigator.clipboard.writeText(window.location.href);
        alert('Link copied to clipboard!');
    }
}

function createNew() {
    // Reset everything
    uploadedPhotos = [];
    selectedTemplate = null;
    currentJobId = null;
    if (pollInterval) {
        clearInterval(pollInterval);
    }
    
    document.getElementById('uploadedPhotos').innerHTML = '';
    document.getElementById('customerPhotos').value = '';
    document.getElementById('resultImage').src = '';
    
    goToStep1();
}

// Initialize with error handling
document.addEventListener('DOMContentLoaded', () => {
    console.log('Photo Swap Studio Frontend Loaded');
    console.log('API Base URL:', API_BASE_URL);
    initTheme();
    
    // Check if required elements exist
    try {
        const step1 = document.getElementById('step1');
        if (!step1) {
            console.error('Required element step1 not found');
            // Don't throw, just warn - page might still work
            console.warn('Page structure may be incomplete, but continuing...');
        } else {
            // Ensure step 1 is visible
            showStep(1);
        }
        
        // Initialize file upload handlers
        initializeFileUpload();
        
        console.log('✓ Frontend initialized successfully');
    } catch (error) {
        console.error('Initialization error:', error);
        // Don't clear the page, just show error banner
        const errorBanner = document.createElement('div');
        errorBanner.style.cssText = 'background: #ff6b6b; color: white; padding: 15px; margin: 20px; border-radius: 4px;';
        errorBanner.innerHTML = '<h3>⚠️ Initialization Error</h3><p>' + error.message + '</p><p>Check browser console (F12) for details.</p>';
        const main = document.querySelector('.main-content') || document.body;
        main.insertBefore(errorBanner, main.firstChild);
    }
});

// Global error handler
window.addEventListener('error', function(e) {
    console.error('Global error:', e.error);
    // Don't clear the page, just log
});

// Prevent unhandled promise rejections from breaking the page
window.addEventListener('unhandledrejection', function(e) {
    console.error('Unhandled promise rejection:', e.reason);
    e.preventDefault(); // Prevent default browser error handling
});

