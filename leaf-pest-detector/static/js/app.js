/**
 * 🌿 LeafGuard AI — Frontend Application Logic
 */

document.addEventListener('DOMContentLoaded', () => {
    // ── DOM Elements ────────────────────────────────────────────
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const uploadContent = document.getElementById('upload-content');
    const previewContent = document.getElementById('preview-content');
    const previewImage = document.getElementById('preview-image');
    const btnRemove = document.getElementById('btn-remove');
    const btnPredict = document.getElementById('btn-predict');
    const btnNew = document.getElementById('btn-new');
    const resultsSection = document.getElementById('results-section');
    const samplesGrid = document.getElementById('samples-grid');
    const modeBadge = document.getElementById('mode-badge');

    let selectedFile = null;

    // ── Initialize ──────────────────────────────────────────────
    loadSampleImages();
    checkMode();

    // ── Mode Check ──────────────────────────────────────────────
    async function checkMode() {
        modeBadge.textContent = 'System Ready';
    }

    // ── File Upload ─────────────────────────────────────────────

    // Click to upload
    uploadZone.addEventListener('click', (e) => {
        if (e.target === btnRemove || e.target.closest('.btn-remove')) return;
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag and drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('drag-over');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // Handle selected file
    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file (JPG, PNG, WebP)');
            return;
        }

        if (file.size > 10 * 1024 * 1024) {
            alert('File too large. Max 10MB.');
            return;
        }

        selectedFile = file;

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            uploadContent.style.display = 'none';
            previewContent.style.display = 'flex';
            btnPredict.disabled = false;
        };
        reader.readAsDataURL(file);

        // Hide results
        resultsSection.style.display = 'none';
    }

    // Remove image
    btnRemove.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });

    function resetUpload() {
        selectedFile = null;
        fileInput.value = '';
        previewImage.src = '';
        previewContent.style.display = 'none';
        uploadContent.style.display = 'flex';
        btnPredict.disabled = true;
    }

    // ── Prediction ──────────────────────────────────────────────

    btnPredict.addEventListener('click', async () => {
        if (!selectedFile) return;

        // Show loading
        const btnText = btnPredict.querySelector('.btn-text');
        const btnLoading = btnPredict.querySelector('.btn-loading');
        btnText.style.display = 'none';
        btnLoading.style.display = 'flex';
        btnPredict.disabled = true;

        try {
            const formData = new FormData();
            formData.append('image', selectedFile);

            const response = await fetch('/predict', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (data.success) {
                showResults(data);
            } else {
                alert('Error: ' + (data.error || 'Prediction failed'));
            }
        } catch (err) {
            alert('Connection error. Make sure the server is running.');
            console.error(err);
        } finally {
            btnText.style.display = 'inline-flex';
            btnLoading.style.display = 'none';
            btnPredict.disabled = false;
        }
    });

    // ── Display Results ─────────────────────────────────────────

    function showResults(data) {
        const pred = data.prediction;
        const probs = data.all_probabilities;

        // Scroll to results
        resultsSection.style.display = 'block';

        // Main result
        const resultMain = document.getElementById('result-main');
        const resultEmoji = document.getElementById('result-emoji');
        const resultName = document.getElementById('result-name');
        const resultDesc = document.getElementById('result-description');

        resultEmoji.textContent = pred.emoji;
        resultName.textContent = pred.name;
        resultDesc.textContent = pred.description;

        // Style based on healthy/pest
        if (pred.is_healthy) {
            resultMain.classList.remove('is-pest');
        } else {
            resultMain.classList.add('is-pest');
        }

        // Confidence
        const confValue = document.getElementById('confidence-value');
        const confFill = document.getElementById('confidence-fill');
        const confPercent = (pred.confidence * 100).toFixed(1);
        confValue.textContent = confPercent + '%';

        // Animate confidence bar
        confFill.style.width = '0%';
        confFill.classList.toggle('danger', !pred.is_healthy);
        setTimeout(() => {
            confFill.style.width = confPercent + '%';
        }, 100);

        // Probabilities
        const probsContainer = document.getElementById('probabilities');
        probsContainer.innerHTML = '';

        const classNames = Object.keys(probs);
        classNames.forEach((cls) => {
            const prob = probs[cls];
            const pct = (prob.probability * 100).toFixed(1);
            const row = document.createElement('div');
            row.className = 'prob-row';
            row.innerHTML = `
                <span class="prob-label">${prob.info}</span>
                <div class="prob-bar-wrap">
                    <div class="prob-bar-fill" style="width: 0%"></div>
                </div>
                <span class="prob-value">${pct}%</span>
            `;
            probsContainer.appendChild(row);

            // Animate
            setTimeout(() => {
                row.querySelector('.prob-bar-fill').style.width = pct + '%';
            }, 200);
        });

        // Recommendation
        const recCard = document.getElementById('recommendation-card');
        const recText = document.getElementById('recommendation-text');
        recText.textContent = pred.recommendation;
        recCard.classList.toggle('is-pest', !pred.is_healthy);

        // Demo badge
        const demoBadge = document.getElementById('demo-badge');
        demoBadge.style.display = data.is_demo_mode ? 'block' : 'none';

        // Scroll to results
        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
    }

    // ── New Analysis ────────────────────────────────────────────

    btnNew.addEventListener('click', () => {
        resetUpload();
        resultsSection.style.display = 'none';
        document.getElementById('upload-section').scrollIntoView({
            behavior: 'smooth',
            block: 'center',
        });
    });

    // ── Sample Images ───────────────────────────────────────────

    async function loadSampleImages() {
        try {
            const response = await fetch('/api/samples');
            const data = await response.json();

            if (data.samples && data.samples.length > 0) {
                samplesGrid.innerHTML = '';
                data.samples.forEach((sample) => {
                    const card = document.createElement('div');
                    card.className = 'sample-card';
                    card.innerHTML = `
                        <img src="${sample.url}" alt="${sample.name}" loading="lazy">
                        <div class="sample-label">${sample.name}</div>
                    `;
                    card.addEventListener('click', () => loadSampleImage(sample));
                    samplesGrid.appendChild(card);
                });
            } else {
                // No sample images, hide section
                document.getElementById('samples-section').style.display = 'none';
            }
        } catch (err) {
            document.getElementById('samples-section').style.display = 'none';
        }
    }

    async function loadSampleImage(sample) {
        try {
            const response = await fetch(sample.url);
            const blob = await response.blob();
            const file = new File([blob], sample.filename, { type: blob.type });
            handleFile(file);
        } catch (err) {
            console.error('Error loading sample:', err);
        }
    }
});
