// Global Variables
const API_URL = window.location.origin;
let activeTab = 'live-tab';
let currentLandmarks = null;
let isCameraActive = true;
let isPredicting = false;
let recordedFrames = [];
let lastFpsUpdate = Date.now();
let framesCount = 0;

// Speech and Sentence Builder State
let isSpeechEnabled = true;
let sentenceWords = [];
let lastTopPrediction = "";
let predictionHeldCount = 0;
const HOLD_THRESHOLD = 10; // Consecutive frames to register pose as a word in sentence

// 3D Skeleton Rotator State
let rotateX = -0.3; // radians
let rotateY = 0.5;  // radians
let isRotating = false;
let startMouseX = 0;
let startMouseY = 0;
const rotatorCanvas = document.getElementById('rotator-canvas');
const rotatorCtx = rotatorCanvas.getContext('2d');

// Chart Instances
let accuracyChart = null;
let latencyChart = null;
let lossChart = null;

// DOM Elements
const navButtons = document.querySelectorAll('.nav-btn');
const tabContents = document.querySelectorAll('.tab-content');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const globalCultureSelect = document.getElementById('global-culture');
const videoElement = document.getElementById('webcam');
const canvasElement = document.getElementById('hand-canvas');
const canvasCtx = canvasElement.getContext('2d');
const detectionMsg = document.getElementById('detection-msg');
const webcamFpsEl = document.getElementById('webcam-fps');
const apiLatencyEl = document.getElementById('api-latency');
const toggleCameraBtn = document.getElementById('toggle-camera');

// Prediction UI Elements
const topPredictionText = document.getElementById('top-prediction-text');
const topCultureBadge = document.getElementById('top-culture-badge');
const topConfidenceBadge = document.getElementById('top-confidence-badge');
const rulePred = document.getElementById('rule-pred');
const ruleLat = document.getElementById('rule-lat');
const ruleConfBar = document.getElementById('rule-conf-bar');
const ruleConfText = document.getElementById('rule-conf-text');
const gnnPred = document.getElementById('gnn-pred');
const gnnLat = document.getElementById('gnn-lat');
const gnnConfBar = document.getElementById('gnn-conf-bar');
const gnnConfText = document.getElementById('gnn-conf-text');
const mlpPred = document.getElementById('mlp-pred');
const mlpLat = document.getElementById('mlp-lat');
const mlpConfBar = document.getElementById('mlp-conf-bar');
const mlpConfText = document.getElementById('mlp-conf-text');
const rfPred = document.getElementById('rf-pred');
const rfLat = document.getElementById('rf-lat');
const rfConfBar = document.getElementById('rf-conf-bar');
const rfConfText = document.getElementById('rf-conf-text');

// Sentence Builder DOM Hooks
const toggleSpeechBtn = document.getElementById('toggle-speech');
const sentenceTextEl = document.getElementById('sentence-text');
const btnSpeakAll = document.getElementById('btn-speak-all');
const btnCopySentence = document.getElementById('btn-copy-sentence');
const btnBackspace = document.getElementById('btn-backspace');
const btnClearSentence = document.getElementById('btn-clear-sentence');
const historyListEl = document.getElementById('history-list');

// Recorder Elements
const recordCultureSelect = document.getElementById('record-culture');
const recordLabelInput = document.getElementById('record-label');
const btnCaptureFrame = document.getElementById('btn-capture-frame');
const btnSaveGesture = document.getElementById('btn-save-gesture');
const btnClearRecorded = document.getElementById('btn-clear-recorded');
const recordedCountEl = document.getElementById('recorded-count');
const recordStatusMsg = document.getElementById('record-status-msg');

// Benchmarks Elements
const btnRetrain = document.getElementById('btn-retrain');
const trainingModal = document.getElementById('training-modal');

// Resize Canvases
function resizeCanvases() {
    canvasElement.width = videoElement.videoWidth || 640;
    canvasElement.height = videoElement.videoHeight || 480;
    
    // Resize rotator canvas
    const rect = rotatorCanvas.parentNode.getBoundingClientRect();
    rotatorCanvas.width = rect.width;
    rotatorCanvas.height = Math.max(rect.height, 220);
}
videoElement.addEventListener('loadedmetadata', resizeCanvases);
window.addEventListener('resize', resizeCanvases);
setTimeout(resizeCanvases, 500);

// 1. Navigation / Tab Switching
navButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        navButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        activeTab = btn.getAttribute('data-target');
        tabContents.forEach(tab => {
            tab.classList.remove('active');
            if (tab.id === activeTab) {
                tab.classList.add('active');
            }
        });
        
        if (activeTab === 'metrics-tab') {
            fetchMetrics();
        } else if (activeTab === 'record-tab') {
            fetchDatasetStats();
        }
    });
});

// 2. Connection Health Check
async function checkBackendHealth() {
    try {
        const response = await fetch(`${API_URL}/api/metrics`);
        if (response.ok) {
            const data = await response.json();
            if (data.is_training_active) {
                setBackendStatus('training', 'Backend Training Models...');
                trainingModal.classList.add('active');
                pollTrainingStatus();
            } else {
                setBackendStatus('online', 'Connected to Backend');
            }
        } else {
            setBackendStatus('offline', 'Backend Error');
        }
    } catch (err) {
        setBackendStatus('offline', 'Backend Offline');
    }
}

function setBackendStatus(status, text) {
    statusDot.className = 'status-indicator ' + status;
    statusText.innerText = text;
}

// 3. MediaPipe Initialization
const hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});

hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});

hands.onResults(onHandResults);

const camera = new Camera(videoElement, {
    onFrame: async () => {
        if (isCameraActive) {
            framesCount++;
            const now = Date.now();
            if (now - lastFpsUpdate >= 1000) {
                webcamFpsEl.innerText = framesCount;
                framesCount = 0;
                lastFpsUpdate = now;
            }
            await hands.send({ image: videoElement });
        }
    },
    width: 640,
    height: 480
});

camera.start().catch(err => {
    console.error("Camera failed to start", err);
    detectionMsg.innerHTML = '<span style="color: #ff3366;"><i class="fa-solid fa-triangle-exclamation"></i> Camera Access Blocked</span>';
});

// Toggle Camera
toggleCameraBtn.addEventListener('click', () => {
    if (isCameraActive) {
        isCameraActive = false;
        videoElement.srcObject.getTracks().forEach(track => track.stop());
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        rotatorCtx.clearRect(0, 0, rotatorCanvas.width, rotatorCanvas.height);
        toggleCameraBtn.innerHTML = '<i class="fa-solid fa-video"></i> Start Camera';
        toggleCameraBtn.className = 'btn btn-primary';
        detectionMsg.style.opacity = '1';
        detectionMsg.innerHTML = 'Camera Stopped';
    } else {
        isCameraActive = true;
        camera.start();
        toggleCameraBtn.innerHTML = '<i class="fa-solid fa-video-slash"></i> Stop Camera';
        toggleCameraBtn.className = 'btn btn-secondary';
        detectionMsg.innerHTML = '<span class="pulse-dot"></span> Detecting Hand Landmarks...';
    }
});

// 4. Handle MediaPipe Results
function onHandResults(results) {
    resizeCanvases();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        detectionMsg.style.opacity = '0';
        const landmarks = results.multiHandLandmarks[0];
        currentLandmarks = landmarks;
        
        // Enable recorder button
        btnCaptureFrame.disabled = false;
        
        // Draw Skeleton representation on 2D Video overlay
        drawSkeleton(landmarks);
        
        // Render 3D rotatable skeleton
        render3DRotator(landmarks);
        
        // Process prediction if on live-tab
        if (activeTab === 'live-tab') {
            const aspect = (videoElement.videoWidth && videoElement.videoHeight) ? (videoElement.videoWidth / videoElement.videoHeight) : (640 / 480);
            const payload = landmarks.map(pt => [pt.x * aspect, pt.y, pt.z]);
            sendPrediction(payload);
        }
    } else {
        currentLandmarks = null;
        btnCaptureFrame.disabled = true;
        detectionMsg.style.opacity = '1';
        detectionMsg.innerHTML = '<span class="pulse-dot"></span> Center Hand in Screen';
        
        // Draw standard empty 3D rotator grid
        render3DGrid();
        
        if (activeTab === 'live-tab') {
            clearPredictionUI();
            predictionHeldCount = 0;
            lastTopPrediction = "";
        }
    }
}

// Draw hand skeleton lines on 2D Overlay
function drawSkeleton(landmarks) {
    canvasCtx.strokeStyle = '#00f2fe';
    canvasCtx.lineWidth = 4;
    canvasCtx.fillStyle = '#bf55ec';
    
    const connections = [
        [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
        [0, 5], [5, 6], [6, 7], [7, 8], // Index
        [0, 9], [9, 10], [10, 11], [11, 12], // Middle
        [0, 13], [13, 14], [14, 15], [15, 16], // Ring
        [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
        [5, 9], [9, 13], [13, 17] // Palm bottom
    ];
    
    // Draw bones
    connections.forEach(([u, v]) => {
        const ptA = landmarks[u];
        const ptB = landmarks[v];
        if (ptA && ptB) {
            canvasCtx.beginPath();
            canvasCtx.moveTo(ptA.x * canvasElement.width, ptA.y * canvasElement.height);
            canvasCtx.lineTo(ptB.x * canvasElement.width, ptB.y * canvasElement.height);
            canvasCtx.stroke();
        }
    });
    
    // Draw joints
    landmarks.forEach(pt => {
        canvasCtx.beginPath();
        canvasCtx.arc(pt.x * canvasElement.width, pt.y * canvasElement.height, 6, 0, 2 * Math.PI);
        canvasCtx.fill();
    });
}

// 5. Send Landmarks to Inference API
async function sendPrediction(landmarksPayload) {
    if (isPredicting) return;
    isPredicting = true;
    
    const startTime = performance.now();
    
    try {
        const selectedCulture = globalCultureSelect.value;
        const response = await fetch(`${API_URL}/api/predict?culture=${selectedCulture}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ landmarks: landmarksPayload })
        });
        
        if (response.ok) {
            const data = await response.json();
            const latency = Math.round(performance.now() - startTime);
            apiLatencyEl.innerText = `${latency}ms`;
            updatePredictionUI(data);
        }
    } catch (err) {
        console.error("Prediction request failed", err);
    } finally {
        isPredicting = false;
    }
}

function updatePredictionUI(preds) {
    const selectedCulture = globalCultureSelect.value;
    
    let topClass = null;
    let topConfidence = 0;
    let topCulture = null;
    let topLabel = null;
    let bestModel = 'rule';
    
    // Evaluate Rule, GNN, MLP and Random Forest predictions - prioritize Rule
    const models = ['rule', 'gnn', 'mlp', 'rf'];
    let candidates = [];
    
    models.forEach(m => {
        if (preds[m]) {
            const pred = preds[m];
            if (selectedCulture === 'ALL' || pred.culture === selectedCulture) {
                candidates.push({
                    model: m,
                    class: pred.class,
                    confidence: pred.confidence,
                    culture: pred.culture,
                    label: pred.label
                });
            }
        }
    });
    
    // Sort candidates: prioritize rule, then confidence
    candidates.sort((a, b) => {
        if (a.model === 'rule' && b.model !== 'rule') return -1;
        if (b.model === 'rule' && a.model !== 'rule') return 1;
        return b.confidence - a.confidence;
    });
    
    if (candidates.length > 0) {
        const top = candidates[0];
        topClass = top.class;
        topConfidence = top.confidence;
        topCulture = top.culture;
        topLabel = top.label;
        bestModel = top.model;
    }
    
    // Render top consensus prediction
    if (topConfidence > 0.20 && topLabel) {
        topPredictionText.innerText = topLabel;
        topCultureBadge.innerText = topCulture;
        topCultureBadge.style.display = 'inline-block';
        topConfidenceBadge.innerText = `${Math.round(topConfidence * 100)}% Match`;
        topConfidenceBadge.style.display = 'inline-block';
        
        // --- STABILIZE POSE AND TRIGGER TTS / SENTENCE BUILDER ---
        if (topLabel === lastTopPrediction) {
            predictionHeldCount++;
            if (predictionHeldCount === HOLD_THRESHOLD) {
                triggerSpeechAndSentence(topLabel, topCulture, topConfidence, bestModel);
            }
        } else {
            lastTopPrediction = topLabel;
            predictionHeldCount = 0;
        }
    } else {
        topPredictionText.innerText = "Analyzing Pose...";
        topCultureBadge.style.display = 'none';
        topConfidenceBadge.style.display = 'none';
        predictionHeldCount = 0;
        lastTopPrediction = "";
    }
    
    // Update Rule UI cards
    rulePred.innerText = preds.rule.label;
    rulePred.className = 'model-prediction rule-text';
    ruleLat.innerText = `${preds.rule.latency_ms.toFixed(1)} ms`;
    ruleConfBar.style.width = `${preds.rule.confidence * 100}%`;
    ruleConfText.innerText = `Confidence: ${Math.round(preds.rule.confidence * 100)}% (${preds.rule.culture})`;
    
    // Update GNN UI cards
    gnnPred.innerText = preds.gnn.label;
    gnnPred.className = 'model-prediction gnn-text';
    gnnLat.innerText = `${preds.gnn.latency_ms.toFixed(1)} ms`;
    gnnConfBar.style.width = `${preds.gnn.confidence * 100}%`;
    gnnConfText.innerText = `Confidence: ${Math.round(preds.gnn.confidence * 100)}% (${preds.gnn.culture})`;
    
    // Update MLP UI cards
    mlpPred.innerText = preds.mlp.label;
    mlpPred.className = 'model-prediction mlp-text';
    mlpLat.innerText = `${preds.mlp.latency_ms.toFixed(1)} ms`;
    mlpConfBar.style.width = `${preds.mlp.confidence * 100}%`;
    mlpConfText.innerText = `Confidence: ${Math.round(preds.mlp.confidence * 100)}% (${preds.mlp.culture})`;
    
    // Update RF UI cards
    rfPred.innerText = preds.rf.label;
    rfPred.className = 'model-prediction rf-text';
    rfLat.innerText = `${preds.rf.latency_ms.toFixed(1)} ms`;
    rfConfBar.style.width = `${preds.rf.confidence * 100}%`;
    rfConfText.innerText = `Confidence: ${Math.round(preds.rf.confidence * 100)}% (${preds.rf.culture})`;
}

function clearPredictionUI() {
    topPredictionText.innerText = "No Hand Detected";
    topCultureBadge.style.display = 'none';
    topConfidenceBadge.style.display = 'none';
    
    const elements = [rulePred, gnnPred, mlpPred, rfPred];
    elements.forEach(el => {
        if (el) { el.innerText = '---'; el.className = 'model-prediction'; }
    });
    
    const latencies = [ruleLat, gnnLat, mlpLat, rfLat];
    latencies.forEach(el => { if (el) el.innerText = '--- ms'; });
    
    const bars = [ruleConfBar, gnnConfBar, mlpConfBar, rfConfBar];
    bars.forEach(bar => { if (bar) bar.style.width = '0%'; });
    
    const texts = [ruleConfText, gnnConfText, mlpConfText, rfConfText];
    texts.forEach(text => { if (text) text.innerText = 'Confidence: 0%'; });
    
    apiLatencyEl.innerText = '0ms';
}

// 6. Sentence Builder & Speech Engine
toggleSpeechBtn.addEventListener('click', () => {
    isSpeechEnabled = !isSpeechEnabled;
    if (isSpeechEnabled) {
        toggleSpeechBtn.classList.add('active');
        toggleSpeechBtn.innerHTML = '<i class="fa-solid fa-volume-high"></i> TTS: On';
    } else {
        toggleSpeechBtn.classList.remove('active');
        toggleSpeechBtn.innerHTML = '<i class="fa-solid fa-volume-xmark"></i> TTS: Off';
    }
});

function speakWord(text) {
    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel(); // Terminate any active speaking immediately
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        window.speechSynthesis.speak(utterance);
    }
}

function triggerSpeechAndSentence(word, culture, confidence, model) {
    // 1. Text-to-Speech
    if (isSpeechEnabled) {
        speakWord(word);
    }
    
    // 2. Add to sentence builder
    sentenceWords.push(word);
    
    // Update display box
    sentenceTextEl.innerHTML = '';
    sentenceWords.forEach(w => {
        const span = document.createElement('span');
        span.className = 'sentence-word';
        span.innerHTML = `${w} <i class="fa-solid fa-circle-xmark font-sm" style="cursor: pointer; opacity: 0.7;"></i>`;
        
        // Remove word on click
        span.querySelector('i').addEventListener('click', (e) => {
            e.stopPropagation();
            const idx = sentenceWords.indexOf(w);
            if (idx > -1) {
                sentenceWords.splice(idx, 1);
                triggerSentenceUpdate();
            }
        });
        sentenceTextEl.appendChild(span);
    });
    
    // Enable buttons
    btnSpeakAll.disabled = false;
    btnCopySentence.disabled = false;
    btnBackspace.disabled = false;
    btnClearSentence.disabled = false;
    
    // 3. Append to prediction log list
    appendActivityLog(word, culture, confidence, model);
}

function triggerSentenceUpdate() {
    if (sentenceWords.length === 0) {
        sentenceTextEl.innerHTML = '<div class="sentence-placeholder">Hold a pose for 1.5 seconds to build a phrase...</div>';
        btnSpeakAll.disabled = true;
        btnCopySentence.disabled = true;
        btnBackspace.disabled = true;
        btnClearSentence.disabled = true;
    } else {
        sentenceTextEl.innerHTML = '';
        sentenceWords.forEach(w => {
            const span = document.createElement('span');
            span.className = 'sentence-word';
            span.innerHTML = `${w} <i class="fa-solid fa-circle-xmark font-sm" style="cursor: pointer; opacity: 0.7;"></i>`;
            span.querySelector('i').addEventListener('click', (e) => {
                e.stopPropagation();
                const idx = sentenceWords.indexOf(w);
                if (idx > -1) {
                    sentenceWords.splice(idx, 1);
                    triggerSentenceUpdate();
                }
            });
            sentenceTextEl.appendChild(span);
        });
        btnSpeakAll.disabled = false;
        btnCopySentence.disabled = false;
        btnBackspace.disabled = false;
        btnClearSentence.disabled = false;
    }
}

btnSpeakAll.addEventListener('click', () => {
    if (sentenceWords.length > 0) {
        speakWord(sentenceWords.join(' '));
    }
});

btnCopySentence.addEventListener('click', () => {
    if (sentenceWords.length > 0) {
        const text = sentenceWords.join(' ');
        navigator.clipboard.writeText(text).then(() => {
            showToast("Sentence copied to clipboard!", "success");
            const origHTML = btnCopySentence.innerHTML;
            btnCopySentence.innerHTML = '<i class="fa-solid fa-check"></i> Copied!';
            btnCopySentence.classList.remove('btn-secondary');
            btnCopySentence.classList.add('btn-success');
            setTimeout(() => {
                btnCopySentence.innerHTML = origHTML;
                btnCopySentence.classList.remove('btn-success');
                btnCopySentence.classList.add('btn-secondary');
            }, 2000);
        }).catch(err => {
            console.error('Failed to copy text: ', err);
            showToast("Failed to copy sentence.", "error");
        });
    }
});

btnBackspace.addEventListener('click', () => {
    if (sentenceWords.length > 0) {
        sentenceWords.pop();
        triggerSentenceUpdate();
    }
});

btnClearSentence.addEventListener('click', () => {
    sentenceWords = [];
    triggerSentenceUpdate();
});

function appendActivityLog(word, culture, confidence, model) {
    // Remove empty item
    const emptyItem = historyListEl.querySelector('.empty');
    if (emptyItem) emptyItem.remove();
    
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    
    const item = document.createElement('div');
    item.className = 'history-item';
    
    const badgeColor = model === 'gnn' ? 'gnn-color' : (model === 'mlp' ? 'mlp-color' : 'rf-color');
    
    item.innerHTML = `
        <div class="history-item-details">
            <span class="history-time">${time}</span>
            <strong>${word}</strong>
            <span class="history-badge ${badgeColor}">${model.toUpperCase()}</span>
            <span style="color: var(--text-secondary); font-size: 0.75rem;">(${culture})</span>
        </div>
        <span class="confidence-val" style="color: var(--color-success); font-weight: 600;">
            ${Math.round(confidence * 100)}% Match
        </span>
    `;
    
    // Insert at top of log
    historyListEl.insertBefore(item, historyListEl.firstChild);
    
    // Prune history
    if (historyListEl.children.length > 20) {
        historyListEl.lastChild.remove();
    }
}

// 7. Interactive 3D Skeleton Canvas visualizer
// Drag-to-orbit handlers
rotatorCanvas.addEventListener('mousedown', (e) => {
    isRotating = true;
    startMouseX = e.clientX;
    startMouseY = e.clientY;
});

window.addEventListener('mousemove', (e) => {
    if (!isRotating) return;
    const deltaX = e.clientX - startMouseX;
    const deltaY = e.clientY - startMouseY;
    
    rotateY += deltaX * 0.008;
    rotateX += deltaY * 0.008;
    
    startMouseX = e.clientX;
    startMouseY = e.clientY;
    
    if (currentLandmarks) {
        render3DRotator(currentLandmarks);
    } else {
        render3DGrid();
    }
});

window.addEventListener('mouseup', () => {
    isRotating = false;
});

// Touch support for mobiles/trackpads
rotatorCanvas.addEventListener('touchstart', (e) => {
    if (e.touches.length === 1) {
        isRotating = true;
        startMouseX = e.touches[0].clientX;
        startMouseY = e.touches[0].clientY;
    }
});

window.addEventListener('touchmove', (e) => {
    if (!isRotating || e.touches.length !== 1) return;
    const deltaX = e.touches[0].clientX - startMouseX;
    const deltaY = e.touches[0].clientY - startMouseY;
    
    rotateY += deltaX * 0.008;
    rotateX += deltaY * 0.008;
    
    startMouseX = e.touches[0].clientX;
    startMouseY = e.touches[0].clientY;
    
    if (currentLandmarks) {
        render3DRotator(currentLandmarks);
    } else {
        render3DGrid();
    }
});

window.addEventListener('touchend', () => {
    isRotating = false;
});

function render3DGrid() {
    rotatorCtx.clearRect(0, 0, rotatorCanvas.width, rotatorCanvas.height);
    
    const cX = rotatorCanvas.width / 2;
    const cY = rotatorCanvas.height / 2;
    
    // Draw wireframe grid plane at bottom (z = 0.5)
    rotatorCtx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
    rotatorCtx.lineWidth = 1;
    
    const size = 120;
    const divisions = 6;
    const step = (size * 2) / divisions;
    
    for (let i = 0; i <= divisions; i++) {
        // Draw grid lines
        const offset = -size + i * step;
        
        // Lines parallel to X axis
        draw3DLine(offset, 120, -size, offset, 120, size, cX, cY);
        // Lines parallel to Z axis
        draw3DLine(-size, 120, offset, size, 120, offset, cX, cY);
    }
    
    rotatorCtx.fillStyle = 'rgba(255, 255, 255, 0.25)';
    rotatorCtx.font = '10px Outfit';
    rotatorCtx.textAlign = 'center';
    rotatorCtx.fillText("Waiting for active skeleton...", cX, cY);
}

function draw3DLine(x1, y1, z1, x2, y2, z2, cX, cY) {
    const pt1 = project3D(x1, y1, z1, cX, cY);
    const pt2 = project3D(x2, y2, z2, cX, cY);
    
    rotatorCtx.beginPath();
    rotatorCtx.moveTo(pt1.x, pt1.y);
    rotatorCtx.lineTo(pt2.x, pt2.y);
    rotatorCtx.stroke();
}

function project3D(x, y, z, cX, cY) {
    // 3D rotation around Y axis (orbit left/right)
    let x1 = x * Math.cos(rotateY) + z * Math.sin(rotateY);
    let z1 = -x * Math.sin(rotateY) + z * Math.cos(rotateY);
    
    // 3D rotation around X axis (orbit up/down)
    let y2 = y * Math.cos(rotateX) - z1 * Math.sin(rotateX);
    
    // Scale and center projection
    return {
        x: cX + x1,
        y: cY + y2
    };
}

function render3DRotator(landmarks) {
    rotatorCtx.clearRect(0, 0, rotatorCanvas.width, rotatorCanvas.height);
    
    const cX = rotatorCanvas.width / 2;
    const cY = rotatorCanvas.height / 2;
    
    // Calculate bounding box and shift origin to hand wrist center (landmark 0)
    const wrist = landmarks[0];
    
    // Scale factor to map [0..1] range coordinates into rotator box pixels
    const scaleFactor = Math.min(rotatorCanvas.width, rotatorCanvas.height) * 0.7;
    
    // Convert landmarks to relative coordinates, scale them, and center them
    const pts = landmarks.map(pt => {
        const x_rel = (pt.x - wrist.x) * scaleFactor;
        const y_rel = (pt.y - wrist.y) * scaleFactor;
        const z_rel = (pt.z - wrist.z) * scaleFactor;
        
        return project3D(x_rel, y_rel, z_rel, cX, cY);
    });
    
    // Draw 3D bones skeleton
    const connections = [
        [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
        [0, 5], [5, 6], [6, 7], [7, 8], // Index
        [0, 9], [9, 10], [10, 11], [11, 12], // Middle
        [0, 13], [13, 14], [14, 15], [15, 16], // Ring
        [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
        [5, 9], [9, 13], [13, 17] // Knuckles
    ];
    
    // Glow path
    rotatorCtx.strokeStyle = 'rgba(0, 242, 254, 0.45)';
    rotatorCtx.lineWidth = 6;
    connections.forEach(([u, v]) => {
        const ptA = pts[u];
        const ptB = pts[v];
        if (ptA && ptB) {
            rotatorCtx.beginPath();
            rotatorCtx.moveTo(ptA.x, ptA.y);
            rotatorCtx.lineTo(ptB.x, ptB.y);
            rotatorCtx.stroke();
        }
    });
    
    // Core skeleton path
    rotatorCtx.strokeStyle = '#00f2fe';
    rotatorCtx.lineWidth = 2.5;
    connections.forEach(([u, v]) => {
        const ptA = pts[u];
        const ptB = pts[v];
        if (ptA && ptB) {
            rotatorCtx.beginPath();
            rotatorCtx.moveTo(ptA.x, ptA.y);
            rotatorCtx.lineTo(ptB.x, ptB.y);
            rotatorCtx.stroke();
        }
    });
    
    // Draw joints
    pts.forEach((pt, idx) => {
        rotatorCtx.beginPath();
        // Give fingertips and wrist distinct colors
        if (idx === 0) {
            rotatorCtx.fillStyle = '#ff007b'; // Wrist is hot pink
            rotatorCtx.arc(pt.x, pt.y, 6, 0, 2 * Math.PI);
        } else if ([4, 8, 12, 16, 20].includes(idx)) {
            rotatorCtx.fillStyle = '#00ff87'; // Fingertips are green
            rotatorCtx.arc(pt.x, pt.y, 5, 0, 2 * Math.PI);
        } else {
            rotatorCtx.fillStyle = '#bf55ec'; // Sibling joints are purple
            rotatorCtx.arc(pt.x, pt.y, 4, 0, 2 * Math.PI);
        }
        rotatorCtx.fill();
    });
}

// 8. Custom Gesture Recorder
btnCaptureFrame.addEventListener('click', () => {
    if (!currentLandmarks) return;
    
    const label = recordLabelInput.value.trim();
    if (!label) {
        showRecordStatus('Please enter a gesture sign label first.', 'error');
        return;
    }
    
    // Extract landmarks payload corrected for aspect ratio
    const aspect = (videoElement.videoWidth && videoElement.videoHeight) ? (videoElement.videoWidth / videoElement.videoHeight) : (640 / 480);
    const payload = currentLandmarks.map(pt => [pt.x * aspect, pt.y, pt.z]);
    recordedFrames.push(payload);
    
    recordedCountEl.innerText = recordedFrames.length;
    btnSaveGesture.disabled = false;
    btnClearRecorded.style.display = 'inline-flex';
    
    showRecordStatus(`Frame ${recordedFrames.length} captured successfully!`, 'success');
});

btnClearRecorded.addEventListener('click', () => {
    recordedFrames = [];
    recordedCountEl.innerText = '0';
    btnSaveGesture.disabled = true;
    btnClearRecorded.style.display = 'none';
    showRecordStatus('Recording reset.', 'success');
});

btnSaveGesture.addEventListener('click', async () => {
    const culture = recordCultureSelect.value;
    const label = recordLabelInput.value.trim();
    
    if (recordedFrames.length === 0 || !label) return;
    
    btnSaveGesture.disabled = true;
    btnCaptureFrame.disabled = true;
    showRecordStatus(`Saving ${recordedFrames.length} frames to dataset...`, 'success');
    
    let savedCount = 0;
    
    for (let frame of recordedFrames) {
        try {
            const response = await fetch(`${API_URL}/api/save_gesture`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    culture: culture,
                    label: label,
                    landmarks: frame
                })
            });
            if (response.ok) {
                savedCount++;
            }
        } catch (err) {
            console.error("Error saving frame", err);
        }
    }
    
    showRecordStatus(`Successfully saved ${savedCount}/${recordedFrames.length} frames for '${culture}_${label}'!`, 'success');
    fetchDatasetStats();
    
    // Clear list
    recordedFrames = [];
    recordedCountEl.innerText = '0';
    btnClearRecorded.style.display = 'none';
    btnCaptureFrame.disabled = false;
});

function showRecordStatus(msg, type) {
    recordStatusMsg.className = `status-msg active ${type}`;
    recordStatusMsg.innerText = msg;
    
    setTimeout(() => {
        recordStatusMsg.classList.remove('active');
    }, 4000);
}

// 9. Retrain & Benchmarking Metrics
btnRetrain.addEventListener('click', async () => {
    if (confirm("Are you sure you want to retrain all models on the updated dataset? This will take a few seconds.")) {
        try {
            const response = await fetch(`${API_URL}/api/train`, { method: 'POST' });
            if (response.ok) {
                showToast("Model retraining started in the background...", "info");
                setBackendStatus('training', 'Backend Training Models...');
                trainingModal.classList.add('active');
                pollTrainingStatus();
            } else {
                showToast("Failed to initiate retraining.", "error");
            }
        } catch (err) {
            console.error("Failed to start training", err);
            showToast("Failed to retrain models due to connection error.", "error");
        }
    }
});

function pollTrainingStatus() {
    const pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_URL}/api/metrics`);
            if (response.ok) {
                const data = await response.json();
                if (!data.is_training_active) {
                    clearInterval(pollInterval);
                    trainingModal.classList.remove('active');
                    setBackendStatus('online', 'Connected to Backend');
                    showToast("Models retrained successfully! Charts and metrics updated.", "success");
                    fetchMetrics(); // Refresh charts with new metrics!
                }
            }
        } catch (err) {
            console.error("Error polling metrics status", err);
        }
    }, 2000);
}

async function fetchMetrics() {
    try {
        const response = await fetch(`${API_URL}/api/metrics`);
        if (response.ok) {
            const data = await response.json();
            if (data.trained) {
                updateMetricsUI(data);
                renderCharts(data);
            }
        }
        fetchTrainingHistory();
    } catch (err) {
        console.error("Failed to fetch metrics", err);
    }
}

async function fetchTrainingHistory() {
    try {
        const response = await fetch(`${API_URL}/api/training_history`);
        if (response.ok) {
            const history = await response.json();
            renderTrainingHistoryTable(history);
        }
    } catch (err) {
        console.error("Failed to fetch training history", err);
    }
}

function renderTrainingHistoryTable(history) {
    const tbody = document.getElementById('history-runs-tbody');
    const clearBtn = document.getElementById('btn-clear-history');
    if (!tbody) return;
    
    if (clearBtn) {
        clearBtn.style.display = (history && history.length > 0) ? 'inline-flex' : 'none';
    }
    
    if (!history || history.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="4" style="text-align: center; color: var(--text-secondary);">No history log loaded yet.</td>
            </tr>
        `;
        return;
    }
    
    // Sort reverse chronological
    const sortedHistory = [...history].reverse();
    
    tbody.innerHTML = sortedHistory.map(run => `
        <tr>
            <td>${run.timestamp}</td>
            <td class="gnn-text" style="font-weight: 600;">${(run.gnn_val_acc * 100).toFixed(1)}%</td>
            <td class="mlp-text" style="font-weight: 600;">${(run.mlp_val_acc * 100).toFixed(1)}%</td>
            <td class="rf-text" style="font-weight: 600;">${(run.rf_val_acc * 100).toFixed(1)}%</td>
        </tr>
    `).join('');
}

// Bind Clear History action
const btnClearHistory = document.getElementById('btn-clear-history');
if (btnClearHistory) {
    btnClearHistory.addEventListener('click', async () => {
        if (confirm("Are you sure you want to clear all historical training records?")) {
            try {
                const response = await fetch(`${API_URL}/api/clear_training_history`, { method: 'POST' });
                if (response.ok) {
                    showToast("Historical training runs log cleared.", "success");
                    fetchTrainingHistory();
                } else {
                    showToast("Failed to clear history log.", "error");
                }
            } catch (err) {
                console.error("Failed to clear training history", err);
                showToast("Failed to clear history log due to network error.", "error");
            }
        }
    });
}

function updateMetricsUI(metrics) {
    // Fill text summaries
    document.getElementById('table-gnn-acc').innerText = `${(metrics.gnn.val_acc * 100).toFixed(1)}%`;
    document.getElementById('table-gnn-time').innerText = `${metrics.gnn.train_time_sec.toFixed(2)}s`;
    document.getElementById('table-gnn-lat').innerText = `${metrics.gnn.inference_latency_ms.toFixed(2)}ms`;
    
    document.getElementById('table-mlp-acc').innerText = `${(metrics.mlp.val_acc * 100).toFixed(1)}%`;
    document.getElementById('table-mlp-time').innerText = `${metrics.mlp.train_time_sec.toFixed(2)}s`;
    document.getElementById('table-mlp-lat').innerText = `${metrics.mlp.inference_latency_ms.toFixed(2)}ms`;
    
    document.getElementById('table-rf-acc').innerText = `${(metrics.rf.val_acc * 100).toFixed(1)}%`;
    document.getElementById('table-rf-time').innerText = `${metrics.rf.train_time_sec.toFixed(2)}s`;
    document.getElementById('table-rf-lat').innerText = `${metrics.rf.inference_latency_ms.toFixed(2)}ms`;
}

// Render Comparison Charts using Chart.js
function renderCharts(metrics) {
    const chartOptions = {
        responsive: true,
        plugins: {
            legend: { display: false }
        },
        scales: {
            y: {
                grid: { color: 'rgba(255,255,255,0.05)' },
                ticks: { color: '#9090a2' }
            },
            x: {
                grid: { display: false },
                ticks: { color: '#9090a2' }
            }
        }
    };

    // 1. Accuracy Chart
    if (accuracyChart) accuracyChart.destroy();
    const accCtx = document.getElementById('accuracy-chart').getContext('2d');
    accuracyChart = new Chart(accCtx, {
        type: 'bar',
        data: {
            labels: ['GNN (Graph)', 'MLP (Dense)', 'Random Forest'],
            datasets: [{
                data: [
                    metrics.gnn.val_acc * 100,
                    metrics.mlp.val_acc * 100,
                    metrics.rf.val_acc * 100
                ],
                backgroundColor: ['#00f2fe', '#bf55ec', '#ff7b00'],
                borderColor: ['rgba(0,242,254,0.3)', 'rgba(191,85,236,0.3)', 'rgba(255,123,0,0.3)'],
                borderWidth: 1,
                borderRadius: 6
            }]
        },
        options: {
            ...chartOptions,
            scales: {
                ...chartOptions.scales,
                y: {
                    ...chartOptions.scales.y,
                    min: 0,
                    max: 100
                }
            }
        }
    });

    // 2. Latency Chart
    if (latencyChart) latencyChart.destroy();
    const latCtx = document.getElementById('latency-chart').getContext('2d');
    latencyChart = new Chart(latCtx, {
        type: 'bar',
        data: {
            labels: ['GNN (Graph)', 'MLP (Dense)', 'Random Forest'],
            datasets: [{
                data: [
                    metrics.gnn.inference_latency_ms,
                    metrics.mlp.inference_latency_ms,
                    metrics.rf.inference_latency_ms
                ],
                backgroundColor: ['#00f2fe', '#bf55ec', '#ff7b00'],
                borderRadius: 6
            }]
        },
        options: chartOptions
    });

    // 3. Loss History Chart (PyTorch curves)
    if (lossChart) lossChart.destroy();
    const lossCtx = document.getElementById('loss-history-chart').getContext('2d');
    
    const epochs = Array.from({ length: metrics.gnn.history.train_loss.length }, (_, i) => i + 1);
    
    lossChart = new Chart(lossCtx, {
        type: 'line',
        data: {
            labels: epochs,
            datasets: [
                {
                    label: 'GNN Train Loss',
                    data: metrics.gnn.history.train_loss,
                    borderColor: '#00f2fe',
                    backgroundColor: 'rgba(0,242,254,0.05)',
                    fill: true,
                    tension: 0.2
                },
                {
                    label: 'GNN Val Loss',
                    data: metrics.gnn.history.val_loss,
                    borderColor: '#0072ff',
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.2
                },
                {
                    label: 'MLP Train Loss',
                    data: metrics.mlp.history.train_loss,
                    borderColor: '#bf55ec',
                    backgroundColor: 'rgba(191,85,236,0.05)',
                    fill: true,
                    tension: 0.2
                },
                {
                    label: 'MLP Val Loss',
                    data: metrics.mlp.history.val_loss,
                    borderColor: '#e040fb',
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.2
                }
            ]
        },
        options: {
            ...chartOptions,
            plugins: {
                legend: {
                    display: true,
                    labels: { color: '#9090a2' }
                }
            }
        }
    });
}

// Keyboard Shortcuts Listener
window.addEventListener('keydown', (e) => {
    // Ignore keyboard shortcuts if the user is currently typing in an input or select
    if (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'SELECT') {
        return;
    }
    
    const key = e.key.toLowerCase();
    if (key === 'k') {
        e.preventDefault();
        toggleCameraBtn.click();
    } else if (key === 's') {
        e.preventDefault();
        toggleSpeechBtn.click();
    } else if (key === 'c') {
        e.preventDefault();
        btnCopySentence.click();
    } else if (key === 'r') {
        e.preventDefault();
        btnClearSentence.click();
    } else if (key === 'backspace' || key === 'b') {
        e.preventDefault();
        btnBackspace.click();
    }
});

// Signs Guide Filter and Search Logic
const signsSearchInput = document.getElementById('signs-search');
const signsFilterButtons = document.querySelectorAll('#signs-culture-filters button');
const signsCultureBlocks = document.querySelectorAll('.signs-culture-block');

function filterSignsGuide() {
    const query = (signsSearchInput?.value || '').toLowerCase().trim();
    const activeBtn = document.querySelector('#signs-culture-filters button.active');
    const activeFilter = activeBtn ? activeBtn.getAttribute('data-filter') : 'ALL';
    
    signsCultureBlocks.forEach(block => {
        const blockCulture = block.getAttribute('data-culture');
        const matchesCulture = (activeFilter === 'ALL' || blockCulture === activeFilter);
        
        let visibleCardsInBlock = 0;
        const cards = block.querySelectorAll('.sign-card');
        
        cards.forEach(card => {
            const name = card.querySelector('strong').innerText.toLowerCase();
            const hint = card.querySelector('.sign-hint').innerText.toLowerCase();
            const matchesSearch = (name.includes(query) || hint.includes(query));
            
            if (matchesCulture && matchesSearch) {
                card.style.display = 'flex';
                visibleCardsInBlock++;
            } else {
                card.style.display = 'none';
            }
        });
        
        if (visibleCardsInBlock > 0) {
            block.style.display = 'block';
        } else {
            block.style.display = 'none';
        }
    });
}

if (signsSearchInput) {
    signsSearchInput.addEventListener('input', filterSignsGuide);
}

signsFilterButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        signsFilterButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        filterSignsGuide();
    });
});

// Fetch and render Dataset Stats for the recorder tab
async function fetchDatasetStats() {
    const totalEl = document.getElementById('stats-total-samples');
    const container = document.getElementById('dataset-stats-container');
    if (!container) return;
    
    try {
        const response = await fetch(`${API_URL}/api/dataset_stats`);
        if (response.ok) {
            const data = await response.json();
            if (totalEl) {
                totalEl.innerText = `Total Samples: ${data.total_samples}`;
            }
            
            const cultures = data.cultures || {};
            const keys = Object.keys(cultures);
            
            if (keys.length === 0) {
                container.innerHTML = `<div style="color: var(--text-secondary); font-size: 0.8rem; text-align: center; padding: 1rem 0;">Dataset is currently empty.</div>`;
                return;
            }
            
            let html = '';
            keys.forEach(culture => {
                const labels = cultures[culture];
                const labelKeys = Object.keys(labels);
                
                html += `
                    <div style="margin-bottom: 0.5rem;">
                        <div style="font-size: 0.8rem; font-weight: 700; color: var(--text-secondary); text-transform: uppercase; margin-bottom: 0.35rem; border-bottom: 1px solid rgba(255,255,255,0.03); padding-bottom: 2px;">${culture}</div>
                        <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 0.4rem;">
                `;
                
                labelKeys.forEach(lbl => {
                    const count = labels[lbl];
                    html += `
                        <div style="background: rgba(255,255,255,0.02); border: 1px solid var(--border-color); border-radius: 8px; padding: 0.4rem 0.6rem; display: flex; justify-content: space-between; align-items: center; font-size: 0.75rem;">
                            <strong style="color: var(--text-primary); font-weight: 500;">${lbl}</strong>
                            <span class="badge" style="padding: 0.1rem 0.3rem; font-size: 0.65rem; background: rgba(0, 242, 254, 0.1); color: var(--color-gnn); border: none; min-width: 20px; text-align: center;">${count}</span>
                        </div>
                    `;
                });
                
                html += `
                        </div>
                    </div>
                `;
            });
            
            container.innerHTML = html;
        }
    } catch (err) {
        console.error("Failed to fetch dataset stats", err);
        container.innerHTML = `<div style="color: #ff3366; font-size: 0.8rem; text-align: center; padding: 1rem 0;">Error loading stats.</div>`;
    }
}

// Global Toast System
function showToast(message, type = 'info') {
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        document.body.appendChild(container);
    }
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    let iconClass = 'fa-circle-info';
    if (type === 'success') iconClass = 'fa-circle-check';
    else if (type === 'error') iconClass = 'fa-triangle-exclamation';
    else if (type === 'warning') iconClass = 'fa-circle-exclamation';
    
    toast.innerHTML = `
        <i class="fa-solid ${iconClass} toast-icon"></i>
        <div class="toast-message">${message}</div>
    `;
    
    container.appendChild(toast);
    
    // Trigger entrance transition
    setTimeout(() => {
        toast.classList.add('show');
    }, 10);
    
    // Auto dismiss after 3.5s
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => {
            toast.remove();
        }, 400);
    }, 3500);
}

// Initial checks
checkBackendHealth();
render3DGrid();
fetchDatasetStats();
// Periodically check status (every 10s)
setInterval(checkBackendHealth, 10000);
