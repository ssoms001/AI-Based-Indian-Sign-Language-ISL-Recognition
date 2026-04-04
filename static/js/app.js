/**
 * ISL Gesture Recognition System - Frontend JavaScript
 */

class ISLGestureApp {
    constructor() {
        this.currentSentence = '';
        this.currentGesture = '';
        this.currentConfidence = 0.0;
        this.handCount = 0;
        this.isProcessing = false;
        this.isSpeaking = false;
        this.fpsCounter = 0;
        this.performanceData = {};
        this.gestureThumbnails = [];

        // Client-side metrics tracking
        this._predictionCount = 0;
        this._predictionStartTime = performance.now();
        this._lastLatency = 0;
        this._totalCorrectPredictions = 0;
        this._totalPredictions = 0;

        // Initialize the application
        this.init();
    }

    /**
     * Initialize the application
     */
    init() {
        console.log('🚀 Initializing ISL Gesture Recognition System...');

        // Bind event listeners
        this.bindEventListeners();

        // Initialize client-side camera
        this.initCamera();

        // Update performance metrics periodically
        setInterval(() => this.updatePerformanceMetrics(), 1000);

        // Initialize UI components
        this.initializeUI();

        console.log('✅ Application initialized successfully');
    }

    /**
     * Bind event listeners to DOM elements
     */
    bindEventListeners() {
        // Speak button
        document.getElementById('speak-btn').addEventListener('click', () => {
            this.speakText();
        });

        // Clear button
        document.getElementById('clear-btn').addEventListener('click', () => {
            this.clearSentence();
        });

        // Backspace button
        document.getElementById('backspace-btn')?.addEventListener('click', () => {
            this.backspaceLastCharacter();
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (event) => {
            this.handleKeyboardShortcuts(event);
        });
    }

    /**
     * Initialize client-side camera via SignBridgeCamera
     */
    async initCamera() {
        if (typeof SignBridgeCamera === 'undefined') {
            console.error('camera.js not loaded');
            return;
        }

        this.camera = new SignBridgeCamera('#video-stream', {
            fps: 3,
            buildSentence: true,
            onPrediction: (data) => this.handlePrediction(data),
            onCameraReady: () => {
                const status = document.getElementById('camera-status');
                if (status) status.innerHTML = '<i class="fas fa-circle text-success" style="font-size:0.6em;"></i> Camera Active';
            },
            onError: (e) => {
                const status = document.getElementById('camera-status');
                if (status) status.innerHTML = '<i class="fas fa-circle text-danger" style="font-size:0.6em;"></i> Camera Error';
                console.error('Camera error:', e);
            }
        });

        // Build multi-camera dropdown
        await this.camera.buildCameraSelector('#camera-selector');

        // Start camera and prediction
        const started = await this.camera.start();
        if (started) {
            this.camera.startPredicting();
        }
    }

    /**
     * Handle prediction results from camera
     */
    handlePrediction(data) {
        if (!data || data.error) return;

        // Track client-side metrics
        this._predictionCount++;
        this._totalPredictions++;
        if (data.confidence && data.confidence > 0.6) this._totalCorrectPredictions++;
        if (data._latency) this._lastLatency = data._latency;

        // Update gesture display
        if (data.gesture && data.gesture !== this.currentGesture) {
            this.currentGesture = data.gesture;
            this.updateCurrentCharacter(data.gesture);
        }

        // Update confidence
        if (data.confidence !== undefined && data.confidence !== this.currentConfidence) {
            this.currentConfidence = data.confidence;
            this.updateConfidence(data.confidence);
        }

        // Update hand count
        if (data.hand_count !== undefined && data.hand_count !== this.handCount) {
            this.handCount = data.hand_count;
        }

        // Update sentence if server built one
        if (data.sentence !== undefined && data.sentence !== this.currentSentence) {
            this.currentSentence = data.sentence;
            this.updateSentenceDisplay(data.sentence);
            this.updateWordSuggestions(data.sentence);
        }
    }

    /**
     * Initialize UI components
     */
    initializeUI() {
        // Add loading states
        this.showLoadingState('Initializing camera...');

        // Hide loading after 3 seconds
        setTimeout(() => {
            this.hideLoadingState();
        }, 3000);

        // Initialize tooltips
        this.initializeTooltips();

        // Initialize gesture learning panel
        this.initializeGestureLearningPanel();
    }

    /**
     * Update gesture recognition data
     */
    async updateGestureData() {
        if (this.isProcessing) return;

        try {
            this.isProcessing = true;
            const response = await fetch('/api/gesture_data');
            const data = await response.json();

            if (data.error) {
                console.error('Error fetching gesture data:', data.error);
                return;
            }

            // Update current sentence if changed (backend should send accumulated text)
            if (data.current_sentence !== this.currentSentence) {
                this.currentSentence = data.current_sentence;
                this.updateSentenceDisplay(data.current_sentence);
                this.updateWordSuggestions(data.current_sentence);
                console.log('Sentence updated:', data.current_sentence); // Logging for debugging

                // Removed automatic speaking - now only speaks when speak button is pressed
            }

            // Update current gesture and confidence
            if (data.current_gesture !== this.currentGesture) {
                this.currentGesture = data.current_gesture;
                if (this.currentGesture) {
                    this.updateCurrentCharacter(this.currentGesture);
                }
            }

            if (data.confidence !== this.currentConfidence) {
                this.currentConfidence = data.confidence;
                this.updateConfidence(data.confidence);
            }

            // Update hand count
            if (data.hand_count !== this.handCount) {
                this.handCount = data.hand_count;
                this.updateHandCountDisplay(data.hand_count);
            }

        } catch (error) {
            console.error('Error updating gesture data:', error);
        } finally {
            this.isProcessing = false;
        }
    }

    /**
     * Update performance metrics (client-side computed)
     */
    updatePerformanceMetrics() {
        const now = performance.now();
        const elapsed = (now - this._predictionStartTime) / 1000;
        const fps = elapsed > 0 ? Math.round(this._predictionCount / elapsed) : 0;
        const accuracy = this._totalPredictions > 0
            ? this._totalCorrectPredictions / this._totalPredictions : 0;
        const latency = this._lastLatency || 0;

        // Reset FPS counter every 3 seconds
        if (elapsed > 3) {
            this._predictionCount = 0;
            this._predictionStartTime = now;
        }

        this.displayPerformanceMetrics({
            avg_fps: fps,
            accuracy: accuracy,
            latency: latency
        });
    }

    /**
     * Update FPS counter
     */
    updateFPSCounter() {
        const fpsElement = document.getElementById('fps-counter');
        if (this.performanceData.fps) {
            fpsElement.textContent = Math.round(this.performanceData.fps);
            fpsElement.className = this.performanceData.fps > 25 ? 'text-success' : 'text-warning';
        }
    }

    /**
     * Update sentence display
     */
    updateSentenceDisplay(sentence) {
        const display = document.getElementById('sentence-display');
        display.value = sentence || '';

        // Add animation
        display.classList.add('slide-up');
        setTimeout(() => {
            display.classList.remove('slide-up');
        }, 300);

        // Extract and display current character from last part of sentence
        if (sentence && sentence.length > 0) {
            const parts = sentence.trim().split(/\s+/);
            const lastPart = parts[parts.length - 1];
            if (lastPart.length > 0) {
                this.updateCurrentCharacter(lastPart.charAt(lastPart.length - 1).toUpperCase());
            }
        } else {
            this.updateCurrentCharacter('--');
        }
    }



    /**
     * Update current character display
     */
    updateCurrentCharacter(character) {
        const element = document.getElementById('current-character');
        element.textContent = character;
        element.classList.add('pulse');

        setTimeout(() => {
            element.classList.remove('pulse');
        }, 500);
    }

    /**
     * Update hand count display
     */
    updateHandCountDisplay(count) {
        // Update any hand count displays if needed
        console.log(`Hand count: ${count}`);
    }

    /**
     * Update confidence bar
     */
    updateConfidence(confidence) {
        const confidenceBar = document.getElementById('confidence-bar');
        const confidenceText = document.getElementById('confidence-text');

        const percentage = Math.round(confidence * 100);
        confidenceBar.style.width = percentage + '%';
        confidenceText.textContent = percentage + '%';

        // Change color based on confidence level
        confidenceBar.className = 'progress-bar ' +
            (percentage >= 80 ? 'bg-success' :
                percentage >= 60 ? 'bg-warning' : 'bg-danger');

        // Add animation for high confidence
        if (percentage >= 80) {
            confidenceBar.classList.add('pulse');
            setTimeout(() => {
                confidenceBar.classList.remove('pulse');
            }, 1000);
        }
    }

    /**
     * Update word suggestions via backend API
     */
    async updateWordSuggestions(sentence) {
        const suggestionsContainer = document.getElementById('suggestions');
        suggestionsContainer.innerHTML = '<div class="text-muted small">Loading suggestions...</div>'; // Loading state

        try {
            const response = await fetch('/api/suggestions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: sentence || '' })
            });

            const data = await response.json();

            if (data.error) {
                console.error('Error fetching suggestions:', data.error);
                suggestionsContainer.innerHTML = '<div class="text-muted small">Suggestions unavailable</div>';
                return;
            }

            suggestionsContainer.innerHTML = '';
            if (data.suggestions && data.suggestions.length > 0) {
                data.suggestions.forEach(word => {
                    const button = document.createElement('button');
                    button.className = 'suggestion-btn fw-bold';
                    button.textContent = word;
                    button.onclick = () => this.applySuggestion(word);
                    suggestionsContainer.appendChild(button);
                });
            } else {
                suggestionsContainer.innerHTML = '<div class="text-muted small">No suggestions</div>';
            }
        } catch (error) {
            console.error('Error updating word suggestions:', error);
            suggestionsContainer.innerHTML = '<div class="text-muted small">Error loading suggestions</div>';
        }
    }

    /**
     * Apply word suggestion
     */
    applySuggestion(word) {
        const words = this.currentSentence.split(' ');
        if (words.length > 0) {
            words[words.length - 1] = word;
        } else {
            words.push(word);
        }
        const newSentence = words.join(' ') + ' ';

        this.updateSentenceDisplay(newSentence);

        // Send to backend
        this.sendSentenceUpdate(newSentence);
    }

    /**
     * Display performance metrics with color coding
     */
    displayPerformanceMetrics(data) {
        // Update average FPS with color class
        const avgFpsElement = document.getElementById('avg-fps');
        if (data.avg_fps !== undefined) {
            const fpsValue = Math.round(data.avg_fps);
            avgFpsElement.textContent = fpsValue;
            avgFpsElement.className = 'metric-value ' +
                (fpsValue > 25 ? 'good' : fpsValue > 15 ? 'warning' : 'danger');
        }

        // Update accuracy with color class
        const accuracyElement = document.getElementById('accuracy');
        if (data.accuracy !== undefined) {
            const accValue = Math.round(data.accuracy * 100);
            accuracyElement.textContent = accValue + '%';
            accuracyElement.className = 'metric-value ' +
                (accValue >= 80 ? 'good' : accValue >= 60 ? 'warning' : 'danger');
        }

        // Update latency with color class (lower is better)
        const latencyElement = document.getElementById('latency');
        if (data.latency !== undefined) {
            const latValue = Math.round(data.latency);
            latencyElement.textContent = latValue + 'ms';
            latencyElement.className = 'metric-value ' +
                (latValue < 100 ? 'good' : latValue < 200 ? 'warning' : 'danger');
        }
    }

    /**
     * Speak current text
     */
    async speakText() {
        const speakBtn = document.getElementById('speak-btn');
        const originalText = speakBtn.innerHTML;

        // Don't speak if already speaking or no text
        if (this.isSpeaking || !this.currentSentence || !this.currentSentence.trim()) {
            console.log('Speech blocked: already speaking or no text');
            return;
        }

        this.isSpeaking = true;

        try {
            speakBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Speaking...';
            speakBtn.disabled = true;

            const response = await fetch('/api/speak', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: this.currentSentence.trim()
                })
            });

            const result = await response.json();

            if (!result.success) {
                throw new Error(result.error || 'Failed to speak text');
            }

            // Show success feedback
            speakBtn.innerHTML = '<i class="fas fa-check me-2"></i>Spoken!';
            setTimeout(() => {
                speakBtn.innerHTML = originalText;
            }, 2000);

        } catch (error) {
            console.error('Error speaking text:', error);
            alert('Error: ' + error.message);
            speakBtn.innerHTML = originalText;
        } finally {
            speakBtn.disabled = false;
            this.isSpeaking = false;
        }
    }



    /**
     * Clear current sentence
     */
    async clearSentence() {
        try {
            const response = await fetch('/api/clear_sentence', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            const result = await response.json();

            if (result.success) {
                this.currentSentence = '';
                this.updateSentenceDisplay('');
                this.updateCurrentCharacter('--');
                this.updateConfidence(0);
                this.updateWordSuggestions('');
            }

        } catch (error) {
            console.error('Error clearing sentence:', error);
        }
    }

    /**
     * Handle video stream errors
     */
    handleVideoError() {
        const statusElement = document.getElementById('camera-status');
        statusElement.innerHTML = '<i class="fas fa-circle text-danger"></i> Camera Error';
        console.error('Video stream error');
    }

    /**
     * Handle video stream load
     */
    handleVideoLoad() {
        const statusElement = document.getElementById('camera-status');
        statusElement.innerHTML = '<i class="fas fa-circle text-success"></i> Camera Active';
    }

    /**
     * Handle keyboard shortcuts
     */
    handleKeyboardShortcuts(event) {
        // Ctrl/Cmd + Space: Speak
        if ((event.ctrlKey || event.metaKey) && event.code === 'Space') {
            event.preventDefault();
            this.speakText();
        }

        // Ctrl/Cmd + Backspace: Clear sentence
        if ((event.ctrlKey || event.metaKey) && event.code === 'Backspace') {
            event.preventDefault();
            this.clearSentence();
        }

        // Backspace: Remove last character
        if (event.code === 'Backspace' && !event.ctrlKey && !event.metaKey && !event.altKey && !event.shiftKey) {
            event.preventDefault();
            this.backspaceLastCharacter();
        }

        // Plus key (+): Add space
        if (event.code === 'Equal' && event.shiftKey && !event.ctrlKey && !event.metaKey && !event.altKey) {
            event.preventDefault();
            this.addSpace();
        }
    }

    /**
     * Show loading state
     */
    showLoadingState(message = 'Loading...') {
        // Implementation for loading state
        console.log('Loading:', message);
    }

    /**
     * Hide loading state
     */
    hideLoadingState() {
        // Implementation for hiding loading state
        console.log('Loading complete');
    }

    /**
     * Initialize tooltips
     */
    initializeTooltips() {
        // Add tooltips to buttons
        document.getElementById('speak-btn').title = 'Speak text (Ctrl+Space)';
        document.getElementById('clear-btn').title = 'Clear text (Ctrl+Backspace)';
    }

    /**
     * Send sentence update to backend
     */
    async sendSentenceUpdate(sentence) {
        try {
            await fetch('/api/update_sentence', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ sentence: sentence })
            });
        } catch (error) {
            console.error('Error sending sentence update:', error);
        }
    }

    /**
     * Remove last character from sentence
     */
    async backspaceLastCharacter() {
        if (!this.currentSentence || this.currentSentence.length === 0) {
            console.log('No characters to remove');
            return;
        }

        // Remove the last character and update local state
        this.currentSentence = this.currentSentence.slice(0, -1);

        // Update the display
        this.updateSentenceDisplay(this.currentSentence);

        // Send to backend
        await this.sendSentenceUpdate(this.currentSentence);

        console.log('Backspace: Removed last character');
    }

    /**
     * Add space to sentence
     */
    async addSpace() {
        // Add space to the current sentence
        const newSentence = this.currentSentence + ' ';

        // Update the display
        this.updateSentenceDisplay(newSentence);

        // Send to backend
        await this.sendSentenceUpdate(newSentence);

        console.log('Space: Added space to sentence');
    }

    /**
     * Check application health
     */
    async checkHealth() {
        try {
            const response = await fetch('/health');
            const health = await response.json();
            console.log('Application health:', health);
            return health;
        } catch (error) {
            console.error('Health check failed:', error);
            return null;
        }
    }

    /**
     * Initialize gesture learning panel (works with new horizontal scroll layout)
     */
    initializeGestureLearningPanel() {
        console.log('🎓 Initializing gesture learning panel...');

        // Letters and numbers are now rendered inline via <script> in index.html
        // Only need to populate common words here
        const wordScroll = document.getElementById('word-scroll');
        const wordGrid = document.getElementById('word-thumbnail-grid');
        const wordContainer = wordScroll || wordGrid;
        if (wordContainer) {
            const words = [
                'HELLO', 'YES', 'NO', 'LOVE', 'SPACE', 'GOODBYE', 'THANKYOU',
                'PLEASE', 'SORRY', 'HAPPY', 'SAD', 'MOTHER', 'FATHER',
                'BROTHER', 'SISTER', 'FRIEND', 'FAMILY', 'TEACHER',
                'FOOD', 'WATER', 'HOUSE', 'MONEY', 'TIME', 'MORNING', 'NIGHT',
                'DOCTOR', 'COLLEGE', 'AFRAID', 'AGREE', 'ASSISTANCE', 'BAD',
                'BECOME', 'FROM', 'PAIN', 'PRAY', 'SECONDARY', 'SKIN',
                'SMALL', 'SPECIFIC', 'STAND', 'TODAY', 'WARN', 'WHICH',
                'WORK', 'YOU'
            ];
            // Clear any existing content
            wordContainer.innerHTML = '';
            // Use gesture-scroll-row style
            if (wordScroll) wordScroll.classList.add('gesture-scroll-row');
            words.forEach(word => {
                const card = document.createElement('div');
                card.className = 'gesture-card';
                card.style.width = 'auto';
                card.style.minWidth = '64px';
                card.title = word;
                card.onclick = () => { if (typeof showGestureRef === 'function') showGestureRef(word); };
                card.innerHTML = `<img src="/static/gestures/${word}.png" alt="${word}" onerror="this.style.display='none'"><span style="font-size:0.7em">${word}</span>`;
                wordContainer.appendChild(card);
            });
        }

        // Set initial gesture image to 'A'
        this.updateGestureImage('A');

        console.log('✅ Gesture learning panel initialized');
    }

    /**
     * Create a gesture thumbnail element
     */
    createGestureThumbnail(character, type) {
        const thumbnail = document.createElement('div');
        thumbnail.className = 'gesture-thumbnail';
        thumbnail.textContent = character;
        thumbnail.dataset.gesture = character;
        thumbnail.dataset.type = type;

        // Add click handler to show gesture image
        thumbnail.addEventListener('click', () => {
            this.showGestureInLearningPanel(character);
            this.highlightActiveThumbnail(thumbnail);
        });

        return thumbnail;
    }

    /**
     * Update gesture image in learning panel
     */
    updateGestureImage(gesture) {
        if (!gesture) return;

        const gestureImage = document.getElementById('gesture-image');
        const gestureTitle = document.getElementById('gesture-title');
        const gestureDescription = document.getElementById('gesture-description');

        // Update image source
        const imagePath = `/static/gestures/${gesture.toUpperCase()}.png`;
        gestureImage.src = imagePath;
        gestureImage.alt = `Gesture for ${gesture}`;

        // Add animation class
        gestureImage.classList.add('gesture-image-update');
        setTimeout(() => {
            gestureImage.classList.remove('gesture-image-update');
        }, 500);

        // Update title and description
        const isLetter = /^[A-Z]$/.test(gesture);
        const isNumber = /^[0-9]$/.test(gesture);

        // Word gesture descriptions
        const wordDescriptions = {
            'HELLO': 'Wave your open palm — the universal greeting sign',
            'YES': 'Make a fist and nod it up and down like nodding',
            'NO': 'Extend index and middle finger, close them together',
            'GOOD': 'Give a thumbs up gesture',
            'BAD': 'Give a thumbs down gesture',
            'HELP': 'Place a fist on an open palm, lift upward',
            'THANKYOU': 'Touch chin with flat hand, move forward',
            'SORRY': 'Place fist on chest, move in circular motion',
            'PLEASE': 'Press both palms together (prayer position)',
            'STOP': 'Show flat palm facing forward (halt)',
            'LOVE': 'Cross arms over your chest',
            'FRIEND': 'Interlock index fingers together',
            'WATER': 'Tap three fingers on chin',
            'FOOD': 'Bring fingertips to mouth repeatedly',
            'HOME': 'Place flat hand on cheek',
            'SCHOOL': 'Clap hands twice (like a teacher)',
            'DOCTOR': 'Tap wrist with two fingers (checking pulse)',
            'COME': 'Palm up, curl fingers inward toward you',
            'GO': 'Point forward and flick your wrist',
            'EAT': 'Bring fingertips to mouth repeatedly'
        };

        if (isLetter) {
            gestureTitle.textContent = `Letter: ${gesture}`;
            gestureDescription.textContent = `Practice the hand sign for the letter "${gesture}"`;
        } else if (isNumber) {
            gestureTitle.textContent = `Number: ${gesture}`;
            gestureDescription.textContent = `Practice the hand sign for the number "${gesture}"`;
        } else if (wordDescriptions[gesture.toUpperCase()]) {
            gestureTitle.textContent = `Word: ${gesture}`;
            gestureDescription.textContent = wordDescriptions[gesture.toUpperCase()];
        } else {
            gestureTitle.textContent = `Gesture: ${gesture}`;
            gestureDescription.textContent = `Practice this gesture sign`;
        }

        // Update active thumbnail
        this.updateActiveThumbnail(gesture);
    }

    /**
     * Show specific gesture in learning panel (when thumbnail clicked)
     */
    showGestureInLearningPanel(gesture) {
        this.updateGestureImage(gesture);

        // Scroll learning panel into view if needed
        const learningPanel = document.querySelector('.gesture-learning-container');
        learningPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    /**
     * Highlight active thumbnail
     */
    highlightActiveThumbnail(activeThumbnail) {
        // Remove active class from all thumbnails
        document.querySelectorAll('.gesture-thumbnail').forEach(thumb => {
            thumb.classList.remove('active');
        });

        // Add active class to clicked thumbnail
        activeThumbnail.classList.add('active');
    }

    /**
     * Update active thumbnail based on current gesture
     */
    updateActiveThumbnail(gesture) {
        if (!gesture) return;

        // Find and highlight the thumbnail for current gesture
        const thumbnail = document.querySelector(`[data-gesture="${gesture.toUpperCase()}"]`);
        if (thumbnail) {
            this.highlightActiveThumbnail(thumbnail);
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.islApp = new ISLGestureApp();

    // Perform initial health check
    window.islApp.checkHealth().then(health => {
        if (health && health.status === 'healthy') {
            console.log('✅ Application is healthy and ready!');
        } else {
            console.warn('⚠️ Application health check failed');
        }
    });
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
        console.log('Page became visible - resuming updates');
        // Resume updates if needed
    } else {
        console.log('Page became hidden - pausing updates');
        // Pause updates to save resources
    }
});

// Handle connection errors
window.addEventListener('online', () => {
    console.log('Connection restored');
});

window.addEventListener('offline', () => {
    console.log('Connection lost');
});
