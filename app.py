"""
Main Flask application for ISL Gesture Recognition System
"""
import sys

# Fix Windows console encoding for emoji characters
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import os
import cv2
import json
import time
import base64
import glob
import subprocess
from flask import Flask, render_template, jsonify, request, Response
from flask_cors import CORS
import numpy as np
import mediapipe as mp
from threading import Thread
import threading
import queue

# Import custom modules
from config import config
from src.models.gesture_recognizer import GestureRecognizer
from src.utils.tts_handler import TTSHandler
from src.utils.nlp_processor import NLPProcessor
from src.utils.performance_monitor import PerformanceMonitor

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(config['development'])
CORS(app)

# Initialize components
gesture_recognizer = None
tts_handler = TTSHandler()
nlp_processor = NLPProcessor()
performance_monitor = PerformanceMonitor()

# Initialize gesture recognizer immediately
print("🚀 Initializing ISL Gesture Recognition System...")
try:
    # Pass model path and configuration to gesture recognizer
    model_config = {
        'MAX_NUM_HANDS': config['development'].MAX_NUM_HANDS,
        'HAND_DETECTION_CONFIDENCE': config['development'].HAND_DETECTION_CONFIDENCE,
        'HAND_TRACKING_CONFIDENCE': config['development'].HAND_TRACKING_CONFIDENCE
    }
    
    gesture_recognizer = GestureRecognizer(
        model_path=config['development'].CNN_MODEL_PATH,
        config=model_config
    )
    print("✅ Gesture recognizer initialized successfully")
except Exception as e:
    print(f"❌ Error initializing gesture recognizer: {e}")
    gesture_recognizer = None

# Global variables for video streaming
camera = None
output_frame = None
lock = threading.Lock()
gesture_queue = queue.Queue()
current_sentence = ""

# Global variables for real-time gesture data
current_gesture = ""
current_confidence = 0.0
current_hand_count = 0
last_detection_time = 0
last_gesture_time = 0
last_captured_gesture = ""  # Track last captured gesture to prevent rapid repeats
gesture_stable_count = 0     # Count consecutive frames with same gesture (stability)
gesture_stable_target = ""   # The gesture being counted for stability
GESTURE_CAPTURE_INTERVAL = 2.0        # Normal capture interval (different gesture)
SAME_GESTURE_COOLDOWN = 4.0           # Longer cooldown for same gesture repeat
GESTURE_STABILITY_FRAMES = 3          # Must see same gesture N times before accepting


def get_camera():
    """Initialize and return camera object"""
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
    return camera

def generate_frames():
    """Generate video frames with gesture recognition"""
    global output_frame, current_sentence, current_gesture, current_confidence, current_hand_count, last_gesture_time
    global last_captured_gesture, gesture_stable_count, gesture_stable_target
    
    print(f"📹 Starting video stream generation...")
    print(f"🤖 Gesture recognizer status: {'Loaded' if gesture_recognizer else 'Not loaded'}")
    print(f"⏰ Gesture capture interval: {GESTURE_CAPTURE_INTERVAL} seconds")
    
    camera = get_camera()
    if not camera.isOpened():
        print("❌ Camera failed to open")
        return
    
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    while True:
        success, frame = camera.read()
        if not success:
            print("❌ Failed to read camera frame")
            break
            
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Perform gesture recognition
        try:
            current_time = time.time()
            if gesture_recognizer:
                result = gesture_recognizer.predict(frame)
                
                if result:
                    # Draw hand landmarks and predictions
                    landmarks_data = result.get('landmarks_data')
                    frame = gesture_recognizer.draw_landmarks(frame, landmarks_data)
                    
                    # Get prediction data
                    gesture = result.get('gesture', '')
                    confidence = result.get('confidence', 0.0)
                    hand_count = result.get('hand_count', 0)
                    
                    # Update real-time global variables (for display purposes)
                    current_gesture = gesture
                    current_confidence = confidence
                    current_hand_count = hand_count
                    
                    # Always show detection info
                    cv2.putText(frame, f"Hands: {hand_count}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(frame, f"Gesture: {gesture}", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 90),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Check if it's time to capture a gesture (every 2 seconds)
                    # Determine cooldown based on whether same gesture is repeating
                    if gesture == last_captured_gesture:
                        required_interval = SAME_GESTURE_COOLDOWN
                    else:
                        required_interval = GESTURE_CAPTURE_INTERVAL
                    
                    # Stability check: same gesture must persist for N frames
                    if gesture == gesture_stable_target:
                        gesture_stable_count += 1
                    else:
                        gesture_stable_target = gesture
                        gesture_stable_count = 1
                    
                    if (current_time - last_gesture_time) >= required_interval and gesture_stable_count >= GESTURE_STABILITY_FRAMES:
                        # Only capture if hands are detected and confidence is high enough
                        if hand_count > 0 and confidence > app.config['CONFIDENCE_THRESHOLD']:
                            # Add recognized gesture to queue
                            gesture_queue.put(gesture)

                            # Handle SPACE gesture specially
                            if gesture == 'SPACE':
                                current_sentence += ' '
                            else:
                                # Update current sentence
                                current_sentence = nlp_processor.process_gesture_sequence([gesture], current_sentence)

                            # Update tracking
                            last_gesture_time = current_time
                            last_captured_gesture = gesture
                            gesture_stable_count = 0  # Reset stability

                            print(f"✅ Captured gesture '{gesture}' with confidence {confidence:.2f}")
                    
                    # Display sentence
                    cv2.putText(frame, f"Sentence: {current_sentence}", (10, 120),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    # Show next capture countdown
                    next_capture_time = GESTURE_CAPTURE_INTERVAL - (current_time - last_gesture_time)
                    if next_capture_time > 0:
                        cv2.putText(frame, f"Next capture: {next_capture_time:.1f}s", (10, 150),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    # Show "No hands detected" when no result
                    cv2.putText(frame, "No hands detected", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Reset real-time variables when no hands detected
                    current_gesture = ""
                    current_confidence = 0.0
                    current_hand_count = 0
                
        except Exception as e:
            print(f"Error in gesture recognition: {e}")
        
        # Calculate and display FPS
        frame_count += 1
        if frame_count % 30 == 0:
            end_time = time.time()
            fps = 30 / (end_time - start_time)
            performance_monitor.log_fps(fps)
            start_time = end_time
            
        cv2.putText(frame, f"FPS: {fps:.1f}", 
                   (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Store frame globally
        output_frame = frame
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def process_gesture_sequence():
    """Process sequence of gestures to form meaningful text"""
    gestures = []
    while not gesture_queue.empty():
        gestures.append(gesture_queue.get())
    
    if gestures:
        # Process gestures using NLP processor with intelligent sentence building
        # Only process the last unique gesture to avoid repetition
        if len(gestures) > 0:
            # Take only the most recent gesture
            latest_gesture = gestures[-1]
            global current_sentence
            current_sentence = nlp_processor.process_gesture_sequence([latest_gesture], current_sentence)
            return current_sentence
    
    return current_sentence

@app.route('/')
def login():
    """Login / role selection page"""
    return render_template('login.html')

@app.route('/home')
def index():
    """Sign language recognition page (for normal speakers)"""
    return render_template('index.html',
                         gesture_classes=app.config['GESTURE_CLASSES'])

@app.route('/speech-to-text')
def speech_to_text():
    """Speech-to-text page (for deaf/mute users)"""
    return render_template('speech_to_text.html')

@app.route('/dashboard')
def dashboard():
    """Statistics dashboard page"""
    return render_template('dashboard.html')

@app.route('/learn')
def learn():
    """Learn mode route"""
    return render_template('learn.html')

@app.route('/alphabet-game')
def alphabet_game():
    """Alphabet game route"""
    return render_template('alphabet_game.html')

@app.route('/numbers-game')
def numbers_game():
    """Numbers game route"""
    return render_template('numbers_game.html')

@app.route('/collect')
def collect():
    """Gesture data collection page"""
    return render_template('collect.html')

@app.route('/test')
def test_accuracy():
    """ISL accuracy testing page"""
    return render_template('test_accuracy.html', gesture_classes=app.config['GESTURE_CLASSES'])

@app.route('/api/predict_frame', methods=['POST'])
def predict_frame():
    """Accept a base64 webcam frame from the browser and return gesture prediction.
    This enables client-side camera (needed for cloud deployment where no server camera exists).
    """
    global current_gesture, current_confidence, current_hand_count, last_gesture_time, current_sentence, last_captured_gesture, gesture_stable_count, gesture_stable_target
    try:
        data = request.get_json()
        image_data = data.get('image', '')
        if not image_data:
            return jsonify({'error': 'No image data'}), 400

        # Decode base64 → OpenCV image
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]
        img_bytes = base64.b64decode(image_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        # Run gesture recognition
        result_data = {'gesture': '', 'confidence': 0.0, 'hand_count': 0}
        if gesture_recognizer:
            result = gesture_recognizer.predict(frame)
            if result:
                gesture = result.get('gesture', '')
                confidence = result.get('confidence', 0.0)
                hand_count = result.get('hand_count', 0)
                result_data = {
                    'gesture': gesture,
                    'confidence': confidence,
                    'hand_count': hand_count
                }
                # Update globals for /api/gesture_data compatibility
                current_gesture = gesture
                current_confidence = confidence
                current_hand_count = hand_count

                # Auto-append to sentence if enough time has passed
                build_sentence = data.get('build_sentence', False)
                if build_sentence:
                    current_time = time.time()
                    # Determine cooldown: longer for same gesture
                    if gesture == last_captured_gesture:
                        required_interval = SAME_GESTURE_COOLDOWN
                    else:
                        required_interval = GESTURE_CAPTURE_INTERVAL
                    
                    # Stability check
                    if gesture == gesture_stable_target:
                        gesture_stable_count += 1
                    else:
                        gesture_stable_target = gesture
                        gesture_stable_count = 1
                    
                    if (current_time - last_gesture_time) >= required_interval and gesture_stable_count >= GESTURE_STABILITY_FRAMES:
                        if hand_count > 0 and confidence > app.config['CONFIDENCE_THRESHOLD']:
                            if gesture == 'SPACE':
                                current_sentence += ' '
                            else:
                                current_sentence = nlp_processor.process_gesture_sequence([gesture], current_sentence)
                            last_gesture_time = current_time
                            last_captured_gesture = gesture
                            gesture_stable_count = 0
                            result_data['sentence'] = current_sentence

        return jsonify(result_data)
    except Exception as e:
        print(f"❌ predict_frame error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/translate', methods=['POST'])
def translate_text():
    """Translate text to Tamil or Hindi"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        target_lang = data.get('lang', 'ta')  # ta=Tamil, hi=Hindi
        
        if not text:
            return jsonify({'success': False, 'error': 'No text provided'})
        
        try:
            from googletrans import Translator
            translator = Translator()
            result = translator.translate(text, dest=target_lang)
            return jsonify({
                'success': True,
                'original': text,
                'translated': result.text,
                'lang': target_lang
            })
        except ImportError:
            return jsonify({'success': False, 'error': 'googletrans not installed. Run: pip install googletrans==4.0.0-rc.1'})
        except Exception as e:
            return jsonify({'success': False, 'error': f'Translation failed: {str(e)}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/abbreviate', methods=['POST'])
def abbreviate_text():
    """Convert sentence to SMS-style abbreviation for testing"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        # Simple abbreviation rules
        abbrevs = {
            'are': 'r', 'you': 'u', 'to': '2', 'for': '4', 'be': 'b',
            'see': 'c', 'why': 'y', 'too': '2', 'before': 'b4',
            'great': 'gr8', 'late': 'l8', 'wait': 'w8', 'please': 'pls',
            'because': 'bcz', 'people': 'ppl', 'about': 'abt',
            'with': 'wth', 'have': 'hv', 'your': 'ur', 'tomorrow': '2mro',
            'tonight': '2nite', 'today': '2day', 'the': 'd', 'we': 'v',
            'and': 'n', 'okay': 'ok', 'thanks': 'thx', 'never': 'nvr',
        }
        
        words = text.lower().split()
        result = []
        for w in words:
            result.append(abbrevs.get(w, w))
        
        return jsonify({'success': True, 'abbreviated': ' '.join(result), 'original': text})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/data_image/<path:filepath>')
def serve_data_image(filepath):
    """Serve images from the data/raw/gestures directory"""
    from flask import send_from_directory
    return send_from_directory('data/raw/gestures', filepath)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/gesture_data')
def get_gesture_data():
    """API endpoint to get current gesture recognition data"""
    try:
        data = {
            'current_sentence': current_sentence,
            'current_gesture': current_gesture,
            'confidence': current_confidence,
            'hand_count': current_hand_count,
            'fps': performance_monitor.get_current_fps(),
            'last_detection_time': last_detection_time,
            'timestamp': time.time()
        }
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/api/speak', methods=['POST'])
def speak_text():
    """API endpoint for text-to-speech conversion"""
    try:
        data = request.get_json()
        text = data.get('text', current_sentence)

        if text and text.strip():
            print(f"🗣️ Speaking text: '{text}'")
            print(f"🔊 TTS Handler available: {tts_handler.is_available()}")
            print(f"🔊 TTS Engine type: {tts_handler.engine_type}")
            success = tts_handler.speak(text.strip())
            print(f"✅ TTS result: {success}")
            return jsonify({'success': success, 'text': text})
        else:
            print("⚠️ No text provided for TTS")
            return jsonify({'error': 'No text to speak'}), 400

    except Exception as e:
        print(f"❌ TTS Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/suggestions', methods=['POST'])
def get_suggestions():
    """AI-powered word suggestions endpoint.
    Expands abbreviations, completes partial words, and suggests next words."""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'suggestions': []})
        
        suggestions = []
        words = text.split()
        last_word = words[-1].lower() if words else ''
        
        # --- 1. Abbreviation / Shorthand expansion ---
        abbreviations = {
            'u': 'you', 'r': 'are', 'v': 'very', 'y': 'why',
            'n': 'and', 'b': 'be', 'c': 'see', 'k': 'okay',
            'w': 'with', 'd': 'the', 'm': 'am', 'i': 'I',
            'ur': 'your', 'bc': 'because', 'bt': 'but',
            'hw': 'how', 'wt': 'what', 'wr': 'where',
            'gd': 'good', 'gm': 'good morning', 'gn': 'good night',
            'tq': 'thank you', 'ty': 'thank you', 'thx': 'thanks',
            'pls': 'please', 'plz': 'please', 'msg': 'message',
            'tmr': 'tomorrow', 'tdy': 'today', 'yday': 'yesterday',
            'brb': 'be right back', 'idk': "I don't know",
            'imo': 'in my opinion', 'lol': 'laughing out loud',
        }
        
        if last_word in abbreviations:
            suggestions.append(abbreviations[last_word])
        
        # --- 2. Partial word completion ---
        common_words = [
            'hello', 'help', 'happy', 'have', 'how', 'here', 'home',
            'thank', 'thanks', 'the', 'this', 'that', 'they', 'there', 'their', 'time', 'today', 'tomorrow',
            'good', 'great', 'go', 'going', 'give', 'get',
            'please', 'play', 'place',
            'want', 'water', 'what', 'when', 'where', 'which', 'why', 'with', 'will', 'work',
            'need', 'name', 'nice', 'new', 'now', 'never', 'next', 'night',
            'morning', 'more', 'make', 'much', 'many', 'meet', 'my',
            'like', 'love', 'live', 'learn', 'let', 'look', 'long',
            'come', 'call', 'can', 'care',
            'food', 'feel', 'find', 'friend', 'from', 'fine', 'family',
            'very', 'visit',
            'school', 'sorry', 'some', 'stop', 'start', 'study',
            'eat', 'evening', 'every', 'enough',
            'know', 'keep',
            'yes', 'you', 'your',
            'are', 'about', 'after', 'again', 'also',
            'because', 'before', 'best', 'better', 'between',
            'day', 'do', 'done',
            'okay', 'open', 'only', 'over',
        ]
        
        if len(last_word) >= 2 and last_word not in abbreviations:
            completions = [w for w in common_words 
                          if w.startswith(last_word) and w != last_word]
            suggestions.extend(completions[:4])
        
        # --- 3. Next-word prediction based on context ---
        next_word_map = {
            'i': ['am', 'want', 'need', 'like', 'love', 'have', 'will', 'can'],
            'you': ['are', 'can', 'will', 'have', 'need', 'want', 'should'],
            'how': ['are', 'do', 'can', 'much', 'many', 'about'],
            'what': ['is', 'are', 'do', 'time', 'about', 'happened'],
            'thank': ['you', 'god'],
            'good': ['morning', 'afternoon', 'evening', 'night', 'job', 'day'],
            'nice': ['to', 'meeting', 'day', 'work'],
            'please': ['help', 'come', 'give', 'call', 'wait', 'stop'],
            'can': ['you', 'I', 'we', 'help'],
            'i am': ['fine', 'good', 'happy', 'sorry', 'here', 'going'],
            'i want': ['to', 'food', 'water', 'help'],
            'i need': ['help', 'water', 'food', 'time', 'you', 'to'],
            'i like': ['you', 'this', 'it', 'food'],
            'i love': ['you', 'this', 'it'],
            'i will': ['come', 'go', 'help', 'call', 'do'],
            'do you': ['want', 'need', 'know', 'like', 'have'],
            'are you': ['okay', 'fine', 'happy', 'coming', 'ready', 'here'],
            'my name': ['is'],
            'very': ['much', 'good', 'happy', 'nice', 'well'],
            'see': ['you', 'this', 'that'],
            'where': ['are', 'is', 'do'],
            'we': ['are', 'can', 'will', 'should', 'need'],
            'he': ['is', 'has', 'can', 'will'],
            'she': ['is', 'has', 'can', 'will'],
            'it': ['is', 'was', 'will'],
            'they': ['are', 'have', 'can', 'will'],
            'let': ['me', 'us', 'go'],
            'come': ['here', 'home', 'back'],
        }
        
        # Check multi-word context first (last 2 words), then single word
        if len(words) >= 2:
            two_word_ctx = ' '.join(words[-2:]).lower()
            if two_word_ctx in next_word_map:
                suggestions.extend(next_word_map[two_word_ctx])
        
        if last_word in next_word_map and last_word not in abbreviations:
            for w in next_word_map[last_word]:
                if w not in suggestions:
                    suggestions.append(w)
        
        # Deduplicate and limit
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            if s.lower() not in seen:
                seen.add(s.lower())
                unique_suggestions.append(s)
        
        return jsonify({'suggestions': unique_suggestions[:8]})
    
    except Exception as e:
        print(f"❌ Suggestions Error: {str(e)}")
        return jsonify({'error': str(e), 'suggestions': []}), 500



@app.route('/api/clear_sentence', methods=['POST'])
def clear_sentence():
    """API endpoint to clear current sentence"""
    global current_sentence
    current_sentence = ""
    
    # Clear gesture queue
    while not gesture_queue.empty():
        gesture_queue.get()
    
    # Clear NLP processor buffers
    nlp_processor.clear_buffers()
    
    return jsonify({'success': True, 'message': 'Sentence cleared'})

@app.route('/api/update_sentence', methods=['POST'])
def update_sentence():
    """API endpoint to update sentence (used by backspace)"""
    global current_sentence
    try:
        data = request.get_json()
        new_sentence = data.get('sentence', '')
        current_sentence = new_sentence
        return jsonify({'success': True, 'sentence': current_sentence})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/performance')
def get_performance_data():
    """API endpoint to get performance metrics"""
    try:
        metrics = performance_monitor.get_metrics()
        return jsonify(metrics)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/gesture_classes')
def get_gesture_classes():
    """API endpoint to get available gesture classes"""
    return jsonify({
        'classes': app.config['GESTURE_CLASSES'],
        'num_classes': app.config['NUM_CLASSES']
    })

@app.route('/api/model_info')
def get_model_info():
    """API endpoint to get model information"""
    try:
        if gesture_recognizer:
            model_info = gesture_recognizer.get_model_info()
            return jsonify(model_info)
        else:
            return jsonify({'error': 'Model not loaded'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'gesture_recognizer': gesture_recognizer is not None,
        'camera_available': camera is not None,
        'tts_available': tts_handler.is_available(),
        'timestamp': time.time()
    }
    return jsonify(status)

@app.route('/debug')
def debug_info():
    """Debug information endpoint"""
    debug_data = {
        'gesture_recognizer_loaded': gesture_recognizer is not None,
        'camera_initialized': camera is not None,
        'model_path': app.config['CNN_MODEL_PATH'],
        'confidence_threshold': app.config['CONFIDENCE_THRESHOLD'],
        'hand_detection_confidence': app.config['HAND_DETECTION_CONFIDENCE'],
        'hand_tracking_confidence': app.config['HAND_TRACKING_CONFIDENCE']
    }
    
    if gesture_recognizer:
        try:
            model_info = gesture_recognizer.get_model_info()
            debug_data['model_info'] = model_info
        except Exception as e:
            debug_data['model_error'] = str(e)
    
    return jsonify(debug_data)

# ══════════════════════════════════════════════
# DATA COLLECTION & RETRAINING APIs
# ══════════════════════════════════════════════

GESTURE_DATA_DIR = os.path.join('data', 'raw', 'gestures')

@app.route('/api/collect_gesture', methods=['POST'])
def collect_gesture():
    """Save a captured webcam frame for a gesture class"""
    try:
        data = request.get_json()
        gesture = data.get('gesture', '').upper()
        image_data = data.get('image', '')

        if not gesture or not image_data:
            return jsonify({'error': 'Missing gesture or image data'}), 400

        # Create directory for this gesture
        gesture_dir = os.path.join(GESTURE_DATA_DIR, gesture)
        os.makedirs(gesture_dir, exist_ok=True)

        # Decode base64 image
        header, encoded = image_data.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Save with timestamp filename
        timestamp = int(time.time() * 1000)
        filename = f"{gesture}_{timestamp}.jpg"
        filepath = os.path.join(gesture_dir, filename)
        cv2.imwrite(filepath, img)

        # Count total images for this gesture
        count = len([f for f in os.listdir(gesture_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        return jsonify({'success': True, 'count': count, 'file': filename})

    except Exception as e:
        print(f"❌ Collect gesture error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/capture_frame', methods=['POST'])
def capture_frame():
    """Capture current camera frame directly on server — no canvas needed"""
    try:
        data = request.get_json()
        gesture = data.get('gesture', '').upper()

        if not gesture:
            return jsonify({'error': 'Missing gesture name'}), 400

        # Use the shared camera object
        cam = get_camera()
        if cam is None or not cam.isOpened():
            return jsonify({'error': 'Camera not available'}), 500

        ret, frame = cam.read()
        if not ret or frame is None:
            return jsonify({'error': 'Failed to capture frame'}), 500

        # Flip for mirror effect (same as video_feed)
        frame = cv2.flip(frame, 1)

        # Create directory & save
        gesture_dir = os.path.join(GESTURE_DATA_DIR, gesture)
        os.makedirs(gesture_dir, exist_ok=True)

        timestamp = int(time.time() * 1000)
        filename = f"{gesture}_{timestamp}.jpg"
        filepath = os.path.join(gesture_dir, filename)
        cv2.imwrite(filepath, frame)

        count = len([f for f in os.listdir(gesture_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        return jsonify({'success': True, 'count': count, 'file': filename})

    except Exception as e:
        print(f"❌ Capture frame error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/gesture_count/<gesture>')
def gesture_count(gesture):
    """Get image count for a specific gesture"""
    gesture_dir = os.path.join(GESTURE_DATA_DIR, gesture.upper())
    if os.path.exists(gesture_dir):
        count = len([f for f in os.listdir(gesture_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    else:
        count = 0
    return jsonify({'gesture': gesture, 'count': count})

@app.route('/api/gesture_counts')
def gesture_counts():
    """Get image counts for all gestures"""
    counts = {}
    if os.path.exists(GESTURE_DATA_DIR):
        for folder in os.listdir(GESTURE_DATA_DIR):
            folder_path = os.path.join(GESTURE_DATA_DIR, folder)
            if os.path.isdir(folder_path):
                c = len([f for f in os.listdir(folder_path)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                counts[folder] = c
    return jsonify({'counts': counts})

@app.route('/api/gesture_samples/<gesture>')
def gesture_samples(gesture):
    """Get sample image paths for a gesture (last 20)"""
    gesture_dir = os.path.join(GESTURE_DATA_DIR, gesture.upper())
    samples = []
    if os.path.exists(gesture_dir):
        files = sorted(
            [f for f in os.listdir(gesture_dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
            reverse=True
        )[:20]
        for f in files:
            samples.append(f"/data_image/{gesture.upper()}/{f}")
    return jsonify({'samples': samples})

@app.route('/api/preprocess', methods=['POST'])
def preprocess_data():
    """Run data preprocessing pipeline"""
    try:
        result = subprocess.run(
            ['python', '-m', 'src.preprocessing.preprocess_data'],
            capture_output=True, text=True, timeout=900, encoding='utf-8', errors='replace'
        )
        if result.returncode == 0:
            return jsonify({'success': True, 'message': 'Preprocessing complete!', 'output': result.stdout})
        else:
            return jsonify({'success': False, 'error': result.stderr or 'Preprocessing failed'})
    except subprocess.TimeoutExpired:
        return jsonify({'success': False, 'error': 'Preprocessing timed out (15 min limit)'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/retrain', methods=['POST'])
def retrain_model():
    """Retrain the gesture recognition model"""
    try:
        result = subprocess.run(
            ['python', '-m', 'src.models.train_cnn'],
            capture_output=True, text=True, timeout=1200, encoding='utf-8', errors='replace'
        )
        if result.returncode == 0:
            return jsonify({'success': True, 'message': 'Model retrained successfully! Restart the server to use the new model.', 'output': result.stdout})
        else:
            return jsonify({'success': False, 'error': result.stderr or 'Training failed'})
    except subprocess.TimeoutExpired:
        return jsonify({'success': False, 'error': 'Training timed out (20 min limit)'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ══════════════════════════════════════════════
# LSTM DYNAMIC GESTURE APIs
# ══════════════════════════════════════════════

SEQUENCE_DATA_DIR = os.path.join('data', 'sequences')
SEQUENCE_LENGTH = 30

@app.route('/api/collect_sequence', methods=['POST'])
def collect_sequence():
    """Save a 30-frame landmark sequence for LSTM training"""
    try:
        data = request.get_json()
        gesture = data.get('gesture', '').upper().strip()
        use_buffer = data.get('use_buffer', False)
        
        if not gesture:
            return jsonify({'success': False, 'error': 'Need gesture name'})
        
        if use_buffer:
            # Read from gesture recognizer's landmark buffer
            if gesture_recognizer is None or len(gesture_recognizer.landmark_buffer) < SEQUENCE_LENGTH:
                return jsonify({'success': False, 'error': f'Not enough frames buffered ({len(gesture_recognizer.landmark_buffer) if gesture_recognizer else 0}/{SEQUENCE_LENGTH}). Make sure webcam is running.'})
            seq = np.array(list(gesture_recognizer.landmark_buffer), dtype=np.float32)
        else:
            frames = data.get('frames', [])
            if len(frames) != SEQUENCE_LENGTH:
                return jsonify({'success': False, 'error': f'Need {SEQUENCE_LENGTH} frames, got {len(frames)}'})
            seq = np.array(frames, dtype=np.float32)
        
        # Save as .npy
        gesture_dir = os.path.join(SEQUENCE_DATA_DIR, gesture)
        os.makedirs(gesture_dir, exist_ok=True)
        
        existing = len([f for f in os.listdir(gesture_dir) if f.endswith('.npy')])
        np.save(os.path.join(gesture_dir, f'{gesture}_{existing}.npy'), seq)
        
        return jsonify({'success': True, 'count': existing + 1, 'gesture': gesture})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/sequence_counts')
def sequence_counts():
    """Get sequence counts per gesture class"""
    counts = {}
    if os.path.exists(SEQUENCE_DATA_DIR):
        for gesture in os.listdir(SEQUENCE_DATA_DIR):
            gesture_dir = os.path.join(SEQUENCE_DATA_DIR, gesture)
            if os.path.isdir(gesture_dir):
                counts[gesture] = len([f for f in os.listdir(gesture_dir) if f.endswith('.npy')])
    return jsonify({'counts': counts})

@app.route('/api/retrain_lstm', methods=['POST'])
def retrain_lstm():
    """Train LSTM model on collected sequences"""
    try:
        result = subprocess.run(
            ['python', '-m', 'src.models.train_lstm'],
            capture_output=True, text=True, timeout=600, encoding='utf-8', errors='replace'
        )
        if result.returncode == 0:
            return jsonify({'success': True, 'message': 'LSTM trained! Restart server to use.', 'output': result.stdout})
        else:
            return jsonify({'success': False, 'error': result.stderr or 'LSTM training failed'})
    except subprocess.TimeoutExpired:
        return jsonify({'success': False, 'error': 'LSTM training timed out'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('errors/500.html'), 500

if __name__ == '__main__':
    print("🚀 Initializing ISL Gesture Recognition System...")
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    print(f"🎯 Confidence threshold: {app.config['CONFIDENCE_THRESHOLD']}")
    print(f"📊 Number of gesture classes: {app.config['NUM_CLASSES']}")
    
    # Start the Flask application
    print("🌐 Starting web server...")
    print("📱 Open your browser and go to: http://localhost:5000")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=app.config['DEBUG'],
        threaded=True
    )
