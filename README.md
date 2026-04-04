# 🤟 Indian Sign Language (ISL) Gesture Recognition System

An AI-powered real-time gesture recognition system that converts Indian Sign Language gestures into text and speech, bridging communication gaps for the deaf/hearing-impaired community.

## 🚀 Key Features

- **Real-time Recognition**: Live ISL gesture detection using webcam and MediaPipe hand tracking
- **59-Class Model**: Recognizes letters (A-Z), numbers (0-9), and 23 common words (96.58% validation accuracy)
- **Text-to-Speech**: Multiple TTS engines (pyttsx3, gTTS) for audio output
- **NLP Processing**: Context-aware sentence formation and AI word suggestions
- **Web Interface**: Modern dashboard with live video feed, dark/light themes, and real-time feedback
- **Educational Games**: Interactive alphabet and numbers learning games
- **Data Collection**: Built-in webcam capture tool to expand the training dataset
- **RESTful API**: Endpoints for integration and custom applications

### Supported Gestures

| Category | Classes |
|----------|--------|
| Letters | A-Z (26) |
| Numbers | 0-9 (10) |
| Words | AFRAID, AGREE, ASSISTANCE, BAD, BECOME, COLLEGE, DOCTOR, FROM, HELLO, LOVE, NO, PAIN, PRAY, SECONDARY, SKIN, SMALL, SPACE, SPECIFIC, STAND, TODAY, WARN, WHICH, WORK, YES, YOU |

## 🏗️ Project Structure

```
AI-Based-Indian-Sign-Language-ISL-Recognition/
├── app.py                 # Main Flask application
├── run.py                 # Application launcher
├── config.py              # Configuration (59 gesture classes)
├── requirements.txt       # Python dependencies
├── merge_datasets.py      # Dataset merging utility
├── src/
│   ├── models/            # CNN training (train_cnn.py)
│   ├── preprocessing/     # MediaPipe landmark extraction
│   └── utils/             # TTS, NLP, gesture recognition
├── static/                # CSS, JS, gesture reference images
├── templates/             # HTML (index, login, learn, collect, etc.)
├── data/
│   ├── raw/gestures/      # 90K merged training images
│   └── processed/         # 330K landmark samples (CSV)
├── models/
│   ├── trained/           # Production model (cnn_model.h5)
│   └── checkpoints/       # Training checkpoints
└── datasets/              # Raw downloaded datasets (~11GB, gitignored)
```

## 🔧 Installation

### Prerequisites
- Python 3.7+
- Webcam
- Git

### Step-by-Step Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd isl-gesture-recognition
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

5. **Run the application**
   ```bash
   python run.py
   ```

6. **Open browser**
   Navigate to `http://localhost:5000`

## 🎯 Usage

1. **Start the App**: Run `python run.py` and open `http://localhost:5000`
2. **Grant Camera Access**: Allow webcam permissions
3. **Position Hands**: Place hands 6-12 inches from camera
4. **Make Gestures**: Perform ISL signs for letters/numbers
5. **View Results**: See real-time text conversion
6. **Hear Speech**: Click "Speak" for audio output

### Keyboard Shortcuts
- **Space**: Speak sentence
- **Enter**: Add gesture to sentence
- **Backspace**: Clear last gesture
- **Ctrl+C**: Clear sentence

## 📡 API Endpoints

- `GET /`: Main dashboard
- `GET /video_feed`: MJPEG video stream
- `GET /api/gesture_data`: Current gesture data
- `POST /api/speak`: Text-to-speech
- `GET /health`: System health check

## 🧠 Model Training

The current model was trained on **330,828 MediaPipe landmark samples** from 10 merged datasets.

| Metric | Value |
|--------|-------|
| Validation Accuracy | **96.58%** |
| Top-K Accuracy | 99.22% |
| Classes | 59 |
| Training Time | ~35 min |

### Retraining

```bash
# 1. Preprocess images → landmarks
python -m src.preprocessing.preprocess_data

# 2. Train CNN
python -m src.models.train_cnn
```

### Collecting More Data

Visit `http://localhost:5000/collect` to capture gesture images via webcam, then retrain.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**Made with ❤️ for accessibility and inclusion**
