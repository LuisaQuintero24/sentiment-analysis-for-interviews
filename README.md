# Interview Sentiment Analyzer

A Python pipeline for analyzing interview audio files. Automatically identifies speakers, transcribes speech, detects questions vs statements, and performs sentiment and emotion analysis on responses.

## Features

- **Speaker Diarization**: Identifies who spoke when using pyannote.audio
- **Transcription**: Converts speech to text using OpenAI Whisper
- **Question Detection**: Classifies utterances as questions or statements using multilingual zero-shot classification (supports 100+ languages)
- **Sentiment Analysis**: Analyzes sentiment (positive/negative/neutral) using pysentimiento transformers
- **Emotion Detection**: Identifies emotions (joy, anger, sadness, fear, etc.)
- **Multilingual**: Full support for Spanish, English, and mixed-language interviews

## Requirements

- Python 3.8+
- FFmpeg (for audio processing)
- HuggingFace account with access to pyannote models
- ~4GB disk space for ML models

## Installation

### 1. Clone and setup environment

```bash
git clone <repository-url>
cd sentiment-main

# Create virtual environment
python -m venv venv

.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .
```

### 2. Setup FFmpeg

**Option A: System installation**
```bash

# Windows (using chocolatey)
choco install ffmpeg
```

**Option B: Manual installation**
1. Download from https://www.gyan.dev/ffmpeg/builds/
2. Extract to `engines/` folder
3. The pipeline will automatically detect it

### 3. Configure HuggingFace Token

1. Create account at https://huggingface.co
2. Get token from https://huggingface.co/settings/tokens
3. **Accept terms for these gated models** (required):
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-community-1

   > **Note:** You must click "Agree and access repository" on each model page while logged in. Without this, the pipeline will fail with a 403 error.

4. Set the token:
```bash
# Windows PowerShell
$env:HF_TOKEN = "your_token_here"

# Or add to .env file (create if doesn't exist)
echo "HF_TOKEN=your_token_here" > .env
```

## Usage

### Quick Start

1. Place your audio file in `data/raw/`:
   - Supported formats: `.mp3`, `.m4a`, `.mp4`, `.ogg`, `.flac`, `.aac`, `.wma`, `.aiff`, `.webm`
   - Any filename is accepted (e.g., `interview.mp3`, `recording.m4a`)
   - The file will be automatically converted to WAV in `data/refined/`

2. Run the pipeline:
```bash
python main.py
```

3. Find results in `data/output/sentiment_analysis.json`

### Configuration

Edit `config.yaml` to customize the pipeline:

```yaml
audio:
  whisper_model: "small"      # tiny, base, small, medium, large
  sample_rate: 16000
  min_segment_duration: 0.1   # Skip clips shorter than this (seconds)

analysis:
  question_model: "joeddav/xlm-roberta-large-xnli"
  default_language: "auto"    # auto, en, es

thresholds:
  question_confidence: 0.5    # Minimum confidence for question classification

output:
  format: "json"
  include_probabilities: true

logging:
  level: "INFO"               # DEBUG, INFO, WARNING, ERROR
```

### Whisper Model Selection

| Model | Size | Speed | Accuracy | VRAM |
|-------|------|-------|----------|------|
| tiny | 39M | Fastest | Low | ~1GB |
| base | 74M | Fast | Medium | ~1GB |
| small | 244M | Medium | Good | ~2GB |
| medium | 769M | Slow | Better | ~5GB |
| large | 1550M | Slowest | Best | ~10GB |

For most interviews, `small` provides a good balance.


## Project Structure

```
sentiment-main/
├── main.py                 # Entry point
├── config.yaml             # Configuration file
├── pyproject.toml          # Dependencies and project config
├── data/
│   ├── raw/                # Input: place your audio file here (mp3, m4a, etc.)
│   ├── refined/            # Converted WAV audio (auto-generated)
│   ├── interim/            # Intermediate files (RTTM diarization)
│   └── output/             # Output: sentiment_analysis.json
├── src/
│   ├── audio/              # Audio processing modules
│   ├── analysis/           # NLP analysis modules
│   ├── models/             # Data models (Pydantic)
│   ├── config/             # Configuration management
│   ├── pipeline/           # Pipeline orchestration
│   └── output/             # Report generation
└── tests/                  # Unit tests
```

## Troubleshooting

### "HF_TOKEN not found"
Set your HuggingFace token as environment variable. See Installation step 3.

### "Model access denied"
Accept the model terms on HuggingFace for pyannote models.

### "FFmpeg not found"
Install FFmpeg or place it in the `engines/` folder.

### "CUDA out of memory"
- Use a smaller Whisper model (`tiny` or `base`)
- Set `device: "cpu"` in the pipeline call
- Close other GPU-intensive applications

### Slow transcription
- Use a smaller Whisper model
- Ensure you're using GPU if available (`device: "cuda"`)

## Running Tests

```bash
pytest tests/
pytest tests/ --cov=src  # With coverage report
```
