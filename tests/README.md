# Test Suite

Unit and integration tests for the Interview Sentiment Analyzer.

## Test Structure

```
tests/
├── analysis/
│   └── test_sentiment.py      # Sentiment, question, speaker, Q&A tests
├── audio/
│   └── test_audio.py          # Audio processing module tests
├── config/
│   ├── test_environment.py    # Environment configuration tests
│   └── test_paths.py          # Path configuration tests
├── models/
│   └── test_models.py         # Pydantic data model tests
├── pipeline/
│   └── test_runner.py         # Pipeline runner tests
└── utils/
    └── test_cleanup.py        # Cleanup utilities tests
```

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific module
pytest tests/analysis/

# Run with coverage
pytest --cov=src --cov-report=html
```

## Test Dependencies

Defined in `requirements.txt`:
- `pytest>=7.0`
- `pytest-cov>=4.0`
- `pytest-mock>=3.10`
