# Setup Guide

## Quick Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **NLTK will automatically download required data on first run:**
   - Stopwords corpus
   - Punkt tokenizer (if needed)

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Open browser:**
   ```
   http://localhost:5000
   ```

## Testing

Run the test suite:
```bash
python test_classifier.py
```

## Troubleshooting

### ModuleNotFoundError: No module named 'nltk'
```bash
pip install nltk scikit-learn
```

### ModuleNotFoundError: No module named 'sklearn'
```bash
pip install scikit-learn
```

### NLTK Data Download Issues
If automatic download fails, manually download:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## Requirements

- Python 3.7+
- Flask 3.0.0
- scikit-learn 1.3.2
- nltk 3.8.1

