# 📧 Email Classifier - NLP-based

An industry-level email classification system using **Natural Language Processing (NLP)** techniques. Classifies emails into **Urgent**, **Normal**, **Promotional**, or **Spam** categories using TF-IDF vectorization and cosine similarity.

## 🎯 Features

- **NLP-based Classification**: Uses real NLP techniques, not just keyword matching
- **TF-IDF Vectorization**: Converts text into numerical vectors using Term Frequency-Inverse Document Frequency
- **Cosine Similarity**: Compares emails with reference documents using cosine similarity
- **Text Preprocessing**: 
  - Lowercase conversion
  - Punctuation removal
  - Stopword removal
- **Confidence Scores**: Shows confidence level for each classification
- **Similarity Scores**: Displays similarity scores for all categories
- **Web Interface**: Beautiful, modern UI with real-time classification

## 🔬 How It Works

### 1. Text Preprocessing
- Converts text to lowercase
- Removes punctuation
- Removes stopwords (common words like "the", "a", "is", etc.)

### 2. Reference Documents
The classifier uses reference documents representing each category:
- **Urgent**: Documents with urgent, emergency, critical keywords
- **Promotional**: Documents with sale, discount, offer keywords
- **Normal**: Regular business communication examples
- **Spam**: Suspicious promotional content

### 3. TF-IDF Vectorization
- Converts both input email and reference documents into TF-IDF vectors
- TF-IDF (Term Frequency-Inverse Document Frequency) weights words based on:
  - How frequently they appear in the document (TF)
  - How rare they are across all documents (IDF)

### 4. Cosine Similarity
- Calculates cosine similarity between input email vector and each reference category
- Cosine similarity measures the angle between vectors (0-1 scale)
- Higher similarity = more similar content

### 5. Classification
- Email is classified as the category with the highest similarity score
- Confidence is calculated as the difference between top and second-highest scores

## 📋 Classification Rules

### Urgent Emails
Reference documents contain: `urgent`, `asap`, `immediately`, `emergency`, `critical`, `deadline`, `time-sensitive`, etc.

### Promotional Emails
Reference documents contain: `offer`, `sale`, `discount`, `deal`, `promotion`, `coupon`, `limited time`, etc.

### Normal Emails
Reference documents represent regular business communication without urgent or promotional language.

### Spam Emails
Reference documents contain suspicious patterns: `click here`, `winner`, `prize`, `guaranteed`, etc.

## 🚀 Quick Start

### Installation

1. **Navigate to the project directory:**
   ```bash
   cd email_classifier
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   This will install:
   - Flask (web framework)
   - scikit-learn (TF-IDF and cosine similarity)
   - nltk (stopwords and text processing)

3. **Note**: NLTK will automatically download required data (stopwords) on first run.

### Running the Application

1. **Start the Flask server:**
   ```bash
   python app.py
   # or
   python run.py
   ```

2. **Open your browser and navigate to:**
   ```
   http://localhost:5000
   ```

3. **Enter an email and click "Classify Email"**

## 📝 Usage Examples

### Example 1: Urgent Email
```
Input: "Urgent: Please respond ASAP. This is critical!"
Output: Urgent
Similarity: Urgent (0.85), Normal (0.12), Promotional (0.08), Spam (0.05)
```

### Example 2: Promotional Email
```
Input: "Check out our amazing sale! 50% discount on all items."
Output: Promotional
Similarity: Promotional (0.78), Normal (0.15), Urgent (0.05), Spam (0.02)
```

### Example 3: Normal Email
```
Input: "Hi, just wanted to follow up on our meeting from yesterday."
Output: Normal
Similarity: Normal (0.92), Urgent (0.03), Promotional (0.03), Spam (0.02)
```

### Example 4: Spam Email
```
Input: "Congratulations! You have won a prize! Click here now!"
Output: Spam
Similarity: Spam (0.82), Promotional (0.10), Urgent (0.05), Normal (0.03)
```

## 💻 Code Usage (Programmatic)

You can also use the classifier programmatically:

```python
from classifier import EmailClassifier

classifier = EmailClassifier()

# Classify an email
result = classifier.classify("Urgent: Please respond ASAP")
print(result['category'])  # Output: Urgent
print(result['similarity_scores'])  # All similarity scores
print(result['confidence'])  # Confidence score (0-1)

# Get detailed analysis
detailed = classifier.get_detailed_analysis("Check out our sale!")
print(detailed)
```

## 🏗️ Project Structure

```
email_classifier/
├── classifier.py      # NLP-based classification logic
├── app.py            # Flask web application
├── run.py            # Quick start script
├── test_classifier.py # Test suite
├── templates/
│   └── index.html    # Web interface
├── requirements.txt  # Python dependencies
└── README.md        # This file
```

## 🔧 Technical Details

### NLP Techniques Used

1. **Text Preprocessing**
   - Lowercase normalization
   - Punctuation removal using `string.translate()`
   - Stopword removal using NLTK's English stopwords

2. **TF-IDF Vectorization**
   - Uses `sklearn.feature_extraction.text.TfidfVectorizer`
   - Max features: 1000
   - N-gram range: (1, 2) - uses unigrams and bigrams
   - Min document frequency: 1
   - Max document frequency: 0.95

3. **Cosine Similarity**
   - Uses `sklearn.metrics.pairwise.cosine_similarity`
   - Measures angle between vectors (0 = orthogonal, 1 = identical direction)
   - Range: 0 to 1

### Reference Documents

The classifier uses 4 reference documents per category (16 total) to create robust category representations. These are combined and vectorized to create category vectors.

## 🎓 Why This Project?

✅ **Real NLP techniques** - TF-IDF, cosine similarity, text preprocessing  
✅ **No training required** - Uses reference documents, no large datasets needed  
✅ **Easy to understand** - Clear explanation of NLP concepts  
✅ **Industry-level** - Professional code structure and implementation  
✅ **Quick to run** - Can be set up and running in under 30 minutes  
✅ **Practical application** - Real-world email classification use case  

## 📊 Testing

Run the test suite:

```bash
python test_classifier.py
```

Tests include:
- Urgent email classification
- Promotional email classification
- Normal email classification
- Spam detection
- Edge cases (empty emails, mixed content)
- Text preprocessing verification
