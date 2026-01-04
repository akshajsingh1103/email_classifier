"""
Email Classifier Module - NLP-based using TF-IDF and Cosine Similarity
Classifies emails into Urgent, Normal, Promotional, or Spam categories
"""

import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class EmailClassifier:
    """NLP-based email classifier using TF-IDF and cosine similarity"""
    
    def __init__(self):
        # Reference documents representing each category
        self.reference_docs = {
            'Urgent': [
                "urgent response needed immediately asap critical emergency",
                "please respond as soon as possible this is time sensitive",
                "deadline approaching rush hurry important action required",
                "emergency situation needs immediate attention critical matter"
            ],
            'Promotional': [
                "special offer sale discount deal promotion limited time",
                "buy now save money coupon voucher free shipping clearance",
                "exclusive deal special price amazing offer discount code",
                "limited time offer amazing savings special promotion today"
            ],
            'Normal': [
                "meeting scheduled follow up discussion tomorrow",
                "thank you message information sharing update",
                "regular communication business as usual routine",
                "friendly conversation casual discussion general information"
            ],
            'Spam': [
                "click here now winner congratulations prize claim",
                "free money guaranteed click link urgent action",
                "limited offer exclusive deal must act now",
                "congratulations you won click here claim prize"
            ]
        }
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),  # Use unigrams and bigrams
            min_df=1,
            max_df=0.95
        )
        
        # Preprocess and vectorize reference documents
        self._prepare_reference_vectors()
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text: lowercase, remove punctuation, remove stopwords
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text or not text.strip():
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Get stopwords
        try:
            stop_words = set(stopwords.words('english'))
        except:
            # Fallback if stopwords not available
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Remove stopwords
        words = text.split()
        words = [word for word in words if word not in stop_words and len(word) > 1]
        
        return ' '.join(words)
    
    def _prepare_reference_vectors(self):
        """Preprocess and vectorize all reference documents"""
        # Combine all reference documents for each category
        self.category_texts = {}
        all_docs = []
        
        for category, docs in self.reference_docs.items():
            # Preprocess each document
            preprocessed_docs = [self._preprocess_text(doc) for doc in docs]
            # Combine into single text per category
            combined_text = ' '.join(preprocessed_docs)
            self.category_texts[category] = combined_text
            all_docs.append(combined_text)
        
        # Fit TF-IDF vectorizer on all reference documents
        self.reference_vectors = self.vectorizer.fit_transform(all_docs)
        self.categories = list(self.reference_docs.keys())
    
    def classify(self, email_text: str) -> dict:
        """
        Classify an email using TF-IDF and cosine similarity
        
        Args:
            email_text: The email content to classify
            
        Returns:
            Dictionary with:
                - category: Predicted category
                - similarity_scores: Dictionary of scores for each category
                - confidence: Confidence score (0-1)
        """
        if not email_text or not email_text.strip():
            return {
                'category': 'Normal',
                'similarity_scores': {'Urgent': 0.0, 'Promotional': 0.0, 'Normal': 1.0, 'Spam': 0.0},
                'confidence': 1.0
            }
        
        # Preprocess the input email
        preprocessed_email = self._preprocess_text(email_text)
        
        if not preprocessed_email:
            return {
                'category': 'Normal',
                'similarity_scores': {'Urgent': 0.0, 'Promotional': 0.0, 'Normal': 1.0, 'Spam': 0.0},
                'confidence': 1.0
            }
        
        # Vectorize the input email
        email_vector = self.vectorizer.transform([preprocessed_email])
        
        # Calculate cosine similarity with all reference categories
        similarities = cosine_similarity(email_vector, self.reference_vectors)[0]
        
        # Create similarity scores dictionary
        similarity_scores = {
            category: float(similarity)
            for category, similarity in zip(self.categories, similarities)
        }
        
        # Find the category with highest similarity
        max_similarity = max(similarities)
        max_index = similarities.argmax()
        predicted_category = self.categories[max_index]
        
        # Calculate confidence (normalized difference between top and second)
        sorted_similarities = sorted(similarities, reverse=True)
        if len(sorted_similarities) > 1:
            confidence = float(sorted_similarities[0] - sorted_similarities[1])
            # Normalize confidence to 0-1 range
            confidence = min(1.0, max(0.0, confidence))
        else:
            confidence = float(max_similarity)
        
        return {
            'category': predicted_category,
            'similarity_scores': similarity_scores,
            'confidence': confidence,
            'max_similarity': float(max_similarity)
        }
    
    def get_detailed_analysis(self, email_text: str) -> dict:
        """
        Get detailed analysis including preprocessing info
        
        Args:
            email_text: The email content to analyze
            
        Returns:
            Dictionary with classification results and preprocessing details
        """
        result = self.classify(email_text)
        
        # Add preprocessing details
        if email_text:
            preprocessed = self._preprocess_text(email_text)
            result['preprocessed_text'] = preprocessed
            result['original_length'] = len(email_text)
            result['preprocessed_length'] = len(preprocessed)
        else:
            result['preprocessed_text'] = ""
            result['original_length'] = 0
            result['preprocessed_length'] = 0
        
        return result
