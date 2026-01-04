"""
Flask Web Application for Email Classifier
"""

from flask import Flask, render_template, request, jsonify
from classifier import EmailClassifier

app = Flask(__name__)
classifier = EmailClassifier()


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify_email():
    """API endpoint to classify an email"""
    try:
        data = request.get_json()
        email_text = data.get('email', '')
        
        if not email_text:
            return jsonify({
                'error': 'Email text is required'
            }), 400
        
        # Classify the email using NLP-based approach
        result = classifier.classify(email_text)
        
        # Get detailed analysis
        detailed = classifier.get_detailed_analysis(email_text)
        
        return jsonify({
            'category': result['category'],
            'similarity_scores': result['similarity_scores'],
            'confidence': result['confidence'],
            'max_similarity': result['max_similarity'],
            'preprocessed_text': detailed.get('preprocessed_text', ''),
            'original_length': detailed.get('original_length', 0),
            'preprocessed_length': detailed.get('preprocessed_length', 0)
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
