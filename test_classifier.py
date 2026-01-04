"""
Test cases for NLP-based Email Classifier
"""

from classifier import EmailClassifier


def test_classifier():
    """Run test cases for the NLP-based email classifier"""
    classifier = EmailClassifier()
    
    test_cases = [
        # (email_text, expected_category, description)
        ("Urgent: Please respond ASAP. This is critical!", "Urgent", "Basic urgent email"),
        ("This is an emergency situation that needs immediate attention", "Urgent", "Emergency keyword"),
        ("Check out our amazing sale! 50% discount on all items", "Promotional", "Sale and discount keywords"),
        ("Limited time offer - buy now and save money!", "Promotional", "Promotional keywords"),
        ("Hi, just wanted to follow up on our meeting", "Normal", "Normal email"),
        ("Meeting scheduled for tomorrow at 3 PM", "Normal", "Normal email without keywords"),
        ("Congratulations! You have won a prize! Click here now!", "Spam", "Spam email"),
        ("Free money guaranteed! Click link urgent action required", "Spam", "Spam with urgent words"),
        ("", "Normal", "Empty email"),
        ("This is critical and time-sensitive deadline approaching", "Urgent", "Multiple urgent keywords"),
    ]
    
    print("=" * 70)
    print("NLP-based Email Classifier Test Suite")
    print("=" * 70)
    print()
    
    passed = 0
    failed = 0
    
    for email, expected, description in test_cases:
        result = classifier.classify(email)
        category = result['category']
        confidence = result['confidence']
        max_sim = result['max_similarity']
        
        status = "[PASS]" if category == expected else "[FAIL]"
        
        if category == expected:
            passed += 1
        else:
            failed += 1
        
        print(f"{status} | {description}")
        print(f"   Input: {email[:60]}{'...' if len(email) > 60 else ''}")
        print(f"   Expected: {expected}, Got: {category}")
        print(f"   Confidence: {confidence:.3f}, Max Similarity: {max_sim:.3f}")
        
        # Show similarity scores
        scores = result['similarity_scores']
        score_str = ", ".join([f"{k}: {v:.3f}" for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)])
        print(f"   Similarities: {score_str}")
        print()
    
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 70)
    
    # Test preprocessing
    print("\n" + "=" * 70)
    print("Testing Text Preprocessing")
    print("=" * 70)
    test_text = "Hello! This is a TEST email with PUNCTUATION!!!"
    preprocessed = classifier._preprocess_text(test_text)
    print(f"Original: {test_text}")
    print(f"Preprocessed: {preprocessed}")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    test_classifier()
