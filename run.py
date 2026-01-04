"""
Quick start script for Email Classifier
Run this file to start the web application
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app

if __name__ == '__main__':
    print("=" * 60)
    print("Email Classifier - Starting Server")
    print("=" * 60)
    print("\nServer will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server\n")
    print("=" * 60)
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n\nServer stopped. Goodbye!")

