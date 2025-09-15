#!/bin/bash

echo "🚀 Setting up YouTube Sentiment Analyzer..."

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Download NLTK data
echo "📚 Downloading NLTK data..."
python -c "
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
print('NLTK data downloaded successfully!')
"

# Download spaCy model
echo "🧠 Downloading spaCy model..."
python -m spacy download en_core_web_sm

echo "✅ Setup complete!"
echo "🔧 To start the API server, run: python api_server.py"
echo "🌐 Then load the Chrome extension and it will connect automatically!"