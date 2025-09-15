# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import numpy as np
# import pandas as pd
# import pickle
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder
# from youtube_comment_downloader import YoutubeCommentDownloader
# import gensim
# from gensim.models import LdaModel
# from gensim.corpora import Dictionary
# import spacy
# from urllib.parse import urlparse, parse_qs

# app = Flask(__name__)
# CORS(app)  # Enable CORS for Chrome extension

# # Download required NLTK data
# try:
#     nltk.download('vader_lexicon', quiet=True)
#     nltk.download('stopwords', quiet=True)
#     nltk.download('wordnet', quiet=True)
#     nltk.download('punkt', quiet=True)
# except:
#     pass

# # Load spaCy model
# try:
#     nlp = spacy.load("en_core_web_sm")
# except:
#     print("Warning: spaCy model not found. Topic modeling may not work.")
#     nlp = None

# class YouTubeSentimentAnalyzer:
#     def __init__(self):
#         self.downloader = YoutubeCommentDownloader()
#         self.sentiments = SentimentIntensityAnalyzer()
#         self.stop_words = set(stopwords.words('english'))
#         self.lzr = WordNetLemmatizer()
#         self.le = LabelEncoder()
#         self.cv = CountVectorizer(max_features=1500)
        
#         # Initialize models
#         self.models = {
#             'naive_bayes_multinomial': MultinomialNB(),
#             'naive_bayes_gaussian': GaussianNB(),
#             'naive_bayes_bernoulli': BernoulliNB(),
#             'logistic_regression': LogisticRegression(max_iter=1000),
#             'svm': SVC(),
#             'random_forest': RandomForestClassifier()
#         }
        
#         # Model training status
#         self.models_trained = False
#         self.X_features = None
        
#     def extract_video_id(self, url):
#         """Extract video ID from YouTube URL"""
#         parsed_url = urlparse(url)
#         if parsed_url.hostname == 'youtu.be':
#             return parsed_url.path[1:]
#         if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
#             if parsed_url.path == '/watch':
#                 return parse_qs(parsed_url.query)['v'][0]
#             if parsed_url.path[:7] == '/embed/':
#                 return parsed_url.path.split('/')[2]
#             if parsed_url.path[:3] == '/v/':
#                 return parsed_url.path.split('/')[2]
#         return None
    
#     def fetch_comments(self, video_url, limit=1000):
#         """Fetch comments using youtube-comment-downloader"""
#         try:
#             comments = self.downloader.get_comments_from_url(video_url)
            
#             all_comments = []
#             for comment in comments:
#                 all_comments.append(comment)
#                 if len(all_comments) >= limit:
#                     break
            
#             df = pd.DataFrame(all_comments)
#             return df[['text', 'author', 'time', 'votes']] if 'text' in df.columns else pd.DataFrame()
            
#         except Exception as e:
#             print(f"Error fetching comments: {e}")
#             return pd.DataFrame()
    
#     def label_sentiment_vader(self, df):
#         """Label sentiment using VADER (from your notebook)"""
#         df["Positive"] = [self.sentiments.polarity_scores(i)["pos"] for i in df["text"]]
#         df["Negative"] = [self.sentiments.polarity_scores(i)["neg"] for i in df["text"]]
#         df["Neutral"] = [self.sentiments.polarity_scores(i)["neu"] for i in df["text"]]
#         df['Compound'] = [self.sentiments.polarity_scores(i)["compound"] for i in df["text"]]

#         score = df["Compound"].values
#         sentiment = []
#         for i in score:
#             if i >= 0.05:
#                 sentiment.append('Positive')
#             elif i <= -0.05:
#                 sentiment.append('Negative')
#             else:
#                 sentiment.append('Neutral')

#         df["Sentiment"] = sentiment
#         return df
    
#     def preprocess_text(self, text):
#         """Preprocess text (from your notebook)"""
#         # Convert text to lowercase
#         text = text.lower()
        
#         # Remove new line characters
#         text = re.sub(r'\n', ' ', text)
        
#         # Remove punctuations and special characters
#         text = re.sub(r'[^a-zA-Z\s]', '', text)
        
#         # Remove multiple spaces
#         text = re.sub(r'\s+', ' ', text).strip()
        
#         # Tokenize
#         tokens = word_tokenize(text)
        
#         # Remove stopwords
#         processed_tokens = [word for word in tokens if word not in self.stop_words]
        
#         # Join tokens back into string
#         return ' '.join(processed_tokens)
    
#     def prepare_features(self, df):
#         """Prepare features for ML models"""
#         # Apply preprocessing
#         df['processed_text'] = df['text'].apply(self.preprocess_text)
        
#         # Create corpus
#         corpus = df['processed_text'].tolist()
        
#         # Transform to features
#         X = self.cv.fit_transform(corpus).toarray()
        
#         # Encode labels
#         y = self.le.fit_transform(df['Sentiment'])
        
#         return X, y, corpus
    
#     def train_models(self, X, y):
#         """Train all ML models"""
#         print("Training models...")
        
#         # Train models
#         for name, model in self.models.items():
#             try:
#                 if name == 'naive_bayes_gaussian':
#                     # Convert sparse matrix to dense for Gaussian NB
#                     model.fit(X, y)
#                 else:
#                     model.fit(X, y)
#                 print(f"Trained {name}")
#             except Exception as e:
#                 print(f"Error training {name}: {e}")
        
#         self.models_trained = True
#         self.X_features = X
#         print("All models trained successfully!")
    
#     def predict_sentiment(self, texts, model_name='naive_bayes_multinomial'):
#         """Predict sentiment for new texts"""
#         if not self.models_trained:
#             return None
        
#         # Preprocess texts
#         processed_texts = [self.preprocess_text(text) for text in texts]
        
#         # Transform to features
#         X_pred = self.cv.transform(processed_texts).toarray()
        
#         # Get model
#         model = self.models.get(model_name)
#         if not model:
#             return None
        
#         # Predict
#         predictions = model.predict(X_pred)
        
#         # Convert back to labels
#         sentiment_labels = self.le.inverse_transform(predictions)
        
#         # Get probabilities if available
#         probabilities = None
#         if hasattr(model, 'predict_proba'):
#             try:
#                 probabilities = model.predict_proba(X_pred)
#             except:
#                 pass
        
#         return sentiment_labels, probabilities
    
#     def perform_topic_modeling(self, corpus, num_topics=5):
#         """Perform topic modeling using LDA (from your notebook)"""
#         if not nlp:
#             return None
        
#         try:
#             # Tokenize each text
#             tokenized_texts = []
#             for text in corpus:
#                 doc = nlp(text)
#                 tokens = [token.text for token in doc if token.is_alpha]
#                 tokenized_texts.append(tokens)
            
#             # Create a dictionary and corpus
#             dictionary = Dictionary(tokenized_texts)
#             bow_corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
            
#             # Train LDA model
#             lda_model = LdaModel(bow_corpus, num_topics=num_topics, id2word=dictionary, random_state=42)
            
#             # Extract topics
#             topics = []
#             for idx, topic in lda_model.print_topics():
#                 topic_words = [word.split('*')[1].strip().strip('"') for word in topic.split(' + ')]
#                 topics.append({
#                     'topic_id': idx,
#                     'words': topic_words[:10]
#                 })
            
#             return topics
#         except Exception as e:
#             print(f"Topic modeling error: {e}")
#             return None

# # Initialize analyzer
# analyzer = YouTubeSentimentAnalyzer()

# @app.route('/analyze', methods=['POST'])
# def analyze_comments():
#     try:
#         data = request.json
#         video_url = data.get('video_url')
#         model_name = data.get('model', 'naive_bayes_multinomial')
#         comment_limit = data.get('comment_limit', 100)
#         include_topics = data.get('include_topics', False)
        
#         if not video_url:
#             return jsonify({'error': 'Video URL is required'}), 400
        
#         # Step 1: Fetch comments
#         print(f"Fetching comments for {video_url}")
#         df_comments = analyzer.fetch_comments(video_url, limit=comment_limit)
        
#         if df_comments.empty:
#             return jsonify({'error': 'No comments found or unable to fetch comments'}), 400
        
#         # Step 2: Label sentiment using VADER
#         print("Labeling sentiment...")
#         df_labeled = analyzer.label_sentiment_vader(df_comments.copy())
        
#         # Step 3: Prepare features and train models
#         print("Preparing features...")
#         X, y, corpus = analyzer.prepare_features(df_labeled)
        
#         print("Training models...")
#         analyzer.train_models(X, y)
        
#         # Step 4: Get predictions from selected model
#         print(f"Getting predictions using {model_name}")
#         predictions, probabilities = analyzer.predict_sentiment(df_labeled['text'].tolist(), model_name)
        
#         # Step 5: Calculate results
#         unique, counts = np.unique(predictions, return_counts=True)
#         sentiment_counts = dict(zip(unique, counts))
        
#         total_comments = len(predictions)
#         sentiment_percentages = {
#             sentiment: round((count / total_comments) * 100, 1)
#             for sentiment, count in sentiment_counts.items()
#         }
        
#         # Step 6: Topic modeling (if requested)
#         topics = None
#         if include_topics:
#             print("Performing topic modeling...")
#             topics = analyzer.perform_topic_modeling(corpus)
        
#         # Step 7: Prepare response
#         results = {
#             'success': True,
#             'total_comments': total_comments,
#             'sentiment_counts': sentiment_counts,
#             'sentiment_percentages': sentiment_percentages,
#             'model_used': model_name,
#             'topics': topics,
#             'sample_comments': [
#                 {
#                     'text': row['text'],
#                     'author': row.get('author', 'Unknown'),
#                     'predicted_sentiment': pred,
#                     'vader_sentiment': row['Sentiment'],
#                     'confidence': prob.max() if prob is not None else None
#                 }
#                 for (_, row), pred, prob in zip(
#                     df_labeled.iterrows(),
#                     predictions,  
#                     (probabilities if probabilities is not None else [None]*len(predictions)) 
#                 )
#             ]
#         }
        
#         return jsonify(results)
        
#     except Exception as e:
#         print(f"Error in analysis: {e}")
#         return jsonify({'error': str(e)}), 500

# @app.route('/health', methods=['GET'])
# def health_check():
#     return jsonify({'status': 'healthy', 'models_trained': analyzer.models_trained})

# @app.route('/models', methods=['GET'])
# def get_available_models():
#     return jsonify({
#         'models': list(analyzer.models.keys()),
#         'default': 'naive_bayes_multinomial'
#     })

# if __name__ == '__main__':
#     print("Starting YouTube Sentiment Analyzer API...")
#     print("Available models:", list(analyzer.models.keys()))
#     app.run(host='0.0.0.0', port=5001, debug=True)













from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from youtube_comment_downloader import YoutubeCommentDownloader
import gensim
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import spacy
from urllib.parse import urlparse, parse_qs

app = Flask(__name__)
CORS(app)  # Enable CORS for Chrome extension

# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Warning: spaCy model not found. Topic modeling may not work.")
    nlp = None

class YouTubeSentimentAnalyzer:
    def __init__(self):
        self.downloader = YoutubeCommentDownloader()
        self.sentiments = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        self.lzr = WordNetLemmatizer()
        self.le = LabelEncoder()
        self.cv = CountVectorizer(max_features=1500)
        
        # Initialize models
        self.models = {
            'naive_bayes_multinomial': MultinomialNB(),
            'naive_bayes_gaussian': GaussianNB(),
            'naive_bayes_bernoulli': BernoulliNB(),
            'logistic_regression': LogisticRegression(max_iter=1000),
            'svm': SVC(),
            'random_forest': RandomForestClassifier()
        }
        
        # Model training status
        self.models_trained = False
        self.X_features = None
        
    def extract_video_id(self, url):
        """Extract video ID from YouTube URL"""
        parsed_url = urlparse(url)
        if parsed_url.hostname == 'youtu.be':
            return parsed_url.path[1:]
        if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
            if parsed_url.path == '/watch':
                return parse_qs(parsed_url.query)['v'][0]
            if parsed_url.path[:7] == '/embed/':
                return parsed_url.path.split('/')[2]
            if parsed_url.path[:3] == '/v/':
                return parsed_url.path.split('/')[2]
        return None
    
    def fetch_comments(self, video_url, limit=1000):
        """Fetch comments using youtube-comment-downloader"""
        try:
            comments = self.downloader.get_comments_from_url(video_url)
            
            all_comments = []
            for comment in comments:
                all_comments.append(comment)
                if len(all_comments) >= limit:
                    break
            
            df = pd.DataFrame(all_comments)
            return df[['text', 'author', 'time', 'votes']] if 'text' in df.columns else pd.DataFrame()
            
        except Exception as e:
            print(f"Error fetching comments: {e}")
            return pd.DataFrame()
    
    def label_sentiment_vader(self, df):
        """Label sentiment using VADER (from your notebook)"""
        df["Positive"] = [self.sentiments.polarity_scores(i)["pos"] for i in df["text"]]
        df["Negative"] = [self.sentiments.polarity_scores(i)["neg"] for i in df["text"]]
        df["Neutral"] = [self.sentiments.polarity_scores(i)["neu"] for i in df["text"]]
        df['Compound'] = [self.sentiments.polarity_scores(i)["compound"] for i in df["text"]]

        score = df["Compound"].values
        sentiment = []
        for i in score:
            if i >= 0.05:
                sentiment.append('Positive')
            elif i <= -0.05:
                sentiment.append('Negative')
            else:
                sentiment.append('Neutral')

        df["Sentiment"] = sentiment
        return df
    
    def preprocess_text(self, text):
        """Preprocess text (from your notebook)"""
        # Convert text to lowercase
        text = text.lower()
        
        # Remove new line characters
        text = re.sub(r'\n', ' ', text)
        
        # Remove punctuations and special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        processed_tokens = [word for word in tokens if word not in self.stop_words]
        
        # Join tokens back into string
        return ' '.join(processed_tokens)
    
    def prepare_features(self, df):
        """Prepare features for ML models"""
        # Apply preprocessing
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Create corpus
        corpus = df['processed_text'].tolist()
        
        # Transform to features
        X = self.cv.fit_transform(corpus).toarray()
        
        # Encode labels
        y = self.le.fit_transform(df['Sentiment'])
        
        return X, y, corpus
    
    def train_models(self, X, y):
        """Train all ML models"""
        print("Training models...")
        
        # Train models
        for name, model in self.models.items():
            try:
                if name == 'naive_bayes_gaussian':
                    # Convert sparse matrix to dense for Gaussian NB
                    model.fit(X, y)
                else:
                    model.fit(X, y)
                print(f"Trained {name}")
            except Exception as e:
                print(f"Error training {name}: {e}")
        
        self.models_trained = True
        self.X_features = X
        print("All models trained successfully!")
    
    def predict_sentiment(self, texts, model_name='naive_bayes_multinomial'):
        """Predict sentiment for new texts"""
        if not self.models_trained:
            return None
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Transform to features
        X_pred = self.cv.transform(processed_texts).toarray()
        
        # Get model
        model = self.models.get(model_name)
        if not model:
            return None
        
        # Predict
        predictions = model.predict(X_pred)
        
        # Convert back to labels
        sentiment_labels = self.le.inverse_transform(predictions)
        
        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(X_pred)
            except:
                pass
        
        return sentiment_labels, probabilities
    
    def perform_topic_modeling(self, corpus, num_topics=5):
        """Perform topic modeling using LDA (from your notebook)"""
        if not nlp:
            return None
        
        try:
            # Tokenize each text
            tokenized_texts = []
            for text in corpus:
                doc = nlp(text)
                tokens = [token.text for token in doc if token.is_alpha]
                tokenized_texts.append(tokens)
            
            # Create a dictionary and corpus
            dictionary = Dictionary(tokenized_texts)
            bow_corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
            
            # Train LDA model
            lda_model = LdaModel(bow_corpus, num_topics=num_topics, id2word=dictionary, random_state=42)
            
            # Extract topics
            topics = []
            for idx, topic in lda_model.print_topics():
                topic_words = [word.split('*')[1].strip().strip('"') for word in topic.split(' + ')]
                topics.append({
                    'topic_id': idx,
                    'words': topic_words[:10]
                })
            
            return topics
        except Exception as e:
            print(f"Topic modeling error: {e}")
            return None

# Initialize analyzer
analyzer = YouTubeSentimentAnalyzer()

@app.route('/analyze', methods=['POST'])
def analyze_comments():
    try:
        data = request.json
        video_url = data.get('video_url')
        model_name = data.get('model', 'naive_bayes_multinomial')
        comment_limit = data.get('comment_limit', 100)
        include_topics = data.get('include_topics', False)
        
        if not video_url:
            return jsonify({'error': 'Video URL is required'}), 400
        
        # Step 1: Fetch comments
        print(f"Fetching comments for {video_url}")
        df_comments = analyzer.fetch_comments(video_url, limit=comment_limit)
        
        if df_comments.empty:
            return jsonify({'error': 'No comments found or unable to fetch comments'}), 400
        
        # Step 2: Label sentiment using VADER
        print("Labeling sentiment...")
        df_labeled = analyzer.label_sentiment_vader(df_comments.copy())
        
        # Step 3: Get predictions from selected model
        print(f"Getting predictions using {model_name}")
        predictions, probabilities = analyzer.predict_sentiment(df_labeled['text'].tolist(), model_name)
        
        if predictions is None:
            return jsonify({'error': 'Model not trained. Please restart the server.'}), 500

        # Step 4: Calculate results
        unique, counts = np.unique(predictions, return_counts=True)
        sentiment_counts = dict(zip(unique, counts))
        
        total_comments = len(predictions)
        sentiment_percentages = {
            sentiment: round((count / total_comments) * 100, 1)
            for sentiment, count in sentiment_counts.items()
        }
        
        # Step 5: Topic modeling (if requested)
        topics = None
        if include_topics:
            print("Performing topic modeling...")
            
            # Prepare corpus for topic modeling
            # This is done here since it requires the raw comments from this specific run
            df_labeled['processed_text'] = df_labeled['text'].apply(analyzer.preprocess_text)
            corpus = df_labeled['processed_text'].tolist()
            
            topics = analyzer.perform_topic_modeling(corpus)
        
        # Step 6: Prepare response
        results = {
            'success': True,
            'total_comments': total_comments,
            'sentiment_counts': sentiment_counts,
            'sentiment_percentages': sentiment_percentages,
            'model_used': model_name,
            'topics': topics,
            'sample_comments': [
                {
                    'text': row['text'],
                    'author': row.get('author', 'Unknown'),
                    'predicted_sentiment': pred,
                    'vader_sentiment': row['Sentiment'],
                    'confidence': prob.max() if prob is not None else None
                }
                for (_, row), pred, prob in zip(
                    df_labeled.iterrows(),
                    predictions,  
                    (probabilities if probabilities is not None else [None]*len(predictions)) 
                )
            ]
        }
        
        return jsonify(results)
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'models_trained': analyzer.models_trained})

@app.route('/models', methods=['GET'])
def get_available_models():
    return jsonify({
        'models': list(analyzer.models.keys()),
        'default': 'naive_bayes_multinomial'
    })

if __name__ == '__main__':
    print("Starting YouTube Sentiment Analyzer API...")
    print("Available models:", list(analyzer.models.keys()))
    
    # New Code: Train models once at startup
    initial_df = pd.DataFrame([
        {'text': 'This is a great comment!', 'author': 'test', 'time': 'now', 'votes': 0},
        {'text': 'This is a terrible comment.', 'author': 'test', 'time': 'now', 'votes': 0},
        {'text': 'This is a neutral comment.', 'author': 'test', 'time': 'now', 'votes': 0}
    ])
    df_labeled_initial = analyzer.label_sentiment_vader(initial_df)
    X_initial, y_initial, _ = analyzer.prepare_features(df_labeled_initial)
    analyzer.train_models(X_initial, y_initial)

    app.run(host='0.0.0.0', port=5001, debug=True)