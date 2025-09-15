# YouTube Comment Sentiment Analyzer 🎭

An advanced Chrome Extension that provides real-time sentiment analysis of YouTube video comments using machine learning models.

## ✨ Features

- **Real-time Analysis:** Get instant sentiment insights on any YouTube video's comments.
- **Multiple ML Models:** Choose from a variety of machine learning models for sentiment prediction, including Naive Bayes, Logistic Regression, and SVM.
- **VADER Sentiment:** Includes VADER analysis for a quick, rule-based sentiment score.
- **Topic Modeling:** Uncover the most discussed topics within the comments to understand the core of the conversation.
- **Comprehensive Metrics:** Displays a breakdown of positive, neutral, and negative comments, along with confidence scores.
- **User-Friendly Interface:** A clean and intuitive popup provides a clear overview of the analysis results.

## 🚀 How to Install and Run

This extension requires a Python backend to function. Follow these steps to set it up locally.

### 1. Backend Setup (Python API)

1.  Clone this repository or download the source code.
2.  Navigate to the `youtube-sentiment-api` directory in your terminal.
3.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
4.  Run the API server:
    ```bash
    python api_server.py
    ```
    Your API server will now be running on `http://127.0.0.1:5001`.

### 2. Frontend Setup (Chrome Extension)

1.  Open Google Chrome and navigate to `chrome://extensions`.
2.  Enable **Developer mode** by toggling the switch in the top-right corner.
3.  Click the **Load unpacked** button.
4.  Select the `youtube-sentiment-analyzer` directory that contains the `manifest.json` file.
5.  The extension will now be loaded and visible in your browser's toolbar.

## 🖼️ Preview

![Preview of the YouTube Comment Sentiment Analyzer](/Users/akshatpeter/Downloads/youtube-sentiment-analyzer/youtube-sentiment-api/images/sentiment_1.png)
![Preview of the YouTube Comment Sentiment Analyzer](/Users/akshatpeter/Downloads/youtube-sentiment-analyzer/youtube-sentiment-api/images/sentiment_2.png)

## 💡 How to Use

1.  Navigate to any YouTube video page.
2.  Click the extension icon in your Chrome toolbar.
3.  Select your desired options, such as the analysis model and the number of comments to analyze.
4.  Click the "Analyze Comments" button. The analysis will take a moment, and the results will be displayed in the popup.

## 🤝 Contributing

Contributions are welcome! If you find a bug or have a suggestion, please open an issue or submit a pull request.
