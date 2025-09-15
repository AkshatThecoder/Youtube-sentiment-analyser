// YouTube Comment Sentiment Analyzer - Content Script
class YouTubeSentimentAnalyzer {
    constructor() {
        this.comments = [];
        this.isAnalyzing = false;
        this.results = null;
        
        // Initialize when DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.init());
        } else {
            this.init();
        }
    }
    
    init() {
        console.log('YouTube Sentiment Analyzer initialized');
        this.addAnalysisButton();
        this.setupMessageListener();
    }
    
    addAnalysisButton() {
        // Wait for YouTube to load
        const checkForVideoPlayer = () => {
            const videoPlayer = document.querySelector('#movie_player');
            const videoTitle = document.querySelector('#title h1');
            
            if (videoPlayer && videoTitle && !document.querySelector('#sentiment-analysis-btn')) {
                this.createAnalysisButton();
            } else {
                setTimeout(checkForVideoPlayer, 1000);
            }
        };
        
        checkForVideoPlayer();
    }
    
    createAnalysisButton() {
        const button = document.createElement('button');
        button.id = 'sentiment-analysis-btn';
        button.innerHTML = '🎭 Analyze Sentiment';
        button.className = 'sentiment-analysis-button';
        
        button.addEventListener('click', () => {
            chrome.runtime.sendMessage({
                action: 'openPopup'
            });
        });
        
        // Add to video controls area
        const controls = document.querySelector('#top-level-buttons-computed');
        if (controls) {
            controls.appendChild(button);
        }
    }
    
    setupMessageListener() {
        chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
            switch (message.action) {
                case 'startAnalysis':
                    this.startAnalysis(message.options);
                    break;
                case 'getVideoInfo':
                    sendResponse(this.getVideoInfo());
                    break;
                case 'getResults':
                    sendResponse(this.results);
                    break;
            }
            return true;
        });
    }
    
    getVideoInfo() {
        const videoId = new URLSearchParams(window.location.search).get('v');
        const title = document.querySelector('#title h1')?.textContent || 'Unknown';
        const channel = document.querySelector('#channel-name a')?.textContent || 'Unknown';
        const views = document.querySelector('#info-text span')?.textContent || '0';
        
        return {
            videoId,
            title: title.trim(),
            channel: channel.trim(),
            views: views.trim(),
            url: window.location.href
        };
    }
    
    // async startAnalysis(options) {
    //     if (this.isAnalyzing) {
    //         chrome.runtime.sendMessage({
    //             action: 'analysisStatus',
    //             status: 'error',
    //             message: 'Analysis already in progress'
    //         });
    //         return;
    //     }
        
    //     this.isAnalyzing = true;
        
    //     try {
    //         chrome.runtime.sendMessage({
    //             action: 'analysisStatus',
    //             status: 'loading',
    //             message: 'Extracting comments...'
    //         });
            
             
    //         // Go straight to analysis. The Python API will fetch its own comments.
    //         // The 'comments' parameter passed here is null, as it's not needed for the API path.
    //         const results = await this.performSentimentAnalysis(null, options);
            
    //         if (comments.length === 0) {
    //             throw new Error('No comments found');
    //         }
            
    //         chrome.runtime.sendMessage({
    //             action: 'analysisStatus',
    //             status: 'loading',
    //             message: `Analyzing ${comments.length} comments...`
    //         });
            
    //         // Perform sentiment analysis
    //         const results = await this.performSentimentAnalysis(comments, options);
            
    //         this.results = {
    //             ...results,
    //             videoInfo: this.getVideoInfo(),
    //             timestamp: new Date().toISOString(),
    //             totalComments: comments.length
    //         };
            
    //         // Store results
    //         chrome.storage.local.set({
    //             lastAnalysisResults: this.results
    //         });
            
    //         chrome.runtime.sendMessage({
    //             action: 'analysisComplete',
    //             results: this.results
    //         });
            
    //     } catch (error) {
    //         console.error('Analysis failed:', error);
    //         chrome.runtime.sendMessage({
    //             action: 'analysisStatus',
    //             status: 'error',
    //             message: error.message
    //         });
    //     } finally {
    //         this.isAnalyzing = false;
    //     }
    // }
    
    // async extractComments(limit = 100) {
    //     return new Promise((resolve) => {
    //         const comments = [];
    //         let attempts = 0;
    //         const maxAttempts = 10;
            
    //         const scrollAndCollect = () => {
    //             // Scroll to load more comments
    //             window.scrollTo(0, document.body.scrollHeight);
                
    //             setTimeout(() => {
    //                 const commentElements = document.querySelectorAll('#content-text');
                    
    //                 commentElements.forEach(el => {
    //                     const text = el.textContent?.trim();
    //                     if (text && text.length > 5 && !comments.some(c => c.text === text)) {
    //                         comments.push({
    //                             text: text,
    //                             author: el.closest('ytd-comment-thread-renderer')
    //                                 ?.querySelector('#author-text')?.textContent?.trim() || 'Unknown',
    //                             timestamp: el.closest('ytd-comment-thread-renderer')
    //                                 ?.querySelector('#published-time-text')?.textContent?.trim() || 'Unknown'
    //                         });
    //                     }
    //                 });
                    
    //                 attempts++;
                    
    //                 if (comments.length >= limit || attempts >= maxAttempts) {
    //                     resolve(comments.slice(0, limit));
    //                 } else {
    //                     scrollAndCollect();
    //                 }
    //             }, 2000);
    //         };
            
    //         // Start collecting
    //         scrollAndCollect();
    //     });
    // }


    
async startAnalysis(options) {
    if (this.isAnalyzing) {
        chrome.runtime.sendMessage({
            action: 'analysisStatus',
            status: 'error',
            message: 'Analysis already in progress'
        });
        return;
    }
    
    this.isAnalyzing = true;
    
    try {
        chrome.runtime.sendMessage({
            action: 'analysisStatus',
            status: 'loading',
            message: 'Sending request to analysis server...'
        });
        
        // Go straight to analysis. The Python API will fetch its own comments.
        // The 'comments' parameter passed here is null, as it's not needed for the API path.
        const results = await this.performSentimentAnalysis(null, options);
        
        this.results = {
            ...results,
            videoInfo: this.getVideoInfo(),
            timestamp: new Date().toISOString()
        };
        
        // Store results
        chrome.storage.local.set({
            lastAnalysisResults: this.results
        });
        
        chrome.runtime.sendMessage({
            action: 'analysisComplete',
            results: this.results
        });
        
    } catch (error) {
        console.error('Analysis failed:', error);
        chrome.runtime.sendMessage({
            action: 'analysisStatus',
            status: 'error',
            message: error.message || 'Analysis failed. Is the Python server running?'
        });
    } finally {
        this.isAnalyzing = false;
    }
}




    async extractComments(limit = 100) {
    return new Promise((resolve) => {
        const comments = new Map();
        let lastCommentCount = 0;
        let scrollAttempts = 0;
        const maxScrollAttempts = 20; // Increased attempts for longer pages

        const scrollInterval = setInterval(() => {
            // Scroll to the bottom of the page
            window.scrollTo(0, document.documentElement.scrollHeight);
            
            // Find all comment elements
            const commentElements = document.querySelectorAll('ytd-comment-thread-renderer #content-text');
            
            commentElements.forEach(el => {
                const text = el.textContent?.trim();
                // Use the text as a key to avoid duplicates
                if (text && text.length > 5 && !comments.has(text)) {
                    const threadElement = el.closest('ytd-comment-thread-renderer');
                    comments.set(text, {
                        text: text,
                        author: threadElement?.querySelector('#author-text')?.textContent?.trim() || 'Unknown',
                        timestamp: threadElement?.querySelector('.published-time-text a')?.textContent?.trim() || 'Unknown'
                    });
                }
            });

            // Check if we should stop
            // 1. We've reached the desired limit
            // 2. We've scrolled multiple times but no new comments are loading (end of page)
            // 3. We've tried too many times (failsafe)
            const shouldStop = comments.size >= limit || 
                               (comments.size === lastCommentCount && scrollAttempts > 3) ||
                               scrollAttempts >= maxScrollAttempts;

            if (shouldStop) {
                clearInterval(scrollInterval);
                // Convert Map values to an array and slice to the limit
                resolve(Array.from(comments.values()).slice(0, limit));
            } else {
                lastCommentCount = comments.size;
                scrollAttempts++;
            }
        }, 1500); // Wait 1.5 seconds between scrolls
    });
}


    
    async performSentimentAnalysis(comments, options) {
        try {
            // First, try to use the Python API with your actual ML models
            const apiResults = await this.callPythonAPI(options);
            if (apiResults) {
                return this.formatAPIResults(apiResults, comments);
            }
        } catch (error) {
            console.log('Python API not available, falling back to client-side analysis:', error);
        }
        
        // Fallback to client-side analysis if API is not available
        return this.performClientSideAnalysis(comments, options);
    }
    
    async callPythonAPI(options) {
        const API_URL = 'http://localhost:5001'; // Your Python Flask server
        
        try {
            const response = await fetch(`${API_URL}/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    video_url: window.location.href,
                    model: this.mapModelName(options.model),
                    comment_limit: options.commentLimit,
                    include_topics: options.topicModeling
                })
            });
            
            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }
            
            const data = await response.json();
            if (data.success) {
                return data;
            } else {
                throw new Error(data.error || 'API analysis failed');
            }
        } catch (error) {
            console.error('API call failed:', error);
            throw error;
        } 
    }
    
    mapModelName(extensionModel) {
        const modelMap = {
            'naive_bayes': 'naive_bayes_multinomial',
            'logistic_regression': 'logistic_regression',
            'svm': 'svm',
            'neural_network': 'logistic_regression' // Fallback since we don't have neural network in sklearn
        };
        return modelMap[extensionModel] || 'naive_bayes_multinomial';
    }
    
    formatAPIResults(apiData, originalComments) {
        const sentimentCounts = {
            positive: apiData.sentiment_counts.Positive || 0,
            neutral: apiData.sentiment_counts.Neutral || 0,
            negative: apiData.sentiment_counts.Negative || 0
        };
        
        const sentimentPercentages = {
            positive: apiData.sentiment_percentages.Positive || 0,
            neutral: apiData.sentiment_percentages.Neutral || 0,
            negative: apiData.sentiment_percentages.Negative || 0
        };
        
        // Format comments with predictions
        const formattedComments = apiData.sample_comments.map(comment => ({
            text: comment.text,
            author: comment.author,
            timestamp: 'API',
            sentiment: {
                label: comment.predicted_sentiment.toLowerCase(),
                confidence: comment.confidence || 0.8,
                score: comment.confidence || 0.8
            },
            processedText: comment.text
        }));
        
        return {
            sentimentCounts,
            sentimentPercentages,
            comments: formattedComments,
            topics: apiData.topics,
            modelUsed: apiData.model_used,
            overallSentiment: this.calculateOverallSentiment(sentimentCounts),
            insights: this.generateInsights(sentimentCounts, apiData.total_comments),
            source: 'python_api',
            totalComments: apiData.total_comments // Use the count from the API
        };
    }
    
    // async performClientSideAnalysis(comments, options) {
    //     // Fallback client-side analysis (your original code)
    //     const sentimentResults = comments.map(comment => ({
    //         ...comment,
    //         sentiment: this.analyzeSentiment(comment.text),
    //         processedText: this.preprocessText(comment.text)
    //     }));
        
    //     const sentimentCounts = {
    //         positive: sentimentResults.filter(r => r.sentiment.label === 'positive').length,
    //         neutral: sentimentResults.filter(r => r.sentiment.label === 'neutral').length,
    //         negative: sentimentResults.filter(r => r.sentiment.label === 'negative').length
    //     };
        
    //     const totalComments = sentimentResults.length;
    //     const sentimentPercentages = {
    //         positive: ((sentimentCounts.positive / totalComments) * 100).toFixed(1),
    //         neutral: ((sentimentCounts.neutral / totalComments) * 100).toFixed(1),
    //         negative: ((sentimentCounts.negative / totalComments) * 100).toFixed(1)
    //     };
        
    //     let topicResults = null;
    //     if (options.topicModeling === 'true') {
    //         topicResults = this.performTopicModeling(sentimentResults);
    //     }
        
    //     return {
    //         sentimentCounts,
    //         sentimentPercentages,
    //         comments: sentimentResults,
    //         topics: topicResults,
    //         modelUsed: options.model,
    //         overallSentiment: this.calculateOverallSentiment(sentimentCounts),
    //         insights: this.generateInsights(sentimentCounts, totalComments),
    //         source: 'client_side',
    //         totalAnalyzed: totalComments
    //     };
    // }
    
    // In content.js, replace the entire performClientSideAnalysis function
async performClientSideAnalysis(options) {
    chrome.runtime.sendMessage({
        action: 'analysisStatus',
        status: 'loading',
        message: 'Extracting comments for fallback analysis...'
    });

    const comments = await this.extractComments(options.commentLimit);
    if (comments.length === 0) {
        throw new Error('No comments found for client-side analysis.');
    }

    chrome.runtime.sendMessage({
        action: 'analysisStatus',
        status: 'loading',
        message: `Analyzing ${comments.length} comments...`
    });

    const sentimentResults = comments.map(comment => ({
        ...comment,
        sentiment: this.analyzeSentiment(comment.text),
        processedText: this.preprocessText(comment.text)
    }));
    
    const sentimentCounts = {
        positive: sentimentResults.filter(r => r.sentiment.label === 'positive').length,
        neutral: sentimentResults.filter(r => r.sentiment.label === 'neutral').length,
        negative: sentimentResults.filter(r => r.sentiment.label === 'negative').length
    };
    
    const totalComments = sentimentResults.length;
    const sentimentPercentages = {
        positive: parseFloat(((sentimentCounts.positive / totalComments) * 100).toFixed(1)),
        neutral: parseFloat(((sentimentCounts.neutral / totalComments) * 100).toFixed(1)),
        negative: parseFloat(((sentimentCounts.negative / totalComments) * 100).toFixed(1))
    };
    
    let topicResults = null;
    if (options.topicModeling === 'true') {
        topicResults = this.performTopicModeling(sentimentResults);
    }
    
    return {
        sentimentCounts,
        sentimentPercentages,
        comments: sentimentResults,
        topics: topicResults,
        modelUsed: `Client-Side (${options.model})`,
        overallSentiment: this.calculateOverallSentiment(sentimentCounts),
        insights: this.generateInsights(sentimentCounts, totalComments),
        source: 'client_side',
        totalComments: totalComments
    };
}


    analyzeSentiment(text) {
        // Simple rule-based sentiment analysis
        const positiveWords = ['good', 'great', 'awesome', 'amazing', 'love', 'best', 'excellent', 'perfect', 'wonderful', 'fantastic', 'brilliant', 'outstanding', 'superb', 'incredible', 'magnificent'];
        const negativeWords = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disgusting', 'pathetic', 'stupid', 'annoying', 'useless', 'garbage', 'trash', 'boring', 'sucks'];
        
        const lowerText = text.toLowerCase();
        let score = 0;
        
        positiveWords.forEach(word => {
            if (lowerText.includes(word)) score += 1;
        });
        
        negativeWords.forEach(word => {
            if (lowerText.includes(word)) score -= 1;
        });
        
        // Check for emoticons and emojis
        if (/😀|😃|😄|😁|😊|🙂|😍|🥰|😘|🤩|👍|❤️|💖|🔥/g.test(text)) score += 1;
        if (/😢|😭|😡|😠|🤬|👎|💔|😒|🙄|😤/g.test(text)) score -= 1;
        
        let label, confidence;
        if (score > 0) {
            label = 'positive';
            confidence = Math.min(0.6 + (score * 0.2), 1.0);
        } else if (score < 0) {
            label = 'negative';
            confidence = Math.min(0.6 + (Math.abs(score) * 0.2), 1.0);
        } else {
            label = 'neutral';
            confidence = 0.5;
        }
        
        return { label, confidence, score };
    }
    
    preprocessText(text) {
        return text.toLowerCase()
            .replace(/[^\w\s]/g, ' ')
            .replace(/\s+/g, ' ')
            .trim();
    }
    
    performTopicModeling(sentimentResults) {
        // Simple keyword extraction for topics
        const allText = sentimentResults.map(r => r.processedText).join(' ');
        const words = allText.split(' ').filter(w => w.length > 3);
        
        const wordFreq = {};
        words.forEach(word => {
            wordFreq[word] = (wordFreq[word] || 0) + 1;
        });
        
        const commonWords = ['this', 'that', 'with', 'from', 'they', 'been', 'have', 'were', 'said', 'each', 'which', 'their', 'time', 'will', 'about', 'would', 'there', 'could', 'other', 'more', 'very', 'what', 'know', 'just', 'first', 'into', 'over', 'think', 'also', 'your', 'work', 'life', 'only', 'new', 'years', 'way', 'may', 'say'];
        
        const topics = Object.entries(wordFreq)
            .filter(([word]) => !commonWords.includes(word))
            .sort(([,a], [,b]) => b - a)
            .slice(0, 10)
            .map(([word, freq]) => ({ word, frequency: freq }));
        
        return topics;
    }
    
    calculateOverallSentiment(counts) {
        const total = counts.positive + counts.neutral + counts.negative;
        const positiveRatio = counts.positive / total;
        const negativeRatio = counts.negative / total;
        
        if (positiveRatio > negativeRatio + 0.1) return 'positive';
        if (negativeRatio > positiveRatio + 0.1) return 'negative';
        return 'neutral';
    }
    
    generateInsights(counts, total) {
        const insights = [];
        const positivePercent = (counts.positive / total) * 100;
        const negativePercent = (counts.negative / total) * 100;
        
        if (positivePercent > 60) {
            insights.push("This video has overwhelmingly positive reception!");
        } else if (negativePercent > 60) {
            insights.push("This video has received mostly negative feedback.");
        } else if (Math.abs(positivePercent - negativePercent) < 10) {
            insights.push("This video has mixed reactions from viewers.");
        }
        
        if (counts.neutral / total > 0.5) {
            insights.push("Many comments are neutral or informational.");
        }
        
        return insights;
    }
}

// Initialize the analyzer
new YouTubeSentimentAnalyzer();