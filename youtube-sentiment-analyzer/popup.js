// Popup script for YouTube Sentiment Analyzer
document.addEventListener('DOMContentLoaded', async function () {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const viewResultsBtn = document.getElementById('viewResultsBtn');
    const statusDiv = document.getElementById('status');
    const resultsPreview = document.getElementById('resultsPreview');
    const modelSelect = document.getElementById('modelSelect');
    const commentLimit = document.getElementById('commentLimit');
    const topicModeling = document.getElementById('topicModeling');

    let currentTab = null;
    let analysisResults = null;

    // Get current tab
    try {
        const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
        currentTab = tabs[0];

        if (!currentTab.url.includes('https://www.youtube.com/watch')) {
            showStatus('error', 'Please navigate to a YouTube video first!');
            analyzeBtn.disabled = true;
            return;
        }
    } catch (error) {
        showStatus('error', 'Unable to access current tab');
        return;
    }

    // Load previous results if available
    chrome.storage.local.get(['lastAnalysisResults'], (result) => {
        if (result.lastAnalysisResults) {
            analysisResults = result.lastAnalysisResults;
            showResults(analysisResults);
            viewResultsBtn.style.display = 'block';
        }
    });

    // // Analyze button click handler
    // analyzeBtn.addEventListener('click', async () => {
    //     if (!currentTab) return;

    //     const options = {
    //         model: modelSelect.value,
    //         commentLimit: parseInt(commentLimit.value),
    //         topicModeling: topicModeling.value
    //     };

    //     analyzeBtn.disabled = true;
    //     analyzeBtn.textContent = '🔄 Analyzing...';

    //     try {
    //         await chrome.tabs.sendMessage(currentTab.id, {
    //             action: 'startAnalysis',
    //             options: options
    //         });
    //     } catch (error) {
    //         showStatus('error', 'Failed to start analysis. Please refresh the page.');
    //         resetAnalyzeButton();
    //     }
    // });


    analyzeBtn.addEventListener('click', () => {
    const selectedModel = modelSelect.value;
    const commentLimitValue = commentLimit.value;
    const topicModelingValue = topicModeling.value === 'true';

    if (!currentTab) {
        showStatus('error', 'Error: No active tab found.');
        return;
    }

    if (!currentTab.url.includes('googleusercontent.com/youtube.com/2')) {
        showStatus('error', 'Please navigate to a YouTube video first!');
        return;
    }

    // New code:
    const options = {
        model: selectedModel,
        commentLimit: parseInt(commentLimitValue, 10),
        topicModeling: topicModelingValue
    };
    
    // Send the message with the new options object
    chrome.tabs.sendMessage(currentTab.id, {
        action: 'startAnalysis',
        options: options
    });
    
    // Show status message
    showStatus('info', 'Starting analysis...');
});




    // View results button click handler
    viewResultsBtn.addEventListener('click', () => {
        if (analysisResults) {
            openResultsPage(analysisResults);
        }
    });

    // Listen for messages from content script
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
        switch (message.action) {
            case 'analysisStatus':
                showStatus(message.status, message.message);
                break;

            case 'analysisComplete':
                analysisResults = message.results;
                showResults(analysisResults);
                showStatus('success', 'Analysis completed!');
                resetAnalyzeButton();
                viewResultsBtn.style.display = 'block';
                break;
        }
    });

    function showStatus(type, message) {
        statusDiv.className = `status ${type}`;
        statusDiv.textContent = message;
        statusDiv.style.display = 'block';

        if (type === 'success') {
            setTimeout(() => {
                statusDiv.style.display = 'none';
            }, 3000);
        }
    }

    function resetAnalyzeButton() {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = '🚀 Analyze Comments';
    }

    function showResults(results) {
        if (!results) return;

        const { sentimentPercentages, sentimentCounts, totalComments } = results;

        // Update sentiment bar
        const sentimentBar = document.getElementById('sentimentBar');
        sentimentBar.innerHTML = `
            <div class="positive" style="width: ${sentimentPercentages.positive}%" title="Positive: ${sentimentPercentages.positive}%"></div>
            <div class="neutral" style="width: ${sentimentPercentages.neutral}%" title="Neutral: ${sentimentPercentages.neutral}%"></div>
            <div class="negative" style="width: ${sentimentPercentages.negative}%" title="Negative: ${sentimentPercentages.negative}%"></div>
        `;

        // Update stats
        const sentimentStats = document.getElementById('sentimentStats');
        sentimentStats.innerHTML = `
            <div style="display: flex; justify-content: space-between; font-size: 11px; margin-top: 5px;">
                <span>😊 ${sentimentCounts.positive}</span>
                <span>😐 ${sentimentCounts.neutral}</span>
                <span>😞 ${sentimentCounts.negative}</span>
            </div>
            <div style="text-align: center; margin-top: 5px; font-size: 11px; opacity: 0.8;">
                Total: ${totalComments} comments
            </div>
        `;

        resultsPreview.style.display = 'block';
    }

    function openResultsPage(results) {
        // Create a new tab with detailed results
        const resultsHTML = generateResultsHTML(results);
        const blob = new Blob([resultsHTML], { type: 'text/html' });
        const url = URL.createObjectURL(blob);
        chrome.tabs.create({ url: url });
    }

    function generateResultsHTML(results) {
        const { videoInfo, sentimentPercentages, sentimentCounts, topics, insights, overallSentiment, comments } = results;

        return `
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sentiment Analysis Results</title>
            <meta charset="UTF-8">
            <style>
                body {
                    font-family: 'Segoe UI', system-ui, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    color: white;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background: rgba(255,255,255,0.1);
                    padding: 30px;
                    border-radius: 15px;
                    backdrop-filter: blur(10px);
                }
                .header {
                    text-align: center;
                    margin-bottom: 30px;
                }
                .video-info, .insights, .chart-container, .comment-list, .topic-cloud {
                    background: rgba(255,255,255,0.1);
                    padding: 20px;
                    border-radius: 10px;
                    margin: 20px 0;
                }
                .sentiment-overview {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }
                .sentiment-card {
                    background: rgba(255,255,255,0.1);
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                }
                .comment-item {
                    background: rgba(255,255,255,0.1);
                    padding: 10px;
                    margin: 10px 0;
                    border-radius: 5px;
                    border-left: 4px solid;
                }
                .comment-item.positive { border-left-color: #28a745; }
                .comment-item.neutral { border-left-color: #ffc107; }
                .comment-item.negative { border-left-color: #dc3545; }
                .topic-tag {
                    display: inline-block;
                    background: rgba(255,255,255,0.2);
                    padding: 5px 10px;
                    margin: 5px;
                    border-radius: 15px;
                    font-size: 12px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎬 YouTube Comment Sentiment Analysis</h1>
                    <p>Analysis completed on ${new Date(results.timestamp).toLocaleString()}</p>
                </div>
                
                <div class="video-info">
                    <h2>📹 Video Information</h2>
                    <p><strong>Title:</strong> ${videoInfo.title}</p>
                    <p><strong>Channel:</strong> ${videoInfo.channel}</p>
                    <p><strong>Views:</strong> ${videoInfo.views}</p>
                    <p><strong>URL:</strong> <a href="${videoInfo.url}" style="color: #87CEEB;">${videoInfo.url}</a></p>
                </div>
                
                <div class="sentiment-overview">
                    <div class="sentiment-card positive">
                        <h3>😊 Positive</h3>
                        <div style="font-size: 2em;">${sentimentPercentages.positive}%</div>
                        <div>${sentimentCounts.positive} comments</div>
                    </div>
                    <div class="sentiment-card neutral">
                        <h3>😐 Neutral</h3>
                        <div style="font-size: 2em;">${sentimentPercentages.neutral}%</div>
                        <div>${sentimentCounts.neutral} comments</div>
                    </div>
                    <div class="sentiment-card negative">
                        <h3>😞 Negative</h3>
                        <div style="font-size: 2em;">${sentimentPercentages.negative}%</div>
                        <div>${sentimentCounts.negative} comments</div>
                    </div>
                </div>

                <div class="insights">
                    <h2>🔍 Insights</h2>
                    <p>${insights}</p>
                </div>
            </div>
        </body>
        </html>
        `;
    }
});
