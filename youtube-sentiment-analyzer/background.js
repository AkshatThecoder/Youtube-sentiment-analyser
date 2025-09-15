// Background script for YouTube Sentiment Analyzer
chrome.runtime.onInstalled.addListener(() => {
    console.log('YouTube Sentiment Analyzer installed');
});

// Handle extension icon click
chrome.action.onClicked.addListener((tab) => {
    if (tab.url.includes('youtube.com/watch')) {
        // Extension popup will handle this
    } else {
        // Show notification if not on YouTube
        chrome.notifications.create({
            type: 'basic',
            iconUrl: 'icon.png',
            title: 'YouTube Sentiment Analyzer',
            message: 'Please navigate to a YouTube video to use this extension.'
        });
    }
});

// Handle messages from content script and popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    switch (message.action) {
        case 'analysisStatus':
        case 'analysisComplete':
            // Forward messages to popup if it's open
            chrome.runtime.sendMessage(message).catch(() => {
                // Popup might not be open, that's okay
            });
            break;
            
        case 'openPopup':
            // This could trigger popup opening logic if needed
            break;
            
        case 'storeResults':
            // Store analysis results
            chrome.storage.local.set({
                lastAnalysisResults: message.results,
                lastAnalysisTime: Date.now()
            });
            break;
            
        case 'getStoredResults':
            chrome.storage.local.get(['lastAnalysisResults'], (result) => {
                sendResponse(result.lastAnalysisResults);
            });
            return true; // Keep message channel open for async response
    }
});

// Context menu integration (optional)
chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: 'analyze-sentiment',
        title: 'Analyze Comment Sentiment',
        contexts: ['page'],
        documentUrlPatterns: ['https://www.youtube.com/watch*']
    });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === 'analyze-sentiment') {
        // Trigger analysis
        chrome.tabs.sendMessage(tab.id, {
            action: 'startAnalysis',
            options: {
                model: 'naive_bayes',
                commentLimit: 100,
                topicModeling: 'false'
            }
        });
    }
});

// Handle storage cleanup
chrome.runtime.onStartup.addListener(() => {
    // Clean up old results (older than 7 days)
    chrome.storage.local.get(null, (items) => {
        const oneWeekAgo = Date.now() - (7 * 24 * 60 * 60 * 1000);
        
        Object.keys(items).forEach(key => {
            if (key.includes('analysisResults') && items[key].timestamp) {
                const resultTime = new Date(items[key].timestamp).getTime();
                if (resultTime < oneWeekAgo) {
                    chrome.storage.local.remove(key);
                }
            }
        });
    });
});