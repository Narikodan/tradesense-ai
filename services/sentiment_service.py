import re

# Keywords for sentiment analysis
BULLISH_KEYWORDS = [
    'up', 'higher', 'growth', 'profit', 'beat', 'surge', 'jump', 'gain',
    'buy', 'upgrade', 'positive', 'bull', 'rally', 'record', 'dividend',
    'outperform', 'strong', 'boom', 'breakout', 'success'
]

BEARISH_KEYWORDS = [
    'down', 'lower', 'loss', 'miss', 'plunge', 'drop', 'fall', 'sell',
    'downgrade', 'negative', 'bear', 'crash', 'cut', 'underperform', 'weak',
    'bust', 'decline', 'scandal', 'lawsuit', 'warning'
]

# Keywords for impact level
HIGH_IMPACT_KEYWORDS = [
    'earnings', 'result', 'acquisition', 'merger', 'scandal', 'bankruptcy',
    'fed', 'guidance', 'sec', 'ceo rating', 'lawsuit', 'dividend'
]

MEDIUM_IMPACT_KEYWORDS = [
    'upgrade', 'downgrade', 'analyst', 'target', 'initiate', 'deal', 'partnership'
]

def analyze_sentiment(text):
    """
    Analyzes text (like a news headline) and returns a sentiment score and label.
    Score ranges from -1.0 to 1.0.
    """
    if not text:
        return 0.0, 'neutral'

    text_lower = str(text).lower()
    
    # Tokenize words roughly
    words = re.findall(r'\b\w+\b', text_lower)
    
    bullish_count = sum(1 for word in words if word in BULLISH_KEYWORDS)
    bearish_count = sum(1 for word in words if word in BEARISH_KEYWORDS)
    
    total_sentiment_words = bullish_count + bearish_count
    
    if total_sentiment_words == 0:
        return 0.0, 'neutral'
        
    # Calculate a simple score between -1 and 1
    score = (bullish_count - bearish_count) / max(total_sentiment_words, 1)
    
    # Add minor scale based on proportion of sentiment words vs total words just to normalize
    word_ratio = min(total_sentiment_words / max(len(words), 1) * 2, 1.0)
    score = score * word_ratio
    
    label = 'neutral'
    if score >= 0.2:
        label = 'bullish'
    elif score <= -0.2:
        label = 'bearish'
        
    return round(score, 2), label

def get_impact_tag(text):
    """
    Determines the impact level based on keywords.
    Returns HIGH, MEDIUM, or LOW.
    """
    if not text:
        return 'LOW'
        
    text_lower = str(text).lower()
    
    for word in HIGH_IMPACT_KEYWORDS:
        if word in text_lower:
            return 'HIGH'
            
    for word in MEDIUM_IMPACT_KEYWORDS:
        if word in text_lower:
            return 'MEDIUM'
            
    return 'LOW'
