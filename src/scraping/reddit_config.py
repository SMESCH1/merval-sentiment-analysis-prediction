"""
Reddit API Configuration
This file contains the configuration for Reddit API access.
You need to create a Reddit app at https://www.reddit.com/prefs/apps to get these credentials.
"""

import os
from typing import Dict, Any

# Reddit API credentials
# To get these credentials:
# 1. Go to https://www.reddit.com/prefs/apps
# 2. Click "Create App" or "Create Another App"
# 3. Choose "script" as the app type
# 4. Fill in the required fields
# 5. Note down the client_id (under the app name) and client_secret

REDDIT_CONFIG = {
    'client_id': os.getenv('REDDIT_CLIENT_ID', ''),
    'client_secret': os.getenv('REDDIT_CLIENT_SECRET', ''),
    'user_agent': 'StockPredictionBot/1.0 by /u/your_username',  # Replace with your Reddit username
    'username': os.getenv('REDDIT_USERNAME', ''),
    'password': os.getenv('REDDIT_PASSWORD', '')
}

# Argentine finance-related subreddits
ARGENTINE_FINANCE_SUBREDDITS = [
    'merval',           # Main Argentine stock market subreddit
    'argentina',        # General Argentina subreddit (has finance discussions)
    'argentinacrypto',  # Argentine cryptocurrency discussions
    'argentinaeconomia', # Argentine economy discussions
    'CryptoArgentina',   # Alternative crypto subreddit
    'InversionesArg',   # Argentine investments
    'finanzaspersonales' # Personal finance (Spanish)
]

# Keywords to filter relevant posts (Spanish/Argentine finance terms)
FINANCE_KEYWORDS = [
    'merval', 'bovespa', 'acciones', 'bonos', 'dólar', 'peso', 'inflación',
    'dolar', 'peso', 'inversion', 'inversión', 'bolsa', 'mercado', 'finanzas',
    'economia', 'economía', 'crypto', 'bitcoin', 'ethereum', 'criptomonedas',
    'broker', 'trading', 'inversor', 'inversora', 'portfolio', 'cartera',
    'renta', 'dividendos', 'cedears', 'adrs', 'fci', 'fondo', 'plazo fijo',
    'pf', 'leliq', 'bcra', 'indec', 'ipc', 'devaluacion', 'devaluación'
]

# Reddit API rate limits (requests per minute)
RATE_LIMITS = {
    'posts_per_minute': 60,
    'comments_per_minute': 60,
    'delay_between_requests': 1  # seconds
}

def get_reddit_config() -> Dict[str, Any]:
    """
    Get Reddit configuration with validation.
    
    Returns:
        Dict containing Reddit API configuration
        
    Raises:
        ValueError: If required credentials are missing
    """
    required_fields = ['client_id', 'client_secret', 'user_agent']
    
    for field in required_fields:
        if not REDDIT_CONFIG.get(field):
            raise ValueError(f"Missing required Reddit configuration: {field}")
    
    return REDDIT_CONFIG

def validate_credentials() -> bool:
    """
    Validate that all required Reddit credentials are present.
    
    Returns:
        bool: True if all credentials are present, False otherwise
    """
    try:
        get_reddit_config()
        return True
    except ValueError:
        return False
