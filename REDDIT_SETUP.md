# Reddit API Integration Setup

This guide will help you set up Reddit API integration to collect daily posts and comments from Argentine finance subreddits.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Reddit API Credentials
```bash
python setup_reddit.py
```

### 3. Test the Connection
```bash
python test_reddit_connection.py
```

### 4. Run the Scraper
```bash
python src/scraping/scraping_reddit.py
```

### 5. Set Up Daily Automation (Optional)
```bash
python setup_cron.py
```

## 📋 Reddit API Setup

### Step 1: Create a Reddit App
1. Go to [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
2. Click **"Create App"** or **"Create Another App"**
3. Fill in the form:
   - **Name**: `StockPredictionBot` (or any name you prefer)
   - **App type**: Select **"script"**
   - **Description**: `Bot for collecting Argentine finance data`
   - **About URL**: (leave blank)
   - **Redirect URI**: `http://localhost:8080`
4. Click **"Create app"**
5. Note down:
   - **Client ID**: The string under your app name
   - **Client Secret**: The secret key

### Step 2: Configure Credentials
The setup script will create a `.env` file with your credentials:
```env
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USERNAME=your_reddit_username
REDDIT_PASSWORD=your_reddit_password
```

## 📊 Subreddits Monitored

The scraper collects data from these Argentine finance-related subreddits:

1. **r/merval** - Main Argentine stock market discussions
2. **r/argentina** - General Argentina (includes finance discussions)
3. **r/argentinacrypto** - Argentine cryptocurrency discussions
4. **r/argentinaeconomia** - Argentine economy discussions
5. **r/CryptoArgentina** - Alternative crypto subreddit
6. **r/InversionesArg** - Argentine investments
7. **r/finanzaspersonales** - Personal finance (Spanish)

## 🔍 Data Collection

### Posts Collected
- Title and text content
- Author information
- Score (upvotes/downvotes)
- Number of comments
- Creation timestamp
- Subreddit source
- URL and permalink

### Comments Collected
- Comment text
- Author information
- Score
- Creation timestamp
- Parent post information
- Subreddit source

### Filtering
Only posts and comments containing finance-related keywords are collected:
- `merval`, `acciones`, `bonos`, `dólar`, `peso`, `inflación`
- `crypto`, `bitcoin`, `ethereum`, `criptomonedas`
- `broker`, `trading`, `inversión`, `portfolio`
- And many more Argentine finance terms...

## 📁 Data Storage

### File Structure
```
data/
└── raw/
    ├── reddit_data_YYYYMMDD_HHMMSS.json          # Complete data (JSON)
    ├── reddit_data_YYYYMMDD_HHMMSS_merval_posts.csv
    ├── reddit_data_YYYYMMDD_HHMMSS_merval_comments.csv
    ├── reddit_data_YYYYMMDD_HHMMSS_argentina_posts.csv
    └── ... (CSV files for each subreddit)
```

### Data Format

#### JSON Format
```json
{
  "merval": {
    "posts": [...],
    "comments": [...],
    "total_posts": 45,
    "total_comments": 123
  },
  "argentina": {
    "posts": [...],
    "comments": [...],
    "total_posts": 23,
    "total_comments": 67
  }
}
```

#### CSV Format
Each CSV file contains structured data with columns:
- `id`, `title`, `text`, `author`, `subreddit`
- `score`, `upvote_ratio`, `created_utc`
- `url`, `permalink`, `type`

## 🤖 Automation

### Daily Scraping
Set up automated daily collection at 9:00 AM:
```bash
python setup_cron.py
```

### Manual Execution
```bash
# Run once
python src/scraping/scraping_reddit.py

# Run daily automation
python src/scraping/daily_reddit_scraper.py
```

### Cron Job
The automation script creates a cron job:
```bash
0 9 * * * cd /path/to/project && python3 src/scraping/daily_reddit_scraper.py >> logs/cron.log 2>&1
```

## 📈 Usage Examples

### Basic Scraping
```python
from src.scraping.scraping_reddit import RedditScraper

# Initialize scraper
scraper = RedditScraper()

# Scrape specific subreddit
posts = scraper.scrape_subreddit_posts('merval', limit=100)
comments = scraper.scrape_subreddit_comments('merval', limit=100)

# Scrape all subreddits
all_data = scraper.scrape_all_subreddits()
```

### Data Processing
```python
import pandas as pd

# Load collected data
df_posts = pd.read_csv('data/raw/reddit_data_20241201_090000_merval_posts.csv')
df_comments = pd.read_csv('data/raw/reddit_data_20241201_090000_merval_comments.csv')

# Analyze sentiment, extract features, etc.
```

## 🔧 Configuration

### Rate Limits
The scraper respects Reddit's rate limits:
- 60 requests per minute
- 1-second delay between requests
- Automatic retry on rate limit errors

### Customization
Edit `src/scraping/reddit_config.py` to:
- Add/remove subreddits
- Modify finance keywords
- Adjust rate limits
- Change data collection parameters

## 📋 Logging

### Log Files
- `logs/reddit_scraper.log` - General scraper logs
- `logs/reddit_daily_YYYYMMDD.log` - Daily automation logs
- `logs/cron.log` - Cron job execution logs

### Log Levels
- **INFO**: General progress and status
- **DEBUG**: Detailed collection information
- **ERROR**: Errors and failures
- **WARNING**: Rate limits and retries

## 🛠️ Troubleshooting

### Common Issues

#### 1. Authentication Errors
```
Error: Invalid credentials
```
**Solution**: Check your Reddit API credentials in `.env` file

#### 2. Rate Limit Errors
```
Error: 429 Too Many Requests
```
**Solution**: The scraper automatically handles this with delays

#### 3. Subreddit Access Errors
```
Error: 403 Forbidden
```
**Solution**: Some subreddits may be private or restricted

#### 4. Network Errors
```
Error: Connection timeout
```
**Solution**: Check your internet connection and Reddit's status

### Debug Mode
Enable detailed logging:
```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## 📊 Data Analysis

### Sample Analysis Script
```python
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Load data
df = pd.read_csv('data/raw/reddit_data_20241201_090000_merval_posts.csv')

# Convert timestamp
df['created_utc'] = pd.to_datetime(df['created_utc'])

# Daily post counts
daily_posts = df.groupby(df['created_utc'].dt.date).size()
daily_posts.plot(title='Daily Posts in r/merval')
plt.show()

# Sentiment analysis (requires additional setup)
# from textblob import TextBlob
# df['sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
```

## 🔒 Security

### Credentials Protection
- Never commit `.env` file to version control
- Add `.env` to `.gitignore`
- Use environment variables in production

### Data Privacy
- Collected data is public Reddit content
- No private user information is stored
- Respects Reddit's terms of service

## 📚 Next Steps

1. **Text Processing**: Set up sentiment analysis and text preprocessing
2. **Feature Engineering**: Extract meaningful features from text data
3. **Time Series Integration**: Combine with market data
4. **Model Training**: Use collected data for prediction models
5. **Dashboard**: Create visualization dashboard

## 🆘 Support

If you encounter issues:
1. Check the logs in `logs/` directory
2. Verify Reddit API credentials
3. Test connection with `python test_reddit_connection.py`
4. Check Reddit's API status at [https://redditstatus.com](https://redditstatus.com)

