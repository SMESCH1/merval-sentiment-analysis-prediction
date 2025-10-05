#!/usr/bin/env python3
"""
Reddit API Setup Script
This script helps you set up Reddit API credentials for the scraper.
"""

import os
import sys
from pathlib import Path

def create_env_file():
    """Create a .env file for Reddit API credentials."""
    env_file = Path('.env')
    
    if env_file.exists():
        print("⚠️  .env file already exists. Backing up to .env.backup")
        env_file.rename('.env.backup')
    
    print("\n🔧 Reddit API Setup")
    print("=" * 50)
    print("To use the Reddit scraper, you need to create a Reddit app:")
    print("1. Go to https://www.reddit.com/prefs/apps")
    print("2. Click 'Create App' or 'Create Another App'")
    print("3. Choose 'script' as the app type")
    print("4. Fill in the required fields:")
    print("   - Name: StockPredictionBot (or any name)")
    print("   - Description: Bot for collecting finance data")
    print("   - About URL: (leave blank)")
    print("   - Redirect URI: http://localhost:8080")
    print("5. Click 'Create app'")
    print("6. Note down the client_id (under the app name) and client_secret")
    print()
    
    # Get credentials from user
    client_id = input("Enter your Reddit client_id: ").strip()
    client_secret = input("Enter your Reddit client_secret: ").strip()
    username = input("Enter your Reddit username (optional): ").strip()
    password = input("Enter your Reddit password (optional): ").strip()
    
    # Create .env file
    env_content = f"""# Reddit API Credentials
REDDIT_CLIENT_ID={client_id}
REDDIT_CLIENT_SECRET={client_secret}
REDDIT_USERNAME={username}
REDDIT_PASSWORD={password}

# Optional: Other API keys
NEWS_API_KEY=
YFINANCE_CACHE_DIR=./data/market
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print(f"\n✅ .env file created successfully!")
    print("🔒 Make sure to add .env to your .gitignore file to keep credentials secure!")
    
    return True

def test_reddit_connection():
    """Test the Reddit API connection."""
    try:
        from src.scraping.reddit_config import validate_credentials, get_reddit_config
        import praw
        
        if not validate_credentials():
            print("❌ Reddit credentials are missing or invalid")
            return False
        
        config = get_reddit_config()
        reddit = praw.Reddit(
            client_id=config['client_id'],
            client_secret=config['client_secret'],
            user_agent=config['user_agent']
        )
        
        # Test connection by getting a simple subreddit
        test_subreddit = reddit.subreddit('test')
        print(f"✅ Reddit API connection successful!")
        print(f"📊 Testing with r/test: {test_subreddit.display_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Reddit API connection failed: {e}")
        return False

def install_dependencies():
    """Install required Python packages."""
    print("\n📦 Installing dependencies...")
    os.system("pip install -r requirements.txt")
    print("✅ Dependencies installed!")

def main():
    """Main setup function."""
    print("🚀 Reddit Scraper Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path('src/scraping').exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Create .env file
    if not Path('.env').exists():
        create_env_file()
    else:
        print("📄 .env file already exists")
    
    # Install dependencies
    install_dependencies()
    
    # Test connection
    print("\n🧪 Testing Reddit API connection...")
    if test_reddit_connection():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run the scraper: python src/scraping/scraping_reddit.py")
        print("2. Check the logs: tail -f logs/reddit_scraper.log")
        print("3. View collected data in data/raw/")
    else:
        print("\n❌ Setup incomplete. Please check your Reddit API credentials.")
        print("Edit the .env file and run this script again.")

if __name__ == "__main__":
    main()
