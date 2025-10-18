#!/usr/bin/env python3
"""
Test Reddit Connection
Simple script to test Reddit API connection and collect a small sample of data.
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.scraping.reddit_config import validate_credentials, get_reddit_config, ARGENTINE_FINANCE_SUBREDDITS
from src.scraping.scraping_reddit import RedditScraper

def test_connection():
    """Test Reddit API connection."""
    print("🧪 Testing Reddit API Connection")
    print("=" * 40)
    
    # Check credentials
    if not validate_credentials():
        print("❌ Reddit credentials are missing or invalid")
        print("Please run: python setup_reddit.py")
        return False
    
    print("✅ Reddit credentials found")
    
    try:
        # Initialize scraper
        scraper = RedditScraper()
        print("✅ Reddit API connection established")
        
        # Test with a small sample
        print("\n📊 Testing data collection...")
        test_subreddit = 'merval'  # Start with the main Argentine finance subreddit
        
        print(f"🔍 Testing with r/{test_subreddit} (collecting 5 posts)...")
        posts = scraper.scrape_subreddit_posts(test_subreddit, limit=5)
        
        print(f"✅ Successfully collected {len(posts)} posts")
        
        if posts:
            print("\n📝 Sample post:")
            sample_post = posts[0]
            print(f"  Title: {sample_post['title'][:100]}...")
            print(f"  Author: {sample_post['author']}")
            print(f"  Score: {sample_post['score']}")
            print(f"  Created: {sample_post['created_utc']}")
        
        # Test comments collection
        print(f"\n💬 Testing comments collection from r/{test_subreddit}...")
        comments = scraper.scrape_subreddit_comments(test_subreddit, limit=10)
        print(f"✅ Successfully collected {len(comments)} comments")
        
        if comments:
            print("\n💬 Sample comment:")
            sample_comment = comments[0]
            print(f"  Text: {sample_comment['text'][:100]}...")
            print(f"  Author: {sample_comment['author']}")
            print(f"  Score: {sample_comment['score']}")
        
        print("\n🎉 Reddit connection test successful!")
        print("✅ The scraper is ready to collect data from Argentine finance subreddits")
        
        return True
        
    except Exception as e:
        print(f"❌ Reddit connection test failed: {e}")
        return False

def show_subreddits():
    """Show the subreddits that will be scraped."""
    print("\n📋 Argentine Finance Subreddits to be scraped:")
    print("=" * 50)
    for i, subreddit in enumerate(ARGENTINE_FINANCE_SUBREDDITS, 1):
        print(f"{i:2d}. r/{subreddit}")
    print(f"\nTotal: {len(ARGENTINE_FINANCE_SUBREDDITS)} subreddits")

def main():
    """Main test function."""
    print("🚀 Reddit Scraper Test")
    print("=" * 30)
    
    # Show subreddits
    show_subreddits()
    
    # Test connection
    if test_connection():
        print("\n🎯 Next steps:")
        print("1. Run full scraper: python src/scraping/scraping_reddit.py")
        print("2. Set up daily automation: python setup_cron.py")
        print("3. Check collected data in data/raw/")
    else:
        print("\n🔧 Setup required:")
        print("1. Run: python setup_reddit.py")
        print("2. Configure your Reddit API credentials")
        print("3. Run this test again")

if __name__ == "__main__":
    main()
