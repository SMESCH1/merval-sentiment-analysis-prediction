#!/usr/bin/env python3
"""
Reddit Scraper for Argentine Finance Data
Collects posts and comments from Argentine finance-related subreddits.
"""

import praw
import json
import csv
import os
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import time
import pandas as pd

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.scraping.reddit_config import (
    get_reddit_config, 
    ARGENTINE_FINANCE_SUBREDDITS, 
    FINANCE_KEYWORDS,
    RATE_LIMITS
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/reddit_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RedditScraper:
    """Reddit scraper for collecting Argentine finance data."""
    
    def __init__(self):
        """Initialize the Reddit scraper with API credentials."""
        try:
            config = get_reddit_config()
            self.reddit = praw.Reddit(
                client_id=config['client_id'],
                client_secret=config['client_secret'],
                user_agent=config['user_agent'],
                username=config.get('username'),
                password=config.get('password')
            )
            logger.info("Reddit API connection established successfully")
        except Exception as e:
            logger.error(f"Failed to establish Reddit API connection: {e}")
            raise
    
    def is_finance_related(self, text: str) -> bool:
        """
        Check if a post/comment is related to finance based on keywords.
        
        Args:
            text: Text content to analyze
            
        Returns:
            bool: True if finance-related, False otherwise
        """
        if not text:
            return False
            
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in FINANCE_KEYWORDS)
    
    def extract_post_data(self, submission) -> Dict[str, Any]:
        """
        Extract relevant data from a Reddit submission.
        
        Args:
            submission: PRAW submission object
            
        Returns:
            Dict containing post data
        """
        return {
            'id': submission.id,
            'title': submission.title,
            'text': submission.selftext,
            'author': str(submission.author) if submission.author else '[deleted]',
            'subreddit': str(submission.subreddit),
            'score': submission.score,
            'upvote_ratio': submission.upvote_ratio,
            'num_comments': submission.num_comments,
            'created_utc': datetime.fromtimestamp(submission.created_utc),
            'url': submission.url,
            'permalink': submission.permalink,
            'is_self': submission.is_self,
            'type': 'post'
        }
    
    def extract_comment_data(self, comment) -> Dict[str, Any]:
        """
        Extract relevant data from a Reddit comment.
        
        Args:
            comment: PRAW comment object
            
        Returns:
            Dict containing comment data
        """
        return {
            'id': comment.id,
            'text': comment.body,
            'author': str(comment.author) if comment.author else '[deleted]',
            'subreddit': str(comment.subreddit),
            'score': comment.score,
            'created_utc': datetime.fromtimestamp(comment.created_utc),
            'parent_id': comment.parent_id,
            'link_id': comment.link_id,
            'type': 'comment'
        }
    
    def scrape_subreddit_posts(self, subreddit_name: str, limit: int = 100, 
                              time_filter: str = 'day') -> List[Dict[str, Any]]:
        """
        Scrape posts from a specific subreddit.
        
        Args:
            subreddit_name: Name of the subreddit
            limit: Maximum number of posts to retrieve
            time_filter: Time filter ('day', 'week', 'month', 'year', 'all')
            
        Returns:
            List of post data dictionaries
        """
        posts = []
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            logger.info(f"Scraping posts from r/{subreddit_name}")
            
            for submission in subreddit.hot(limit=limit):
                if self.is_finance_related(submission.title + ' ' + submission.selftext):
                    post_data = self.extract_post_data(submission)
                    posts.append(post_data)
                    logger.debug(f"Collected post: {submission.title[:50]}...")
                
                # Rate limiting
                time.sleep(RATE_LIMITS['delay_between_requests'])
                
        except Exception as e:
            logger.error(f"Error scraping posts from r/{subreddit_name}: {e}")
        
        return posts
    
    def scrape_subreddit_comments(self, subreddit_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Scrape comments from a specific subreddit.
        
        Args:
            subreddit_name: Name of the subreddit
            limit: Maximum number of comments to retrieve
            
        Returns:
            List of comment data dictionaries
        """
        comments = []
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            logger.info(f"Scraping comments from r/{subreddit_name}")
            
            for submission in subreddit.hot(limit=limit):
                submission.comments.replace_more(limit=0)  # Load all comments
                
                for comment in submission.comments.list():
                    if self.is_finance_related(comment.body):
                        comment_data = self.extract_comment_data(comment)
                        comments.append(comment_data)
                        logger.debug(f"Collected comment: {comment.body[:50]}...")
                
                # Rate limiting
                time.sleep(RATE_LIMITS['delay_between_requests'])
                
        except Exception as e:
            logger.error(f"Error scraping comments from r/{subreddit_name}: {e}")
        
        return comments
    
    def scrape_all_subreddits(self, include_comments: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """
        Scrape all Argentine finance subreddits.
        
        Args:
            include_comments: Whether to include comments in scraping
            
        Returns:
            Dictionary with subreddit names as keys and data as values
        """
        all_data = {}
        
        for subreddit_name in ARGENTINE_FINANCE_SUBREDDITS:
            logger.info(f"Starting to scrape r/{subreddit_name}")
            
            # Scrape posts
            posts = self.scrape_subreddit_posts(subreddit_name)
            logger.info(f"Collected {len(posts)} posts from r/{subreddit_name}")
            
            # Scrape comments if requested
            comments = []
            if include_comments:
                comments = self.scrape_subreddit_comments(subreddit_name)
                logger.info(f"Collected {len(comments)} comments from r/{subreddit_name}")
            
            all_data[subreddit_name] = {
                'posts': posts,
                'comments': comments,
                'total_posts': len(posts),
                'total_comments': len(comments)
            }
            
            # Delay between subreddits to respect rate limits
            time.sleep(2)
        
        return all_data
    
    def save_to_json(self, data: Dict[str, Any], filename: str) -> None:
        """
        Save data to JSON file.
        
        Args:
            data: Data to save
            filename: Output filename
        """
        # Convert datetime objects to strings for JSON serialization
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj
        
        # Recursively convert datetime objects
        def recursive_convert(data):
            if isinstance(data, dict):
                return {k: recursive_convert(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [recursive_convert(item) for item in data]
            else:
                return convert_datetime(data)
        
        converted_data = recursive_convert(data)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Data saved to {filename}")
    
    def save_to_csv(self, data: Dict[str, Any], base_filename: str) -> None:
        """
        Save data to CSV files (separate files for posts and comments).
        
        Args:
            data: Data to save
            base_filename: Base filename (without extension)
        """
        for subreddit_name, subreddit_data in data.items():
            # Save posts
            if subreddit_data['posts']:
                posts_df = pd.DataFrame(subreddit_data['posts'])
                posts_filename = f"{base_filename}_{subreddit_name}_posts.csv"
                posts_df.to_csv(posts_filename, index=False, encoding='utf-8')
                logger.info(f"Posts saved to {posts_filename}")
            
            # Save comments
            if subreddit_data['comments']:
                comments_df = pd.DataFrame(subreddit_data['comments'])
                comments_filename = f"{base_filename}_{subreddit_name}_comments.csv"
                comments_df.to_csv(comments_filename, index=False, encoding='utf-8')
                logger.info(f"Comments saved to {comments_filename}")

def main():
    """Main function to run the Reddit scraper."""
    try:
        # Create data directory if it doesn't exist
        os.makedirs('data/raw', exist_ok=True)
        
        # Initialize scraper
        scraper = RedditScraper()
        
        # Scrape all subreddits
        logger.info("Starting Reddit scraping process...")
        all_data = scraper.scrape_all_subreddits(include_comments=True)
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save data
        json_filename = f"data/raw/reddit_data_{timestamp}.json"
        csv_base_filename = f"data/raw/reddit_data_{timestamp}"
        
        scraper.save_to_json(all_data, json_filename)
        scraper.save_to_csv(all_data, csv_base_filename)
        
        # Print summary
        total_posts = sum(data['total_posts'] for data in all_data.values())
        total_comments = sum(data['total_comments'] for data in all_data.values())
        
        logger.info(f"Scraping completed successfully!")
        logger.info(f"Total posts collected: {total_posts}")
        logger.info(f"Total comments collected: {total_comments}")
        logger.info(f"Data saved to: {json_filename}")
        
    except Exception as e:
        logger.error(f"Error in main scraping process: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

