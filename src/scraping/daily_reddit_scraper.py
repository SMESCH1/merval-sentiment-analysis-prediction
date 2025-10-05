#!/usr/bin/env python3
"""
Daily Reddit Scraper
Automated script for daily collection of Reddit data.
This script is designed to be run via cron job.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.scraping.scraping_reddit import RedditScraper

def setup_daily_logging():
    """Set up logging for daily scraping."""
    log_dir = project_root / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    # Create daily log file
    today = datetime.now().strftime("%Y%m%d")
    log_file = log_dir / f"reddit_daily_{today}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def run_daily_scraping():
    """Run the daily Reddit scraping process."""
    logger = setup_daily_logging()
    
    try:
        logger.info("🚀 Starting daily Reddit scraping...")
        
        # Initialize scraper
        scraper = RedditScraper()
        
        # Create data directory
        data_dir = project_root / 'data' / 'raw'
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Scrape all subreddits
        logger.info("📊 Collecting data from Argentine finance subreddits...")
        all_data = scraper.scrape_all_subreddits(include_comments=True)
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save data
        json_filename = data_dir / f"reddit_data_{timestamp}.json"
        csv_base_filename = data_dir / f"reddit_data_{timestamp}"
        
        scraper.save_to_json(all_data, str(json_filename))
        scraper.save_to_csv(all_data, str(csv_base_filename))
        
        # Print summary
        total_posts = sum(data['total_posts'] for data in all_data.values())
        total_comments = sum(data['total_comments'] for data in all_data.values())
        
        logger.info(f"✅ Daily scraping completed successfully!")
        logger.info(f"📈 Total posts collected: {total_posts}")
        logger.info(f"💬 Total comments collected: {total_comments}")
        logger.info(f"💾 Data saved to: {json_filename}")
        
        # Create a summary file
        summary_file = data_dir / f"scraping_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Reddit Scraping Summary - {datetime.now().isoformat()}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total posts collected: {total_posts}\n")
            f.write(f"Total comments collected: {total_comments}\n")
            f.write(f"Data files created:\n")
            f.write(f"  - {json_filename.name}\n")
            for subreddit_name in all_data.keys():
                f.write(f"  - reddit_data_{timestamp}_{subreddit_name}_posts.csv\n")
                f.write(f"  - reddit_data_{timestamp}_{subreddit_name}_comments.csv\n")
        
        logger.info(f"📋 Summary saved to: {summary_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in daily scraping process: {e}")
        return False

def cleanup_old_data(days_to_keep=30):
    """Clean up old data files to save disk space."""
    logger = logging.getLogger(__name__)
    
    try:
        data_dir = project_root / 'data' / 'raw'
        if not data_dir.exists():
            return
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        files_deleted = 0
        for file_path in data_dir.glob("reddit_data_*.json"):
            if file_path.stat().st_mtime < cutoff_date.timestamp():
                file_path.unlink()
                files_deleted += 1
                logger.info(f"🗑️  Deleted old file: {file_path.name}")
        
        logger.info(f"🧹 Cleanup completed. Deleted {files_deleted} old files.")
        
    except Exception as e:
        logger.error(f"❌ Error during cleanup: {e}")

def main():
    """Main function for daily scraping."""
    success = run_daily_scraping()
    
    if success:
        # Clean up old data
        cleanup_old_data()
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()

