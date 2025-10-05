#!/usr/bin/env python3
"""
Cron Job Setup Script
This script helps set up automated daily Reddit scraping via cron.
"""

import os
import subprocess
from pathlib import Path

def get_project_path():
    """Get the absolute path to the project root."""
    return Path(__file__).parent.absolute()

def create_cron_job():
    """Create a cron job for daily Reddit scraping."""
    project_path = get_project_path()
    python_path = subprocess.check_output(['which', 'python3']).decode().strip()
    script_path = project_path / 'src' / 'scraping' / 'daily_reddit_scraper.py'
    
    # Cron job entry
    cron_entry = f"0 9 * * * cd {project_path} && {python_path} {script_path} >> {project_path}/logs/cron.log 2>&1"
    
    print("🕘 Setting up daily Reddit scraping cron job...")
    print(f"📁 Project path: {project_path}")
    print(f"🐍 Python path: {python_path}")
    print(f"📜 Script path: {script_path}")
    print()
    print("Cron job entry:")
    print(f"  {cron_entry}")
    print()
    print("This will run the scraper daily at 9:00 AM")
    print()
    
    # Add to crontab
    try:
        # Get current crontab
        result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
        current_crontab = result.stdout if result.returncode == 0 else ""
        
        # Check if our job already exists
        if "daily_reddit_scraper.py" in current_crontab:
            print("⚠️  Cron job already exists!")
            print("Current crontab entries:")
            print(current_crontab)
            return
        
        # Add new job
        new_crontab = current_crontab + f"\n{cron_entry}\n"
        
        # Write to temporary file
        with open('/tmp/new_crontab', 'w') as f:
            f.write(new_crontab)
        
        # Install new crontab
        subprocess.run(['crontab', '/tmp/new_crontab'], check=True)
        
        print("✅ Cron job added successfully!")
        print("📋 Current crontab:")
        subprocess.run(['crontab', '-l'])
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error setting up cron job: {e}")
        print("Please run this script with appropriate permissions or set up the cron job manually.")

def remove_cron_job():
    """Remove the Reddit scraping cron job."""
    try:
        # Get current crontab
        result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
        if result.returncode != 0:
            print("No crontab found.")
            return
        
        current_crontab = result.stdout
        lines = current_crontab.split('\n')
        
        # Filter out our job
        new_lines = [line for line in lines if "daily_reddit_scraper.py" not in line]
        
        if len(new_lines) == len(lines):
            print("ℹ️  No Reddit scraping cron job found to remove.")
            return
        
        # Write updated crontab
        new_crontab = '\n'.join(new_lines)
        with open('/tmp/updated_crontab', 'w') as f:
            f.write(new_crontab)
        
        subprocess.run(['crontab', '/tmp/updated_crontab'], check=True)
        print("✅ Reddit scraping cron job removed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error removing cron job: {e}")

def show_cron_status():
    """Show current cron job status."""
    try:
        result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
        if result.returncode != 0:
            print("No crontab found.")
            return
        
        current_crontab = result.stdout
        if "daily_reddit_scraper.py" in current_crontab:
            print("✅ Reddit scraping cron job is active!")
            print("Current crontab entries:")
            print(current_crontab)
        else:
            print("❌ No Reddit scraping cron job found.")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error checking cron status: {e}")

def main():
    """Main function."""
    print("🕘 Reddit Scraper Cron Job Setup")
    print("=" * 50)
    
    while True:
        print("\nChoose an option:")
        print("1. Add daily Reddit scraping cron job")
        print("2. Remove Reddit scraping cron job")
        print("3. Show current cron job status")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            create_cron_job()
        elif choice == '2':
            remove_cron_job()
        elif choice == '3':
            show_cron_status()
        elif choice == '4':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

