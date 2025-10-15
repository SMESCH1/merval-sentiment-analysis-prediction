# Dockerfile for Reddit Scraper
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    cron \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p logs data/raw

# Create cron job for daily scraping at 9:00 AM
RUN echo "0 9 * * * cd /app && python src/scraping/daily_reddit_scraper.py >> logs/cron.log 2>&1" | crontab -

# Start cron daemon and keep container running
CMD ["sh", "-c", "cron && tail -f /dev/null"]
