#!/usr/bin/env python3
"""
Docker Setup para Scraping Histórico de Reddit
Configuración optimizada para grandes volúmenes de datos históricos.
"""

import os
import subprocess
import sys
from pathlib import Path

def create_historical_dockerfile():
    """Crear Dockerfile optimizado para scraping histórico."""
    
    dockerfile_content = '''# Dockerfile for Historical Reddit Scraper
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    cron \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p logs data/historical data/raw

# Create cron job for historical scraping (manual execution)
# Note: Historical scraping is typically run manually via notebook
RUN echo "# Historical scraping cron job (disabled by default)" > /etc/cron.d/historical-scraper
RUN echo "# 0 2 * * * cd /app && python notebooks/historical_reddit_scraper.py >> logs/historical.log 2>&1" >> /etc/cron.d/historical-scraper

# Start cron daemon and keep container running
CMD ["sh", "-c", "cron && tail -f /dev/null"]
'''
    
    with open('Dockerfile.historical', 'w') as f:
        f.write(dockerfile_content)
    
    print("✅ Dockerfile histórico creado: Dockerfile.historical")

def create_historical_docker_compose():
    """Crear docker-compose optimizado para scraping histórico."""
    
    compose_content = '''version: '3.8'

services:
  reddit-historical-scraper:
    build:
      context: .
      dockerfile: Dockerfile.historical
    container_name: reddit-historical-scraper-mia
    restart: unless-stopped
    volumes:
      # Persist historical data
      - ./data/historical:/app/data/historical
      - ./data/raw:/app/data/raw
      - ./logs:/app/logs
      # Mount .env file for credentials
      - ./.env:/app/.env:ro
      # Mount notebooks for manual execution
      - ./notebooks:/app/notebooks
    environment:
      - TZ=America/Argentina/Buenos_Aires
      - PYTHONPATH=/app
    # Increase memory limit for large datasets
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
'''
    
    with open('docker-compose.historical.yml', 'w') as f:
        f.write(compose_content)
    
    print("✅ Docker Compose histórico creado: docker-compose.historical.yml")

def check_docker():
    """Verificar que Docker esté disponible."""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker encontrado: {result.stdout.strip()}")
            return True
        else:
            print("❌ Docker no encontrado")
            return False
    except FileNotFoundError:
        print("❌ Docker no está instalado")
        return False

def build_historical_container():
    """Construir contenedor para scraping histórico."""
    print("🔨 Construyendo contenedor histórico...")
    
    try:
        # Build the image
        result = subprocess.run([
            'docker-compose', '-f', 'docker-compose.historical.yml', 'build'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Contenedor histórico construido exitosamente!")
            return True
        else:
            print(f"❌ Error construyendo contenedor: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def start_historical_container():
    """Iniciar contenedor histórico."""
    print("🚀 Iniciando contenedor histórico...")
    
    try:
        result = subprocess.run([
            'docker-compose', '-f', 'docker-compose.historical.yml', 'up', '-d'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Contenedor histórico iniciado!")
            print("📊 El contenedor está listo para scraping histórico")
            print("🔧 Usa el notebook para ejecutar el scraping manualmente")
            return True
        else:
            print(f"❌ Error iniciando contenedor: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def run_historical_notebook():
    """Ejecutar el notebook de scraping histórico en el contenedor."""
    print("📓 Ejecutando notebook histórico...")
    
    try:
        # Ejecutar notebook en el contenedor
        result = subprocess.run([
            'docker-compose', '-f', 'docker-compose.historical.yml', 'exec', 
            'reddit-historical-scraper', 'python', '-c', 
            '''
import sys
sys.path.append("/app")
from notebooks.historical_reddit_scraper import *
            '''
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Notebook ejecutado exitosamente!")
            print(result.stdout)
        else:
            print(f"❌ Error ejecutando notebook: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def show_historical_status():
    """Mostrar estado del contenedor histórico."""
    try:
        result = subprocess.run([
            'docker-compose', '-f', 'docker-compose.historical.yml', 'ps'
        ], capture_output=True, text=True)
        
        print("📊 Estado del contenedor histórico:")
        print(result.stdout)
        
    except Exception as e:
        print(f"❌ Error: {e}")

def show_historical_logs():
    """Mostrar logs del contenedor histórico."""
    try:
        print("📋 Logs del contenedor histórico:")
        subprocess.run([
            'docker-compose', '-f', 'docker-compose.historical.yml', 
            'logs', '-f', '--tail=50'
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Saliendo de los logs")
    except Exception as e:
        print(f"❌ Error: {e}")

def stop_historical_container():
    """Detener contenedor histórico."""
    print("🛑 Deteniendo contenedor histórico...")
    
    try:
        result = subprocess.run([
            'docker-compose', '-f', 'docker-compose.historical.yml', 'down'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Contenedor histórico detenido")
        else:
            print(f"❌ Error: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Función principal."""
    print("📊 Docker Setup para Scraping Histórico de Reddit")
    print("=" * 55)
    
    # Check .env file
    if not Path('.env').exists():
        print("⚠️  Archivo .env no encontrado!")
        print("Ejecuta primero: python setup_reddit.py")
        return
    
    # Check Docker
    if not check_docker():
        print("\n📥 Instala Docker Desktop:")
        print("https://www.docker.com/products/docker-desktop")
        return
    
    while True:
        print("\nOpciones:")
        print("1. Crear archivos Docker para scraping histórico")
        print("2. Construir contenedor histórico")
        print("3. Iniciar contenedor histórico")
        print("4. Ver estado del contenedor")
        print("5. Ver logs del contenedor")
        print("6. Ejecutar notebook histórico")
        print("7. Detener contenedor histórico")
        print("8. Salir")
        
        choice = input("\nElige una opción (1-8): ").strip()
        
        if choice == '1':
            create_historical_dockerfile()
            create_historical_docker_compose()
        elif choice == '2':
            build_historical_container()
        elif choice == '3':
            start_historical_container()
        elif choice == '4':
            show_historical_status()
        elif choice == '5':
            show_historical_logs()
        elif choice == '6':
            run_historical_notebook()
        elif choice == '7':
            stop_historical_container()
        elif choice == '8':
            print("👋 ¡Hasta luego!")
            break
        else:
            print("❌ Opción inválida")

if __name__ == "__main__":
    main()
