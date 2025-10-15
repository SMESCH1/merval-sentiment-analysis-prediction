#!/usr/bin/env python3
"""
Docker Setup for Reddit Scraper
Simple script to manage the Docker container for daily Reddit scraping.
"""

import os
import subprocess
import sys
from pathlib import Path

def check_docker():
    """Check if Docker is installed and running."""
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

def check_docker_running():
    """Check if Docker daemon is running."""
    try:
        result = subprocess.run(['docker', 'info'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker está corriendo")
            return True
        else:
            print("❌ Docker no está corriendo")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def build_and_start():
    """Build and start the Docker container."""
    print("🔨 Construyendo y iniciando contenedor Docker...")
    
    try:
        # Build the image
        print("📦 Construyendo imagen...")
        result = subprocess.run(['docker-compose', 'build'], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Error construyendo imagen: {result.stderr}")
            return False
        
        # Start the container
        print("🚀 Iniciando contenedor...")
        result = subprocess.run(['docker-compose', 'up', '-d'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Contenedor iniciado exitosamente!")
            print("🕘 El scraper se ejecutará diariamente a las 9:00 AM")
            return True
        else:
            print(f"❌ Error iniciando contenedor: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def stop_container():
    """Stop the Docker container."""
    print("🛑 Deteniendo contenedor...")
    
    try:
        result = subprocess.run(['docker-compose', 'down'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Contenedor detenido")
        else:
            print(f"❌ Error: {result.stderr}")
    except Exception as e:
        print(f"❌ Error: {e}")

def show_status():
    """Show container status."""
    try:
        result = subprocess.run(['docker-compose', 'ps'], capture_output=True, text=True)
        print("📊 Estado del contenedor:")
        print(result.stdout)
    except Exception as e:
        print(f"❌ Error: {e}")

def show_logs():
    """Show container logs."""
    try:
        print("📋 Logs del contenedor (Ctrl+C para salir):")
        subprocess.run(['docker-compose', 'logs', '-f', '--tail=20'])
    except KeyboardInterrupt:
        print("\n👋 Saliendo de los logs")
    except Exception as e:
        print(f"❌ Error: {e}")

def run_now():
    """Run the scraper immediately."""
    print("🧪 Ejecutando scraper ahora...")
    
    try:
        result = subprocess.run([
            'docker-compose', 'exec', 'reddit-scraper', 
            'python', 'src/scraping/daily_reddit_scraper.py'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Scraping completado!")
            print(result.stdout)
        else:
            print(f"❌ Error: {result.stderr}")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Main function."""
    print("🐳 Configuración Docker para Reddit Scraper")
    print("=" * 45)
    
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
    
    if not check_docker_running():
        print("\n🔄 Inicia Docker Desktop")
        return
    
    while True:
        print("\nOpciones:")
        print("1. Construir e iniciar contenedor")
        print("2. Detener contenedor")
        print("3. Ver estado")
        print("4. Ver logs")
        print("5. Ejecutar scraper ahora")
        print("6. Salir")
        
        choice = input("\nElige una opción (1-6): ").strip()
        
        if choice == '1':
            build_and_start()
        elif choice == '2':
            stop_container()
        elif choice == '3':
            show_status()
        elif choice == '4':
            show_logs()
        elif choice == '5':
            run_now()
        elif choice == '6':
            print("👋 ¡Hasta luego!")
            break
        else:
            print("❌ Opción inválida")

if __name__ == "__main__":
    main()
