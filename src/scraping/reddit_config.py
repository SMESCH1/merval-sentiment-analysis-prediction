import os
from typing import Dict, Any
import praw
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Configuración de Reddit como diccionario
reddit_config = {
    'client_id': os.getenv('client_id', ''),
    'client_secret': os.getenv('client_secret', ''),
    'user_agent': "scraper-MIA:v1.0 (by u/Sure_Session_9073)",
    'username': os.getenv('reddit_username', ''),
    'password': os.getenv('reddit_password', '')
}

# subreddits de finanzas Argentina
nombre_subreddits = [
    'merval',           
    'argentina',        # este es general de Arg, pero se puede buscar por keywords
    'argentinacrypto',  
    'argentinaeconomia', 
    'CryptoArgentina',   
    'InversionesArg',   
    'finanzaspersonales' # este es en español, no específico de arg
]

# Keywords to filter relevant posts (Spanish/Argentine finance terms)
keywords = [
    'merval', 'bovespa', 'acciones', 'bonos', 'dólar', 'peso', 'inflación',
    'dolar', 'peso', 'inversion', 'inversión', 'bolsa', 'mercado', 'finanzas',
    'economia', 'economía', 'crypto', 'bitcoin', 'ethereum', 'criptomonedas',
    'broker', 'trading', 'inversor', 'inversora', 'portfolio', 'cartera',
    'renta', 'dividendos', 'cedears', 'adrs', 'fci', 'fondo', 'plazo fijo',
    'pf', 'leliq', 'bcra', 'indec', 'ipc', 'devaluacion', 'devaluación'
]

# Reddit API rate limits (requests per minute)
""" rate = {
    'posts_per_minute': 60,
    'comments_per_minute': 60,
    'delay_between_requests': 1  # seconds
} """

def get_reddit_config() -> Dict[str, Any]:
    """Obtener configuracion de Reddit con validacion."""
    required_fields = ['client_id', 'client_secret', 'user_agent']
    
    for field in required_fields:
        if not reddit_config.get(field) or reddit_config.get(field) == '':
            raise ValueError(f"Missing required Reddit configuration: {field}")
    
    return reddit_config

def validate_credentials() -> bool:
    """Validar que todas las credenciales requeridas esten presentes."""
    try:
        get_reddit_config()
        return True
    except ValueError:
        return False
