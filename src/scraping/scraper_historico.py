#!/usr/bin/env python3

"""
Scraper histórico de Reddit usando PRAW.
Diseñado para ejecutarse por largos períodos con guardado progresivo.
Filtra automáticamente por keywords financieras.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
import signal

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import praw
import pandas as pd
from src.scraping.reddit_config import (
    get_reddit_config,
    nombre_subreddits,
    keywords
)


class HistoricalRedditScraper:
    """Scraper histórico de Reddit con PRAW."""
    
    def __init__(self, output_dir: str = "data/historical"):
        """Inicializar scraper."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializar PRAW
        config = get_reddit_config()
        self.reddit = praw.Reddit(
            client_id=config['client_id'],
            client_secret=config['client_secret'],
            user_agent=config['user_agent'],
            username=config.get('username'),
            password=config.get('password')
        )
        
        # Rate limiting: Reddit permite ~60 requests/min sin auth, ~600 con auth
        self.delay_between_requests = 1.0  # Segundos entre requests
        self.delay_between_subreddits = 5.0  # Segundos entre subreddits
        self.delay_between_batches = 60.0  # Segundos entre batches
        
        # Checkpoint file
        self.checkpoint_file = self.output_dir / "scraping_checkpoint.json"
        
        # Setup logging
        self.setup_logging()
        
        # Flag para interrupción graceful
        self.interrupted = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Manejar interrupciones (Ctrl+C)."""
        self.logger.warning("Interrupcion recibida. Guardando progreso...")
        self.interrupted = True
    
    def setup_logging(self):
        """Configurar logging."""
        # Usar ruta absoluta relativa al proyecto para que siempre esté en el mismo lugar
        project_root = Path(__file__).parent.parent.parent
        log_dir = project_root / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"historical_scraper_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def is_finance_related(self, text: str) -> bool:
        """Verificar si un texto está relacionado con finanzas."""
        if not text:
            return False
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in keywords)
    
    def extract_post_data(self, submission) -> Dict[str, Any]:
        """Extraer datos de un submission."""
        try:
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
        except Exception as e:
            self.logger.warning(f"Error extrayendo datos de post {submission.id}: {e}")
            return None
    
    def extract_comment_data(self, comment) -> Dict[str, Any]:
        """Extraer datos de un comentario."""
        try:
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
        except Exception as e:
            self.logger.warning(f"Error extrayendo datos de comentario {comment.id}: {e}")
            return None
    
    def scrape_subreddit_historical(
        self,
        subreddit_name: str,
        start_date: datetime,
        end_date: datetime,
        max_posts: int = 10000,
        include_comments: bool = True,
        filter_by_keywords: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Scrapear datos históricos de un subreddit."""
        self.logger.info(f"Scrapeando r/{subreddit_name} desde {start_date.date()} hasta {end_date.date()}")
        if not filter_by_keywords:
            self.logger.info(f"  Filtro de keywords DESACTIVADO - obteniendo todos los posts")
        
        posts = []
        comments = []
        seen_ids = set()  # Para evitar duplicados
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            strategies = [
                ('top', 'all', max_posts),
                ('hot', 'all', max_posts),
            ]
            
            for method_name, time_filter, limit in strategies:
                if self.interrupted or len(posts) >= max_posts:
                    break
                
                self.logger.info(f"  Buscando posts {method_name} (time_filter={time_filter}) en r/{subreddit_name}...")
                try:
                    method = getattr(subreddit, method_name)
                    count_before_filter = 0
                    count_after_filter = 0
                    
                    for submission in method(time_filter=time_filter, limit=limit):
                        if self.interrupted:
                            break
                        
                        # Evitar duplicados
                        if submission.id in seen_ids:
                            continue
                        
                        try:
                            post_date = datetime.fromtimestamp(submission.created_utc)
                            count_before_filter += 1
                            
                            # Filtrar por fecha
                            if post_date < start_date or post_date > end_date:
                                continue
                            
                            # Filtrar por keywords (si está habilitado)
                            if filter_by_keywords:
                                text_to_check = submission.title + ' ' + submission.selftext
                                if not self.is_finance_related(text_to_check):
                                    continue
                            
                            post_data = self.extract_post_data(submission)
                            if post_data:
                                posts.append(post_data)
                                seen_ids.add(submission.id)
                                count_after_filter += 1
                                
                                # Scrapear comentarios
                                if include_comments:
                                    try:
                                        submission.comments.replace_more(limit=0)
                                        for comment in submission.comments.list():
                                            if comment.id in seen_ids:
                                                continue
                                            
                                            comment_date = datetime.fromtimestamp(comment.created_utc)
                                            if start_date <= comment_date <= end_date:
                                                # Filtrar comentarios por keywords (si está habilitado)
                                                if not filter_by_keywords or self.is_finance_related(comment.body):
                                                    comment_data = self.extract_comment_data(comment)
                                                    if comment_data:
                                                        comments.append(comment_data)
                                                        seen_ids.add(comment.id)
                                    except Exception as e:
                                        self.logger.debug(f"Error obteniendo comentarios: {e}")
                            
                            time.sleep(self.delay_between_requests)
                            
                            if count_after_filter % 50 == 0:
                                self.logger.info(f"    {count_after_filter} posts válidos de {count_before_filter} procesados...")
                                
                        except Exception as e:
                            self.logger.warning(f"Error procesando submission: {e}")
                            continue
                    
                    self.logger.info(f"    {method_name}/{time_filter}: {count_after_filter} posts válidos de {count_before_filter} procesados")
                            
                except Exception as e:
                    self.logger.error(f"Error en subreddit.{method_name}(): {e}")
            
            
            self.logger.info(f"r/{subreddit_name}: {len(posts)} posts, {len(comments)} comentarios")
            
        except Exception as e:
            self.logger.error(f"Error scrapeando r/{subreddit_name}: {e}")
            import traceback
            traceback.print_exc()
        
        return {
            'posts': posts,
            'comments': comments,
            'total_posts': len(posts),
            'total_comments': len(comments)
        }
    
    def save_batch_data(self, batch_name: str, data: Dict[str, Dict[str, Any]]):
        """Guardar datos de un batch."""
        # Guardar JSON
        json_file = self.output_dir / f"reddit_historical_{batch_name}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        # Guardar CSV por subreddit
        for subreddit, subreddit_data in data.items():
            if subreddit_data['posts']:
                posts_df = pd.DataFrame(subreddit_data['posts'])
                csv_file = self.output_dir / f"reddit_historical_{batch_name}_{subreddit}_posts.csv"
                posts_df.to_csv(csv_file, index=False, encoding='utf-8')
            
            if subreddit_data['comments']:
                comments_df = pd.DataFrame(subreddit_data['comments'])
                csv_file = self.output_dir / f"reddit_historical_{batch_name}_{subreddit}_comments.csv"
                comments_df.to_csv(csv_file, index=False, encoding='utf-8')
        
        self.logger.info(f"Datos guardados: {json_file}")
    
    def load_checkpoint(self) -> Optional[Dict]:
        """Cargar checkpoint de progreso."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Error cargando checkpoint: {e}")
        return None
    
    def save_checkpoint(self, completed_batches: List[str], current_batch: Optional[str] = None):
        """Guardar checkpoint de progreso."""
        checkpoint = {
            'completed_batches': completed_batches,
            'current_batch': current_batch,
            'last_update': datetime.now().isoformat()
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def create_monthly_batches(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Crear batches mensuales."""
        batches = []
        current = start_date
        
        while current <= end_date:
            month_start = current.replace(day=1)
            
            if current.month == 12:
                month_end = current.replace(year=current.year + 1, month=1, day=1) - timedelta(days=1)
            else:
                month_end = current.replace(month=current.month + 1, day=1) - timedelta(days=1)
            
            month_start = max(month_start, start_date)
            month_end = min(month_end, end_date)
            
            batches.append({
                'start': month_start,
                'end': month_end,
                'name': month_start.strftime('%Y-%m')
            })
            
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
        return batches
    
    def scrape_historical(
        self,
        start_date: datetime,
        end_date: datetime,
        max_posts_per_subreddit: int = 10000,
        include_comments: bool = True,
        resume: bool = True,
        filter_by_keywords: bool = True
    ):
        """
        Ejecutar scraping histórico completo.
        
        Args:
            filter_by_keywords: Si True, solo scrapea posts/comentarios con keywords financieras.
                               Si False, obtiene todos los posts de los subreddits configurados.
        """
        self.logger.info("=" * 60)
        self.logger.info("Iniciando scraping historico con PRAW")
        self.logger.info("=" * 60)
        self.logger.info(f"Rango: {start_date.date()} a {end_date.date()}")
        self.logger.info(f"Subreddits configurados: {len(nombre_subreddits)}")
        for sub in nombre_subreddits:
            self.logger.info(f"  - r/{sub}")
        if filter_by_keywords:
            self.logger.info(f"Filtrando por keywords financieras")
            self.logger.info(f"Keywords: {len(keywords)} terminos definidos")
        else:
            self.logger.info(f"Filtro de keywords DESACTIVADO - obteniendo todos los posts")
        
        # Crear batches
        batches = self.create_monthly_batches(start_date, end_date)
        self.logger.info(f"Total batches: {len(batches)}")
        
        # Cargar checkpoint si existe
        completed_batches = []
        if resume:
            checkpoint = self.load_checkpoint()
            if checkpoint:
                completed_batches = checkpoint.get('completed_batches', [])
                self.logger.info(f"Checkpoint encontrado. Batches completados: {len(completed_batches)}")
        
        # Procesar batches
        start_time = datetime.now()
        successful_batches = 0
        failed_batches = 0
        
        for i, batch in enumerate(batches):
            batch_name = batch['name']
            
            # Saltar si ya está completado
            if batch_name in completed_batches:
                self.logger.info(f"Saltando batch {batch_name} (ya completado)")
                continue
            
            if self.interrupted:
                self.logger.warning("Scraping interrumpido por usuario")
                break
            
            self.logger.info("=" * 60)
            self.logger.info(f"Procesando batch {i+1}/{len(batches)}: {batch_name}")
            self.logger.info(f"Periodo: {batch['start'].date()} a {batch['end'].date()}")
            self.logger.info("=" * 60)
            
            try:
                all_data = {}
                
                # Solo scrapear subreddits de nombre_subreddits
                for subreddit in nombre_subreddits:
                    if self.interrupted:
                        break
                    
                    subreddit_data = self.scrape_subreddit_historical(
                        subreddit_name=subreddit,
                        start_date=batch['start'],
                        end_date=batch['end'],
                        max_posts=max_posts_per_subreddit,
                        include_comments=include_comments,
                        filter_by_keywords=filter_by_keywords
                    )
                    
                    all_data[subreddit] = subreddit_data
                    
                    # Pausa entre subreddits
                    time.sleep(self.delay_between_subreddits)
                
                # Guardar batch
                self.save_batch_data(batch_name, all_data)
                
                # Actualizar checkpoint
                completed_batches.append(batch_name)
                self.save_checkpoint(completed_batches)
                
                # Resumen del batch
                total_posts = sum(d['total_posts'] for d in all_data.values())
                total_comments = sum(d['total_comments'] for d in all_data.values())
                
                self.logger.info(f"Batch {batch_name} completado:")
                self.logger.info(f"  Posts: {total_posts}")
                self.logger.info(f"  Comentarios: {total_comments}")
                
                successful_batches += 1
                
                # Pausa entre batches
                if i < len(batches) - 1 and not self.interrupted:
                    self.logger.info(f"Esperando {self.delay_between_batches} segundos...")
                    time.sleep(self.delay_between_batches)
                    
            except Exception as e:
                self.logger.error(f"Error en batch {batch_name}: {e}")
                import traceback
                traceback.print_exc()
                failed_batches += 1
        
        # Resumen final
        end_time = datetime.now()
        duration = end_time - start_time
        
        self.logger.info("=" * 60)
        self.logger.info("Scraping historico completado")
        self.logger.info("=" * 60)
        self.logger.info(f"Duracion: {duration}")
        self.logger.info(f"Batches exitosos: {successful_batches}")
        self.logger.info(f"Batches fallidos: {failed_batches}")
        if len(batches) > 0:
            self.logger.info(f"Tasa de exito: {successful_batches/len(batches)*100:.1f}%")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description='Scraper histórico de Reddit usando PRAW (filtra por keywords financieras)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='2020-01-01',
        help='Fecha de inicio (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='Fecha de fin (YYYY-MM-DD). Default: hoy'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/historical',
        help='Directorio de salida'
    )
    parser.add_argument(
        '--max-posts',
        type=int,
        default=10000,
        help='Máximo posts por subreddit por batch'
    )
    parser.add_argument(
        '--no-comments',
        action='store_true',
        help='No scrapear comentarios'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='No continuar desde checkpoint'
    )
    parser.add_argument(
        '--no-keywords-filter',
        action='store_true',
        help='No filtrar por keywords (obtener todos los posts de los subreddits configurados)'
    )
    
    args = parser.parse_args()
    
    # Parsear fechas
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else datetime.now()
    
    # Crear scraper
    scraper = HistoricalRedditScraper(output_dir=args.output_dir)
    
    # Ejecutar scraping
    scraper.scrape_historical(
        start_date=start_date,
        end_date=end_date,
        max_posts_per_subreddit=args.max_posts,
        include_comments=not args.no_comments,
        resume=not args.no_resume,
        filter_by_keywords=not args.no_keywords_filter
    )


if __name__ == "__main__":
    main()