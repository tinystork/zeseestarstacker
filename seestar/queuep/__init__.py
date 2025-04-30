# --- START OF FILE seestar/queuep/__init__.py ---
"""
Package queuep pour Seestar - fournit les fonctionnalités de gestion
de file d'attente pour le traitement des images astronomiques.
(image_db.py et image_info.py retirés car non utilisés par queue_manager actuel)
"""

from .queue_manager import SeestarQueuedStacker

__all__ = [
    'SeestarQueuedStacker'
]
# --- END OF FILE seestar/queuep/__init__.py ---