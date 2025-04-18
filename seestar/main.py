#!/usr/bin/env python3
"""
Script principal pour lancer l'application Seestar Stacker.
"""

import os
import sys
# Ajouter le chemin du projet au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from seestar.gui import SeestarStackerGUI


def check_dependencies():
    """
    Vérifie que toutes les dépendances sont installées.
    
    Returns:
        bool: True si toutes les dépendances sont installées, False sinon
    """
    dependencies = [
        ('numpy', 'numpy'),
        ('astropy', 'astropy'),
        ('cv2', 'opencv-python'),
        ('astroalign', 'astroalign'),
        ('tqdm', 'tqdm'),
        ('matplotlib', 'matplotlib')
    ]
    
    missing_deps = []
    
    for module_name, package_name in dependencies:
        try:
            __import__(module_name)
        except ImportError:
            missing_deps.append(package_name)
    
    if missing_deps:
        print("ERREUR: Dépendances manquantes")
        print("Veuillez installer les dépendances suivantes:")
        print(f"pip install {' '.join(missing_deps)}")
        return False
    
    return True


def main():
    """Point d'entrée principal de l'application."""
    print("Démarrage de Seestar Stacker...")
    
    # Vérifier les dépendances
    if not check_dependencies():
        sys.exit(1)
    
    # Lancer l'interface graphique
    app = SeestarStackerGUI()
    app.run()


if __name__ == "__main__":
    main()