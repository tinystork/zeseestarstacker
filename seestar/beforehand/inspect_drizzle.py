# -----------------------------------------------------------------------------
# Auteur       : TRISTAN NAULEAU 
# Date         : 2025-07-12
# Licence      : GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007
#
# Ce travail est distribué librement en accord avec les termes de la
# GNU GPL v3 (https://www.gnu.org/licenses/gpl-3.0.html).
# Vous êtes libre de redistribuer et de modifier ce code, à condition
# de conserver cette notice et de mentionner que je suis l’auteur
# de tout ou partie du code si vous le réutilisez.
# -----------------------------------------------------------------------------
# Author       : TRISTAN NAULEAU
# Date         : 2025-07-12
# License      : GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007
#
# This work is freely distributed under the terms of the
# GNU GPL v3 (https://www.gnu.org/licenses/gpl-3.0.html).
# You are free to redistribute and modify this code, provided that
# you keep this notice and mention that I am the author
# of all or part of the code if you reuse it.
# -----------------------------------------------------------------------------
# Script pour explorer les paramètres de Drizzle
import inspect
from drizzle.resample import Drizzle

# Afficher la signature du constructeur
print("Signature de Drizzle.__init__:")
sig = inspect.signature(Drizzle.__init__)
print(sig)

# Essayer de créer un objet Drizzle sans arguments
try:
    driz = Drizzle()
    print("\nCréation de l'objet Drizzle sans arguments : OK")
    
    # Afficher les méthodes disponibles
    print("\nMéthodes disponibles :")
    methods = [name for name in dir(driz) if callable(getattr(driz, name)) and not name.startswith('_')]
    for method in methods:
        print(f"  {method}")
    
    # Afficher les propriétés disponibles
    print("\nPropriétés disponibles :")
    attrs = [name for name in dir(driz) if not callable(getattr(driz, name)) and not name.startswith('_')]
    for attr in attrs:
        try:
            value = getattr(driz, attr)
            print(f"  {attr}: {value}")
        except Exception as e:
            print(f"  {attr}: <erreur d'accès: {e}>")
            
except Exception as e:
    print(f"\nErreur lors de la création d'un objet Drizzle sans arguments: {e}")