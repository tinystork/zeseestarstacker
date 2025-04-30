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