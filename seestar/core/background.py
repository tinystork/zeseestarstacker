"""
Module pour la soustraction de fond 2D des images astronomiques
en utilisant photutils.
"""
import numpy as np
import traceback # Pour un meilleur débogage des erreurs photutils

# Essayer d'importer les composants nécessaires de photutils
try:
    from photutils.background import Background2D, MedianBackground
    from astropy.stats import SigmaClip
    _PHOTOUTILS_AVAILABLE = True
    print("DEBUG [background.py]: Photutils importé avec succès.")
except ImportError:
    _PHOTOUTILS_AVAILABLE = False
    print("ERREUR [background.py]: Photutils non trouvé. La soustraction de fond 2D sera désactivée.")
    # Définir des classes factices pour que la fonction ne plante pas si appelée sans photutils
    # (bien qu'on vérifiera _PHOTOUTILS_AVAILABLE avant d'appeler)
    class Background2D: pass
    class MedianBackground: pass
    class SigmaClip: pass

__all__ = ["subtract_background_2d", "_PHOTOUTILS_AVAILABLE"] # Exposer la fonction et le flag

def subtract_background_2d(image_data: np.ndarray,
                           box_size: int = 128,    # Taille de la boîte pour l'estimation locale
                           filter_size: int = 5,   # Taille du filtre médian pour lisser la carte de fond
                           sigma_clip_val: float = 3.0,
                           exclude_percentile: float = 98.0, # Pour ignorer les pixels très brillants
                           maxiters_sigma_clip: int = 5):    # Max itérations pour sigma clip
    """
    Soustrait un modèle de fond 2D d'une image (monochrome ou par canal pour RGB)
    en utilisant photutils.Background2D.

    Args:
        image_data (np.ndarray): Image d'entrée (H,W) ou (H,W,3),
                                 attendue en float32, plage [0,1] ou ADU.
        box_size (int): Taille de la boîte pour l'estimation du fond (px).
        filter_size (int): Taille du filtre médian appliqué à la carte de fond (px).
                           Doit être impair.
        sigma_clip_val (float): Valeur sigma pour le SigmaClip.
        exclude_percentile (float): Percentile de pixels à exclure (les plus brillants)
                                   de chaque boîte avant l'estimation du fond. 0-100.
        maxiters_sigma_clip (int): Nombre max d'itérations pour le sigma clipping.


    Returns:
        tuple: (image_corrected, background_model)
               - image_corrected (np.ndarray): Image avec fond soustrait.
               - background_model (np.ndarray): Modèle de fond 2D calculé.
               Retourne (image_data, None) si photutils n'est pas dispo ou en cas d'erreur.
    """
    print(f"DEBUG [subtract_background_2d]: Début. Shape entrée: {image_data.shape}, box: {box_size}, filt: {filter_size}, sig: {sigma_clip_val}, excl_perc: {exclude_percentile}")

    if not _PHOTOUTILS_AVAILABLE:
        print("WARN [subtract_background_2d]: Photutils non disponible, retour image originale.")
        return image_data, None

    if image_data is None:
        print("WARN [subtract_background_2d]: image_data est None, retour.")
        return None, None

    # Assurer que filter_size est impair
    if filter_size % 2 == 0:
        filter_size += 1
        print(f"DEBUG [subtract_background_2d]: filter_size ajusté à {filter_size} (doit être impair).")

    # Traitement par canal pour les images RGB
    if image_data.ndim == 3 and image_data.shape[2] == 3:
        print("DEBUG [subtract_background_2d]: Traitement image RGB par canal...")
        corrected_channels = []
        # Pour le modèle de fond RGB, on pourrait moyenner les modèles de fond des canaux,
        # ou retourner le modèle du canal vert (souvent le plus représentatif pour la luminance du fond).
        # Pour l'instant, on ne retourne pas de modèle de fond combiné pour RGB.
        background_model_combined = None 

        for c in range(3):
            channel_name = ['R', 'G', 'B'][c]
            print(f"DEBUG [subtract_background_2d]: Traitement canal {channel_name}...")
            # Appel récursif pour chaque canal
            channel_corrected, channel_bkg_model = subtract_background_2d(
                image_data[:, :, c],
                box_size=box_size,
                filter_size=filter_size,
                sigma_clip_val=sigma_clip_val,
                exclude_percentile=exclude_percentile,
                maxiters_sigma_clip=maxiters_sigma_clip
            )
            if channel_corrected is None: # Si erreur sur un canal
                print(f"ERREUR [subtract_background_2d]: Échec soustraction fond pour canal {channel_name}. Retour image originale.")
                return image_data, None # Retourner l'originale complète si un canal échoue
            
            corrected_channels.append(channel_corrected)
            # On pourrait stocker channel_bkg_model pour faire une moyenne plus tard si besoin
            # mais pour l'instant, le modèle de fond n'est pas retourné pour RGB.

        # Réassembler l'image couleur
        # S'assurer que les canaux ont bien été traités
        if len(corrected_channels) == 3:
            image_corrected_rgb = np.dstack(corrected_channels)
            print("DEBUG [subtract_background_2d]: Tous les canaux RGB traités et réassemblés.")
            return image_corrected_rgb.astype(np.float32), None # Pas de modèle de fond unique pour RGB ici
        else:
            print("ERREUR [subtract_background_2d]: Nombre incorrect de canaux après traitement RGB. Retour image originale.")
            return image_data, None


    # Traitement pour image monochrome (ou un seul canal)
    elif image_data.ndim == 2:
        print("DEBUG [subtract_background_2d]: Traitement image monochrome.")
        try:
            # Configurer SigmaClip
            sigma_clip = SigmaClip(sigma=sigma_clip_val, maxiters=maxiters_sigma_clip)
            # Configurer l'estimateur de fond
            bkg_estimator = MedianBackground()
            
            # S'assurer que les données sont en float (Background2D peut être sensible)
            data_float = image_data.astype(float)

            print(f"DEBUG [subtract_background_2d]: Appel Background2D avec box_size=({box_size},{box_size}), filter_size=({filter_size},{filter_size}), exclude_percentile={exclude_percentile}")
            bkg = Background2D(data_float,
                               box_size=(box_size, box_size),
                               filter_size=(filter_size, filter_size),
                               sigma_clip=sigma_clip,
                               bkg_estimator=bkg_estimator,
                               exclude_percentile=exclude_percentile, # Ignorer les N% pixels les plus brillants dans chaque boîte
                               fill_value=0.0) # Valeur pour les pixels masqués/ignorés

            background_model = bkg.background
            image_corrected = data_float - background_model
            
            print(f"DEBUG [subtract_background_2d]: Fond soustrait. Médiane du modèle: {bkg.background_median:.4f}, RMS: {bkg.background_rms_median:.4f}")
            # Retourner l'image corrigée et le modèle de fond, en s'assurant du type float32
            return image_corrected.astype(np.float32), background_model.astype(np.float32)

        except ValueError as ve: # Souvent lié à box_size trop grand pour l'image
            print(f"ERREUR [subtract_background_2d] (ValueError avec photutils): {ve}")
            print("   -> Causes possibles: box_size trop grande pour la taille de l'image, ou image avec peu de variance/structure.")
            traceback.print_exc(limit=1)
            return image_data.astype(np.float32), None # Retourner l'original en float32
        except Exception as e:
            print(f"ERREUR [subtract_background_2d] (Exception inattendue avec photutils): {e}")
            traceback.print_exc(limit=2)
            return image_data.astype(np.float32), None # Retourner l'original en float32
    else:
        print(f"WARN [subtract_background_2d]: Shape d'image non supportée ({image_data.shape}). Retour image originale.")
        return image_data, None

# --- END OF FILE seestar/core/background.py ---