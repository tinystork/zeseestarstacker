# --- START OF FILE seestar/enhancement/drizzle_integration.py (CORRECTED STRUCTURE) ---
import numpy as np
import os
import traceback
import warnings
import time
import gc

# --- Astropy imports ---
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.coordinates import SkyCoord # Bien que non utilisé ici, gardons pour WCS
from astropy import units as u           # Bien que non utilisé ici, gardons pour WCS

# --- Dépendance optionnelle pour Debayer ---
try:
    import colour_demosaicing
    _DEBAYER_AVAILABLE = True
    print("DEBUG DrizzleIntegration: Found colour_demosaicing.")
except ImportError:
    print("WARNING DrizzleIntegration: colour-demosaicing library not found. Debayering in load helper will fail.")
    _DEBAYER_AVAILABLE = False
    # Définir une fonction factice si la bibliothèque est manquante
    class colour_demosaicing:
        @staticmethod
        def demosaicing_CFA_Bayer_Malvar2004(data, pattern):
            print("ERROR: colour_demosaicing not available for demosaicing.")
            # Retourner juste les données N&B dans ce cas
            # (Ou lever une erreur plus explicite si le debayering est obligatoire)
            return data # Fallback très basique

# --- Import de la fonction tdriz (si disponible) ---
_standalone_drizzle_available = False
tdriz_function = None # Placeholder

try:
    # Importer le module cdrizzle qui contient tdriz
    from drizzle import cdrizzle
    # Vérifier si tdriz existe et est appelable
    if hasattr(cdrizzle, 'tdriz') and callable(getattr(cdrizzle, 'tdriz')):
        tdriz_function = cdrizzle.tdriz # Assigner la fonction
        print("DEBUG DrizzleIntegration: Successfully found 'drizzle.cdrizzle.tdriz' function.")
        _standalone_drizzle_available = True
    else:
        # Lever une ImportError si tdriz n'est pas trouvé DANS cdrizzle
        raise ImportError("Function 'tdriz' not found within drizzle.cdrizzle module.")
except ImportError as e_imp:
    # Erreur si drizzle.cdrizzle ou tdriz n'est pas trouvé
    print(f"ERREUR CRITIQUE: Impossible d'importer ou trouver 'drizzle.cdrizzle.tdriz'. Erreur: {e_imp}")
    print("Vérifiez l'installation: pip install drizzle")
    traceback.print_exc(limit=1)
    # Définir une fonction factice pour éviter les erreurs d'attribut plus tard
    def dummy_tdriz(*args, **kwargs):
        raise ImportError("Standalone drizzle function (tdriz) not available or import failed.")
    tdriz_function = dummy_tdriz
except Exception as e_other:
    # Capturer d'autres erreurs pendant l'import (ex: dépendances manquantes pour drizzle)
    print(f"ERREUR INATTENDUE pendant l'import de drizzle/cdrizzle: {type(e_other).__name__}: {e_other}")
    traceback.print_exc(limit=3)
    def dummy_tdriz(*args, **kwargs):
        raise ImportError(f"Unexpected error during drizzle import: {e_other}")
    tdriz_function = dummy_tdriz


# Ignorer certains avertissements Astropy WCS
warnings.filterwarnings('ignore', category=FITSFixedWarning)
################################################################################################################################################


# --- DANS LE FICHIER: seestar/enhancement/drizzle_integration.py ---

def _load_drizzle_temp_file(filepath):
    """
    Charge spécifiquement un fichier FITS temporaire Drizzle.
    Attend des données couleur (C,H,W ou H,W,C) float32 et un header WCS.
    Retourne : (image_data_HxWx3_float32, wcs, header) ou (None, None, None)
    """
    # print(f"   -> Chargement Temp Drizzle: {os.path.basename(filepath)}") # DEBUG Optionnel
    try:
        with fits.open(filepath, memmap=False) as hdul:
            if not hdul or not hdul[0].is_image or hdul[0].data is None:
                print(f"      - ERREUR: HDU primaire invalide ou vide dans {os.path.basename(filepath)}")
                return None, None, None

            hdu = hdul[0]
            data = hdu.data # astropy lit les données telles qu'elles sont stockées
            header = hdu.header

            # --- CORRECTION : Gérer l'ordre des axes C,H,W ou H,W,C ---
            data_hxwx3 = None
            if data.ndim == 3:
                # Cas 1: Données déjà en H,W,C (peu probable si sauvé avec moveaxis)
                if data.shape[2] == 3:
                    data_hxwx3 = data.astype(np.float32)
                    print(f"      - Données temp lues comme HxWxC: {data_hxwx3.shape}")
                # Cas 2: Données en C,H,W (le plus probable après notre sauvegarde)
                elif data.shape[0] == 3:
                    data_hxwx3 = np.moveaxis(data, 0, -1).astype(np.float32) # Transpose C,H,W -> H,W,C
                    print(f"      - Données temp lues comme CxHxW, transposées en HxWxC: {data_hxwx3.shape}")
                else:
                    # Autre forme 3D inattendue
                    print(f"      - ERREUR: Shape 3D inattendue {data.shape} dans {os.path.basename(filepath)}")
                    return None, None, None
            else:
                # Pas 3D
                print(f"      - ERREUR: Données temp non 3D ({data.ndim}D) dans {os.path.basename(filepath)}")
                return None, None, None
            # --- FIN CORRECTION ---

            # --- WCS reste identique ---
            wcs = None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', FITSFixedWarning)
                    wcs_hdr = WCS(header, naxis=2) # WCS 2D pour le plan image
                if wcs_hdr.is_celestial:
                    wcs = wcs_hdr
                else:
                    print(f"      - WARNING: WCS non céleste dans header temp {os.path.basename(filepath)}")
            except Exception as e_wcs:
                print(f"      - WARNING: Erreur chargement WCS header temp {os.path.basename(filepath)}: {e_wcs}")

            if wcs is None:
                 print(f"      - ERREUR: WCS non trouvé/valide dans header temp {os.path.basename(filepath)}")
                 # Important d'échouer ici si pas de WCS
                 return None, None, None

            # Retourner les données HxWx3 float32, le wcs, le header
            return data_hxwx3, wcs, header

    except FileNotFoundError:
        print(f"   - ERREUR: Fichier temp non trouvé: {filepath}")
        return None, None, None
    except Exception as e:
        print(f"   - ERREUR chargement fichier temp FITS {os.path.basename(filepath)}: {e}")
        traceback.print_exc(limit=1)
        return None, None, None

# --- FIN FONCTION _load_drizzle_temp_file CORRIGÉE ---



###################################################################################################################################



# === Fonctions Helper (Intégrées ou adaptées) ===





def _create_wcs_from_header(header):
    """Tente de créer un WCS Astropy à partir d'un header type Seestar."""
    # (Code identique à celui que vous aviez dans gcolor_core_drizzle.py)
    required_keys = ['NAXIS1', 'NAXIS2', 'RA', 'DEC', 'FOCALLEN', 'XPIXSZ', 'YPIXSZ']
    if not all(key in header for key in required_keys):
        # print("   - Helper Warning: Missing WCS keys:", [k for k in required_keys if k not in header])
        return None
    try:
        naxis1 = int(header['NAXIS1']); naxis2 = int(header['NAXIS2'])
        ra_deg = float(header['RA']); dec_deg = float(header['DEC'])
        focal_len_mm = float(header['FOCALLEN'])
        pixel_size_x_um = float(header['XPIXSZ']); pixel_size_y_um = float(header['YPIXSZ'])
        focal_len_m = focal_len_mm * 1e-3
        pixel_size_x_m = pixel_size_x_um * 1e-6; pixel_size_y_m = pixel_size_y_um * 1e-6
        scale_x_rad_per_pix = pixel_size_x_m / focal_len_m; scale_y_rad_per_pix = pixel_size_y_m / focal_len_m
        deg_per_rad = 180.0 / np.pi
        scale_x_deg_per_pix = scale_x_rad_per_pix * deg_per_rad; scale_y_deg_per_pix = scale_y_rad_per_pix * deg_per_rad

        w = WCS(naxis=2)
        w.wcs.crpix = [naxis1 / 2.0 + 0.5, naxis2 / 2.0 + 0.5]
        w.wcs.crval = [ra_deg, dec_deg]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.cdelt = np.array([-scale_x_deg_per_pix, scale_y_deg_per_pix])
        w.wcs.cunit = ['deg', 'deg']
        # Assurer que la matrice PC existe (même si identité)
        w.wcs.pc = np.array([[1.0, 0.0], [0.0, 1.0]])
        # print(f"   - Helper WCS Gen OK: Scale ({w.wcs.cdelt[0]*3600:.2f}\", {w.wcs.cdelt[1]*3600:.2f}\"/pix)")
        return w
    except Exception as e:
        # print(f"   - Helper Error generating WCS: {e}")
        return None

def _load_fits_data_wcs_debayered(filepath, bayer_pattern='GRBG'):
    """
    Charge les données FITS, dématrice si nécessaire, obtient WCS et header.
    Retourne : (données_couleur_HxWx3_float32, wcs, header) ou (None, None, None)
    """
    print(f"   -> Chargement/Debayer: {os.path.basename(filepath)}")
    try:
        with fits.open(filepath, memmap=False) as hdul:
            hdu = None; header = None
            # Chercher la première HDU image 2D valide
            for h in hdul:
                if h.is_image and h.data is not None and h.data.ndim == 2:
                    hdu = h; break
            if hdu is None: print("      - Aucune HDU image 2D trouvée."); return None, None, None

            bayer_data = hdu.data; header = hdu.header
            print(f"      - Données Bayer trouvées: {bayer_data.shape}, {bayer_data.dtype}")

            # Dématriçage
            if _DEBAYER_AVAILABLE:
                try:
                    bayer_float = bayer_data.astype(np.float32)
                    # Assurer que le pattern est valide pour la lib
                    valid_patterns = ['GRBG', 'RGGB', 'GBRG', 'BGGR']
                    pattern_upper = bayer_pattern.upper()
                    if pattern_upper not in valid_patterns:
                        print(f"      - WARNING: Motif Bayer '{pattern_upper}' invalide. Utilisation GRBG.")
                        pattern_upper = 'GRBG'
                    # Utiliser colour_demosaicing
                    rgb_image = colour_demosaicing.demosaicing_CFA_Bayer_Malvar2004(bayer_float, pattern=pattern_upper)
                    # S'assurer que c'est HxWx3
                    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
                        raise ValueError(f"Résultat Debayer inattendu shape: {rgb_image.shape}")
                    print(f"      - Debayer OK -> {rgb_image.shape}")
                except Exception as debayer_err:
                    print(f"      - ERREUR Debayer: {debayer_err}. Fichier ignoré.")
                    return None, None, None
            else:
                print("      - ERREUR: Bibliothèque colour-demosaicing manquante. Fichier ignoré.")
                return None, None, None

            # WCS
            wcs = None; wcs_source = "None"
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', FITSFixedWarning)
                    wcs_hdr = WCS(header, naxis=2)
                if wcs_hdr.is_celestial: wcs = wcs_hdr; wcs_source="Header"
            except Exception: pass # Ignorer erreurs WCS header

            if wcs is None: # Essayer génération
                wcs_gen = _create_wcs_from_header(header)
                if wcs_gen and wcs_gen.is_celestial: wcs = wcs_gen; wcs_source="Generated"

            if wcs: print(f"      - WCS trouvé ({wcs_source})")
            else: print("      - WARNING: WCS non trouvé/généré."); # Retourner quand même données/header? Non, drizzle a besoin du WCS.
            return None, None, None

            # Retourner données HxWx3 float32, WCS, Header
            return rgb_image.astype(np.float32), wcs, header

    except FileNotFoundError: print(f"   - ERREUR: Fichier non trouvé: {filepath}"); return None, None, None
    except Exception as e: print(f"   - ERREUR chargement FITS: {e}"); traceback.print_exc(limit=1); return None, None, None


# === CLASSE DrizzleProcessor ===

class DrizzleProcessor:
    """
    Encapsule la logique de Drizzle en utilisant la fonction drizzle.cdrizzle.tdriz.
    Prend une liste de fichiers FITS temporaires (déjà alignés, avec WCS) en entrée.
    """
    def __init__(self, scale_factor=2.0, pixfrac=1.0, kernel='square'):
        """
        Initialise le processeur Drizzle.

        Args:
            scale_factor (float): Facteur d'agrandissement de l'image de sortie.
            pixfrac (float): Fraction de pixel à "dropper" (0.0 à 1.0).
            kernel (str): Noyau de Drizzle ('square', 'gaussian', 'point', etc.).
        """
        if not _standalone_drizzle_available:
            raise ImportError("Standalone drizzle function (tdriz) not available or import failed.")

        # Validation des paramètres
        self.scale_factor = float(max(1.0, scale_factor))
        self.pixfrac = float(np.clip(pixfrac, 0.01, 1.0))
        valid_kernels = ['square', 'gaussian', 'point', 'tophat', 'turbo', 'lanczos2', 'lanczos3']
        self.kernel = kernel.lower() if kernel.lower() in valid_kernels else 'square'

        print(f"DrizzleProcessor (tdriz) Initialized: ScaleFactor={self.scale_factor}, Pixfrac={self.pixfrac}, Kernel='{self.kernel}'")

    def _create_output_wcs(self, ref_wcs, ref_shape_2d, scale_factor):
        """Crée le WCS et la shape pour l'image Drizzle de sortie."""
        if not ref_wcs or not ref_wcs.is_celestial:
            raise ValueError("Référence WCS invalide ou non céleste.")
        if len(ref_shape_2d) != 2:
             raise ValueError(f"Référence shape 2D attendue, reçu {ref_shape_2d}")

        h_in, w_in = ref_shape_2d
        out_h = int(round(h_in * scale_factor))
        out_w = int(round(w_in * scale_factor))
        out_shape_2d = (out_h, out_w) # Ordre (H, W) pour NumPy

        # Copier le WCS d'entrée et ajuster
        out_wcs = ref_wcs.deepcopy()

        # Ajuster échelle via CDELT ou CD matrix
        try:
            if hasattr(out_wcs.wcs, 'cd') and out_wcs.wcs.cd is not None and np.any(out_wcs.wcs.cd):
                out_wcs.wcs.cd = ref_wcs.pixel_scale_matrix / scale_factor
            elif hasattr(out_wcs.wcs, 'cdelt') and out_wcs.wcs.cdelt is not None and np.any(out_wcs.wcs.cdelt):
                out_wcs.wcs.cdelt = ref_wcs.wcs.cdelt / scale_factor
                 # S'assurer que la matrice PC existe si CDELT est utilisé
                if not hasattr(out_wcs.wcs, 'pc') or out_wcs.wcs.pc is None:
                     out_wcs.wcs.pc = np.array([[1.0, 0.0], [0.0, 1.0]])
            else:
                raise ValueError("Input WCS lacks valid CD or CDELT matrix/vector.")
        except Exception as e:
            raise ValueError(f"Failed to adjust pixel scale in output WCS: {e}")

        # Centrer CRPIX sur la nouvelle image de sortie
        out_wcs.wcs.crpix = [out_w / 2.0 + 0.5, out_h / 2.0 + 0.5]
        # Définir la taille pixel de sortie pour Astropy
        out_wcs.pixel_shape = (out_w, out_h) # Ordre (W, H) pour Astropy WCS

        print(f"   - Output WCS créé: Shape={out_shape_2d} (H,W), CRPIX={out_wcs.wcs.crpix}")
        return out_wcs, out_shape_2d # Retourne WCS et shape (H, W)
##################################################################################################################################################
# --- DANS LA CLASSE DrizzleProcessor DANS seestar/enhancement/drizzle_integration.py ---

    def apply_drizzle(self, temp_filepath_list):
        """
        Applique Drizzle (tdriz) sur une liste de fichiers FITS temporaires.
        Chaque fichier doit contenir une image couleur (HxWx3) alignée et un header WCS valide.
        UTILISE _load_drizzle_temp_file pour lire les entrées.

        Args:
            temp_filepath_list (list): Liste des chemins vers les fichiers FITS temporaires.

        Returns:
            tuple: (final_image_normalized, final_wht) ou (None, None) en cas d'erreur.
                   final_image_normalized est HxWx3 float32 [0,1].
                   final_wht est HxWx3 float32.
        """
        start_time = time.time()
        if not _standalone_drizzle_available:
            print("ERREUR: Fonction Drizzle (tdriz) non disponible.")
            return None, None
        if not temp_filepath_list:
            print("WARNING DrizzleProcessor: Liste de fichiers vide fournie.")
            return None, None

        print(f"DrizzleProcessor (tdriz): Application sur {len(temp_filepath_list)} fichiers temporaires...")

        output_wcs = None
        output_shape_2d = None # Shape (H, W)
        ref_shape_2d = None    # Shape (H, W) du premier fichier valide
        final_sci = None     # Sera (H, W, 3) float32
        final_wht = None     # Sera (H, W, 3) float32
        processed_count = 0

        # 1. Définir la grille de sortie basée sur le premier fichier valide
        print("   -> Définition grille sortie via premier fichier...")
        for i, filepath in enumerate(temp_filepath_list):
            # --- UTILISER LA NOUVELLE FONCTION ---
            img_data, wcs_in, header_in = _load_drizzle_temp_file(filepath)
            # ------------------------------------
            if img_data is not None and wcs_in is not None:
                # img_data est déjà HxWx3 float32
                ref_shape_2d = img_data.shape[:2] # Obtenir (H, W)
                try:
                    output_wcs, output_shape_2d = self._create_output_wcs(wcs_in, ref_shape_2d, self.scale_factor)
                    # Initialiser les tableaux de sortie NumPy (H, W, C) float32
                    out_h, out_w = output_shape_2d
                    final_sci = np.zeros((out_h, out_w, 3), dtype=np.float32)
                    final_wht = np.zeros((out_h, out_w, 3), dtype=np.float32)
                    print(f"   -> Grille sortie définie: Shape={final_sci.shape} (H,W,C)")
                    del img_data, wcs_in, header_in; gc.collect()
                    break # Sortir après avoir trouvé le premier valide
                except Exception as e_wcs:
                    print(f"      - ERREUR création WCS/Shape sortie: {e_wcs}. Skip Drizzle.")
                    del img_data, wcs_in, header_in; gc.collect(); return None, None
            else: print(f"      - Skip fichier {i+1} pour définition grille (échec chargement temp).")
            # Libérer mémoire même si échec chargement partiel
            if img_data is not None: del img_data
            if wcs_in is not None: del wcs_in
            if header_in is not None: del header_in
            gc.collect()

        if output_wcs is None or final_sci is None or final_wht is None:
            print("ERREUR: Impossible de définir la grille de sortie Drizzle (aucun fichier temp valide trouvé?).")
            return None, None

        # 2. Boucle Drizzle sur tous les fichiers
        print(f"   -> Démarrage boucle Drizzle sur {len(temp_filepath_list)} fichiers...")
        for i, filepath in enumerate(temp_filepath_list):
            if (i + 1) % 10 == 0 or i == 0 or i == len(temp_filepath_list) - 1:
                 print(f"      Processing Drizzle Input {i+1}/{len(temp_filepath_list)}: {os.path.basename(filepath)}")

            # --- UTILISER LA NOUVELLE FONCTION ---
            img_data, wcs_in, header_in = _load_drizzle_temp_file(filepath)
            # ------------------------------------

            if img_data is None or wcs_in is None:
                print(f"      - Skip (échec chargement/WCS fichier temp {i+1})")
                if img_data is not None: del img_data
                if wcs_in is not None: del wcs_in
                if header_in is not None: del header_in
                gc.collect()
                continue

            # img_data est maintenant HxWx3 float32

            # Vérifier si les dimensions sont cohérentes (optionnel mais bonne pratique)
            if img_data.shape[:2] != ref_shape_2d:
                 print(f"      - WARNING: Shape temp {img_data.shape[:2]} différente de référence {ref_shape_2d}. Poursuite...")

            try:
                # Préparer les arguments pour tdriz pour CHAQUE canal
                context_mask = np.ones(img_data.shape[:2], dtype=np.int32)
                input_weights = np.ones(img_data.shape[:2], dtype=np.float32)
                exptime = 1.0 # Défaut
                if header_in and 'EXPTIME' in header_in:
                    try: exptime = float(header_in['EXPTIME']); exptime = max(1e-6, exptime)
                    except (ValueError, TypeError): pass

                # Appeler tdriz pour chaque canal R, G, B
                for c in range(3):
                    channel_data = img_data[:, :, c] # Extraire le canal (déjà float32)

                    # --- APPEL MODIFIÉ : Conversion explicite et vérif ordre ---
                    tdriz_function(
                        # 1-7: Core arrays, WCS, masks (Ordre semble correct)
                        channel_data,           # OK (float32)
                        wcs_in,                 # OK (WCS obj)
                        input_weights,          # OK (float32)
                        output_wcs,             # OK (WCS obj)
                        final_sci[:, :, c],     # OK (float32)
                        context_mask,           # OK (int32)
                        # 8-13: Parameters (Conversion explicite + Ordre vérifié)
                        float(exptime),         # expin (forcer float)
                        float(self.pixfrac),    # pixfrac (forcer float)
                        str(self.kernel),       # kernel (forcer str)
                        0.0,                  # fillval (garder str '0.0')
                        str('exptime'),         # wt_scl (forcer str)
                        str('counts'),          # in_units (forcer str)
                        # --- Keyword Args ---
                        outwht=final_wht[:, :, c] # OK (float32)
                    )
                    # --- FIN APPEL MODIFIÉ ---
                processed_count += 1

            except ImportError as imp_err:
                 print(f"ERREUR Drizzle: {imp_err}")
                 del img_data, wcs_in, header_in; gc.collect(); return None, None
            except Exception as e_tdriz:
                 print(f"   - ERREUR appel tdriz pour fichier {i+1}: {e_tdriz}")
                 traceback.print_exc(limit=2)
            finally:
                 # Libérer mémoire après chaque fichier
                 del img_data, wcs_in, header_in
                 if (i + 1) % 20 == 0: gc.collect()

        # 3. Finalisation (logique de normalisation reste la même)
        print(f"   -> Boucle Drizzle terminée. {processed_count}/{len(temp_filepath_list)} fichiers traités.")
        if processed_count == 0:
            print("ERREUR: Aucun fichier n'a pu être traité par Drizzle.")
            return None, None

        final_image_normalized = None
        try:
            print("   -> Normalisation finale 0-1...")
            min_val = np.nanmin(final_sci)
            max_val = np.nanmax(final_sci)
            if max_val > min_val:
                final_image_normalized = (final_sci - min_val) / (max_val - min_val)
                final_image_normalized = np.clip(final_image_normalized, 0.0, 1.0)
                final_image_normalized = np.nan_to_num(final_image_normalized, nan=0.0)
            else:
                print("   - WARNING: Image finale constante ou vide après Drizzle.")
                final_image_normalized = np.zeros_like(final_sci)
        except Exception as norm_err:
             print(f"   - ERREUR pendant normalisation finale: {norm_err}")
             return None, None

        end_time = time.time()
        print(f"✅ DrizzleProcessor (tdriz): Terminé en {end_time - start_time:.2f}s.")

        # Retourner l'image normalisée (HxWx3 float32) et la carte de poids (HxWx3 float32)
        return final_image_normalized.astype(np.float32), final_wht.astype(np.float32)

# --- FIN DE LA MÉTHODE apply_drizzle MODIFIÉE ---




# Note: Le bloc if __name__ == "__main__": de l'ancien fichier est supprimé
# car ce fichier est maintenant destiné à être importé comme un module.