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
# --- En haut de drizzle_integration.py ---
# (Après les autres imports)
try:
    from drizzle.resample import Drizzle # Importer la CLASSE Drizzle
    _OO_DRIZZLE_AVAILABLE = True
    print("DEBUG DrizzleIntegration: Imported drizzle.resample.Drizzle")
except ImportError:
    print("ERROR DrizzleIntegration: Cannot import drizzle.resample.Drizzle class")
    _OO_DRIZZLE_AVAILABLE = False
    Drizzle = None # Définir comme None si indisponible
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





###################################################################################################################################



    def apply_drizzle_mosaic(self, temp_filepath_list, output_wcs, output_shape_2d_hw):
        """
        Assemble une mosaïque en utilisant la classe Drizzle à partir de fichiers temporaires.
        MAJ: Passe explicitement output_wcs à Drizzle.

        Args:
            temp_filepath_list (list): Liste des chemins vers les fichiers FITS temporaires
                                      (HxWx3 float32 attendu, avec WCS valide).
            output_wcs (astropy.wcs.WCS): WCS final pour l'image combinée.
            output_shape_2d_hw (tuple): Shape (H, W) finale pour l'image combinée.

        Returns:
            tuple: (final_sci_image_hxwxc, final_wht_map_hxwxc) ou (None, None) si échec.
                   Les tableaux retournés sont en float32.
        """
        start_time = time.time()
        if not _OO_DRIZZLE_AVAILABLE or Drizzle is None:
            print("ERREUR (apply_drizzle_mosaic): Classe Drizzle non disponible.")
            return None, None
        if not temp_filepath_list:
            print("WARNING (apply_drizzle_mosaic): Liste de fichiers vide fournie.")
            return None, None
        if output_wcs is None or output_shape_2d_hw is None:
            print("ERREUR (apply_drizzle_mosaic): WCS ou Shape de sortie manquant.")
            return None, None

        num_files = len(temp_filepath_list)
        print(f"DrizzleProcessor (Mosaic): Démarrage assemblage sur {num_files} fichiers...")
        # --- Ajout Debug pour vérifier la shape de sortie ATTENDUE ---
        print(f"   -> Grille Sortie CIBLE: Shape={output_shape_2d_hw} (H,W), WCS fourni: {'Oui' if output_wcs else 'Non'}")
        if output_wcs:
            print(f"      - WCS Cible CRPIX: {output_wcs.wcs.crpix}, PixelShape: {output_wcs.pixel_shape}")
        # --- Fin Ajout Debug ---

        # --- Initialiser les objets Drizzle finaux et tableaux de sortie ---
        num_output_channels = 3
        final_drizzlers = []
        final_output_sci_list = []
        final_output_wht_list = []
        initialized = False

        try:
            print(f"   -> Initialisation Drizzle pour {num_output_channels} canaux...")
            for _ in range(num_output_channels):
                out_img_ch = np.zeros(output_shape_2d_hw, dtype=np.float32)
                out_wht_ch = np.zeros(output_shape_2d_hw, dtype=np.float32)
                final_output_sci_list.append(out_img_ch)
                final_output_wht_list.append(out_wht_ch)

                # --- MODIFICATION ICI : Ajouter out_wcs ---
                driz_ch = Drizzle(
                    out_img=out_img_ch,
                    out_wht=out_wht_ch,
                    out_shape=output_shape_2d_hw, # Garder shape pour clarté
                    out_wcs=output_wcs,           ### AJOUT out_wcs ###
                    kernel=self.kernel,
                    fillval="0.0"
                )
                # --- FIN MODIFICATION ---

                final_drizzlers.append(driz_ch)
            initialized = True
            print("   -> Initialisation Drizzle terminée (avec WCS de sortie).")
        except Exception as init_err:
            print(f"   -> ERREUR initialisation Drizzle pour Mosaïque: {init_err}")
            traceback.print_exc(limit=1)
            return None, None

        if not initialized: return None, None # Sécurité

        # --- Boucle Drizzle sur les fichiers temporaires (INCHANGÉ) ---
        print(f"   -> Démarrage boucle Drizzle sur {len(temp_filepath_list)} fichiers...")
        processed_count = 0
        for i, filepath in enumerate(temp_filepath_list):
            if (i + 1) % 10 == 0 or i == 0 or i == len(temp_filepath_list) - 1: print(f"      Adding Mosaïque Input {i+1}/{len(temp_filepath_list)}") # Renommé pour clarté
            img_data_hxwxc, wcs_in, header_in = _load_drizzle_temp_file(filepath)
            if img_data_hxwxc is None or wcs_in is None: print(f"      - Skip Mosaïque Input {i+1} (échec chargement/WCS)"); del img_data_hxwxc, wcs_in, header_in; gc.collect(); continue
            current_input_shape_2d = img_data_hxwxc.shape[:2]
            # On ne compare plus à ref_shape_2d car les images peuvent avoir des tailles légèrement différentes

            pixmap = None
            try:
                y_in, x_in = np.indices(current_input_shape_2d)
                world_coords = wcs_in.all_pix2world(x_in.flatten(), y_in.flatten(), 0)
                x_out, y_out = output_wcs.all_world2pix(world_coords[0], world_coords[1], 0)
                pixmap = np.dstack((x_out.reshape(current_input_shape_2d), y_out.reshape(current_input_shape_2d))).astype(np.float32)
            except Exception as map_err: print(f"      - ERREUR calcul pixmap mosaïque pour input {i+1}: {map_err}"); del img_data_hxwxc, wcs_in, header_in; gc.collect(); continue

            if pixmap is not None:
                try:
                    exptime = 1.0
                    if header_in and 'EXPTIME' in header_in:
                        try: exptime = max(1e-6, float(header_in['EXPTIME']))
                        except (ValueError, TypeError): pass
                    for c in range(num_output_channels):
                        channel_data_2d = img_data_hxwxc[:, :, c].astype(np.float32) # Assurer float32 ici
                        finite_mask = np.isfinite(channel_data_2d)
                        if not np.all(finite_mask): channel_data_2d[~finite_mask] = 0.0
                        final_drizzlers[c].add_image(data=channel_data_2d, pixmap=pixmap, exptime=exptime, in_units='counts', pixfrac=self.pixfrac)
                    processed_count += 1
                except Exception as e_add: print(f"      - ERREUR add_image mosaïque input {i+1}: {e_add}"); traceback.print_exc(limit=1)
                finally: del img_data_hxwxc, wcs_in, header_in, pixmap; gc.collect()
            else: del img_data_hxwxc, wcs_in, header_in; gc.collect()
        # --- Fin Boucle Drizzle ---

        print(f"   -> Boucle assemblage Mosaïque terminée. {processed_count}/{num_files} fichiers ajoutés.")
        if processed_count == 0:
            print("ERREUR (apply_drizzle_mosaic): Aucun fichier traité avec succès.")
            del final_drizzlers, final_output_sci_list, final_output_wht_list; gc.collect()
            return None, None

        # --- Assemblage et Retour (INCHANGÉ) ---
        try:
            print("   -> Assemblage final des canaux (Mosaïque)...")
            final_sci_hxwxc = np.stack(final_output_sci_list, axis=-1)
            final_wht_hxwxc = np.stack(final_output_wht_list, axis=-1)
            final_sci_hxwxc[~np.isfinite(final_sci_hxwxc)] = 0.0
            final_wht_hxwxc[~np.isfinite(final_wht_hxwxc)] = 0.0
            final_wht_hxwxc[final_wht_hxwxc < 0] = 0.0
            print(f"   -> Combinaison terminée. Shape finale SCI: {final_sci_hxwxc.shape}, WHT: {final_wht_hxwxc.shape}")
        except Exception as e_final:
            print(f"   -> ERREUR assemblage final Mosaïque: {e_final}")
            del final_drizzlers, final_output_sci_list, final_output_wht_list; gc.collect()
            return None, None

        end_time = time.time()
        print(f"✅ DrizzleProcessor (Mosaic): Assemblage terminé en {end_time - start_time:.2f}s.")
        del final_drizzlers, final_output_sci_list, final_output_wht_list; gc.collect()
        # Retourner HxWx3 float32
        return final_sci_hxwxc.astype(np.float32), final_wht_hxwxc.astype(np.float32)




###################################################################################################################################





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
# --- DANS LE FICHIER: seestar/enhancement/drizzle_integration.py ---

    # Ajouter output_wcs et output_shape_2d_hw comme arguments optionnels
    def apply_drizzle(self, temp_filepath_list, output_wcs=None, output_shape_2d_hw=None):
        """
        Applique Drizzle en utilisant la classe drizzle.resample.Drizzle.
        MAJ: Accepte output_wcs et output_shape_2d_hw optionnels pour Mosaïque/Pré-calcul.

        Args:
            temp_filepath_list (list): Liste des chemins vers les fichiers FITS temporaires.
            output_wcs (astropy.wcs.WCS, optional): WCS pré-calculé pour la sortie.
                                                    Si None, calculé depuis la première image.
            output_shape_2d_hw (tuple, optional): Shape (H, W) pré-calculée pour la sortie.
                                                  Si None, calculé depuis la première image.

        Returns:
            tuple: (final_image_normalized, final_wht) ou (None, None) en cas d'erreur.
        """
        start_time = time.time()
        if not _OO_DRIZZLE_AVAILABLE or Drizzle is None: print("ERREUR: Classe Drizzle non disponible."); return None, None
        if not temp_filepath_list: print("WARNING DrizzleProcessor: Liste fichiers vide."); return None, None

        print(f"DrizzleProcessor (resample): Application sur {len(temp_filepath_list)} fichiers...")

        # --- Déterminer la Grille de Sortie ---
        # Utiliser la grille fournie si disponible, sinon la calculer depuis la première image
        final_output_wcs = output_wcs
        final_output_shape_hw = output_shape_2d_hw
        ref_shape_2d = None # Pour la boucle

        if final_output_wcs is None or final_output_shape_hw is None:
            print("   -> Grille sortie non fournie, calcul depuis première image valide...")
            for i, filepath in enumerate(temp_filepath_list):
                img_data, wcs_in, _ = _load_drizzle_temp_file(filepath)
                if img_data is not None and wcs_in is not None:
                    ref_shape_2d = img_data.shape[:2]
                    try:
                        final_output_wcs, final_output_shape_hw = self._create_output_wcs(wcs_in, ref_shape_2d, self.scale_factor)
                        print(f"   -> Grille sortie calculée: Shape={final_output_shape_hw} (H,W)")
                        del img_data, wcs_in # Nettoyer mémoire
                        break # Sortir après avoir trouvé la première valide
                    except Exception as e_init:
                        print(f"      - ERREUR calcul grille depuis image {i+1}: {e_init}. Skip Drizzle.")
                        return None, None
                else: del img_data, wcs_in # Nettoyer mémoire
            if final_output_wcs is None: print("ERREUR: Impossible de déterminer la grille de sortie."); return None, None
        else:
             print(f"   -> Utilisation grille sortie fournie: Shape={final_output_shape_hw} (H,W)")
             # Essayer de déterminer ref_shape_2d pour la boucle
             try: _, wcs_tmp, _ = _load_drizzle_temp_file(temp_filepath_list[0]); ref_shape_2d = wcs_tmp.pixel_shape[::-1] if wcs_tmp and wcs_tmp.pixel_shape else None; del _,wcs_tmp,_
             except: pass
             if ref_shape_2d is None: print("WARN: Impossible de déterminer shape réf pour boucle add_image"); # Continuer quand même ?


        # --- Initialiser Drizzle et Tableaux de Sortie ---
        # (Utilise final_output_wcs et final_output_shape_hw déterminés ci-dessus)
        num_output_channels = 3; final_drizzlers = []; output_images_list = []; output_weights_list = []; initialized = False
        try:
            print(f"   -> Initialisation Drizzle pour {num_output_channels} canaux...")
            for _ in range(num_output_channels):
                out_img_ch = np.zeros(final_output_shape_hw, dtype=np.float32); out_wht_ch = np.zeros(final_output_shape_hw, dtype=np.float32)
                output_images_list.append(out_img_ch); output_weights_list.append(out_wht_ch)
                driz_ch = Drizzle(out_img=out_img_ch, out_wht=out_wht_ch, out_shape=final_output_shape_hw,
                                  #out_wcs=final_output_wcs, # Utilise le WCS final
                                  kernel=self.kernel, fillval="0.0")
                final_drizzlers.append(driz_ch)
            initialized = True; print("   -> Initialisation Drizzle terminée.")
        except Exception as init_err: print(f"   -> ERREUR init Drizzle: {init_err}"); traceback.print_exc(limit=1); return None, None
        if not initialized: return None, None

        # --- Boucle Drizzle (add_image) ---
        print(f"   -> Démarrage boucle Drizzle sur {len(temp_filepath_list)} fichiers...")
        processed_count = 0
        for i, filepath in enumerate(temp_filepath_list):
            # ... (Logique identique pour charger, calculer pixmap, appeler add_image) ...
            # ... (Utilise final_output_wcs pour calculer pixmap) ...
            if (i + 1) % 10 == 0 or i == 0 or i == len(temp_filepath_list) - 1: print(f"      Adding Drizzle Input {i+1}/{len(temp_filepath_list)}")
            img_data_hxwxc, wcs_in, header_in = _load_drizzle_temp_file(filepath) # Renommé img_data_hxwxc
            if img_data_hxwxc is None or wcs_in is None: print(f"      - Skip Input {i+1} (load/WCS)"); del img_data_hxwxc, wcs_in, header_in; gc.collect(); continue

            # Utiliser ref_shape_2d déterminé plus haut ou celui de l'image courante
            current_input_shape_hw = img_data_hxwxc.shape[:2] # H, W
            if ref_shape_2d is None: ref_shape_2d = current_input_shape_hw # Initialiser si besoin

            pixmap = None
            try:
                y_in, x_in = np.indices(current_input_shape_hw)
                world_coords = wcs_in.all_pix2world(x_in.flatten(), y_in.flatten(), 0)
                x_out, y_out = final_output_wcs.all_world2pix(world_coords[0], world_coords[1], 0) # Utilise WCS final
                pixmap = np.dstack((x_out.reshape(current_input_shape_hw), y_out.reshape(current_input_shape_hw))).astype(np.float32)
            except Exception as map_err: print(f"      - ERREUR pixmap: {map_err}"); del img_data_hxwxc, wcs_in, header_in; gc.collect(); continue

            try: # Appel add_image
                exptime = 1.0; # ... (calcul exptime) ...
                if header_in and 'EXPTIME' in header_in:
                    try: exptime = max(1e-6, float(header_in['EXPTIME']))
                    except (ValueError, TypeError): pass
                for c in range(3):
                    channel_data_2d = img_data_hxwxc[:, :, c]; finite_mask = np.isfinite(channel_data_2d); channel_data_2d[~finite_mask] = 0.0
                    final_drizzlers[c].add_image(data=channel_data_2d, pixmap=pixmap, exptime=exptime, in_units='counts', pixfrac=self.pixfrac)
                processed_count += 1
            except Exception as e_add: print(f"   - ERREUR add_image {i+1}: {e_add}"); traceback.print_exc(limit=1)
            finally: del img_data_hxwxc, wcs_in, header_in, pixmap; gc.collect()
        # --- Fin Boucle Drizzle ---

        print(f"   -> Boucle Drizzle terminée. {processed_count}/{len(temp_filepath_list)} fichiers traités.")
        if processed_count == 0: print("ERREUR: Aucun fichier traité."); return None, None

        # --- Assemblage et Retour ---
        try:
            print("   -> Combinaison canaux finaux..."); final_sci = np.stack(output_images_list, axis=-1); final_wht = np.stack(output_weights_list, axis=-1)
            print(f"   -> Combinaison terminée. Shape finale: {final_sci.shape}")
        except Exception as e_final: print(f"   -> ERREUR assemblage final: {e_final}"); return None, None

        # --- Normalisation Finale ---
        # ... (logique normalisation inchangée) ...
        final_image_normalized = None # Initialiser
        try:
            print("   -> Normalisation finale 0-1...")
            min_val = np.nanmin(final_sci); max_val = np.nanmax(final_sci)
            print(f"      - Avant Norm -> Min: {min_val:.4f}, Max: {max_val:.4f}") # Log valeurs
            if max_val > min_val:
                final_image_normalized = (final_sci - min_val) / (max_val - min_val)
                final_image_normalized = np.clip(final_image_normalized, 0.0, 1.0)
                final_image_normalized = np.nan_to_num(final_image_normalized, nan=0.0) # Nettoyer NaN après division
            else:
                print("   - WARNING: Image finale constante/vide après Drizzle.")
                final_image_normalized = np.zeros_like(final_sci) # Mettre à zéro explicitement
            # Assurer float32 en sortie
            final_image_normalized = final_image_normalized.astype(np.float32)
            print(f"      - Après Norm -> Shape: {final_image_normalized.shape}, Type: {final_image_normalized.dtype}, Min: {np.min(final_image_normalized):.2f}, Max: {np.max(final_image_normalized):.2f}") # Log après

        except Exception as norm_err:
            print(f"   - ERREUR normalisation finale: {norm_err}")
            final_image_normalized = None # Mettre à None si erreur
            # Ne pas retourner final_wht si l'image sci a échoué
            final_wht = None # Assurer que wht est aussi None

        end_time = time.time(); print(f"✅ DrizzleProcessor terminé en {end_time - start_time:.2f}s.")
        # Ajouter un log juste avant le return pour voir ce qui est retourné
        print(f"DEBUG [apply_drizzle return]: Returning SCI is None? {final_image_normalized is None}, WHT is None? {final_wht is None}")
        return final_image_normalized, final_wht


# --- FIN DE LA MÉTHODE apply_drizzle (MODIFIÉE) ---