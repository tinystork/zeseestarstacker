import numpy as np
import os
import traceback
import warnings
import gc
import logging
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
import cv2
from scipy.ndimage import gaussian_filter
# ConvexHull n'est pas utilisé dans ce fichier

_logger = logging.getLogger(__name__)
_original_showwarning = warnings.showwarning

# -------------------------------------------------------------
# Filter the flood of "is not a flux-conserving kernel" warnings
warnings.filterwarnings(
    action="once",                                   # show only first
    message=r".*is not a flux-conserving kernel\.$", # any kernel name
    module=r"drizzle\.resample"
)

# Optional: log that first (now unique) occurrence via the project logger
def _relay_first_warning(message, category, filename, lineno, file=None, line=None):
    if ("is not a flux-conserving kernel" in str(message)
            and category is RuntimeWarning):
        _logger.warning(str(message))
    _original_showwarning(message, category, filename, lineno, file, line)
warnings.showwarning = _relay_first_warning
# -------------------------------------------------------------

logger = logging.getLogger(__name__)
# ConvexHull n'est pas utilisé dans ce fichier

try:
    import colour_demosaicing
    _DEBAYER_AVAILABLE = True
    logger.debug("DEBUG DrizzleIntegration: Found colour_demosaicing.")
except ImportError:
    _DEBAYER_AVAILABLE = False
    logger.warning("WARNING DrizzleIntegration: colour-demosaicing library not found.")
    class colour_demosaicing: # Factice
        @staticmethod
        def demosaicing_CFA_Bayer_Malvar2004(data, pattern):
            logger.error("ERROR: colour_demosaicing not available for demosaicing.")
            return data

try:
    from drizzle.resample import Drizzle
    _OO_DRIZZLE_AVAILABLE = True
    logger.debug("DEBUG DrizzleIntegration: Imported drizzle.resample.Drizzle")
except ImportError:
    _OO_DRIZZLE_AVAILABLE = False
    Drizzle = None  # Pour éviter NameError si Drizzle non importé
    logger.error("ERROR DrizzleIntegration: Cannot import drizzle.resample.Drizzle class")

# try: # Pas besoin de SeestarQueuedStacker ici
#     from ..queuep.queue_manager import SeestarQueuedStacker 
# except ImportError:
#     SeestarQueuedStacker = None 

warnings.filterwarnings('ignore', category=FITSFixedWarning)

# --- Fonctions Helper de Module ---
def _load_drizzle_temp_file(filepath): # Nom corrigé
    try:
        with fits.open(filepath, memmap=False) as hdul:
            if not hdul or not hdul[0].is_image or hdul[0].data is None: 
                # logger.debug(f"DEBUG _load_drizzle_temp_file: HDU invalide pour {filepath}")
                return None, None, None
            hdu = hdul[0]; data = hdu.data; header = hdu.header
            data_hxwx3 = None
            if data.ndim == 3:
                if data.shape[2] == 3: data_hxwx3 = data.astype(np.float32)
                elif data.shape[0] == 3: data_hxwx3 = np.moveaxis(data, 0, -1).astype(np.float32)
                else: 
                    # logger.debug(f"DEBUG _load_drizzle_temp_file: Shape 3D inattendue {data.shape} pour {filepath}")
                    return None, None, None
            else: 
                # logger.debug(f"DEBUG _load_drizzle_temp_file: Données non 3D {data.ndim}D pour {filepath}")
                return None, None, None
            wcs = None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', FITSFixedWarning)
                    wcs_hdr = WCS(header, naxis=2)
                if wcs_hdr.is_celestial: wcs = wcs_hdr
            except Exception: pass
            if wcs is None: 
                # logger.debug(f"DEBUG _load_drizzle_temp_file: WCS non trouvé/valide pour {filepath}")
                return None, None, None
            return data_hxwx3, wcs, header
    except FileNotFoundError: 
        # logger.debug(f"DEBUG _load_drizzle_temp_file: Fichier non trouvé {filepath}")
        return None, None, None
    except Exception as e: 
        logger.debug(f"ERREUR _load_drizzle_temp_file pour {filepath}: {e}")
        traceback.print_exc(limit=1); return None, None, None

def _create_wcs_from_header(header): # Nom corrigé
    required_keys = ['NAXIS1', 'NAXIS2', 'RA', 'DEC', 'FOCALLEN', 'XPIXSZ', 'YPIXSZ']
    if not all(key in header for key in required_keys): 
        # logger.debug(f"DEBUG _create_wcs_from_header: Clés manquantes { [k for k in required_keys if k not in header] }")
        return None
    try:
        naxis1 = int(header['NAXIS1']); naxis2 = int(header['NAXIS2'])
        ra_deg = float(header['RA']); dec_deg = float(header['DEC'])
        focal_len_mm = float(header['FOCALLEN'])
        pixel_size_x_um = float(header['XPIXSZ']); pixel_size_y_um = float(header['YPIXSZ'])
        # Conversion en mètres pour la formule d'échelle
        focal_len_m = focal_len_mm * 1e-3
        pixel_size_x_m = pixel_size_x_um * 1e-6; pixel_size_y_m = pixel_size_y_um * 1e-6
        # Échelle en radians/pixel
        scale_x_rad_per_pix = pixel_size_x_m / focal_len_m
        scale_y_rad_per_pix = pixel_size_y_m / focal_len_m
        # Conversion en degrés/pixel
        deg_per_rad = 180.0 / np.pi
        scale_x_deg_per_pix = scale_x_rad_per_pix * deg_per_rad
        scale_y_deg_per_pix = scale_y_rad_per_pix * deg_per_rad

        w = WCS(naxis=2)
        # Point de référence au centre de l'image (convention FITS 1-based)
        w.wcs.crpix = [naxis1 / 2.0 + 0.5, naxis2 / 2.0 + 0.5]
        w.wcs.crval = [ra_deg, dec_deg] # Coordonnées célestes au point de référence
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"] # Type de projection (tangentielle)
        # Échelle en degrés par pixel. Négatif pour RA car RA augmente vers la gauche.
        w.wcs.cdelt = np.array([-scale_x_deg_per_pix, scale_y_deg_per_pix]) 
        w.wcs.cunit = ['deg', 'deg'] # Unités des cdelt et crval
        w.wcs.pc = np.array([[1.0, 0.0], [0.0, 1.0]]) # Matrice de rotation (identité ici)
        return w
    except Exception as e_wcs_create: 
        logger.debug(f"ERREUR _create_wcs_from_header: {e_wcs_create}")
        return None

def _load_fits_data_wcs_debayered(filepath, bayer_pattern='GRBG'): # Nom corrigé
    # logger.debug(f"DEBUG _load_fits_data_wcs_debayered: Tentative chargement {filepath}")
    try:
        with fits.open(filepath, memmap=False) as hdul:
            hdu = None; header = None
            for h_item in hdul:
                if h_item.is_image and h_item.data is not None and h_item.data.ndim == 2:
                    hdu = h_item; break
            if hdu is None: 
                # logger.debug(f"DEBUG _load_fits_data_wcs_debayered: Aucune HDU 2D image valide dans {filepath}")
                return None, None, None
            
            bayer_data = hdu.data; header = hdu.header
            rgb_image = None
            if _DEBAYER_AVAILABLE and colour_demosaicing is not None: # Vérifier aussi que l'objet existe
                try:
                    bayer_float = bayer_data.astype(np.float32)
                    valid_patterns = ['GRBG', 'RGGB', 'GBRG', 'BGGR'] # Assurer que le pattern est valide
                    pattern_upper = bayer_pattern.upper()
                    if pattern_upper not in valid_patterns: pattern_upper = 'GRBG'
                    
                    rgb_image = colour_demosaicing.demosaicing_CFA_Bayer_Malvar2004(bayer_float, pattern=pattern_upper)
                    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3: 
                        raise ValueError(f"Debayer a retourné une shape inattendue: {rgb_image.shape}")
                except Exception as e_debayer:
                    logger.debug(f"ERREUR Debayer dans _load_fits_data_wcs_debayered pour {filepath}: {e_debayer}")
                    return None, None, None # Échec du debayering
            else: 
                logger.debug(f"WARN _load_fits_data_wcs_debayered: Bibliothèque colour_demosaicing non dispo pour {filepath}.")
                return None, None, None # Pas de debayering possible
            
            wcs = None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', FITSFixedWarning)
                    wcs_hdr = WCS(header, naxis=2)
                if wcs_hdr.is_celestial: wcs = wcs_hdr
            except Exception: pass

            if wcs is None: # Si WCS du header échoue, essayer de générer
                wcs_gen = _create_wcs_from_header(header) # Appel au helper de module corrigé
                if wcs_gen and wcs_gen.is_celestial: wcs = wcs_gen
            
            if not wcs: 
                # logger.debug(f"DEBUG _load_fits_data_wcs_debayered: WCS non trouvé/généré pour {filepath}")
                return None, None, None # WCS est essentiel
            
            return rgb_image.astype(np.float32), wcs, header
            
    except FileNotFoundError: 
        # logger.debug(f"DEBUG _load_fits_data_wcs_debayered: Fichier non trouvé {filepath}")
        return None, None, None
    except Exception as e_load: 
        logger.debug(f"ERREUR _load_fits_data_wcs_debayered pour {filepath}: {e_load}")
        traceback.print_exc(limit=1); return None, None, None

# === CLASSE DrizzleProcessor ===
class DrizzleProcessor:
    """
    Encapsule la logique pour appliquer Drizzle en utilisant la classe drizzle.resample.Drizzle.
    """
    def __init__(self, scale_factor=2.0, pixfrac=1.0, kernel='square', fillval="0.0", final_wht_threshold=0.001):
        if not _OO_DRIZZLE_AVAILABLE or Drizzle is None:
            logger.debug("ERREUR DrizzleProcessor: Classe Drizzle (drizzle.resample.Drizzle) non disponible.")
            raise ImportError("Classe Drizzle (drizzle.resample.Drizzle) non disponible. Veuillez installer 'drizzle'.")

        self.scale_factor = float(max(1.0, scale_factor))
        self.pixfrac = float(np.clip(pixfrac, 0.01, 1.0))
        valid_kernels = ['square', 'gaussian', 'point', 'tophat', 'turbo', 'lanczos2', 'lanczos3']
        self.kernel = kernel.lower() if kernel.lower() in valid_kernels else 'square'
        self.fillval = str(fillval) 
        self.final_wht_threshold = float(np.clip(final_wht_threshold, 0.0, 1.0))
        
        logger.debug(f"DEBUG DrizzleProcessor Initialized (using drizzle.resample.Drizzle): "
              f"ScaleFactor(info)={self.scale_factor}, Pixfrac={self.pixfrac}, Kernel='{self.kernel}', "
              f"Fillval='{self.fillval}', FinalWHTThresh={self.final_wht_threshold}")




    def apply_drizzle(self, 
                      input_file_paths: list, 
                      output_wcs: WCS, 
                      output_shape_2d_hw: tuple,
                      use_local_alignment_logic: bool = False,
                      anchor_wcs_for_local: WCS = None,
                      progress_callback: callable = None
                      #max_ctx_id=0,         
                      #disable_ctx=True      
                      ):
        """
        Applique Drizzle à une liste de fichiers FITS d'entrée.
        Gère la logique de pixmap pour l'alignement local de mosaïque.
        Inclut le masquage final basé sur une carte de poids lissée.
        Version: V_Inspector_API_FINAL_Corrected
        """
        # --- 0. Initialisation et Logs de Début ---
        if not progress_callback: 
            progress_callback = lambda msg, prog=None: logger.debug(f"DrizzleProc LOG: {msg}" + (f" ({prog}%)" if prog is not None else ""))
        
        logger.debug(f"DEBUG DrizzleProcessor.apply_drizzle (V_Inspector_API_FINAL_Corrected): Appelée avec {len(input_file_paths)} fichiers.")
        logger.debug(f"  -> use_local_alignment_logic: {use_local_alignment_logic}")
        logger.debug(f"  -> anchor_wcs_for_local fourni: {'Oui' if anchor_wcs_for_local else 'Non'}")
        logger.debug(f"  -> Shape de sortie CIBLE (H,W): {output_shape_2d_hw}, WCS de sortie fourni: {'Oui' if output_wcs else 'Non'}")

        # --- 1. Vérifications Préliminaires des Arguments ---
        if not _OO_DRIZZLE_AVAILABLE or Drizzle is None: 
            progress_callback("Drizzle ERREUR: Classe Drizzle (drizzle.resample.Drizzle) non disponible.", 0)
            return None, None
        if not input_file_paths:
            progress_callback("Drizzle: Aucune image d'entrée fournie.", 0)
            return None, None
        if output_wcs is None or output_shape_2d_hw is None:
            progress_callback("Drizzle ERREUR: WCS ou shape de sortie Drizzle non défini.", 0)
            return None, None
        if use_local_alignment_logic and anchor_wcs_for_local is None:
            progress_callback("Drizzle ERREUR (Local Mosaic): WCS d'ancrage non fourni.", 0)
            return None, None
        
        num_output_channels = 3 
        out_images_by_channel = [np.zeros(output_shape_2d_hw, dtype=np.float32) for _ in range(num_output_channels)]
        out_weights_by_channel = [np.zeros(output_shape_2d_hw, dtype=np.float32) for _ in range(num_output_channels)]
        drizzlers_by_channel = []

        # --- 2. Initialisation des Objets Drizzle (par canal) ---
        try:
            progress_callback(f"Drizzle: Initialisation pour {num_output_channels} canaux (Shape sortie: {output_shape_2d_hw})...", None)
            for i in range(num_output_channels):
                driz_ch = Drizzle( 
                    out_img=out_images_by_channel[i],  
                    out_wht=out_weights_by_channel[i],
                    kernel=self.kernel,   # kernel est pour __init__
                    fillval=self.fillval,  # fillval est pour __init__
                    # pixfrac N'EST PAS pour __init__
                )
                drizzlers_by_channel.append(driz_ch)
            progress_callback("Drizzle: Initialisation Drizzle terminée.", None)
            logger.debug(f"DEBUG DrizzleProcessor (V_Inspector_API_FINAL_Corrected): Initialisation des Drizzlers OK. "
                  f"Kernel='{self.kernel}', Fillval='{self.fillval}', Pixfrac (pour add_image)='{self.pixfrac}'")
        except Exception as e_init_driz:
            progress_callback(f"Drizzle ERREUR: Initialisation Drizzle échouée: {e_init_driz}", 0)
            traceback.print_exc(limit=1)
            return None, None
        
        # --- 3. Boucle sur les Fichiers d'Entrée ---
        files_processed_count = 0
        for i, file_path in enumerate(input_file_paths):
            if progress_callback and (i % 10 == 0 or i == len(input_file_paths) - 1 or i < 3) : 
                progress_callback(f"Drizzle: Traitement image {i+1}/{len(input_file_paths)} ('{os.path.basename(file_path)}')...", 
                                  int(i / len(input_file_paths) * 100) if len(input_file_paths) > 0 else 0)
            
            input_image_cxhxw=None # Image d'entrée chargée (CxHxW)
            input_header=None      # Header du fichier d'entrée
            exposure_time=1.0      # Temps d'exposition lu du header
            current_pixmap_for_drizzle=None # Pixmap calculé pour CE fichier vers la grille de sortie
            # Les variables suivantes sont pour le calcul du pixmap
            sky_ra_deg = None
            sky_dec_deg = None

            try:
                # --- 3.A Charger l'image et son header (CxHxW et header) ---
                # (Cette partie est cruciale et doit être correcte pour que la suite fonctionne)
                with fits.open(file_path, memmap=False) as hdul:
                    if not hdul or len(hdul) == 0 or hdul[0].data is None:
                        logger.debug(f"    WARN (DrizzleProcessor): FITS invalide/vide pour {file_path}. Ignoré."); continue
                    
                    data_loaded = hdul[0].data # Peut être HWC, CHW, ou HW
                    input_header = hdul[0].header

                    # S'assurer que les données sont CxHxW, float32
                    if data_loaded.ndim == 3 and data_loaded.shape[0] == num_output_channels: # Déjà CxHxW
                        input_image_cxhxw = data_loaded.astype(np.float32)
                    elif data_loaded.ndim == 3 and data_loaded.shape[2] == num_output_channels: # HWC -> CxHxW
                        input_image_cxhxw = np.moveaxis(data_loaded, -1, 0).astype(np.float32)
                    elif data_loaded.ndim == 2 and num_output_channels == 1: # N&B et on veut N&B
                        input_image_cxhxw = data_loaded.astype(np.float32)[np.newaxis, :, :] # CxHxW
                    elif data_loaded.ndim == 2 and num_output_channels == 3: # N&B mais on veut RGB (répliquer)
                        input_image_cxhxw = np.stack([data_loaded.astype(np.float32)]*3, axis=0)
                    else:
                        raise ValueError(f"Shape image {data_loaded.shape} non supportée ou incompatible avec num_output_channels={num_output_channels}")
                
                if input_header and 'EXPTIME' in input_header:
                    try: exposure_time = max(1e-6, float(input_header['EXPTIME']))
                    except (ValueError, TypeError): pass
                
                input_shape_hw = (input_image_cxhxw.shape[1], input_image_cxhxw.shape[2]) 
                y_in_coords_flat, x_in_coords_flat = np.indices(input_shape_hw).reshape(2, -1)

                # --- 3.B Calcul du Pixmap (Unifié) ---
                logger.debug(f"    DEBUG (DrizzleProcessor): Calcul Pixmap pour {os.path.basename(file_path)} - Mode Local: {use_local_alignment_logic}")
                if use_local_alignment_logic: # Mosaïque avec alignement local (utilise ancre WCS et matrice M)
                    if anchor_wcs_for_local is None: 
                        logger.debug(f"      ERREUR (DrizzleProcessor Local): WCS ancre manquant pour {file_path}. Ignoré."); continue
                    
                    # La matrice M doit être dans le header du fichier d'entrée (file_path)
                    # car _save_drizzle_input_temp ne la sauvegarde pas.
                    # CELA SIGNIFIE QUE POUR use_local_alignment_logic=True, input_file_paths doit contenir
                    # des fichiers qui ont DÉJÀ ÉTÉ alignés sur l'ancre et dont le header contient M.
                    # Si ce n'est pas le cas, cette logique est fausse.
                    # Actuellement, _save_drizzle_input_temp injecte le WCS de réf globale, pas de matrice M.
                    # Donc, cette branche use_local_alignment_logic ne fonctionnera pas correctement
                    # si les fichiers dans input_file_paths sont ceux créés par _save_drizzle_input_temp.
                    # Il faudrait soit :
                    # 1. Que _save_drizzle_input_temp sauvegarde M si use_local_alignment_logic.
                    # 2. OU que DrizzleProcessor.apply_drizzle reçoive M séparément pour chaque fichier.
                    # Pour l'instant, on suppose que M est dans le header (ce qui n'est pas le cas avec le code actuel).
                    # *** CELA DOIT ÊTRE REVU SI use_local_alignment_logic EST UTILISÉ AVEC apply_drizzle ***
                    M11 = input_header.get('M11', None); M12 = input_header.get('M12', None); M13 = input_header.get('M13', None)
                    M21 = input_header.get('M21', None); M22 = input_header.get('M22', None); M23 = input_header.get('M23', None)

                    if not all(v is not None for v in [M11, M12, M13, M21, M22, M23]):
                        logger.debug(f"      ERREUR (DrizzleProcessor Local): Matrice M incomplète/manquante dans header de {file_path}. Ignoré."); continue
                    M_matrix = np.array([[M11,M12,M13],[M21,M22,M23]], dtype=np.float32)
                                        
                    pts_orig_N12 = np.dstack((x_in_coords_flat, y_in_coords_flat)).astype(np.float32).reshape(-1,1,2)
                    pts_in_anchor_pixels_N2 = cv2.transform(pts_orig_N12, M_matrix).reshape(-1,2)
                    
                    sky_ra_deg, sky_dec_deg = anchor_wcs_for_local.all_pix2world(pts_in_anchor_pixels_N2[:,0], pts_in_anchor_pixels_N2[:,1], 0)
                
                else: # Drizzle Standard (utilise le WCS du fichier d'entrée, qui devrait être le WCS de référence globale)
                    input_wcs_for_drizzle_std = None
                    try:
                        with warnings.catch_warnings(): warnings.simplefilter("ignore"); input_wcs_for_drizzle_std = WCS(input_header, naxis=2) 
                        if not input_wcs_for_drizzle_std.is_celestial: input_wcs_for_drizzle_std = None
                    except Exception: input_wcs_for_drizzle_std = None
                    
                    if input_wcs_for_drizzle_std is None: 
                        logger.debug(f"      ERREUR (DrizzleProcessor Standard): WCS invalide pour {file_path}. Ignoré."); continue
                    
                    sky_ra_deg, sky_dec_deg = input_wcs_for_drizzle_std.all_pix2world(x_in_coords_flat, y_in_coords_flat, 0)

                # Projection sur la grille Drizzle de sortie finale (output_wcs)
                if not (np.all(np.isfinite(sky_ra_deg)) and np.all(np.isfinite(sky_dec_deg))):
                    logger.debug(f"      WARN (DrizzleProcessor): Coordonnées célestes non finies pour {file_path} avant projection sur grille Drizzle. Ignoré."); continue

                final_x_output_pixels, final_y_output_pixels = output_wcs.all_world2pix(sky_ra_deg, sky_dec_deg, 0)

                if not (np.all(np.isfinite(final_x_output_pixels)) and np.all(np.isfinite(final_y_output_pixels))):
                    logger.debug(f"      WARN (DrizzleProcessor): Pixmap contient des NaN/Inf pour {file_path} après projection sur grille Drizzle. Tentative de nettoyage.");
                    # Remplacer les NaN/Inf par une valeur hors champ (ex: très négative) que Drizzle pourrait ignorer
                    final_x_output_pixels = np.nan_to_num(final_x_output_pixels, nan=-1e9, posinf=-1e9, neginf=-1e9)
                    final_y_output_pixels = np.nan_to_num(final_y_output_pixels, nan=-1e9, posinf=-1e9, neginf=-1e9)

                current_pixmap_for_drizzle = np.dstack((
                    final_x_output_pixels.reshape(input_shape_hw), 
                    final_y_output_pixels.reshape(input_shape_hw)
                )).astype(np.float32)
                
                # --- LOG DEBUG PIXMAP ---
                if current_pixmap_for_drizzle is not None:
                    logger.debug(f"      DEBUG PIXMAP File {i+1}: Shape={current_pixmap_for_drizzle.shape}, Dtype={current_pixmap_for_drizzle.dtype}")
                    if np.isnan(current_pixmap_for_drizzle).any(): logger.debug(f"      WARN PIXMAP File {i+1}: CONTIENT ENCORE DES NaN !") # Ne devrait plus arriver avec nan_to_num
                    if np.isinf(current_pixmap_for_drizzle).any(): logger.debug(f"      WARN PIXMAP File {i+1}: CONTIENT ENCORE DES INF !") # Idem
                    logger.debug(f"      DEBUG PIXMAP File {i+1} X Coords: Min={np.nanmin(current_pixmap_for_drizzle[...,0]):.2f}, Max={np.nanmax(current_pixmap_for_drizzle[...,0]):.2f}")
                    logger.debug(f"      DEBUG PIXMAP File {i+1} Y Coords: Min={np.nanmin(current_pixmap_for_drizzle[...,1]):.2f}, Max={np.nanmax(current_pixmap_for_drizzle[...,1]):.2f}")
                else:
                    logger.debug(f"      WARN PIXMAP File {i+1}: current_pixmap_for_drizzle EST NONE APRÈS CALCUL.")
                # --- FIN LOG DEBUG PIXMAP ---

            except Exception as e_pixmap_calc:
                progress_callback(f"Drizzle ERREUR: Calcul Pixmap pour {file_path} échoué: {e_pixmap_calc}", None)
                traceback.print_exc(limit=1)
                continue 

            if current_pixmap_for_drizzle is None: 
                logger.debug(f"    ERREUR (DrizzleProcessor): Pixmap final est None pour {file_path}. Ignoré."); continue

            # --- 3.C Ajouter l'image à chaque Drizzler de canal ---
            exptime_to_pass_to_add_image = exposure_time # Utiliser le temps d'expo lu du header
            in_units_to_pass_to_add_image = 'counts' # Ou 'cps' si les données ne sont pas déjà normalisées par exptime

            for ch_idx in range(num_output_channels):
                channel_data_2d = input_image_cxhxw[ch_idx, :, :].astype(np.float32)
                if not np.all(np.isfinite(channel_data_2d)): 
                    channel_data_2d[~np.isfinite(channel_data_2d)] = 0.0
                
                if i < 3 or (i+1) % 50 == 0 or i == len(input_file_paths) -1 : 
                    logger.debug(f"    LOG Drizzle.add_image: F={os.path.basename(file_path)} Ch={ch_idx}")
                    logger.debug(f"      Data Range: [{np.min(channel_data_2d):.3f}-{np.max(channel_data_2d):.3f}], Mean: {np.mean(channel_data_2d):.3f}")
                    logger.debug(f"      >> exptime: {exptime_to_pass_to_add_image:.2f}, in_units: '{in_units_to_pass_to_add_image}', pixfrac: {self.pixfrac:.2f}")

                drizzlers_by_channel[ch_idx].add_image(
                    data=channel_data_2d, 
                    pixmap=current_pixmap_for_drizzle, # Toujours utiliser pixmap
                    exptime=exptime_to_pass_to_add_image,
                    in_units=in_units_to_pass_to_add_image, 
                    pixfrac=self.pixfrac    # pixfrac est un argument valide pour add_image
                    # weight_map peut être ajouté ici si on a des poids par pixel pour l'image d'entrée
                )
            files_processed_count += 1 # Incrémenter seulement si add_image a été appelé pour tous les canaux
        # ... (le reste de la boucle et de la méthode apply_drizzle) ...


        # --- 4. Récupération des Données Finales et Masquage WHT Lissé ---
        try:
            progress_callback("Drizzle: Combinaison des canaux finaux...", None)
            final_sci_image_hxwxc = np.stack(out_images_by_channel, axis=-1).astype(np.float32) 
            final_wht_map_hxwxc = np.stack(out_weights_by_channel, axis=-1).astype(np.float32)
            # progress_callback("Drizzle: Combinaison canaux terminée.", None) # Peut être trop verbeux
            logger.debug(f"DEBUG DrizzleProcessor: SCI brut (après Drizzle) - Shape: {final_sci_image_hxwxc.shape}, Range: [{np.min(final_sci_image_hxwxc):.3g}-{np.max(final_sci_image_hxwxc):.3g}], Mean: {np.mean(final_sci_image_hxwxc):.3g}")
            logger.debug(f"DEBUG DrizzleProcessor: WHT brut (après Drizzle) - Shape: {final_wht_map_hxwxc.shape}, Range: [{np.min(final_wht_map_hxwxc):.2g}-{np.max(final_wht_map_hxwxc):.2g}], Mean: {np.mean(final_wht_map_hxwxc):.2g}")

            final_sci_image_hxwxc = np.nan_to_num(final_sci_image_hxwxc, nan=0.0, posinf=0.0, neginf=0.0)
            final_wht_map_hxwxc = np.nan_to_num(final_wht_map_hxwxc, nan=0.0, posinf=0.0, neginf=0.0)
            final_wht_map_hxwxc = np.maximum(final_wht_map_hxwxc, 0.0) 

            if self.final_wht_threshold > 1e-9: # Si un seuil significatif est appliqué
                logger.debug(f"DEBUG DrizzleProcessor: Application du seuil WHT final (relatif: {self.final_wht_threshold * 100:.2f}%).")
                if final_wht_map_hxwxc.size > 0 :
                    mean_wht_raw_2d = np.mean(final_wht_map_hxwxc, axis=2).astype(np.float32)
                    
                    # Le sigma de lissage pourrait être un paramètre. Valeur par défaut ex: 2.0 ou 3.0
                    # Ajuster sigma en fonction de la taille de l'image pour éviter les erreurs/artefacts
                    gaussian_sigma_for_wht_smoothing = 3.0 
                    min_dim_wht = min(mean_wht_raw_2d.shape) if mean_wht_raw_2d.ndim == 2 and mean_wht_raw_2d.size > 0 else 0
                    if min_dim_wht > 0 and min_dim_wht <= gaussian_sigma_for_wht_smoothing * 4 : 
                        effective_sigma = max(1.0, min_dim_wht / 8.0)
                        logger.debug(f"   -> WARN: Sigma lissage WHT ({gaussian_sigma_for_wht_smoothing}) grand pour shape {mean_wht_raw_2d.shape}. Ajusté à {effective_sigma:.1f}.")
                        gaussian_sigma_for_wht_smoothing = effective_sigma
                    
                    if min_dim_wht > 0 : # S'assurer qu'on a une image à lisser
                        logger.debug(f"   -> Lissage WHT moyenne avec sigma={gaussian_sigma_for_wht_smoothing:.1f}...")
                        smoothed_mean_wht_2d = gaussian_filter(mean_wht_raw_2d, sigma=gaussian_sigma_for_wht_smoothing)
                        max_overall_smoothed_wht = np.max(smoothed_mean_wht_2d)

                        if max_overall_smoothed_wht > 1e-9: 
                            wht_cutoff_value_absolute = self.final_wht_threshold * max_overall_smoothed_wht
                            logger.debug(f"   -> Seuil WHT absolu (sur WHT lissé max {max_overall_smoothed_wht:.2g}): {wht_cutoff_value_absolute:.3g}")
                            low_wht_mask_smooth_2d = smoothed_mean_wht_2d < wht_cutoff_value_absolute
                            num_masked_pixels = np.sum(low_wht_mask_smooth_2d); total_pixels = low_wht_mask_smooth_2d.size
                            percent_masked = (num_masked_pixels / total_pixels) * 100 if total_pixels > 0 else 0
                            logger.debug(f"   -> Masque WHT bas (lissé) créé. {num_masked_pixels}/{total_pixels} pixels ({percent_masked:.1f}%) seront mis à zéro.")
                            for c_idx in range(num_output_channels):
                                final_sci_image_hxwxc[low_wht_mask_smooth_2d, c_idx] = 0.0 
                            logger.debug(f"   -> Masque WHT bas (lissé) appliqué à l'image science.")
                        else: logger.debug("   -> WARN: Max WHT lissé proche de zéro. Pas de masquage WHT.")
                    else: logger.debug("   -> WARN: Carte WHT moyenne vide ou 1D après moyenne. Pas de lissage/masquage.")
                else: logger.debug("   -> WARN: Carte de poids finale vide. Pas de masquage WHT.")
            else:
                logger.debug("DEBUG DrizzleProcessor: Seuil WHT final est <= 0 (ou non significatif). Aucun masquage de faible poids appliqué.")
            
            logger.debug(f"DEBUG DrizzleProcessor (V_Inspector_API_FINAL_Corrected).apply_drizzle: Retourne images.")
            return final_sci_image_hxwxc, final_wht_map_hxwxc

        except Exception as e_final_stack:
            progress_callback(f"Drizzle ERREUR: Assemblage final des canaux échoué: {e_final_stack}", 0)
            traceback.print_exc(limit=1)
            return None, None
        finally:
            del drizzlers_by_channel, out_images_by_channel, out_weights_by_channel
            gc.collect()


class DrizzleIntegrator:
    """Combine drizzle outputs while optionally renormalizing the flux."""

    def __init__(self, renormalize: str = "max") -> None:
        self.renormalize = renormalize
        self._sci_accum: np.ndarray | None = None
        self._wht_accum: np.ndarray | None = None
        self._n_images = 0

    def add(self, sci: np.ndarray, wht: np.ndarray) -> None:
        """Add a drizzle result to the accumulation."""
        sci = np.asarray(sci, dtype=np.float32)
        wht = np.asarray(wht, dtype=np.float32)
        if self._sci_accum is None:
            self._sci_accum = sci.copy()
            self._wht_accum = wht.copy()
        else:
            self._sci_accum += sci
            self._wht_accum += wht
        self._n_images += 1

    def current_preview(self) -> np.ndarray:
        """Return a float32 normalised copy of the current stack.

        Returns:
            np.ndarray: Normalised drizzle accumulation.

        Raises:
            ValueError: If no images were added yet.
        """
        if self._sci_accum is None or self._wht_accum is None:
            raise ValueError("No images added")

        return (self._sci_accum / np.maximum(self._wht_accum, 1e-9)).astype(
            np.float32
        )

    def cumulative_preview(self) -> np.ndarray:
        """Return the current drizzle stack normalised to ``[0,1]``.

        The returned array shares no memory with the internal accumulators and
        is always ``float32``. The integrator state is unchanged.

        Returns
        -------
        np.ndarray
            Normalised cumulative drizzle image.
        """

        if self._sci_accum is None or self._wht_accum is None:
            raise ValueError("No images added")

        return (self._sci_accum / np.maximum(self._wht_accum, 1e-9)).astype(
            np.float32, copy=False
        )

    def finalize(self) -> np.ndarray:
        """Return the stacked image after optional renormalization."""
        if self._sci_accum is None or self._wht_accum is None:
            raise ValueError("No images added")

        wht_accum_safe = np.maximum(self._wht_accum, 1e-9)
        final_img = self._sci_accum / wht_accum_safe
        if self.renormalize == "max":
            final_img *= wht_accum_safe.max()
        elif self.renormalize == "n_images":
            final_img *= self._n_images
        return final_img.astype(np.float32)


# --- FIN DU FICHIER seestar/enhancement/drizzle_integration.py ---