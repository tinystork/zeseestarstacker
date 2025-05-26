# --- START OF FILE seestar/core/fast_aligner_module.py (Corrected for DAOStarFinder) ---
"""
Module pour l'alignement rapide d'images.
Utilise DAOStarFinder pour la détection d'étoiles et ORB pour les descripteurs.
"""
import cv2
import numpy as np
import traceback 
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats

print("DEBUG [FastAlignerModule]: Module en cours de chargement (V_DAO_Integrated)...")

# =============================================================================
#  FAST ALIGNER – Utilise DAOStarFinder + ORB Descriptors
# =============================================================================
class FastAligner:
    def __init__(self, debug: bool = False):
        self.debug = bool(debug)
        self.progress_callback = None
        if self.debug:
            print("DEBUG [FastAligner]: Instance créée avec debug =", self.debug)

    def set_progress_callback(self, cb):
        self.progress_callback = cb
        if self.debug:
            print("DEBUG [FastAligner]: Progress callback défini.")

    def _log(self, msg: str, level: str = "INFO"):
        tag = f"[FastAligner/{level}] "
        if self.debug:
            print(tag + msg)
        if callable(self.progress_callback):
            try:
                self.progress_callback(tag + msg, None)
            except Exception as e_cb:
                if self.debug:
                    print(tag + f"(Erreur callback ignorée: {e_cb})")

########################################################################################################################################################







# DANS seestar/core/fast_aligner_module.py
# DANS la classe FastAligner

    def _detect_stars_dao_orb(self, 
                              image_u8: np.ndarray, 
                              fwhm: float, 
                              threshold_sigma: float, # Facteur sigma (ex: 4.0, 8.0, 10.0)
                              max_stars_to_describe: int,
                              stats_f32: tuple | None = None): # Format attendu: (median_float32, std_dev_float32)
        """
        Détecte les étoiles avec DAOStarFinder et calcule les descripteurs ORB.
        MODIFIED V3.1:
        - Utilise stats_f32 pour un std_dev de seuil plus robuste.
        - Introduit un std_dev_floor_u8 pour éviter des seuils trop bas.
        """
        if image_u8 is None:
            self._log("DAOStarFinder Helper: Image d'entrée (image_u8) est None.", "WARN")
            return [], None

        self._log(f"DAOStarFinder Helper V3.1: Début détection. FWHM={fwhm:.1f}, ThrSigFactor={threshold_sigma:.1f}, MaxStars={max_stars_to_describe}", "DEBUG")
        
        keypoints_final, descriptors = [], None
        
        try:
            median_for_subtraction_u8: float
            std_dev_for_threshold_u8: float 

            if stats_f32 and len(stats_f32) == 2:
                median_original_f32, std_original_f32 = stats_f32
                self._log(f"  DAO Stats (V3.1): Utilisation stats_f32 fournies. Median_f32={median_original_f32:.4f}, Std_f32={std_original_f32:.6f}", "DEBUG")

                median_for_subtraction_u8 = float(np.median(image_u8)) # Médiane de l'image U8 pour centrer les données pour DAO
                
                std_dev_scaled_from_f32 = std_original_f32 * 255.0
                
                # Plancher pour l'écart-type utilisé pour le seuil (sur l'échelle 0-255)
                # Augmenté à 1.5 pour être un peu plus discriminant que 0.5
                std_dev_floor_u8 = 1.5 
                std_dev_for_threshold_u8 = max(std_dev_scaled_from_f32, std_dev_floor_u8) 

                self._log(f"    Median_u8 pour soustraction={median_for_subtraction_u8:.2f}", "DEBUG")
                self._log(f"    Std_u8 (scaled from f32)={std_dev_scaled_from_f32:.2f}, Plancher_Std_u8={std_dev_floor_u8:.2f} => Std_u8_pour_seuil={std_dev_for_threshold_u8:.2f}", "DEBUG")
            else: 
                if stats_f32 is not None: 
                    self._log(f"  DAO Stats (V3.1): stats_f32 fourni mais invalide ({stats_f32}). Fallback sur calcul direct u8.", "WARN")
                
                _mean_u8_fb, median_u8_fb, std_u8_fb_raw = sigma_clipped_stats(image_u8, sigma=3.0, maxiters=5)
                median_for_subtraction_u8 = median_u8_fb
                std_dev_floor_u8 = 1.5 # Appliquer le même plancher ici aussi
                std_dev_for_threshold_u8 = max(std_u8_fb_raw, std_dev_floor_u8) 
                self._log(f"  DAO Stats (V3.1): Fallback (stats_f32 non fournies/invalides). Median_u8={median_for_subtraction_u8:.2f}, Std_u8_Raw={std_u8_fb_raw:.2f} => Std_u8_pour_seuil={std_dev_for_threshold_u8:.2f}", "DEBUG")

            detection_threshold_for_dao = threshold_sigma * std_dev_for_threshold_u8
            data_for_dao_finding = image_u8.astype(float) - median_for_subtraction_u8
            
            self._log(f"  DAO DetThr pour DAOStarFinder (sur données centrées autour de 0): {detection_threshold_for_dao:.2f} (facteur sigma: {threshold_sigma:.1f})", "DEBUG")

            daofind = DAOStarFinder(fwhm=fwhm, threshold=detection_threshold_for_dao)
            sources_table = daofind(data_for_dao_finding) 

            if sources_table is None or len(sources_table) == 0:
                self._log("DAOStarFinder (V3.1) n'a trouvé aucune source stellaire.", "INFO")
                return [], None
            
            self._log(f"DAOStarFinder (V3.1) a trouvé {len(sources_table)} sources initiales.", "INFO")

            sort_key = 'peak' if 'peak' in sources_table.colnames else 'flux' if 'flux' in sources_table.colnames else None
            if sort_key: 
                sources_table.sort(sort_key, reverse=True)
            else: 
                if 'id' in sources_table.colnames: sources_table.sort('id') 
                self._log("WARN: Impossible de trier les sources DAO par 'peak' ou 'flux'. L'ordre peut être sub-optimal.", "WARN")
            
            sources_to_use = sources_table[:max_stars_to_describe]
            if len(sources_table) > max_stars_to_describe:
                 self._log(f"Limitation à {len(sources_to_use)} étoiles (sur {len(sources_table)}) pour descripteurs.", "DEBUG")

            keypoints_dao = []
            for s_row in sources_to_use:
                diameter_approx = float(s_row.get('fwhm', fwhm)) * 1.5 
                response_strength = float(s_row.get('peak', s_row.get('flux', 0.0)))                
                kp = cv2.KeyPoint(
                    x=float(s_row['xcentroid']), 
                    y=float(s_row['ycentroid']), 
                    size=max(1.0, diameter_approx), 
                    response=response_strength,    
                    angle=-1,                      
                    octave=0                       
                )
                keypoints_dao.append(kp)
            
            if not keypoints_dao:
                self._log("Aucun keypoint cv2 créé depuis DAOStarFinder (V3.1).", "WARN"); return [], None

            orb_n_features_target = max(max_stars_to_describe * 2, 1000) 
            orb_desc_detector = cv2.ORB_create(nfeatures=orb_n_features_target) 
            
            keypoints_final, descriptors = orb_desc_detector.compute(image_u8, keypoints_dao) 
            
            if descriptors is None or len(keypoints_final) == 0:
                self._log("ORB (V3.1) n'a pu calculer aucun descripteur pour les étoiles DAO.", "WARN"); return [], None
            
            self._log(f"ORB (V3.1) a calculé {len(descriptors)} descripteurs pour {len(keypoints_final)} keypoints DAO (demandé {orb_n_features_target} à ORB).", "INFO")
            return keypoints_final, descriptors

        except Exception as e_detect:
            self._log(f"Erreur dans _detect_stars_dao_orb (V3.1): {type(e_detect).__name__} - {e_detect}", "ERROR")
            if self.debug: traceback.print_exc(limit=1)
            return [], None






########################################################################################################################################################


# DANS seestar/core/fast_aligner_module.py
# DANS la classe FastAligner

    # --- MÉTHODE estimate_transform PRINCIPALE (UTILISE MAINTENANT DAOSTARFINDER ET PASSE STATS_F32) ---
    def estimate_transform(self, 
                           ref_gray_f32: np.ndarray,  # MODIFIED: Accepte directement l'image N&B float32 (0-1)
                           img_gray_f32: np.ndarray,  # MODIFIED: Accepte directement l'image N&B float32 (0-1)
                           # Paramètres pour le matching et RANSAC (configurables)
                           min_matches_ratio_config: float = 0.20, 
                           min_absolute_matches_config: int = 10, 
                           ransac_thresh_config: float = 3.0,      
                           min_ransac_inliers_value_config: int = 4, 
                           # Paramètres pour DAOStarFinder (configurables)
                           daofind_fwhm_config: float = 3.5,
                           daofind_threshold_sigma_config: float = 4.0, # C'est le facteur sigma
                           max_stars_to_describe_config: int = 750
                           # orb_features_config n'est plus utilisé ici car ORB_create gère nfeatures
                           ):       
        """
        Estime la transformation affine entre deux images N&B.
        MODIFIED V_DAO_StatsV3:
        - Accepte des images N&B float32 (0-1) en entrée.
        - Calcule sigma_clipped_stats sur ces images float32.
        - Normalise en uint8 pour ORB.
        - Passe les stats float32 à _detect_stars_dao_orb.
        """
        self._log(f"EstimateTransform (V_DAO_StatsV3): Début estimation.")
        self._log(f"  Params DAO cfg: FWHM={daofind_fwhm_config:.1f}, ThrSigFactor={daofind_threshold_sigma_config:.1f}, MaxStars={max_stars_to_describe_config}")
        self._log(f"  Params Match/RANSAC cfg: MinMatchRatio={min_matches_ratio_config:.2f}, MinAbsMatch={min_absolute_matches_config}, RANSACThresh={ransac_thresh_config:.1f}, MinRansacVal={min_ransac_inliers_value_config}")

        if ref_gray_f32 is None or img_gray_f32 is None:
            self._log("EstimateTransform: Image de référence ou source N&B float32 est None.", "ERROR")
            return None
        if ref_gray_f32.ndim != 2 or img_gray_f32.ndim != 2:
            self._log("EstimateTransform: Les images d'entrée doivent être N&B (2D).", "ERROR")
            return None

        # --- 1. Calculer les statistiques sur les images N&B float32 (0-1) ---
        # Ces statistiques seront plus représentatives de la dynamique originale du signal.
        try:
            self._log("  EstimateTransform: Calcul sigma_clipped_stats sur ref_gray_f32...", "DEBUG")
            _mean_ref_f32, median_ref_f32, std_ref_f32 = sigma_clipped_stats(ref_gray_f32, sigma=3.0, maxiters=5)
            ref_stats_tuple_f32 = (median_ref_f32, std_ref_f32)
            self._log(f"    Stats Réf (float32 0-1): Median={median_ref_f32:.4f}, Std={std_ref_f32:.6f}", "DEBUG")

            self._log("  EstimateTransform: Calcul sigma_clipped_stats sur img_gray_f32...", "DEBUG")
            _mean_img_f32, median_img_f32, std_img_f32 = sigma_clipped_stats(img_gray_f32, sigma=3.0, maxiters=5)
            img_stats_tuple_f32 = (median_img_f32, std_img_f32)
            self._log(f"    Stats Src (float32 0-1): Median={median_img_f32:.4f}, Std={std_img_f32:.6f}", "DEBUG")
        except Exception as e_stats:
            self._log(f"EstimateTransform: Erreur lors du calcul de sigma_clipped_stats sur images float32: {e_stats}", "ERROR")
            return None

        # --- 2. Préparer les images uint8 pour ORB et _detect_stars_dao_orb ---
        # _detect_stars_dao_orb attend une image uint8 pour ORB.compute, mais utilisera
        # les stats_f32 pour calculer son seuil de détection DAOStarFinder.
        def _normalize_to_u8(gray_img_f32_local):
            # Normaliser MINMAX pour s'assurer que toute la dynamique est utilisée pour l'image u8
            # sur laquelle ORB va travailler.
            min_v, max_v = np.min(gray_img_f32_local), np.max(gray_img_f32_local) # Utiliser min/max simples ici
            if max_v <= min_v + 1e-7: # Si l'image est plate
                self._log("EstimateTransform: Image (float32) est plate avant normalisation u8.", "WARN")
                return np.full_like(gray_img_f32_local, 128, dtype=cv2.CV_8U) # Retourner gris moyen
            
            # Appliquer cv2.normalize pour la conversion en uint8
            # NORM_MINMAX étire la plage de valeurs de l'image source pour couvrir la plage de sortie (0-255)
            try:
                normalized_u8 = cv2.normalize(gray_img_f32_local, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                return normalized_u8
            except cv2.error as e_norm:
                self._log(f"EstimateTransform: Erreur cv2.normalize: {e_norm}. L'image d'entrée float était peut-être non finie ?", "ERROR")
                return None # Indiquer un échec si la normalisation plante

        ref_gray_u8 = _normalize_to_u8(ref_gray_f32)
        img_gray_u8 = _normalize_to_u8(img_gray_f32)

        if ref_gray_u8 is None: 
            self._log("EstimateTransform: Image de référence inutilisable après normalisation u8.", "ERROR"); return None
        if img_gray_u8 is None: 
            self._log("EstimateTransform: Image source inutilisable après normalisation u8.", "ERROR"); return None
        
        # --- 3. Détection des étoiles et calcul des descripteurs ORB ---
        # On passe maintenant ref_stats_tuple_f32 et img_stats_tuple_f32
        kp1, des1 = self._detect_stars_dao_orb(ref_gray_u8, daofind_fwhm_config, daofind_threshold_sigma_config, max_stars_to_describe_config, stats_f32=ref_stats_tuple_f32)
        kp2, des2 = self._detect_stars_dao_orb(img_gray_u8, daofind_fwhm_config, daofind_threshold_sigma_config, max_stars_to_describe_config, stats_f32=img_stats_tuple_f32)

        if des1 is None or len(kp1) == 0: 
            self._log("EstimateTransform: Pas de descripteurs pour Réf (DAO+ORB V3).", "ERROR"); return None
        if des2 is None or len(kp2) == 0: 
            self._log("EstimateTransform: Pas de descripteurs pour Src (DAO+ORB V3).", "ERROR"); return None
        self._log(f"EstimateTransform: Keypoints/Desc (DAO+ORB V3): {len(kp1)} Réf, {len(kp2)} Src.")

        # --- 4. Matching des descripteurs et filtrage RANSAC (logique inchangée) ---
        try:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) 
            raw_matches = matcher.match(des1, des2)
            self._log(f"  EstimateTransform: BFMatcher (DAO+ORB V3) a trouvé {len(raw_matches)} correspondances brutes.")
        except Exception as e_match: 
            self._log(f"  EstimateTransform: Erreur Matcher (DAO+ORB V3): {e_match}", "ERROR"); return None
        
        if not raw_matches: 
            self._log("  EstimateTransform: Aucune correspondance brute trouvée (DAO+ORB V3).", "WARN"); return None
        
        if isinstance(raw_matches, tuple): raw_matches = list(raw_matches) # Au cas où
        raw_matches.sort(key=lambda m: m.distance)
        
        # Calculer le nombre de "bonnes" correspondances à garder pour RANSAC
        num_good_matches_to_keep = max(min_absolute_matches_config, int(len(raw_matches) * min_matches_ratio_config))
        # S'assurer de ne pas dépasser le nombre de correspondances brutes disponibles
        num_good_matches_to_keep = min(num_good_matches_to_keep, len(raw_matches)) 
        
        good_matches = raw_matches[:num_good_matches_to_keep]
        
        if len(good_matches) < min_absolute_matches_config:
            self._log(f"  EstimateTransform: Pas assez de 'bonnes' correspondances ({len(good_matches)} < {min_absolute_matches_config}) pour RANSAC (DAO+ORB V3).", "WARN")
            return None
        self._log(f"  EstimateTransform: {len(good_matches)} 'bonnes' correspondances retenues pour RANSAC (DAO+ORB V3).")

        # RANSAC a besoin d'au moins 3 points pour une transformation affine partielle (2x3)
        if len(good_matches) < 3: 
            self._log(f"  EstimateTransform: Moins de 3 bonnes correspondances ({len(good_matches)}) pour RANSAC (DAO+ORB V3).", "WARN")
            return None
        
        try:
            pts1_ref_ransac = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
            pts2_src_ransac = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
        except IndexError as e_idx_ransac: 
            self._log(f"  EstimateTransform: Erreur d'indice lors de la création des points pour RANSAC (DAO+ORB V3): {e_idx_ransac}", "ERROR")
            return None

        try:
            # Estimer la transformation affine (partielle = rotation, échelle, translation, cisaillement limité)
            M, inliers_mask = cv2.estimateAffinePartial2D(
                pts2_src_ransac, pts1_ref_ransac, # Src to Ref
                method=cv2.RANSAC,
                ransacReprojThreshold=ransac_thresh_config, # Seuil de reprojection
                maxIters=2000, 
                confidence=0.99
            )
        except cv2.error as e_cv_ransac: 
            self._log(f"  EstimateTransform: Erreur OpenCV lors de RANSAC (DAO+ORB V3): {e_cv_ransac}", "ERROR"); return None
        except Exception as e_ransac: 
            self._log(f"  EstimateTransform: Erreur inattendue lors de RANSAC (DAO+ORB V3): {e_ransac}", "ERROR"); return None
                                                         
        if M is None: 
            self._log("  EstimateTransform: RANSAC (DAO+ORB V3) n'a pas retourné de matrice M.", "WARN"); return None
        if inliers_mask is None : # Ne devrait pas arriver si M n'est pas None, mais sécurité
            self._log("  EstimateTransform: RANSAC (DAO+ORB V3) n'a pas retourné de masque d'inliers.", "ERROR"); return None
        
        num_inliers = np.sum(inliers_mask)
        self._log(f"  EstimateTransform: Matrice M (DAO+ORB V3) trouvée. Inliers RANSAC: {num_inliers}/{len(good_matches)}.")
        
        # Calculer le nombre minimum d'inliers RANSAC requis.
        # Basé sur la valeur de config, la moitié des correspondances absolues min, et un minimum de 3.
        min_ransac_inliers_needed = max(min_ransac_inliers_value_config, min_absolute_matches_config, 3) 
        
        if num_inliers < min_ransac_inliers_needed:
            self._log(f"  EstimateTransform: Nombre d'inliers RANSAC ({num_inliers}) trop faible. Requis au moins {min_ransac_inliers_needed} (basé sur config min_ransac={min_ransac_inliers_value_config} et min_abs_match={min_absolute_matches_config}). Rejet de la transformation.", "WARN")
            return None

        if self.debug: self._log(f"  EstimateTransform: Matrice M finale (DAO+ORB V3):\n{M}")
        return M

    # warp_image reste inchangé
    def warp_image(self, img_to_warp: np.ndarray, M: np.ndarray, output_shape_wh: tuple, 
                   border_mode=cv2.BORDER_CONSTANT, border_value=(0,0,0,0)):
        if M is None: self._log("Warp: Matrice M est None.", "ERROR"); return img_to_warp 
        try:
            return cv2.warpAffine(img_to_warp, M, dsize=output_shape_wh, flags=cv2.INTER_LINEAR, 
                                  borderMode=border_mode, borderValue=border_value)
        except Exception as e_warp: 
            self._log(f"Erreur cv2.warpAffine: {e_warp}", "ERROR"); return img_to_warp
        

#########################################################################################################################################################


    def warp_image(self, img_to_warp: np.ndarray, M: np.ndarray, output_shape_wh: tuple, 
                   border_mode=cv2.BORDER_CONSTANT, border_value=(0,0,0,0)):
        if M is None: self._log("Warp: Matrice M est None.", "ERROR"); return img_to_warp 
        # self._log(f"Phase 3: Warp. Shape out (W,H): {output_shape_wh}", "DEBUG") # Peut être verbeux
        try:
            return cv2.warpAffine(img_to_warp, M, dsize=output_shape_wh, flags=cv2.INTER_LINEAR, 
                                  borderMode=border_mode, borderValue=border_value)
        except Exception as e_warp: 
            self._log(f"Erreur cv2.warpAffine: {e_warp}", "ERROR"); return img_to_warp

# =============================================================================
#  FastSeestarAligner (Adaptateur)                                              ==========================================================================
# =============================================================================
class FastSeestarAligner:

    def __init__(self, debug: bool = False):
        self._fa = FastAligner(debug=debug)


# DANS seestar/core/fast_aligner_module.py
# DANS la classe FastSeestarAligner

    def _align_image(self, 
                     src_img_f32_in: np.ndarray, # Image source (float32, 0-1, peut être Couleur HWC ou N&B HW)
                     ref_img_f32_in: np.ndarray, # Image référence (float32, 0-1, peut être Couleur HWC ou N&B HW)
                     file_name: str | None = None,
                     # Paramètres ORB (nfeatures est maintenant géré dans _detect_stars_dao_orb)
                     # orb_features: int = 5000, # Ce paramètre n'est plus directement utilisé par estimate_transform V_DAO_StatsV3
                     # Paramètres communs pour matching/RANSAC (lus depuis self.fa_xxx dans _worker)
                     min_absolute_matches: int = 12, # Valeur par défaut si non passée par _worker
                     min_ransac_inliers_value: int = 4, 
                     ransac_thresh: float = 2.5,     # Nouvelle valeur par défaut plus stricte
                     min_matches_ratio: float = 0.15,
                     # Paramètres DAOStarFinder (lus depuis self.fa_dao_xxx dans _worker)
                     daofind_fwhm: float = 3.5,
                     daofind_threshold_sigma: float = 6.0, # C'est le facteur sigma
                     max_stars_to_describe: int = 750
                     ) -> tuple[np.ndarray | None, np.ndarray | None, bool]: # Retour: (img_alignée, Matrice_M, succès_bool)
        """
        Aligne une image source sur une image de référence en utilisant FastAligner.
        MODIFIED V_DAO_StatsV3:
        - Prépare les images N&B float32 pour estimate_transform.
        - Passe tous les paramètres de configuration à estimate_transform.
        - Gère le warp de l'image source originale (potentiellement couleur).
        """
        tag = file_name if file_name else "[image sans nom]"
        self._fa._log(f"FastSeestarAligner._align_image (V_DAO_StatsV3): Début alignement pour '{tag}'.")
        self._fa._log(f"  Params DAO reçus: FWHM={daofind_fwhm:.1f}, ThrSigFactor={daofind_threshold_sigma:.1f}, MaxStars={max_stars_to_describe}")
        self._fa._log(f"  Params Match/RANSAC reçus: MinAbsM={min_absolute_matches}, MinRansacVal={min_ransac_inliers_value}, RansacThr={ransac_thresh:.1f}, MinMatchRatio={min_matches_ratio:.2f}")

        if src_img_f32_in is None or ref_img_f32_in is None:
            self._fa._log(f"FastSeestarAligner: Image source ou référence est None pour '{tag}'.", level="ERROR")
            return None, None, False
        
        # --- 1. S'assurer que les images sont float32 (devrait déjà être le cas) et non vides ---
        src_f32 = src_img_f32_in.astype(np.float32, copy=False) # copy=False si déjà float32
        ref_f32 = ref_img_f32_in.astype(np.float32, copy=False)

        if src_f32.size == 0 or ref_f32.size == 0:
            self._fa._log(f"FastSeestarAligner: Image source ou référence est vide pour '{tag}'.", level="ERROR")
            return None, None, False

        # --- 2. Convertir en N&B (grayscale) float32 (0-1) si elles sont en couleur ---
        #    Cette image N&B sera utilisée pour la détection de features et l'estimation de la transformation.
        def to_gray_if_color_f32(im_f32_local: np.ndarray) -> np.ndarray | None:
            if im_f32_local.ndim == 2:
                return im_f32_local # Déjà N&B
            elif im_f32_local.ndim == 3 and im_f32_local.shape[2] == 3: # RGB
                try: return cv2.cvtColor(im_f32_local, cv2.COLOR_RGB2GRAY)
                except cv2.error as e_cvt: self._fa._log(f"Erreur cvtColor RGB2GRAY: {e_cvt}", "ERROR"); return None
            elif im_f32_local.ndim == 3 and im_f32_local.shape[2] == 4: # RGBA
                try: return cv2.cvtColor(im_f32_local, cv2.COLOR_RGBA2GRAY)
                except cv2.error as e_cvt: self._fa._log(f"Erreur cvtColor RGBA2GRAY: {e_cvt}", "ERROR"); return None
            elif im_f32_local.ndim == 3 and im_f32_local.shape[2] == 1: # HxWx1, juste enlever le dernier axe
                 return im_f32_local[..., 0]
            else:
                self._fa._log(f"FastSeestarAligner: Shape d'image non supportée pour conversion N&B: {im_f32_local.shape}", level="ERROR")
                return None

        src_gray_f32_for_align = to_gray_if_color_f32(src_f32)
        ref_gray_f32_for_align = to_gray_if_color_f32(ref_f32)

        if src_gray_f32_for_align is None or ref_gray_f32_for_align is None:
            self._fa._log(f"FastSeestarAligner: Échec conversion N&B pour '{tag}'.", level="ERROR")
            return None, None, False
        
        self._fa._log(f"  FastSeestarAligner: Images N&B float32 prêtes pour estimate_transform. SrcShape={src_gray_f32_for_align.shape}, RefShape={ref_gray_f32_for_align.shape}", "DEBUG")

        # --- 3. Estimer la transformation en utilisant FastAligner.estimate_transform ---
        #    On passe les images N&B float32 (0-1) et tous les paramètres de configuration.
        #    L'orb_features n'est plus passé directement ici à estimate_transform.
        M_transform_matrix = self._fa.estimate_transform(
            ref_gray_f32=ref_gray_f32_for_align, 
            img_gray_f32=src_gray_f32_for_align, 
            min_matches_ratio_config=min_matches_ratio,
            min_absolute_matches_config=min_absolute_matches,
            ransac_thresh_config=ransac_thresh,
            min_ransac_inliers_value_config=min_ransac_inliers_value,
            daofind_fwhm_config=daofind_fwhm, 
            daofind_threshold_sigma_config=daofind_threshold_sigma,
            max_stars_to_describe_config=max_stars_to_describe
        ) 
        
        if M_transform_matrix is None: 
            self._fa._log(f"FastSeestarAligner: Alignement ÉCHOUÉ (estimate_transform a retourné None) pour '{tag}'.", level="ERROR")
            return None, None, False # Retourner None pour l'image et la matrice, et False pour succès
        
        # --- 4. Appliquer la transformation (Warp) à l'image source ORIGINALE (qui peut être couleur) ---
        output_h, output_w = ref_f32.shape[:2] # La transformation est vers l'espace de l'image de référence
        output_shape_wh_cv = (output_w, output_h) # OpenCV attend (W, H)
        
        aligned_image_result = None
        if src_f32.ndim == 2: # Si l'image source originale était N&B
            aligned_image_result = self._fa.warp_image(src_f32, M_transform_matrix, output_shape_wh_cv)
        elif src_f32.ndim == 3 and src_f32.shape[2] in [3, 4]: # Si l'image source originale était Couleur (RGB ou RGBA)
            # Warper chaque canal séparément
            channels = cv2.split(src_f32)
            warped_channels = []
            for i_ch in range(src_f32.shape[2]): # 3 pour RGB, 4 pour RGBA
                warped_channel = self._fa.warp_image(channels[i_ch], M_transform_matrix, output_shape_wh_cv)
                if warped_channel is channels[i_ch]: # Si warp_image a retourné l'original (erreur)
                    self._fa._log(f"FastSeestarAligner: Échec du warp pour canal {i_ch} de '{tag}'.", level="ERROR")
                    return None, M_transform_matrix, False # Échec partiel du warp
                warped_channels.append(warped_channel)
            aligned_image_result = cv2.merge(warped_channels)
        else: 
            self._fa._log(f"FastSeestarAligner: Format d'image source original non supporté pour le warp: {src_f32.shape} pour '{tag}'", level="ERROR")
            return None, M_transform_matrix, False 
        
        # Vérifier si le warp a réellement produit une nouvelle image ou retourné l'originale (signe d'erreur dans warp_image)
        if aligned_image_result is src_f32 : # Comparaison d'identité d'objet
            self._fa._log(f"FastSeestarAligner: Alignement PARTIELLEMENT ÉCHOUÉ pour '{tag}' (warp_image a retourné l'image source originale).",level="WARN")
            return None, M_transform_matrix, False # Retourner M mais indiquer échec
            
        # S'assurer que l'image alignée est bien float32 et clippée 0-1
        aligned_image_final = np.clip(aligned_image_result.astype(np.float32), 0.0, 1.0)
        
        self._fa._log(f"FastSeestarAligner: Alignement RÉUSSI pour '{tag}' (DAO+ORB V3). Shape alignée: {aligned_image_final.shape}")
        return aligned_image_final, M_transform_matrix, True


    def set_progress_callback(self, cb):
        self._fa.set_progress_callback(cb)




print("DEBUG [FastAlignerModule]: Module chargé (V_DAO_Integrated_Corrected).")