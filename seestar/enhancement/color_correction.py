import numpy as np
from scipy.ndimage import gaussian_filter

class ChromaticBalancer:
    """
    Classe pour équilibrer les canaux de couleur dans les zones de recouvrement
    et corriger les artefacts colorés lors du stacking d'images.
    """
    
    def __init__(self, border_size=25, blur_radius=8, 
                 r_factor_limits=(0.7, 1.3), # Limites par défaut pour le facteur R
                 b_factor_limits=(0.4, 1.5)  # Limites par défaut pour le facteur B
                ):
        """
        Initialise le balancer avec les paramètres donnés.
        
        Args:
            border_size (int): Taille de la bordure à analyser en pixels.
            blur_radius (int): Rayon de flou pour les transitions.
            r_factor_limits (tuple): (min_r_factor, max_r_factor) pour clipper le gain Rouge.
            b_factor_limits (tuple): (min_b_factor, max_b_factor) pour clipper le gain Bleu.
        """
        self.border_size = int(border_size)
        self.blur_radius = int(blur_radius)
        
        # Stocker les limites de gain pour les facteurs
        self.r_factor_min = float(r_factor_limits[0])
        self.r_factor_max = float(r_factor_limits[1])
        self.b_factor_min = float(b_factor_limits[0])
        self.b_factor_max = float(b_factor_limits[1])

        print(f"DEBUG [ChromaticBalancer]: Initialisé avec border={self.border_size}, blur={self.blur_radius}, R_limits=[{self.r_factor_min:.2f},{self.r_factor_max:.2f}], B_limits=[{self.b_factor_min:.2f},{self.b_factor_max:.2f}]")


#################################################################################################################################################


    def create_edge_mask(self, shape_hw):
        """
        Crée un masque avec poids réduits aux bords de l'image.
        
        Args:
            shape_hw (tuple): Forme (H, W) de l'image
            
        Returns:
            np.ndarray: Masque pondéré des bords (0-1)
        """
        h, w = shape_hw
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # Distance depuis les bords
        dist_from_left = x
        dist_from_right = w - x - 1
        dist_from_top = y
        dist_from_bottom = h - y - 1
        
        # Minimum distance depuis n'importe quel bord
        min_dist = np.minimum.reduce([
            dist_from_left, dist_from_right,
            dist_from_top, dist_from_bottom
        ])
        
        # Normaliser à [0, 1] dans la zone de bord
        edge_mask = np.clip(min_dist / self.border_size, 0, 1)
        
        # Appliquer un flou pour adoucir les transitions
        edge_mask = gaussian_filter(edge_mask, sigma=self.blur_radius)
        
        return edge_mask
    


####################################################################################################################################


    def calculate_channel_ratios(self, image_data):
        """
        Calcule les ratios entre canaux pour détecter les déséquilibres.
        
        Args:
            image_data (np.ndarray): Image couleur (H, W, 3)
            
        Returns:
            tuple: Ratios (R/G, B/G)
        """
        # Éviter division par zéro
        epsilon = 1e-8
        
        # Extraire les canaux
        r_channel = image_data[..., 0]
        g_channel = image_data[..., 1]
        b_channel = image_data[..., 2]
        
        # Créer un masque pour les pixels valides (non nuls)
        valid_mask = (g_channel > epsilon)
        
        # Calculer les ratios moyens sur les pixels valides
        if np.sum(valid_mask) > 100:  # Assez de pixels pour une statistique fiable
            r_g_ratio = np.median(r_channel[valid_mask] / (g_channel[valid_mask] + epsilon))
            b_g_ratio = np.median(b_channel[valid_mask] / (g_channel[valid_mask] + epsilon))
        else:
            # Valeurs par défaut si pas assez de pixels
            r_g_ratio = 1.0
            b_g_ratio = 1.0
            
        return r_g_ratio, b_g_ratio
    

#######################################################################################################################################

    def normalize_stack(self, stacked_data, reference_ratios=None):
        """
        Normalise une image stackée pour corriger les artefacts de couleur.
        Utilise self.r_factor_min/max et self.b_factor_min/max pour clipper les gains.
        """
        print(f"DEBUG [ChromaticBalancer normalize_stack]: Début. R_limits=[{self.r_factor_min:.2f},{self.r_factor_max:.2f}], B_limits=[{self.b_factor_min:.2f},{self.b_factor_max:.2f}]")
        if stacked_data is None or stacked_data.ndim != 3 or stacked_data.shape[2] != 3:
            print("DEBUG [ChromaticBalancer normalize_stack]: Données invalides ou non RGB.")
            return stacked_data

        corrected = stacked_data.astype(np.float32, copy=True) # Travailler sur une copie float32
        
        if reference_ratios is None:
            h_ref, w_ref = corrected.shape[:2]
            center_h_ref, center_w_ref = h_ref // 2, w_ref // 2
            center_size_ref = min(h_ref, w_ref) // 4
            center_slice_ref = corrected[center_h_ref - center_size_ref : center_h_ref + center_size_ref,
                                         center_w_ref - center_size_ref : center_w_ref + center_size_ref]
            if center_slice_ref.size > 0:
                 reference_ratios = self.calculate_channel_ratios(center_slice_ref)
                 print(f"DEBUG [ChromaticBalancer normalize_stack]: Ratios de référence (centre): R/G={reference_ratios[0]:.2f}, B/G={reference_ratios[1]:.2f}")
            else:
                 print("WARN [ChromaticBalancer normalize_stack]: Slice centrale vide pour référence. Utilisation ratios par défaut (1,1).")
                 reference_ratios = (1.0, 1.0) # Fallback

        edge_mask = self.create_edge_mask(corrected.shape[:2])
        h, w = corrected.shape[:2]
        
        # Utiliser des pas plus grands si l'image est grande pour la performance
        step = max(1, self.border_size // 2, min(h, w) // 20) # Assurer au moins 1, ne pas dépasser 1/20 de la dim
        
        print(f"DEBUG [ChromaticBalancer normalize_stack]: Boucle sur régions avec step={step}, border_size={self.border_size}")
        for y_start_loop in range(0, h, step):
            for x_start_loop in range(0, w, step):
                y_end_loop = min(y_start_loop + self.border_size, h)
                x_end_loop = min(x_start_loop + self.border_size, w)
                
                region = corrected[y_start_loop:y_end_loop, x_start_loop:x_end_loop]
                if region.size == 0: continue

                local_ratios = self.calculate_channel_ratios(region)
                
                r_factor = reference_ratios[0] / max(local_ratios[0], 1e-8)
                b_factor = reference_ratios[1] / max(local_ratios[1], 1e-8)
                
                ### MODIFIÉ : Utilisation des attributs self pour les limites de clipping ###
                r_factor = np.clip(r_factor, self.r_factor_min, self.r_factor_max)
                b_factor = np.clip(b_factor, self.b_factor_min, self.b_factor_max)
                ### FIN MODIFICATION ###
                
                # Debug log moins fréquent pour éviter de spammer la console
                # if y_start_loop % (step * 5) == 0 and x_start_loop % (step * 5) == 0 :
                #    print(f"  Region ({y_start_loop}:{y_end_loop},{x_start_loop}:{x_end_loop}) LocalRatios: R/G={local_ratios[0]:.2f}, B/G={local_ratios[1]:.2f} | Factors: R={r_factor:.2f}, B={b_factor:.2f}")

                # Appliquer la correction à la *partie centrale* de la fenêtre de calcul
                # pour éviter les effets de bord de la fenêtre elle-même, ou pondérer
                # l'application pour une fusion plus douce.
                # Pour l'instant, on applique à toute la région de calcul pour la simplicité,
                # mais la pondération par edge_mask est la plus importante.

                # Boucle sur les pixels de la région actuelle pour appliquer la correction pondérée
                # Cette partie est coûteuse. Si possible, des opérations vectorielles seraient mieux.
                # Mais pour la pondération par edge_mask pixel par pixel, une boucle est plus simple à écrire.
                for y_pix in range(y_start_loop, y_end_loop):
                    for x_pix in range(x_start_loop, x_end_loop):
                        strength = 1.0 - edge_mask[y_pix, x_pix] # 0 aux bords définis par border_size, 1 au centre
                        
                        # Appliquer la correction additivement ou multiplicativement
                        # L'application actuelle est multiplicative mais tend vers l'original si strength=0
                        # et vers la pleine correction si strength=1
                        # original_r = stacked_data[y_pix, x_pix, 0] # Lire depuis l'original pour éviter accumulation d'erreurs
                        # original_b = stacked_data[y_pix, x_pix, 2]
                        # corrected[y_pix, x_pix, 0] = original_r * (1.0 - strength + strength * r_factor)
                        # corrected[y_pix, x_pix, 2] = original_b * (1.0 - strength + strength * b_factor)

                        # Simplification: appliquer sur 'corrected' qui est une copie de stacked_data
                        # Cette version modifie 'corrected' en place.
                        corrected[y_pix, x_pix, 0] = corrected[y_pix, x_pix, 0] * (1.0 - strength + strength * r_factor)
                        corrected[y_pix, x_pix, 2] = corrected[y_pix, x_pix, 2] * (1.0 - strength + strength * b_factor)

        final_corrected = np.clip(corrected, 0.0, 1.0)
        print("DEBUG [ChromaticBalancer normalize_stack]: Normalisation terminée.")
        return final_corrected


#####################################################################################################################################################

def apply_scnr(image_rgb, target_channel='green', amount=1.0, preserve_luminosity=True):
    """
    Applique une réduction du bruit de couleur de type SCNR.

    Args:
        image_rgb (np.ndarray): Image couleur (H, W, 3) en float32, normalisée 0-1.
        target_channel (str): Canal à réduire ('green' ou 'blue').
        amount (float): Force de la réduction (0.0 à 1.0). 1.0 = remplacement complet.
        preserve_luminosity (bool): Si True, tente de préserver la luminance originale.

    Returns:
        np.ndarray: Image corrigée.
    """
    print(f"DEBUG [apply_scnr]: Application SCNR sur canal '{target_channel}' avec amount={amount}, preserve_lum={preserve_luminosity}")
    if image_rgb is None or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        print("DEBUG [apply_scnr]: Image non RGB valide, retour original.")
        return image_rgb

    corrected_image = image_rgb.astype(np.float32, copy=True)
    r_channel, g_channel, b_channel = corrected_image[..., 0], corrected_image[..., 1], corrected_image[..., 2]

    original_luminance = None
    if preserve_luminosity:
        # Coefficients de luminance standard (peuvent être ajustés)
        original_luminance = 0.2126 * r_channel + 0.7152 * g_channel + 0.0722 * b_channel
        # Alternative plus simple souvent utilisée : original_luminance = np.mean(corrected_image, axis=2)

    if target_channel == 'green':
        # Référence pour G = médiane(R, B)
        # Pour éviter les artefacts avec np.median sur des tableaux 2D par pixel,
        # on peut faire min(R,B) + 0.5 * abs(R-B) ou une moyenne simple si R et B sont proches.
        # Une approche simple et souvent efficace :
        ref_for_g = (r_channel + b_channel) / 2.0
        # Alternative plus robuste au bruit (mais un peu plus lente) :
        # ref_for_g = np.median(np.stack((r_channel, b_channel), axis=-1), axis=-1)

        # Masque où G > Réf
        green_mask = g_channel > ref_for_g
        # Calcul du nouveau G
        new_g = ref_for_g # Remplacement complet si amount = 1.0
        if amount < 1.0: # Application partielle
            new_g = g_channel * (1.0 - amount * green_mask) + ref_for_g * (amount * green_mask)
        
        corrected_image[..., 1] = np.where(green_mask, new_g, g_channel)

    elif target_channel == 'blue': # Similaire pour le bleu si besoin un jour
        ref_for_b = (r_channel + g_channel) / 2.0
        blue_mask = b_channel > ref_for_b
        new_b = ref_for_b
        if amount < 1.0:
            new_b = b_channel * (1.0 - amount * blue_mask) + ref_for_b * (amount * blue_mask)
        corrected_image[..., 2] = np.where(blue_mask, new_b, b_channel)
    else:
        print(f"DEBUG [apply_scnr]: Canal cible '{target_channel}' non supporté.")
        return image_rgb

    if preserve_luminosity and original_luminance is not None:
        # Recalculer la nouvelle luminance
        new_r, new_g, new_b = corrected_image[...,0], corrected_image[...,1], corrected_image[...,2]
        current_luminance = 0.2126 * new_r + 0.7152 * new_g + 0.0722 * new_b
        
        # Éviter division par zéro
        current_luminance_safe = np.where(current_luminance < 1e-7, 1e-7, current_luminance)
        
        # Facteur de correction pour restaurer la luminance
        luminance_correction_factor = original_luminance / current_luminance_safe
        
        # Appliquer le facteur et clipper
        corrected_image[..., 0] *= luminance_correction_factor
        corrected_image[..., 1] *= luminance_correction_factor
        corrected_image[..., 2] *= luminance_correction_factor
        
    corrected_image = np.clip(corrected_image, 0.0, 1.0)
    print("DEBUG [apply_scnr]: SCNR terminé.")
    return corrected_image