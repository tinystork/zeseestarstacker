import numpy as np
from scipy.ndimage import gaussian_filter

class ChromaticBalancer:
    """
    Classe pour équilibrer les canaux de couleur dans les zones de recouvrement
    et corriger les artefacts colorés lors du stacking d'images.
    """
    
    def __init__(self, border_size=50, blur_radius=25):#<-----------------------------------------------------------------------------------------------------------
        """
        Initialise le balancer avec les paramètres donnés.
        
        Args:
            border_size (int): Taille de la bordure à analyser en pixels
            blur_radius (int): Rayon de flou pour les transitions
        """
        self.border_size = border_size
        self.blur_radius = blur_radius
    


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
        
        Args:
            stacked_data (np.ndarray): Image stackée (H, W, 3)
            reference_ratios (tuple): Ratios de référence (R/G, B/G) ou None
            
        Returns:
            np.ndarray: Image avec couleurs corrigées
        """
        if stacked_data.ndim != 3 or stacked_data.shape[2] != 3:
            return stacked_data  # Non-RGB, retourner tel quel
            
        # Créer une copie pour éviter de modifier l'original
        corrected = stacked_data.copy()
        
        # Calculer les ratios si non fournis
        if reference_ratios is None:
            # Utiliser le centre de l'image comme référence
            h, w = stacked_data.shape[:2]
            center_h, center_w = h // 2, w // 2
            center_size = min(h, w) // 4
            
            center_slice = stacked_data[
                center_h - center_size:center_h + center_size,
                center_w - center_size:center_w + center_size
            ]
            
            reference_ratios = self.calculate_channel_ratios(center_slice)
        
        # Créer un masque pour les bords
        edge_mask = self.create_edge_mask(stacked_data.shape[:2])
        
        # Pour chaque pixel, analyser l'équilibre local des couleurs
        h, w = stacked_data.shape[:2]
        for y in range(0, h, self.border_size):
            for x in range(0, w, self.border_size):
                # Définir la région locale
                y_end = min(y + self.border_size, h)
                x_end = min(x + self.border_size, w)
                
                # Extraire la région
                region = stacked_data[y:y_end, x:x_end]
                
                # Calculer les ratios de cette région
                local_ratios = self.calculate_channel_ratios(region)
                
                # Calculer les facteurs de correction
                r_factor = reference_ratios[0] / max(local_ratios[0], 1e-8)
                b_factor = reference_ratios[1] / max(local_ratios[1], 1e-8)
                
                # Limiter les facteurs pour éviter les corrections extrêmes
                r_factor = np.clip(r_factor, 0.5, 1.5)
                b_factor = np.clip(b_factor, 0.5, 1.5)
                
                # Appliquer la correction avec fusion pondérée par masque de bord
                for i in range(y, y_end):
                    for j in range(x, x_end):
                        # Force de la correction basée sur le masque
                        strength = 1.0 - edge_mask[i, j]
                        
                        # Appliquer la correction
                        corrected[i, j, 0] = stacked_data[i, j, 0] * (1.0 - strength + strength * r_factor)
                        # Canal vert inchangé (référence)
                        corrected[i, j, 2] = stacked_data[i, j, 2] * (1.0 - strength + strength * b_factor)
        
        return np.clip(corrected, 0.0, 1.0)
    


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