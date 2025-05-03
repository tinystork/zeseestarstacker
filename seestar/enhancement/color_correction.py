import numpy as np
from scipy.ndimage import gaussian_filter

class ChromaticBalancer:
    """
    Classe pour équilibrer les canaux de couleur dans les zones de recouvrement
    et corriger les artefacts colorés lors du stacking d'images.
    """
    
    def __init__(self, border_size=50, blur_radius=15):
        """
        Initialise le balancer avec les paramètres donnés.
        
        Args:
            border_size (int): Taille de la bordure à analyser en pixels
            blur_radius (int): Rayon de flou pour les transitions
        """
        self.border_size = border_size
        self.blur_radius = blur_radius
    
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
                r_factor = np.clip(r_factor, 0.7, 1.3)
                b_factor = np.clip(b_factor, 0.7, 1.3)
                
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