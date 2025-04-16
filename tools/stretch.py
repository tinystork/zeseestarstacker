"""
Module pour l'étirement des images astronomiques.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

class Stretch:
    """Implémentation de l'algorithme de stretch inspiré de PixInsight."""

    def __init__(self, target_bkg=0.25, shadows_clip=-1.25):
        """
        Initialise l'objet Stretch avec les paramètres spécifiés.

        Args:
            target_bkg (float): Objectif pour le fond de ciel (0.25 par défaut)
            shadows_clip (float): Valeur de seuil pour les ombres (-1.25 par défaut)
        """
        self.shadows_clip = shadows_clip
        self.target_bkg = target_bkg

    def _get_avg_dev(self, data):
        """
        Retourne la déviation moyenne par rapport à la médiane.

        Args:
            data (np.array): tableau de valeurs flottantes, typiquement les données de l'image

        Returns:
            float: déviation moyenne
        """
        median = np.median(data)
        n = data.size
        median_deviation = lambda x: abs(x - median)
        avg_dev = np.sum(median_deviation(data) / n)
        return avg_dev

    def _mtf(self, m, x):
        """
        Fonction de transfert des tons moyens (Midtones Transfer Function)

        MTF(m, x) = {
            0               pour x == 0,
            1/2             pour x == m,
            1               pour x == 1,

            (m - 1)x
            --------------  sinon.
            (2m - 1)x - m
        }

        Voir la section "Midtones Balance" dans:
        https://pixinsight.com/doc/tools/HistogramTransformation/HistogramTransformation.html

        Args:
            m (float): paramètre d'équilibrage des tons moyens
                       une valeur inférieure à 0.5 assombrit les tons moyens
                       une valeur supérieure à 0.5 éclaircit les tons moyens
            x (np.array): les données que nous voulons copier et transformer.

        Returns:
            np.array: données transformées
        """
        shape = x.shape
        x = x.flatten()
        zeros = x == 0
        halfs = x == m
        ones = x == 1
        others = np.logical_xor((x == x), (zeros + halfs + ones))

        x[zeros] = 0
        x[halfs] = 0.5
        x[ones] = 1
        x[others] = (m - 1) * x[others] / ((((2 * m) - 1) * x[others]) - m)
        return x.reshape(shape)

    def _get_stretch_parameters(self, data):
        """
        Obtenir automatiquement les paramètres d'étirement.

        Args:
            data (np.array): données de l'image

        Returns:
            dict: paramètres d'étirement (c0, c1, m)
        """
        median = np.median(data)
        avg_dev = self._get_avg_dev(data)

        c0 = np.clip(median + (self.shadows_clip * avg_dev), 0, 1)
        m = self._mtf(self.target_bkg, median - c0)

        return {
            "c0": c0,
            "c1": 1,
            "m": m
        }

    def stretch(self, data):
        """
        Étirer l'image.

        Args:
            data (np.array): tableau de données de l'image originale.

        Returns:
            np.array: données de l'image étirée
        """
        # S'assurer qu'on ne modifie pas l'original
        data_copy = np.copy(data)
        
        # Normaliser les données
        if np.max(data_copy) > 0:
            d = data_copy / np.nanmax(data_copy)
        else:
            # Éviter la division par zéro
            return np.zeros_like(data_copy)

        # Obtenir les paramètres d'étirement
        stretch_params = self._get_stretch_parameters(d)
        m = stretch_params["m"]
        c0 = stretch_params["c0"]
        c1 = stretch_params["c1"]

        # Sélecteurs pour les pixels qui se trouvent en dessous ou au-dessus du point de seuil des ombres
        below = d < c0
        above = d >= c0

        # Couper tout ce qui est en dessous du point de seuil des ombres
        d[below] = 0

        # Pour le reste des pixels : appliquer la fonction de transfert des tons moyens
        d[above] = self._mtf(m, (d[above] - c0) / (1 - c0))
        return d
    
@staticmethod
def apply_stretch(data, target_bkg=0.25, shadows_clip=-1.25):
    """
    Fonction d'emballage pour une interface plus simple.

    Args:
        data (np.array): Données de l'image à étirer
        target_bkg (float): Objectif pour le fond de ciel
        shadows_clip (float): Valeur de seuil pour les ombres

    Returns:
        np.array: Image étirée
    """
    return Stretch(target_bkg, shadows_clip).stretch(data)

@staticmethod
def save_fits_as_png(fits_data, output_path, cmap='gray'):
    """
    Sauvegarde les données FITS en tant qu'image PNG.

    Args:
        fits_data (np.array): Données de l'image
        output_path (str): Chemin de sortie pour le PNG
        cmap (str): Carte de couleurs à utiliser (par défaut: 'gray')
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(fits_data, cmap=cmap, norm=Normalize(vmin=0, vmax=1))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()