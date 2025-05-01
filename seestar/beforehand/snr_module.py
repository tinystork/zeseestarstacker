# --- START OF FILE snr_module.py ---

import numpy as np
from astropy.stats import sigma_clipped_stats
import traceback
import warnings

def calculate_snr(data):
    """
    Calcule le SNR d'une image de manière robuste.
    Prend en entrée les données de l'image (tableau numpy 2D).
    Retourne snr (float), sky_bg (float), sky_noise (float), num_signal_pixels (int).
    Retourne (np.nan, np.nan, np.nan, 0) en cas d'erreur majeure ou si les stats ne peuvent être calculées.
    """
    try:
        # S'assurer que les données sont en float64 pour la précision
        # et créer une copie pour ne pas modifier l'original si c'est une vue
        data_float = np.array(data, dtype=np.float64)

        # Vérifier si le tableau est vide ou ne contient que des NaN/Inf
        if data_float.size == 0 or not np.any(np.isfinite(data_float)):
            print("WARNING (calculate_snr): Données vides ou non finies.")
            return np.nan, np.nan, np.nan, 0

        # --- Estimation robuste du fond de ciel (sky_bg) et du bruit (sky_noise) ---
        # sigma_clipped_stats ignore les NaN par défaut.
        try:
            # Utiliser 5 itérations pour une meilleure convergence sur des images potentiellement bruitées
            mean, median, std = sigma_clipped_stats(data_float, sigma=3.0, maxiters=5)
            sky_bg = median
            sky_noise = std
        except Exception as stat_err:
            # Si sigma_clipped_stats échoue (rare, mais possible avec des données très étranges)
            print(f"WARNING (calculate_snr): sigma_clipped_stats a échoué ({stat_err}). Fallback vers nanmedian/nanstd.")
            # Ignorer les avertissements RuntimeWarning (ex: "Mean of empty slice") pendant le fallback
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                sky_bg = np.nanmedian(data_float)
                sky_noise = np.nanstd(data_float)

        # Vérifier si les stats calculées sont valides
        if not np.isfinite(sky_bg) or not np.isfinite(sky_noise):
            print(f"ERREUR (calculate_snr): Impossible de calculer des statistiques valides (median={sky_bg}, std={sky_noise}).")
            return np.nan, np.nan, np.nan, 0

        # Gérer le cas où le bruit est nul ou très faible (image plate?)
        if sky_noise < 1e-9:
            # print("DEBUG (calculate_snr): Bruit (std) proche de zéro.")
            # Dans ce cas, le SNR sera soit 0 (si pas de signal > fond), soit infini.
            # On le traitera plus tard. Le bruit est considéré comme quasi nul.
            sky_noise = 1e-9 # Assigner une valeur minimale pour éviter division par zéro plus tard

        # --- Identification des pixels de signal ---
        # Seuil initial: 5-sigma au-dessus du fond de ciel médian
        threshold = sky_bg + 5.0 * sky_noise
        # Masque des pixels potentiels de signal (doivent être finis ET au-dessus du seuil)
        signal_mask = np.isfinite(data_float) & (data_float > threshold)
        num_signal_pixels = np.sum(signal_mask)

        # Fallback si très peu de pixels dépassent le seuil 5-sigma (ex: nébuleuse faible/étendue)
        # Utiliser un percentile élevé comme seuil alternatif.
        # Le seuil de 10 pixels est arbitraire et peut être ajusté.
        if num_signal_pixels < 10:
            # print(f"DEBUG (calculate_snr): Peu de pixels ({num_signal_pixels}) > 5 sigma. Essai avec percentile.")
            # Calculer sur les données finies uniquement
            finite_data = data_float[np.isfinite(data_float)]
            if finite_data.size > 0:
                try:
                    # Utiliser P95 comme seuil alternatif (capture les 5% pixels les plus brillants)
                    alt_threshold = np.percentile(finite_data, 95)
                    # S'assurer que le seuil alternatif est au moins un peu au-dessus du fond
                    if alt_threshold > sky_bg + 1e-6: # Ajouter petite marge
                        threshold = alt_threshold
                        signal_mask = np.isfinite(data_float) & (data_float > threshold)
                        num_signal_pixels = np.sum(signal_mask)
                        # print(f"DEBUG (calculate_snr): Utilisation seuil P95 ({threshold:.2f}). Nouveaux pixels signal: {num_signal_pixels}")
                    # else: # Si P95 est très proche du fond, garder le masque 5-sigma (probablement vide)
                        # print(f"DEBUG (calculate_snr): Seuil P95 ({alt_threshold:.2f}) trop proche du fond ({sky_bg:.2f}). Maintien masque 5-sigma.")
                except Exception as perc_err:
                    print(f"WARNING (calculate_snr): Erreur calcul percentile ({perc_err}). Maintien masque 5-sigma.")
            # else: # Si pas de données finies, num_signal_pixels reste 0

        # --- Calcul de la valeur moyenne du signal (au-dessus du fond) ---
        signal_value_above_bg = 0.0
        if num_signal_pixels > 0:
            # Calculer la moyenne des pixels du masque (qui sont finis par définition du masque)
            mean_signal_pixels = np.mean(data_float[signal_mask])
            signal_value_above_bg = mean_signal_pixels - sky_bg
            # Assurer que la valeur n'est pas négative (peut arriver si seuil très bas et bruit)
            if signal_value_above_bg < 0:
                signal_value_above_bg = 0.0

        # --- Calcul du SNR ---
        # SNR = (Signal moyen au-dessus du fond) / Bruit
        # Utiliser la valeur de sky_noise (std robuste) calculée précédemment
        # sky_noise a déjà été vérifié pour être > 0
        snr = signal_value_above_bg / sky_noise

        # Retourner les valeurs en types Python standard
        return float(snr), float(sky_bg), float(sky_noise), int(num_signal_pixels)

    except MemoryError:
        # Gérer spécifiquement les erreurs de mémoire
        print("ERREUR CRITIQUE (calculate_snr): Mémoire insuffisante pour traiter l'image.")
        traceback.print_exc()
        # Retourner NaN pour indiquer un échec clair dû à la mémoire
        return np.nan, np.nan, np.nan, 0
    except Exception as e:
        # Capturer toute autre exception imprévue
        print(f"ERREUR (calculate_snr): Exception inattendue - {e}")
        traceback.print_exc()
        # Retourner NaN pour indiquer un échec
        return np.nan, np.nan, np.nan, 0

# --- FIN DU FICHIER snr_module.py ---