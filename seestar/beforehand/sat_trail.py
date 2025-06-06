import os
import glob
import shutil
import threading
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from astropy.io import fits
import numpy as np
from astropy.stats import sigma_clipped_stats
import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import subprocess
import sys
import platform
import traceback
import inspect # Pour vérifier les arguments si besoin
import warnings # Ajoutez cette ligne au début de votre fichier .py
import time

# Importer acstools si disponible
try:
    from acstools import satdet
    # Vérifier si la fonction detsat attend 'searchpattern' comme premier argument
    sig = inspect.signature(satdet.detsat)
    params = list(sig.parameters)
    if params and params[0] == 'searchpattern':
        SATDET_AVAILABLE = True
        SATDET_USES_SEARCHPATTERN = True
        print("INFO: acstools.satdet.detsat détecté (mode searchpattern).")
    else:
        # Peut-être une ancienne version qui prenait 'data'?
        # Ou une fonction différente? Pour l'instant on la désactive si elle ne matche pas la doc.
        print(f"WARNING: acstools.satdet.detsat trouvée, mais sa signature ({params}) ne correspond pas à la documentation attendue ('searchpattern'...). Détection désactivée.")
        SATDET_AVAILABLE = False
        SATDET_USES_SEARCHPATTERN = False

except ImportError:
    print("INFO: acstools non trouvé.")
    SATDET_AVAILABLE = False
    SATDET_USES_SEARCHPATTERN = False
except Exception as e:
    print(f"Erreur lors de l'import ou de l'inspection d'acstools.satdet: {e}")
    SATDET_AVAILABLE = False
    SATDET_USES_SEARCHPATTERN = False


class AstroImageAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Analyseur d'Images Astronomiques")
        self.root.geometry("900x730") # Ajusté la hauteur
        self.root.minsize(900, 730)

        # Variables
        self.input_dir = tk.StringVar()
        self.output_log = tk.StringVar()
        self.sat_trail_dir = tk.StringVar()
        self.status_text = tk.StringVar(value="Prêt")
        # Désactiver par défaut si la bonne fonction n'est pas trouvée
        self.detect_satellites = tk.BooleanVar(value=(SATDET_AVAILABLE and SATDET_USES_SEARCHPATTERN))
        self.move_satellite_images = tk.BooleanVar(value=True)
        self.sort_by_snr = tk.BooleanVar(value=True)
        self.progress_var = tk.DoubleVar(value=0.0)

        # Variables pour les paramètres satellites (basés sur la nouvelle doc)
        # Defaults de la doc : sigma=2.0, low_thresh=0.1, h_thresh=0.5
        self.sat_sigma = tk.StringVar(value="2.0")
        self.sat_low_thresh = tk.StringVar(value="0.1") # !! Echelle 0-1 !!
        self.sat_h_thresh = tk.StringVar(value="0.25") # !! Echelle 0-1 !!
        # minarea n'existe plus dans cette fonction

        # Résultats
        self.analysis_results = []
        self.analysis_completed = False

        # Création de l'interface
        self.create_widgets()
        self.toggle_sat_params_state() # Appeler pour état initial

    def create_widgets(self):
        # ... (début de create_widgets identique) ...
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        config_frame.pack(fill=tk.X, pady=5)
        config_frame.columnconfigure(1, weight=1)
        ttk.Label(config_frame, text="Dossier d'entrée:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(config_frame, textvariable=self.input_dir, width=50).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W+tk.E)
        ttk.Button(config_frame, text="Parcourir", command=self.browse_input_dir).grid(row=0, column=2, padx=5, pady=2)
        ttk.Label(config_frame, text="Fichier log:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(config_frame, textvariable=self.output_log, width=50).grid(row=1, column=1, padx=5, pady=2, sticky=tk.W+tk.E)
        ttk.Button(config_frame, text="Parcourir", command=self.browse_output_log).grid(row=1, column=2, padx=5, pady=2)

        satellite_frame = ttk.LabelFrame(config_frame, text="Détection de traînées (Basée sur Hough Transform)", padding="5")
        satellite_frame.grid(row=2, column=0, columnspan=3, sticky=tk.W+tk.E, pady=5)
        satellite_frame.columnconfigure(1, weight=1)

        sat_check = ttk.Checkbutton(satellite_frame, text="Détecter les traînées de satellites",
                                    variable=self.detect_satellites, command=self.toggle_sat_params_state)
        sat_check.grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)

        acstools_frame = ttk.Frame(satellite_frame)
        acstools_frame.grid(row=0, column=1, columnspan=2, sticky=tk.W)

        if not SATDET_AVAILABLE:
             sat_check.configure(state="disabled")
             ttk.Label(acstools_frame, text="(acstools non trouvé ou incompatible)", foreground="red").pack(side=tk.LEFT, padx=5)
             # On ne propose plus d'installer si c'est juste une incompatibilité de fonction
        elif not SATDET_USES_SEARCHPATTERN:
             sat_check.configure(state="disabled")
             ttk.Label(acstools_frame, text="(fonction detsat incompatible détectée)", foreground="orange").pack(side=tk.LEFT, padx=5)
        # else: # acstools est là ET compatible
             # ttk.Label(acstools_frame, text="(acstools disponible)", foreground="green").pack(side=tk.LEFT, padx=5)


        self.move_sat_check = ttk.Checkbutton(satellite_frame, text="Déplacer les images avec traînées",
                                              variable=self.move_satellite_images, command=self.toggle_sat_params_state)
        self.move_sat_check.grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)

        self.sat_dir_label = ttk.Label(satellite_frame, text="Dossier pour images avec traînées:")
        self.sat_dir_label.grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.sat_dir_entry = ttk.Entry(satellite_frame, textvariable=self.sat_trail_dir, width=40)
        self.sat_dir_entry.grid(row=2, column=1, padx=5, pady=2, sticky=tk.W+tk.E)
        self.sat_dir_button = ttk.Button(satellite_frame, text="Parcourir", command=self.browse_sat_trail_dir)
        self.sat_dir_button.grid(row=2, column=2, padx=5, pady=2)

        # --- Paramètres satdet (nouvelle version) ---
        params_frame = ttk.Frame(satellite_frame)
        params_frame.grid(row=3, column=0, columnspan=3, sticky=tk.W+tk.E, pady=5, padx=5)

        self.sigma_label = ttk.Label(params_frame, text="Sigma (Filtre Gaussien):")
        self.sigma_label.grid(row=0, column=0, sticky=tk.W, padx=5)
        self.sigma_entry = ttk.Entry(params_frame, textvariable=self.sat_sigma, width=8)
        self.sigma_entry.grid(row=0, column=1, sticky=tk.W, padx=5)

        self.low_thresh_label = ttk.Label(params_frame, text="Low Thresh (Contour, 0-1):")
        self.low_thresh_label.grid(row=0, column=2, sticky=tk.W, padx=5)
        self.low_thresh_entry = ttk.Entry(params_frame, textvariable=self.sat_low_thresh, width=8)
        self.low_thresh_entry.grid(row=0, column=3, sticky=tk.W, padx=5)

        self.h_thresh_label = ttk.Label(params_frame, text="High Thresh (Contour, 0-1):")
        self.h_thresh_label.grid(row=1, column=0, sticky=tk.W, padx=5)
        self.h_thresh_entry = ttk.Entry(params_frame, textvariable=self.sat_h_thresh, width=8)
        self.h_thresh_entry.grid(row=1, column=1, sticky=tk.W, padx=5)

        # minarea n'est plus là
        ttk.Label(params_frame, text="(Autres params: défauts)").grid(row=1, column=2, columnspan=2, sticky=tk.W, padx=5)

        # --- Fin paramètres satdet ---

        options_frame = ttk.LabelFrame(config_frame, text="Options", padding="5")
        options_frame.grid(row=4, column=0, columnspan=3, sticky=tk.W+tk.E, pady=5) # row=4
        ttk.Checkbutton(options_frame, text="Trier les résultats par SNR décroissant", variable=self.sort_by_snr).grid(row=0, column=0, sticky=tk.W, padx=5)

        # ... (reste de create_widgets: boutons, progress, status, results_text) ...
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        ttk.Button(button_frame, text="Analyser les images", command=self.start_analysis, width=20).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Visualiser les résultats", command=self.visualize_results).pack(side=tk.LEFT, padx=5)
        self.open_log_button = ttk.Button(button_frame, text="Ouvrir le fichier log", command=self.open_log_file, width=20)
        self.open_log_button.pack(side=tk.LEFT, padx=5)
        self.open_log_button.config(state=tk.DISABLED)
        ttk.Button(button_frame, text="Quitter", command=self.root.destroy, width=10).pack(side=tk.RIGHT, padx=5)
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=5)
        self.progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=100, mode='determinate', variable=self.progress_var)
        self.progress_bar.pack(fill=tk.X, padx=5)
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X)
        ttk.Label(status_frame, textvariable=self.status_text).pack(side=tk.LEFT, padx=5)
        results_frame = ttk.LabelFrame(main_frame, text="Résultats", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, width=80, height=15)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.results_text.config(state=tk.DISABLED)

    def toggle_sat_params_state(self):
        """Active ou désactive les champs liés aux paramètres satellites."""
        # La détection globale doit être possible ET activée par l'utilisateur
        detect_enabled = self.detect_satellites.get() and SATDET_AVAILABLE and SATDET_USES_SEARCHPATTERN
        move_enabled = self.move_satellite_images.get()

        sat_param_state = tk.NORMAL if detect_enabled else tk.DISABLED
        self.sigma_label.config(state=sat_param_state)
        self.sigma_entry.config(state=sat_param_state)
        self.low_thresh_label.config(state=sat_param_state)
        self.low_thresh_entry.config(state=sat_param_state)
        self.h_thresh_label.config(state=sat_param_state)
        self.h_thresh_entry.config(state=sat_param_state)
        # minarea n'est plus là

        # Le bouton déplacer et le chemin ne dépendent que des cases à cocher
        # (même si la détection échoue ensuite, l'intention de l'utilisateur est là)
        move_check_state = tk.NORMAL if (SATDET_AVAILABLE and SATDET_USES_SEARCHPATTERN) else tk.DISABLED
        self.move_sat_check.config(state=move_check_state)

        sat_dir_state = tk.NORMAL if (move_check_state == tk.NORMAL and move_enabled) else tk.DISABLED
        self.sat_dir_label.config(state=sat_dir_state)
        self.sat_dir_entry.config(state=sat_dir_state)
        self.sat_dir_button.config(state=sat_dir_state)

    # ... (browse_input_dir, browse_output_log, browse_sat_trail_dir, open_log_file) ...
    # ... (update_status, update_progress, update_results_text) ...
    # ... (calculate_snr - reste identique) ...

# ---> AJOUTEZ CE BLOC DE MÉTHODES À L'INTÉRIEUR DE LA CLASSE AstroImageAnalyzerGUI <---
#     (Par exemple, après la méthode toggle_sat_params_state et avant run_satellite_detection)

    def browse_input_dir(self):
        directory = filedialog.askdirectory(title="Sélectionner le dossier contenant les images FITS")
        if directory:
            self.input_dir.set(directory)
            # Suggérer automatiquement un fichier log et un dossier sat dans ce dossier
            self.output_log.set(os.path.join(directory, "analyse_snr_hough.log")) # Nom de log légèrement différent
            self.sat_trail_dir.set(os.path.join(directory, "satellite_trails_hough")) # Nom de dossier différent

    def browse_output_log(self):
        filename = filedialog.asksaveasfilename(
            title="Enregistrer le fichier log",
            defaultextension=".log",
            filetypes=[("Fichiers log", "*.log"), ("Tous les fichiers", "*.*")]
        )
        if filename:
            self.output_log.set(filename)

    def browse_sat_trail_dir(self):
        directory = filedialog.askdirectory(title="Sélectionner le dossier pour les images avec traînées")
        if directory:
            self.sat_trail_dir.set(directory)

    def open_log_file(self):
        """Ouvre le fichier log avec l'application par défaut du système"""
        log_path = self.output_log.get()
        if not log_path or not os.path.exists(log_path):
            messagebox.showerror("Erreur", "Le fichier log n'existe pas ou n'est pas spécifié.")
            return

        try:
            if platform.system() == 'Windows':
                os.startfile(log_path)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.call(['open', log_path])
            else:  # Linux et autres
                subprocess.call(['xdg-open', log_path])
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'ouvrir le fichier log '{log_path}':\n{str(e)}")

    def update_status(self, message):
        # Assurer l'exécution dans le thread principal de Tkinter si appelé depuis un autre thread
        self.root.after(0, self.status_text.set, message)
        # self.root.update_idletasks() # Peut causer des pbs si appelé trop souvent depuis thread, 'after' est plus sûr

    def update_progress(self, value):
        # Assurer l'exécution dans le thread principal de Tkinter
        self.root.after(0, self.progress_var.set, value)
        # self.root.update_idletasks()
        # Cette fonction est appelée via root.after, donc elle s'exécute dans le thread principal.
        # Met à jour la variable Tkinter liée à la barre de progression
        self.progress_var.set(value)

        # Force Tkinter à traiter les événements IDLE en attente,
        # ce qui inclut le redessin des widgets dont l'état a changé.
        # C'est souvent la clé pour voir la mise à jour visuelle.
        try:
            # Vérifier si la racine existe toujours (au cas où la fenêtre serait fermée pendant l'analyse)
            if self.root.winfo_exists():
                 self.root.update_idletasks()
        except tk.TclError:
             # Ignorer l'erreur si la fenêtre a été détruite entre le check et l'appel
             pass

    def update_results_text(self, text, clear=False):
        # Fonction interne pour la mise à jour dans le thread Tkinter
        def _update_text():
            self.results_text.config(state=tk.NORMAL)
            if clear:
                self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, text + "\n")
            self.results_text.see(tk.END) # Faire défiler vers le bas
            self.results_text.config(state=tk.DISABLED)
            # self.root.update_idletasks() # update_idletasks n'est généralement pas nécessaire avec 'after'

        # Assurer l'exécution dans le thread principal de Tkinter
        self.root.after(0, _update_text)

    def calculate_snr(self, data):
        """Calcule le SNR d'une image"""
        try:
             # S'assurer que data est un tableau numpy de floats pour les stats
             # Utiliser float64 pour plus de précision dans les stats
            data_float = np.array(data, dtype=np.float64)

            # Utiliser sigma_clipped_stats pour une estimation robuste du fond et du bruit
            # maxiters=5 est une bonne valeur par défaut
            mean, median, std = sigma_clipped_stats(data_float, sigma=3.0, maxiters=5)

            # Définir un seuil pour les pixels considérés comme "signal"
            # Utiliser la médiane + 5*std est souvent un bon point de départ
            threshold = median + 5 * std
            signal_mask = data_float > threshold

            # Fallback si très peu de pixels dépassent ce seuil (image faible ou très bruitée)
            # On utilise alors un percentile élevé comme seuil
            if np.sum(signal_mask) < 10: # Si moins de 10 pixels dépassent 5 sigma
                # Utiliser un percentile élevé (ex: 95ème) pour définir le signal
                # Attention: sur une image sans objet brillant, cela peut sélectionner du bruit
                signal_threshold = np.percentile(data_float, 95)
                signal_mask = data_float > signal_threshold
                # On pourrait même ajouter un message si ce fallback est utilisé
                # print(f"DEBUG: Fallback percentile 95 utilisé pour le masque de signal (seuil 5 sigma non atteint)")


            # Calculer la valeur moyenne du signal AU-DESSUS du fond (médiane)
            if np.sum(signal_mask) > 0:
                # On prend la moyenne des pixels du masque et on soustrait la médiane (fond)
                signal_value = np.mean(data_float[signal_mask]) - median
            else:
                # Si aucun pixel n'est au-dessus du seuil (même après fallback),
                # considérer le signal comme nul par rapport au bruit.
                signal_value = 0.0

            # Calculer le SNR: (Signal au-dessus du fond) / Bruit
            # Gérer le cas où l'écart-type est nul (image complètement plate ou constante)
            if std > 1e-9: # Utiliser une petite tolérance pour éviter division par quasi-zéro
                snr = signal_value / std
            else:
                snr = 0.0 # SNR est 0 si pas de bruit ou pas de signal

            # Nombre de pixels contribuant au signal (peut être utile pour l'évaluation)
            num_signal_pixels = np.sum(signal_mask)

            # Retourner les valeurs calculées
            return snr, median, std, num_signal_pixels

        except Exception as e:
             # Gérer les erreurs potentielles (ex: données non numériques, tableau vide?)
             self.update_results_text(f"Erreur dans calculate_snr: {e}")
             # Imprimer l'erreur complète dans la console pour débogage
             traceback.print_exc()
             # Retourner des valeurs par défaut ou NaN pour indiquer l'échec
             return 0.0, 0.0, 0.0, 0


    def install_acstools(self):
        """Installe la bibliothèque acstools via pip"""
        # Note: Cette fonction n'est plus liée à un bouton si acstools est déjà installé ou incompatible
        self.update_status("Tentative d'installation de acstools...")
        # Désactiver le bouton s'il existe et est visible (normalement non dans ce cas)
        # if hasattr(self, 'install_button') and self.install_button.winfo_exists():
        #     self.install_button.config(state=tk.DISABLED)

        def run_installation():
            try:
                pip_cmd = [sys.executable, "-m", "pip", "install", "acstools"]
                process = subprocess.Popen(
                    pip_cmd,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    universal_newlines=True, creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == 'Windows' else 0 # Cache la fenêtre console sous Windows
                )
                stdout, stderr = process.communicate()

                if process.returncode == 0:
                    self.update_status("Installation de acstools réussie! Redémarrage nécessaire.")
                    self.update_results_text("\nInstallation de acstools réussie!\nVeuillez redémarrer l'application.", clear=True)
                    messagebox.showinfo("Installation terminée",
                                       "acstools a été installé. Veuillez redémarrer l'application.")
                    # On ne peut pas réactiver la détection dynamiquement facilement ici,
                    # il faut vraiment redémarrer pour que l'import et la vérification au début fonctionnent.
                else:
                    self.update_status("Échec de l'installation de acstools")
                    self.update_results_text(f"\nÉchec de l'installation de acstools. Erreur:\n{stderr}", clear=True)
                    messagebox.showerror("Erreur d'installation",
                                        f"L'installation a échoué. Voir les logs ou essayer manuellement:\n{' '.join(pip_cmd)}")
                    # Réactiver le bouton si il existe
                    # if hasattr(self, 'install_button') and self.install_button.winfo_exists():
                    #     self.install_button.config(state=tk.NORMAL)

            except Exception as e:
                self.update_status(f"Erreur installation: {str(e)}")
                self.update_results_text(f"\nErreur lors de l'installation: {str(e)}", clear=True)
                messagebox.showerror("Erreur", f"Une erreur est survenue: {str(e)}")
                # Réactiver le bouton si il existe
                # if hasattr(self, 'install_button') and self.install_button.winfo_exists():
                #     self.install_button.config(state=tk.NORMAL)

        # Lancer l'installation dans un thread séparé
        threading.Thread(target=run_installation, daemon=True).start()


    def run_satellite_detection(self, search_pattern):
        """
        Exécute acstools.satdet.detsat sur un ensemble de fichiers.
        Retourne les dictionnaires de résultats et d'erreurs.
        """
        if not (SATDET_AVAILABLE and SATDET_USES_SEARCHPATTERN):
             self.update_results_text("Avertissement: Détection de satellites non disponible ou incompatible.")
             # Utiliser root.after pour être sûr que le message s'affiche même si appelé depuis thread
             self.root.after(0, self.update_results_text, "Avertissement: Détection de satellites non disponible ou incompatible.")
             return {}, {} # Retourne des dict vides

        # --- Récupérer et valider les paramètres depuis l'interface ---
        try:
            sigma = float(self.sat_sigma.get())
            if sigma <= 0: raise ValueError("Sigma doit être positif")
        except ValueError as e:
            self.root.after(0, self.update_results_text, f"Avertissement: Sigma invalide ({e}), utilisation de 2.0")
            sigma = 2.0

        try:
            low_thresh = float(self.sat_low_thresh.get())
            if not (0 <= low_thresh <= 1): raise ValueError("Doit être entre 0 et 1")
        except ValueError as e:
            self.root.after(0, self.update_results_text, f"Avertissement: Low Thresh invalide ({e}), utilisation de 0.1")
            low_thresh = 0.1

        try:
            h_thresh = float(self.sat_h_thresh.get())
            if not (0 <= h_thresh <= 1): raise ValueError("Doit être entre 0 et 1")
            if h_thresh < low_thresh: raise ValueError("Doit être >= Low Thresh")
        except ValueError as e:
            self.root.after(0, self.update_results_text, f"Avertissement: High Thresh invalide ({e}), utilisation de 0.5")
            h_thresh = 0.5
            # S'assurer que la valeur par défaut est toujours valide par rapport à low_thresh
            if h_thresh < low_thresh:
                h_thresh = low_thresh

        # --- Paramètres fixes ou à tester ---
        chips_to_use = [0]    # Extension FITS à analyser (0 pour SeeStar a priori)
        n_processes = 1       # Multiprocessing désactivé (plus simple)
        verbose_det = False   # Mettre à True pour plus de logs de satdet lui-même

        # --- VALEURS À AJUSTER POUR LA DÉTECTION ---
        # Essayez de diminuer ces valeurs si la détection échoue
        test_line_len = 150      # Longueur min. segment (défaut acstools: 200 - souvent trop grand)
        test_small_edge = 60    # Taille min. contour (défaut acstools: 60)
        test_line_gap = 75      # Écart max autorisé (défaut acstools: 75)
        # ------------------------------------------

        # Mise à jour du statut (via root.after pour thread-safety)
        self.root.after(0, self.update_status, "Lancement de la détection de traînées (peut prendre du temps)...")
        # Affichage des paramètres utilisés (via root.after pour thread-safety)
        params_msg = (f"Détection avec: sigma={sigma}, low={low_thresh}, high={h_thresh}, "
                      f"chips={chips_to_use}, line_len={test_line_len}, small_edge={test_small_edge}")
        self.root.after(0, self.update_results_text, params_msg)

        results = {}
        errors = {}
        try:
            # Utiliser warnings.catch_warnings pour ignorer l'avertissement sur l'extension 0
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action='ignore',
                    message=r'.*is not a valid science extension for ACS/WFC.*',
                    # Vous pouvez ajouter category=UserWarning si besoin d'être plus spécifique
                )

                # Appel à la fonction satdet avec tous les paramètres
                results, errors = satdet.detsat(
                    searchpattern=search_pattern,
                    chips=chips_to_use,
                    n_processes=n_processes,
                    sigma=sigma,
                    low_thresh=low_thresh,
                    h_thresh=h_thresh,
                    # Paramètres supplémentaires ajustés
                    line_len=test_line_len,
                    small_edge=test_small_edge,
                    line_gap=test_line_gap,
                    # Autres paramètres laissés par défaut (percentile, buf...)
                    plot=False,       # Plotting désactivé
                    verbose=verbose_det
                )

            # Mise à jour du statut à la fin (via root.after)
            self.root.after(0, self.update_status, "Détection de traînées terminée.")

            # Afficher les erreurs spécifiques de satdet (sauf celle qu'on a ignorée)
            if errors:
                 # Utiliser une fonction pour mettre à jour le texte via root.after
                 def report_errors():
                    self.update_results_text("Erreurs spécifiques reportées par satdet:")
                    count = 0
                    for (fname, ext), msg in errors.items():
                        if "is not a valid science extension for ACS/WFC" not in str(msg):
                             self.update_results_text(f"  - {os.path.basename(fname)} (ext {ext}): {msg}")
                             count += 1
                    if count == 0:
                        self.update_results_text("  (Aucune erreur pertinente à afficher)")

                 self.root.after(0, report_errors)

            return results, errors

        except ImportError as imp_err:
             # Erreur spécifique si scipy ou skimage manque
             err_msg = f"Erreur d'importation nécessaire pour satdet: {imp_err}. Assurez-vous que 'scipy' et 'scikit-image' sont installés."
             self.root.after(0, self.update_results_text, err_msg)
             self.root.after(0, self.update_status, "Erreur dépendance détection.")
             return {}, {'IMPORT_ERROR': err_msg}
        except Exception as e:
            # Gérer toute autre erreur majeure pendant l'appel satdet
            err_msg = f"Erreur majeure lors de l'appel à acstools.satdet.detsat: {str(e)}"
            self.root.after(0, self.update_results_text, err_msg)
            # Imprimer la trace complète dans la console pour un débogage avancé
            print("\n--- Traceback Erreur Satdet ---")
            traceback.print_exc()
            print("-----------------------------\n")
            self.root.after(0, self.update_status, "Erreur lors de la détection de traînées.")
            return {}, {'FATAL_ERROR': str(e)} # Retourner une erreur fatale
        
    # Assurez-vous d'avoir 'import time' en haut de votre fichier .py

    def analyze_images(self):
        """Fonction principale d'analyse des images"""
        input_dir = self.input_dir.get()
        output_log = self.output_log.get()
        sat_trail_dir = self.sat_trail_dir.get() if self.move_satellite_images.get() else None
        # Vérifie si la détection est possible ET activée par l'utilisateur
        detect_sat = self.detect_satellites.get() and SATDET_AVAILABLE and SATDET_USES_SEARCHPATTERN
        move_sat = self.move_satellite_images.get()

        # Désactiver le bouton log pendant l'analyse
        # Utiliser root.after pour thread-safety si appelé depuis un thread autre que main
        self.root.after(0, self.open_log_button.config, {'state': tk.DISABLED})
        self.analysis_completed = False

        # Vérifications initiales des chemins et création du dossier satellite
        if not input_dir or not os.path.isdir(input_dir):
             messagebox.showerror("Erreur", "Le dossier d'entrée n'existe pas ou n'est pas valide.")
             self.root.after(0, self.update_status, "Erreur: Dossier d'entrée invalide.")
             return
        if not output_log:
             messagebox.showerror("Erreur", "Veuillez spécifier un fichier log.")
             self.root.after(0, self.update_status, "Erreur: Fichier log non spécifié.")
             return
        if detect_sat and move_sat:
            if not sat_trail_dir:
                 messagebox.showerror("Erreur", "Veuillez spécifier un dossier pour les images avec traînées.")
                 self.root.after(0, self.update_status, "Erreur: Dossier pour traînées non spécifié.")
                 return
            # Essayer de créer le dossier s'il n'existe pas
            if not os.path.exists(sat_trail_dir):
                try:
                    os.makedirs(sat_trail_dir)
                    self.root.after(0, self.update_results_text, f"Dossier créé: {sat_trail_dir}")
                except OSError as e:
                     messagebox.showerror("Erreur", f"Impossible de créer le dossier {sat_trail_dir}:\n{e}")
                     self.root.after(0, self.update_status, f"Erreur création dossier: {e}")
                     return

        # --- Étape 1: Détection des satellites (si activée) ---
        # Cette étape s'exécute en premier et peut prendre du temps
        sat_results = {}
        sat_errors = {}
        if detect_sat:
            # Construire le search pattern pour satdet
            # Essayer différents patterns communs pour FITS
            patterns_to_try = ["*.fit*", "*.fits", "*.fit"]
            search_pattern = None
            for pattern in patterns_to_try:
                full_pattern = os.path.join(input_dir, pattern)
                if glob.glob(full_pattern): # Vérifie si au moins un fichier correspond
                    search_pattern = full_pattern
                    break # Utiliser le premier pattern qui trouve des fichiers

            if search_pattern:
                 self.root.after(0, self.update_status, "Lancement détection traînées...") # Info avant appel long
                 sat_results, sat_errors = self.run_satellite_detection(search_pattern)
                 # Statut mis à jour dans run_satellite_detection
            else:
                 # Utiliser root.after pour afficher le message depuis le thread
                 self.root.after(0, self.update_results_text, f"Avertissement: Aucun fichier FITS trouvé dans {input_dir} pour la détection de traînées.")
                 self.root.after(0, self.update_status, "Détection traînées: Aucun fichier trouvé.")


        # --- Étape 2: Analyse individuelle (SNR, métadonnées, déplacement) ---
        # Obtenir la liste finale des fichiers FITS présents dans le dossier
        # (peut être différent si satdet est très lent ou si des fichiers sont ajoutés/supprimés)
        fits_files = glob.glob(os.path.join(input_dir, "*.fit")) + glob.glob(os.path.join(input_dir, "*.fits"))

        if not fits_files:
            # S'il n'y a pas de fichiers, afficher un message et écrire le résumé du log (qui contiendra au moins les infos satdet si elles existent)
            messagebox.showwarning("Attention", f"Aucun fichier FITS (.fit, .fits) trouvé dans {input_dir} pour l'analyse SNR.")
            self.root.after(0, self.update_status, "Analyse SNR: Aucun fichier trouvé.")
            # Écrire le log même si l'analyse SNR n'a pas eu lieu (peut contenir des infos satdet)
            self.write_log_summary(output_log, input_dir, detect_sat, move_sat, sat_trail_dir, sat_errors)
            # Activer le bouton log s'il existe un fichier log
            if os.path.exists(output_log):
                self.root.after(0, self.open_log_button.config, {'state': tk.NORMAL})
            return

        # Réinitialiser les résultats pour cette analyse
        self.analysis_results = []
        # Ne pas effacer les messages précédents si satdet a déjà écrit
        self.root.after(0, self.update_results_text, "Démarrage de l'analyse individuelle (SNR...)...", (not detect_sat))

        processed_count = 0
        moved_count = 0
        error_count = 0

        try:
            # Ouvrir le fichier log en mode écriture (écrase le précédent)
            with open(output_log, 'w') as log_file:
                # Écrire l'en-tête du log
                log_file.write(f"Analyse effectuée le {datetime.datetime.now()}\n")
                log_file.write(f"Dossier d'entrée: {input_dir}\n")
                log_file.write("="*80 + "\n")
                if detect_sat:
                    # Log des paramètres de détection utilisés
                    log_file.write(f"Détection traînées (Hough): sigma={self.sat_sigma.get()}, low={self.sat_low_thresh.get()}, high={self.sat_h_thresh.get()}\n")
                    # Lire les paramètres line_len, small_edge, etc. (ils sont fixés dans run_satellite_detection pour l'instant)
                    # Pourrait être amélioré en les passant en argument ou en les stockant
                    # Ici on met les valeurs connues du code précédent:
                    log_file.write(f"  (Paramètres internes approx: line_len=20, small_edge=10, line_gap=75, percentile=(1.0, 99.0))\n")

                    if sat_errors:
                         log_file.write("Erreurs reportées par satdet:\n")
                         for key, msg in sat_errors.items():
                             if "is not a valid science extension for ACS/WFC" not in str(msg):
                                 log_file.write(f"  - {key}: {msg}\n")
                    if move_sat: log_file.write(f"Déplacement activé vers: {sat_trail_dir}\n")
                    log_file.write("="*80 + "\n")

                # Écrire les en-têtes de colonnes
                log_file.write("Fichier\tSNR\tFond\tBruit\tPixelsSig")
                if detect_sat: log_file.write("\tTraînées (seg.)\tNb Segments")
                log_file.write("\tExpo\tFiltre\tTemp\n")

                # Boucle principale sur les fichiers FITS
                total_files = len(fits_files)
                for i, fits_file_path in enumerate(fits_files):
                    # Vérifier si le fichier existe toujours (au cas où il aurait été déplacé par un autre processus?)
                    if not os.path.exists(fits_file_path):
                        continue # Ignorer si le fichier a disparu

                    progress = ((i + 1) / total_files) * 100 # Progression de 1 à 100
                    file_name = os.path.basename(fits_file_path)
                    self.root.after(0, self.update_progress, progress)
                    self.root.after(0, self.update_status, f"Analyse: {file_name} ({i+1}/{total_files})")

                    try:
                        # --- Recherche résultat satellite pour CE fichier ---
                        has_trails = False
                        num_trails = 0 # Nombre de segments de ligne trouvés
                        if detect_sat and sat_results: # Vérifier si sat_results existe
                            # Chercher la clé (filename, ext=0)
                            found_key = None
                            for key_tuple in sat_results.keys():
                                # Comparaison robuste (insensible à la casse sur Windows?) et sur le nom de base
                                if os.path.normcase(os.path.basename(key_tuple[0])) == os.path.normcase(file_name) and key_tuple[1] == 0:
                                    found_key = key_tuple
                                    break
                                # Fallback chemin complet
                                elif os.path.normcase(key_tuple[0]) == os.path.normcase(fits_file_path) and key_tuple[1] == 0:
                                    found_key = key_tuple
                                    break

                            if found_key:
                                trail_segments = sat_results[found_key]
                                # Vérifier si on a bien reçu une liste/array non vide
                                if isinstance(trail_segments, (list, np.ndarray)) and len(trail_segments) > 0:
                                    has_trails = True
                                    num_trails = len(trail_segments) # Nombre de segments

                        # --- Analyse SNR et métadonnées ---
                        # Utiliser le contexte 'with' pour garantir la fermeture du fichier
                        with fits.open(fits_file_path) as hdul:
                            # Vérifier si la HDU primaire contient des données
                            if not hdul or len(hdul) == 0 or hdul[0].data is None:
                                error_msg = f"Erreur: {file_name} - Pas de données image trouvées dans HDU 0."
                                self.root.after(0, self.update_results_text, error_msg)
                                log_file.write(f"{file_name}\tERREUR: Pas de données HDU 0\n")
                                error_count += 1
                                continue # Passer au fichier suivant

                            # Lire les données et l'en-tête
                            data = hdul[0].data
                            header = hdul[0].header
                            exposure = header.get('EXPTIME', 'N/A')
                            filter_name = header.get('FILTER', 'N/A')
                            temperature = header.get('CCD-TEMP', 'N/A')

                            # Calculer le SNR
                            snr, sky_bg, sky_noise, signal_pixels = self.calculate_snr(data)

                        # ---> Le fichier est fermé ici automatiquement par le 'with' <---

                        # Stocker les résultats pour ce fichier
                        result = {
                            'file': file_name, 'path': fits_file_path, 'snr': snr,
                            'sky_bg': sky_bg, 'sky_noise': sky_noise,
                            'signal_pixels': signal_pixels, 'has_trails': has_trails,
                            'num_trails': num_trails, 'exposure': exposure,
                            'filter': filter_name, 'temperature': temperature
                        }
                        self.analysis_results.append(result)
                        processed_count += 1

                        # Écrire la ligne de résultat dans le log
                        log_line = f"{file_name}\t{snr:.2f}\t{sky_bg:.2f}\t{sky_noise:.2f}\t{signal_pixels}"
                        if detect_sat:
                            log_line += f"\t{'Oui' if has_trails else 'Non'}\t{num_trails}"
                        log_line += f"\t{exposure}\t{filter_name}\t{temperature}\n"
                        log_file.write(log_line)

                        # Afficher les résultats dans l'interface (via root.after)
                        snr_info = f"  {file_name}: SNR={snr:.2f}, Fond={sky_bg:.2f}"
                        self.root.after(0, self.update_results_text, snr_info)
                        if detect_sat:
                            trail_info = f"    Traînées (segments): {'Oui' if has_trails else 'Non'} ({num_trails})"
                            self.root.after(0, self.update_results_text, trail_info)

                        # --- Déplacement du fichier (SI NÉCESSAIRE) ---
                        if has_trails and move_sat and sat_trail_dir:
                            try:
                                # Donner un court instant au système pour libérer le fichier
                                time.sleep(0.1) # Pause de 100ms

                                dest_path = os.path.join(sat_trail_dir, file_name)
                                # S'assurer que la source existe toujours avant de déplacer
                                if os.path.exists(fits_file_path):
                                     shutil.move(fits_file_path, dest_path)
                                     move_info = f"    -> Déplacé vers {sat_trail_dir}"
                                     self.root.after(0, self.update_results_text, move_info)
                                     # Mettre à jour le chemin dans nos résultats stockés
                                     result['path'] = dest_path
                                     moved_count += 1
                                else:
                                     # Le fichier a peut-être été déplacé entre temps? Loguer une info.
                                     skip_info = f"    Info: {file_name} n'existait plus pour déplacement."
                                     self.root.after(0, self.update_results_text, skip_info)


                            except Exception as move_err:
                                # Afficher l'erreur via root.after pour thread safety
                                error_message = f"    Erreur déplacement {file_name}: {move_err}"
                                self.root.after(0, self.update_results_text, error_message)
                                log_file.write(f"ERREUR_DEPLACEMENT\t{file_name}\t{move_err}\n")
                                # Ne pas incrémenter error_count ici car l'analyse a réussi
                        # --- FIN DU BLOC DÉPLACEMENT ---

                    except Exception as e_file:
                         # Gérer les erreurs pendant l'analyse d'un fichier spécifique
                         error_msg = f"Erreur analyse fichier {file_name}: {e_file}"
                         self.root.after(0, self.update_results_text, error_msg)
                         log_file.write(f"{file_name}\tERREUR_ANALYSE: {e_file}\n")
                         error_count += 1
                         # Imprimer la trace pour débogage console
                         print(f"\n--- Traceback Erreur Fichier {file_name} ---")
                         traceback.print_exc()
                         print("-------------------------------------------\n")

                # --- Fin de la boucle sur les fichiers ---

                # Écrire le résumé dans le fichier log (à la fin du fichier)
                self.write_log_summary(log_file, input_dir, detect_sat, move_sat, sat_trail_dir,
                                       sat_errors=None, # Erreurs satdet déjà loguées au début
                                       results_list=self.analysis_results,
                                       moved_count=moved_count)

            # Le fichier log est fermé ici automatiquement par le 'with'

        except IOError as e_log:
             # Erreur si le fichier log n'a pas pu être ouvert/écrit
             messagebox.showerror("Erreur Fichier Log", f"Impossible d'écrire dans le fichier log {output_log}:\n{e_log}")
             self.root.after(0, self.update_status, "Erreur écriture log")
             # Pas de bouton log à activer si l'écriture a échoué
             return # Arrêter ici

        # --- Finalisation après la boucle et la fermeture du log ---
        self.root.after(0, self.update_progress, 100)
        final_status = f"Analyse terminée. {processed_count} traités, {moved_count} déplacés, {error_count} erreurs fichier."
        self.root.after(0, self.update_status, final_status)
        self.root.after(0, self.update_results_text, f"\n{final_status}") # Répéter dans la zone de texte

        # Afficher les stats globales dans l'interface si des images ont été traitées
        if self.analysis_results:
            all_snrs = [r['snr'] for r in self.analysis_results if 'snr' in r and isinstance(r['snr'], (int, float))]
            if all_snrs:
                 snr_mean = np.mean(all_snrs)
                 self.root.after(0, self.update_results_text, f"SNR moyen: {snr_mean:.2f}")
            if detect_sat:
                sat_count = sum(1 for r in self.analysis_results if r.get('has_trails', False))
                percentage = (sat_count / processed_count) * 100 if processed_count > 0 else 0
                self.root.after(0, self.update_results_text, f"Images avec traînées détectées: {sat_count} ({percentage:.1f}%)")

            # Activer le bouton pour ouvrir le log
            self.root.after(0, self.open_log_button.config, {'state': tk.NORMAL})
            self.analysis_completed = True
        else:
             # Si aucune image n'a été traitée avec succès
             self.root.after(0, self.update_results_text, "Aucune image n'a pu être traitée avec succès.")
             # Laisser le bouton log désactivé ou le réactiver si le log existe quand même ?
             # Réactiver si le fichier log existe (il contient au moins l'en-tête/infos satdet)
             if os.path.exists(output_log):
                  self.root.after(0, self.open_log_button.config, {'state': tk.NORMAL})
             self.analysis_completed = False

        # TODO: Réactiver les boutons Analyse/Visualiser qui auraient pu être désactivés au début

    def write_log_summary(self, log_file_or_path, input_dir, detect_sat, move_sat, sat_trail_dir, sat_errors=None, results_list=None, moved_count=0):
        """Écrit le résumé dans le fichier log (peut être appelé même si l'analyse SNR échoue)."""
        is_path = isinstance(log_file_or_path, str)
        try:
            log_file = open(log_file_or_path, 'a') if is_path else log_file_or_path

            log_file.write("\n" + "="*80 + "\n")
            log_file.write("Résumé de l'analyse:\n")

            if results_list is None: # Cas où on écrit juste le log de satdet
                 log_file.write("Aucune analyse SNR individuelle effectuée.\n")
                 if sat_errors:
                     log_file.write("Erreurs reportées par satdet:\n")
                     for key, msg in sat_errors.items(): log_file.write(f"  - {key}: {msg}\n")

            else: # Cas analyse complète
                total_analyzed = len(results_list)
                log_file.write(f"Nombre total d'images analysées (SNR): {total_analyzed}\n")

                if total_analyzed > 0:
                    all_snrs = [r['snr'] for r in results_list if 'snr' in r]
                    if detect_sat:
                        sat_count = sum(1 for r in results_list if r.get('has_trails', False))
                        percentage_sat = (sat_count / total_analyzed) * 100
                        log_file.write(f"Images avec traînées détectées: {sat_count} ({percentage_sat:.1f}%)\n")

                    if all_snrs:
                        log_file.write(f"SNR moyen: {np.mean(all_snrs):.2f}, médian: {np.median(all_snrs):.2f}, min: {min(all_snrs):.2f}, max: {max(all_snrs):.2f}\n")

                    if move_sat and moved_count > 0:
                        log_file.write(f"{moved_count} images avec traînées déplacées vers: {sat_trail_dir}\n")

                    # Recommandations
                    log_file.write("\n" + "="*80 + "\n")
                    log_file.write("Recommandations pour l'empilement Winsorised Kappa-Sigma:\n")
                    if len(results_list) >= 10 and all_snrs:
                        # Trier pour la recommandation
                        results_list.sort(key=lambda x: x.get('snr', -1), reverse=True)
                        percentile_75 = np.percentile(all_snrs, 75)
                        good_images = [r for r in results_list if r.get('snr', -float('inf')) >= percentile_75]
                        log_file.write(f"Utiliser les {len(good_images)} meilleures images (SNR >= {percentile_75:.2f}):\n")
                        for img in good_images:
                             log_file.write(f"  {img.get('file', 'N/A')} (SNR: {img.get('snr', 0.0):.2f})\n")
                    elif len(results_list) > 0:
                        log_file.write("Moins de 10 images, utilisez toutes les images valides.\n")
                    else:
                        log_file.write("Aucune image valide pour recommandation.\n")
                else:
                     log_file.write("Aucune image analysée avec succès.\n")

        except Exception as e:
             print(f"Erreur lors de l'écriture du résumé du log: {e}")
        finally:
            if is_path and 'log_file' in locals() and log_file:
                log_file.close()


    def start_analysis(self):
        """Démarre l'analyse dans un thread séparé"""
        if self.detect_satellites.get() and not (SATDET_AVAILABLE and SATDET_USES_SEARCHPATTERN):
             messagebox.showerror("Erreur", "Détection activée mais acstools est absent, incompatible, ou sa fonction detsat ne correspond pas à la documentation.")
             return

        # TODO: Désactiver les boutons Analyse/Visualiser ici
        self.update_status("Démarrage de l'analyse...")
        threading.Thread(target=self.analyze_images, daemon=True).start()
        # TODO: Réactiver les boutons à la fin de analyze_images (même en cas d'erreur)


    # --- visualize_results reste TRÈS SIMILAIRE ---
    # La seule différence est que 'num_trails' représente maintenant des segments
    # et le camembert 'Avec/Sans traînées' reste valide.
    # Pas de changement majeur nécessaire ici, juste s'assurer qu'il utilise
    # self.analysis_results qui est rempli par la nouvelle logique.
    def visualize_results(self):
        """Visualise les résultats avec des graphiques"""
        if not self.analysis_results:
            messagebox.showinfo("Information", "Aucun résultat à visualiser.")
            return
        if not self.analysis_completed:
             messagebox.showwarning("Attention", "Analyse incomplète ou échouée.")
             # return # Décommenter pour empêcher la visu si analyse non complétée

        try:
            vis_window = tk.Toplevel(self.root)
            vis_window.title("Visualisation des résultats")
            vis_window.geometry("800x650")
            notebook = ttk.Notebook(vis_window)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # --- Onglet Distribution SNR (inchangé) ---
            try:
                snr_tab = ttk.Frame(notebook); notebook.add(snr_tab, text="Distribution SNR")
                fig1, ax1 = plt.subplots(); all_snrs = [r['snr'] for r in self.analysis_results if 'snr' in r]
                if all_snrs: ax1.hist(all_snrs, bins=20); ax1.set_title('Distribution SNR'); ax1.set_xlabel('SNR'); ax1.set_ylabel('Nb images')
                else: ax1.text(0.5, 0.5, "N/A", ha='center', va='center')
                canvas1 = FigureCanvasTkAgg(fig1, master=snr_tab); canvas1.draw(); canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            except Exception as e: print(f"Err SNR Hist: {e}"); ttk.Label(snr_tab, text=f"Err:\n{e}").pack()

            # --- Onglet Comparaison SNR (inchangé) ---
            try:
                comp_tab = ttk.Frame(notebook); notebook.add(comp_tab, text="Comparaison SNR")
                valid_res = [r for r in self.analysis_results if 'snr' in r and 'file' in r]; sorted_res = sorted(valid_res, key=lambda x: x['snr'], reverse=True)
                num_show = min(10, len(sorted_res) // 2)
                if num_show > 0:
                    best = sorted_res[:num_show]; worst = sorted_res[-num_show:]
                    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 6))
                    ax2.barh([r['file'] for r in best], [r['snr'] for r in best], color='g'); ax2.set_title(f'Top {num_show}'); ax2.invert_yaxis()
                    ax3.barh([r['file'] for r in worst], [r['snr'] for r in worst], color='r'); ax3.set_title(f'Bottom {num_show}'); ax3.invert_yaxis()
                    fig2.tight_layout(); canvas2 = FigureCanvasTkAgg(fig2, master=comp_tab); canvas2.draw(); canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                else: ttk.Label(comp_tab, text="Pas assez d'images pour comparer.").pack()
            except Exception as e: print(f"Err Comp SNR: {e}"); ttk.Label(comp_tab, text=f"Err:\n{e}").pack()

            # --- Onglet Satellites (inchangé conceptuellement) ---
            detect_sat_enabled = self.detect_satellites.get() and SATDET_AVAILABLE and SATDET_USES_SEARCHPATTERN
            has_sat_results = any('has_trails' in r for r in self.analysis_results)
            if detect_sat_enabled and has_sat_results:
                try:
                    sat_tab = ttk.Frame(notebook); notebook.add(sat_tab, text="Traînées")
                    sat_count = sum(1 for r in self.analysis_results if r.get('has_trails', False))
                    no_sat_count = len(self.analysis_results) - sat_count
                    if sat_count > 0 or no_sat_count > 0:
                        fig3, ax4 = plt.subplots(); ax4.pie([no_sat_count, sat_count], labels=['Sans', 'Avec'], autopct='%1.1f%%', colors=['lightblue', 'salmon'], explode=(0, 0.1 if sat_count > 0 else 0))
                        ax4.set_title('Images avec/sans traînées (segments Hough)')
                        canvas3 = FigureCanvasTkAgg(fig3, master=sat_tab); canvas3.draw(); canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                    else: ttk.Label(sat_tab, text="N/A").pack()
                except Exception as e: print(f"Err Sat Pie: {e}"); ttk.Label(sat_tab, text=f"Err:\n{e}").pack()

            # --- Onglet Données brutes (adapter pour num_trails -> Nb Segments) ---
            try:
                data_tab = ttk.Frame(notebook); notebook.add(data_tab, text="Données brutes")
                cols = ("Fichier", "SNR", "Fond", "Bruit", "PixSig")
                if detect_sat_enabled: cols = cols + ("Traînées", "Nb Seg.") # Changé nom colonne
                tree = ttk.Treeview(data_tab, columns=cols, show='headings')
                for col in cols: tree.heading(col, text=col); tree.column(col, width=80, anchor='center')
                tree.column("Fichier", width=200, anchor='w')
                display_res = sorted(self.analysis_results, key=lambda x: x.get('snr', -1), reverse=True) if self.sort_by_snr.get() else self.analysis_results
                for r in display_res:
                    vals = (r.get('file','?'), f"{r.get('snr',0):.2f}", f"{r.get('sky_bg',0):.2f}", f"{r.get('sky_noise',0):.2f}", f"{r.get('signal_pixels',0)}")
                    if detect_sat_enabled: vals = vals + (f"{'Oui' if r.get('has_trails',False) else 'Non'}", f"{r.get('num_trails',0)}")
                    tree.insert('', tk.END, values=vals)
                scr = ttk.Scrollbar(data_tab, orient=tk.VERTICAL, command=tree.yview); tree.configure(yscroll=scr.set); scr.pack(side=tk.RIGHT, fill=tk.Y); tree.pack(fill=tk.BOTH, expand=True)
            except Exception as e: print(f"Err Data Tree: {e}"); ttk.Label(data_tab, text=f"Err:\n{e}").pack()

# --- Onglet Recommandations (inchangé) ---
            try:
                if len(self.analysis_results) >= 5:
                    stack_tab = ttk.Frame(notebook)
                    notebook.add(stack_tab, text="Recommandations")
                    recom_frame = ttk.LabelFrame(stack_tab, text="Winsorised Kappa-Sigma", padding=10)
                    recom_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                    all_snrs = [r['snr'] for r in self.analysis_results if 'snr' in r]
                    if all_snrs:
                        p75 = np.percentile(all_snrs, 75)
                        good_img = sorted([r for r in self.analysis_results if r.get('snr', -float('inf')) >= p75], key=lambda x: x['snr'], reverse=True)
                        ttk.Label(recom_frame, text=f"Utiliser les {len(good_img)} meilleures images (SNR >= {p75:.2f}):").pack(anchor=tk.W)
                        rec_tree = ttk.Treeview(recom_frame, columns=("Fichier", "SNR"), show='headings', height=10)
                        rec_tree.heading("Fichier", text="Fichier")
                        rec_tree.heading("SNR", text="SNR")
                        rec_tree.column("Fichier", width=350)
                        rec_tree.column("SNR", width=100)
                        for img in good_img:
                            rec_tree.insert('', tk.END, values=(img.get('file', '?'), f"{img.get('snr', 0):.2f}"))
                        rec_scr = ttk.Scrollbar(recom_frame, orient=tk.VERTICAL, command=rec_tree.yview)
                        rec_tree.configure(yscroll=rec_scr.set)
                        rec_scr.pack(side=tk.RIGHT, fill=tk.Y)
                        rec_tree.pack(fill=tk.BOTH, expand=True)
                        # Export Button (Lambda pour capturer good_img et p75 au moment de la création)
                        export_cmd = lambda gi=good_img, p=p75: self.export_recommended_list(gi, p)
                        ttk.Button(recom_frame, text="Exporter Liste Recommandée", command=export_cmd).pack(pady=10)
                    else:
                        ttk.Label(recom_frame, text="N/A").pack()
                else:
                    stack_tab = ttk.Frame(notebook)
                    notebook.add(stack_tab, text="Recommandations")
                    ttk.Label(stack_tab, text="Moins de 5 images.").pack()
                    # Créer l'onglet même si erreur
            except Exception as e:
                print(f"Err Recom: {e}")
                try:
                    stack_tab = ttk.Frame(notebook)
                    notebook.add(stack_tab, text="Recommandations")
                except tk.TclError:
                    pass
                ttk.Label(stack_tab, text=f"Err:\n{e}").pack()

        except Exception as e_vis: print(f"Err Visu Glob: {e_vis}"); traceback.print_exc(); messagebox.showerror("Erreur Visu", f"Erreur:\n{e_vis}")

    # Fonction export séparée pour la clarté
    def export_recommended_list(self, good_images_list, percentile_val):
         save_path = filedialog.asksaveasfilename(title="Enregistrer Liste Recommandée", defaultextension=".txt", filetypes=[("Texte", "*.txt"), ("Tous", "*.*")])
         if save_path:
             try:
                 with open(save_path, 'w') as f:
                     f.write(f"# Images recommandées (SNR >= {percentile_val:.2f})\n")
                     f.write(f"# Généré: {datetime.datetime.now()}\n\n")
                     for img in good_images_list: f.write(f"{img.get('file', 'N/A')}\n")
                 messagebox.showinfo("Export Réussi", f"Liste de {len(good_images_list)} images exportée.")
             except IOError as ex_err: messagebox.showerror("Erreur Export", f"Erreur écriture:\n{ex_err}")


# --- check_dependencies reste identique ---
# --- if __name__ == "__main__": reste identique ---
# (Le code pour check_dependencies et le bloc main n'est pas recopié ici pour la lisibilité)
# Assurez-vous de les garder dans votre fichier final.

# --- Coller ici les fonctions check_dependencies() et le bloc if __name__ == "__main__": du code précédent ---
def check_dependencies():
    """Vérifie les dépendances requises et propose l'installation"""
    missing_deps = []
    import importlib.util
    if importlib.util.find_spec("astropy") is None:
        missing_deps.append("astropy")
    if importlib.util.find_spec("numpy") is None:
        missing_deps.append("numpy")
    if importlib.util.find_spec("matplotlib") is None:
        missing_deps.append("matplotlib")
    # acstools est optionnel et géré différemment

    # Vérifier skimage et scipy qui sont nécessaires pour satdet version Hough
    if importlib.util.find_spec("skimage") is None:
        missing_deps.append("scikit-image")
    if importlib.util.find_spec("scipy") is None:
        missing_deps.append("scipy")


    if missing_deps:
        msg = "Certaines bibliothèques requises ne sont pas installées:\n"
        msg += "\n".join([f"- {dep}" for dep in missing_deps])
        msg += "\n\nSouhaitez-vous essayer de les installer automatiquement avec pip ?"
        msg += "\n(Cela nécessite une connexion Internet et que pip soit configuré)"

        if messagebox.askyesno("Dépendances manquantes", msg):
            install_success = True
            for dep in missing_deps:
                print(f"Tentative d'installation de {dep}...")
                try:
                    # Utiliser check_call pour voir la sortie/erreur directement
                    subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
                    print(f"Installation de {dep} réussie.")
                except subprocess.CalledProcessError as e:
                    print(f"Échec de l'installation de {dep}. Erreur: {e}")
                    messagebox.showerror("Erreur d'installation", f"L'installation de {dep} a échoué. Veuillez l'installer manuellement.")
                    install_success = False
                except FileNotFoundError:
                     messagebox.showerror("Erreur Pip", f"La commande '{sys.executable} -m pip' n'a pas été trouvée. Assurez-vous que Python et pip sont correctement installés et dans le PATH.")
                     install_success = False
                     break # Inutile de continuer si pip n'est pas trouvé

            if install_success:
                 messagebox.showinfo("Installation terminée",
                                  "Les dépendances semblent installées. Veuillez redémarrer l'application.")
                 sys.exit(0) # Quitter pour forcer le redémarrage
            else:
                 messagebox.showwarning("Dépendances", "Certaines dépendances n'ont pas pu être installées. L'application pourrait ne pas fonctionner correctement.")
                 # On ne quitte pas forcément, l'utilisateur peut vouloir réessayer manuellement
        else:
            messagebox.showerror("Erreur", "L'application a besoin de ces bibliothèques pour fonctionner. Veuillez les installer.")
            sys.exit(1) # Quitter si l'utilisateur refuse


if __name__ == "__main__":
    # Vérifier les dépendances avant de lancer l'interface (sauf si lancé par un testeur comme pytest)
    if 'pytest' not in sys.modules:
        root = None # Initialiser à None
        try:
            # Essayer de créer la fenêtre principale AVANT de vérifier les dépendances
            # pour que les messagebox aient une fenêtre parente.
            root = tk.Tk()
            root.withdraw() # Cacher la fenêtre principale vide pendant la vérification

            # Maintenant, vérifier les dépendances, les messagebox s'afficheront correctement
            check_dependencies()

            # Si check_dependencies n'a pas quitté, afficher la fenêtre et lancer l'app
            root.deiconify() # Ré-afficher la fenêtre principale
            app = AstroImageAnalyzerGUI(root)
            root.mainloop()

        except tk.TclError as e:
            # Cette erreur se produit souvent si DISPLAY n'est pas défini (Linux sans X server)
            print(f"Erreur Tcl/Tk: Impossible d'initialiser l'interface graphique. {e}", file=sys.stderr)
            print("Assurez-vous d'exécuter ce script dans un environnement graphique.", file=sys.stderr)
            if root is None: pass
            else:
                 try: messagebox.showerror("Erreur Graphique", f"Impossible d'initialiser l'interface graphique.\n{e}\nVérifiez votre environnement.")
                 except tk.TclError: pass
            sys.exit(1)
        except Exception as e_main:
            # Capturer toute autre exception inattendue au démarrage
            print(f"Erreur inattendue au démarrage: {e_main}", file=sys.stderr)
            traceback.print_exc()
            if root:
                try: messagebox.showerror("Erreur Inattendue", f"Une erreur s'est produite au démarrage:\n{e_main}")
                except: pass
            sys.exit(1)