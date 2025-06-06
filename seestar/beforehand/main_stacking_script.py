
import tkinter as tk
from tkinter import ttk # Assurer que ttk est importé
from tkinter import filedialog, messagebox
import os
import sys

import traceback # Pour afficher les erreurs détaillées

# --- Importer l'interface de l'analyseur ---
# On suppose que analyse_gui.py gère ses propres imports et dépendances
try:
    import analyse_gui
    # Importer aussi les traductions pour pouvoir afficher des messages de base
    from zone import translations
    # Fonction basique pour obtenir texte (utilisée si analyse_gui échoue à s'importer)
    def _main_gettext(key, default_text=""):
         # Essayer fr, puis en, puis default
         return translations.get('fr', {}).get(key, translations.get('en', {}).get(key, default_text))

except ImportError:
     # Erreur critique si analyse_gui ou zone sont introuvables
     print("ERREUR: Impossible d'importer analyse_gui.py ou zone.py.")
     print("Assurez-vous qu'ils sont dans le même dossier que ce script.")
     # Essayer d'afficher une erreur Tkinter si possible
     try:
         root_err = tk.Tk(); root_err.withdraw()
         # Essayer d'obtenir un message traduit si zone.py était là
         err_title = translations.get('fr', {}).get('msg_error', "Error")
         err_msg = translations.get('fr', {}).get('msg_missing_logic', "Core module (analyse_gui.py or zone.py) is missing.")
         messagebox.showerror(err_title, err_msg)
         root_err.destroy()
     except Exception as e_msg:
         print(f"Impossible d'afficher la messagebox d'erreur Tkinter: {e_msg}")
     sys.exit(1)
except Exception as e:
     # Autre erreur pendant l'import (ex: syntax error dans analyse_gui)
     print(f"ERREUR inattendue lors de l'import de analyse_gui/zone: {e}")
     traceback.print_exc()
     try:
         root_err = tk.Tk(); root_err.withdraw()
         err_title = translations.get('fr', {}).get('msg_error', "Error")
         messagebox.showerror(err_title, f"Erreur chargement analyse_gui/zone:\n{e}")
         root_err.destroy()
     except Exception as e_msg:
         print(f"Impossible d'afficher la messagebox d'erreur Tkinter: {e_msg}")
     sys.exit(1)


# --- Variables Globales (ou dans une classe App si plus complexe) ---
selected_folder = None
analysis_gui_open = False # Flag pour savoir si le GUI d'analyse est ouvert

# --- Fonctions ---

def select_folder():
    """Demande à l'utilisateur de choisir un dossier d'images."""
    global selected_folder
    # Utiliser la fenêtre racine comme parent pour le dialogue
    folder = filedialog.askdirectory(parent=root, title=_main_gettext("Sélectionner Dossier", "Select Image Folder"))
    if folder:
        selected_folder = folder
        # Afficher une version courte du chemin si trop long
        display_path = folder
        if len(folder) > 60:
            display_path = "..." + folder[-57:]
        folder_label.config(text=f"{_main_gettext('Dossier:', 'Folder:')} {display_path}")
        # Activer le bouton Analyse/Pré-filtrage
        analyze_button.config(state=tk.NORMAL)
        # Désactiver Stacking tant que pas analysé (ou optionnel)
        stack_button.config(state=tk.DISABLED)
        print(f"Dossier sélectionné: {selected_folder}")
    else:
        # Si l'utilisateur annule, ne pas changer le dossier sélectionné
        # selected_folder = None # Ne pas réinitialiser si déjà sélectionné
        # folder_label.config(text=_main_gettext("Aucun dossier sélectionné", "No folder selected"))
        # analyze_button.config(state=tk.DISABLED)
        # stack_button.config(state=tk.DISABLED)
        pass # Ne rien faire si annulé

def run_analyzer():
    """Lance l'interface graphique de l'analyseur."""
    global analysis_gui_open
    if analysis_gui_open:
         print("L'analyseur est déjà ouvert.")
         # On pourrait essayer de ramener la fenêtre au premier plan
         # find_analyzer_window_and_focus() # Fonction à créer si besoin
         return

    if not selected_folder:
        messagebox.showwarning(_main_gettext("Dossier Manquant", "Missing Folder"),
                               _main_gettext("Veuillez d'abord sélectionner un dossier.", "Please select a folder first."),
                               parent=root)
        return

    print("Lancement de l'interface d'analyse...")
    # Désactiver les boutons du script principal pendant que l'analyseur est ouvert
    select_button.config(state=tk.DISABLED)
    analyze_button.config(state=tk.DISABLED)
    stack_button.config(state=tk.DISABLED)
    analysis_gui_open = True # Mettre le flag

    # Créer une nouvelle fenêtre Toplevel pour l'analyseur
    analyzer_window = tk.Toplevel(root)
    analyzer_window.title(_main_gettext("Pré-filtrage / Analyse", "Pre-filtering / Analysis")) # Titre simple ici

    # --- Corrections Focus/Modalité ---
    analyzer_window.transient(root) # Lier à la fenêtre principale pour le focus
    analyzer_window.grab_set()      # Rendre la fenêtre modale

    # Définir la fonction callback qui sera appelée par le bouton "Retour" ou la fermeture
    def on_analyzer_close():
        global analysis_gui_open
        print("Retour depuis l'analyseur détecté.")
        analysis_gui_open = False # Réinitialiser le flag
        # grab_release n'est pas nécessaire si la fenêtre est détruite
        # analyzer_window.grab_release()
        # Réactiver les boutons du script principal
        enable_main_buttons_after_analysis()
        # S'assurer que la fenêtre est bien détruite (au cas où appelée par WM_DELETE_WINDOW sans destroy)
        try:
            if analyzer_window.winfo_exists():
                 analyzer_window.destroy()
        except tk.TclError:
             pass # Fenêtre déjà détruite

    # Lier la fermeture de la fenêtre (bouton X) au callback aussi
    analyzer_window.protocol("WM_DELETE_WINDOW", on_analyzer_close)

    # Instancier le GUI de l'analyseur en lui passant le callback
    try:
         # Passer le chemin initial au GUI analyseur
         # The on_analyzer_close callback is still needed for the "Retour" button's logic
         app_analyzer = analyse_gui.AstroImageAnalyzerGUI(analyzer_window, main_app_callback=on_analyzer_close)

         # --- MODIFICATION ICI ---
         # Lier la fermeture de la fenêtre (bouton X) DIRECTEMENT à la méthode return_or_quit de l'instance GUI
         # Cette méthode gère le nettoyage ET appelle ensuite le callback (on_analyzer_close)
         analyzer_window.protocol("WM_DELETE_WINDOW", app_analyzer.return_or_quit)
         # --- FIN MODIFICATION ---


         # Pré-remplir les chemins si le dossier sélectionné existe
         if selected_folder and os.path.isdir(selected_folder):
             # ... (pre-filling logic remains the same) ...
             app_analyzer.input_dir.set(selected_folder)
             default_log = os.path.join(selected_folder, "analyse_resultats.log")
             default_snr_reject = os.path.join(selected_folder, "rejected_low_snr")
             default_trail_reject = os.path.join(selected_folder, "rejected_satellite_trails")
             if not app_analyzer.output_log.get(): app_analyzer.output_log.set(default_log)
             if not app_analyzer.snr_reject_dir.get(): app_analyzer.snr_reject_dir.set(default_snr_reject)
             if not app_analyzer.trail_reject_dir.get(): app_analyzer.trail_reject_dir.set(default_trail_reject)

         # La fenêtre est modale (grab_set), donc le code ici attend implicitement
         # que la fenêtre soit fermée (via on_analyzer_close qui détruit la fenêtre).

    except Exception as e:
         # ... (exception handling remains the same) ...
         print(f"Erreur lors de l'instanciation ou de l'exécution de analyse_gui: {e}")
         traceback.print_exc()
         messagebox.showerror(_main_gettext("Erreur Analyseur", "Analyzer Error"),
                              _main_gettext("Impossible de lancer l'outil d'analyse:\n{e}", "Could not launch analyzer tool:\n{e}").format(e=e),
                              parent=root)
         # Assurer la réactivation des boutons et reset du flag si erreur au lancement
         analysis_gui_open = False
         try: # Assurer la destruction de la fenêtre Toplevel si elle a été créée
             if analyzer_window.winfo_exists(): analyzer_window.destroy()
         except tk.TclError: pass
         enable_main_buttons_after_analysis()


def enable_main_buttons_after_analysis():
     """Réactive les boutons du script principal."""
     select_button.config(state=tk.NORMAL)
     # Réactiver Analyse seulement si un dossier est sélectionné
     analyze_button.config(state=tk.NORMAL if selected_folder else tk.DISABLED)
     # Activer le bouton Stacking seulement si un dossier est sélectionné
     # (L'analyse n'est qu'une étape optionnelle avant stacking)
     stack_button.config(state=tk.NORMAL if selected_folder else tk.DISABLED)
     if selected_folder:
          print("Interface principale réactivée.")
          if analysis_gui_open is False: # Vérifier si l'analyseur est bien fermé
               print("Prêt à lancer le stacking sur le dossier (potentiellement nettoyé).")
     else:
          print("Interface principale réactivée (pas de dossier sélectionné).")


def run_stacking():
    """Lance le processus de stacking (simulation)."""
    if not selected_folder:
        messagebox.showwarning(_main_gettext("Dossier Manquant", "Missing Folder"),
                               _main_gettext("Aucun dossier sélectionné pour le stacking.", "No folder selected for stacking."),
                               parent=root)
        return

    # On pourrait demander confirmation si l'analyseur n'a pas été explicitement lancé/fermé
    # Mais pour l'instant, on permet de stacker directement si un dossier est choisi.
    # if not analysis_gui_closed: # Remplacé par analysis_gui_open flag
    #      if messagebox.askyesno("Analyse non faite?", "L'étape de pré-filtrage n'a pas été (correctement) terminée.\nVoulez-vous lancer le stacking sur le dossier tel quel ?"):
    #          pass # Continuer quand même
    #      else:
    #          return # Annuler stacking

    print(f"Lancement du stacking sur les fichiers dans : {selected_folder}")
    # Afficher une info (remplacer par la vraie logique)
    messagebox.showinfo(_main_gettext("Stacking", "Stacking"),
                        _main_gettext("Simulation du lancement du stacking sur :\n{folder}\n\n(Remplacez ceci par votre logique de stacking)",
                                      "Simulating stacking launch on:\n{folder}\n\n(Replace this with your stacking logic)").format(folder=selected_folder),
                        parent=root)

    # --- EXEMPLE : Lancer un script externe ou une fonction ---
    # try:
    #    # Supposons que vous ayez un script stacking.py ou une fonction
    #    print("Exécution de la commande de stacking (exemple)...")
    #    # Remplacer par l'appel réel, ex:
    #    # result = subprocess.run([sys.executable, "path/to/your/stacking_script.py", selected_folder],
    #    #                         capture_output=True, text=True, check=True) # check=True lève une exception si erreur
    #    # print("Stacking terminé.")
    #    # print("Sortie Stacking:", result.stdout)
    #    # Ou appel de fonction:
    #    # import stacking_module
    #    # stacking_module.stack_images(selected_folder, output_file="stacked_image.fits")
    #    # print("Stacking terminé.")
    #    # messagebox.showinfo("Stacking Terminé", "Le stacking s'est terminé avec succès.", parent=root)
    #    pass # Placeholder pour la simulation
    # except FileNotFoundError as e:
    #    print(f"Erreur: Le script de stacking n'a pas été trouvé: {e}")
    #    messagebox.showerror("Erreur Stacking", f"Script de stacking introuvable:\n{e}", parent=root)
    # except subprocess.CalledProcessError as e:
    #    print(f"Erreur durant l'exécution du stacking (code {e.returncode}):")
    #    print(e.stderr)
    #    messagebox.showerror("Erreur Stacking", f"Le processus de stacking a échoué (code {e.returncode}):\n{e.stderr[:500]}", parent=root) # Afficher début de l'erreur
    # except Exception as e:
    #    print(f"Erreur inattendue durant le stacking: {e}")
    #    traceback.print_exc()
    #    messagebox.showerror("Erreur Stacking", f"Une erreur inattendue s'est produite durant le stacking:\n{e}", parent=root)
    # --- FIN EXEMPLE ---


# --- Interface Graphique Principale ---
root = tk.Tk()
root.title(_main_gettext("Lanceur Stacking Astro", "Astro Stacking Launcher"))
root.geometry("500x250")
root.minsize(450, 230)

# Appliquer un thème ttk si disponible pour un look plus moderne
style = ttk.Style()
available_themes = style.theme_names()
# print("Available themes:", available_themes) # Pour débugger les thèmes dispo
if 'vista' in available_themes: # Préférer 'vista' ou 'clam' si dispo
    style.theme_use('vista')
elif 'clam' in available_themes:
     style.theme_use('clam')
# Sinon, utilise le thème par défaut

main_frame = ttk.Frame(root, padding="20")
main_frame.pack(fill=tk.BOTH, expand=True)

folder_label = ttk.Label(main_frame, text=_main_gettext("Aucun dossier sélectionné", "No folder selected"), wraplength=450, anchor=tk.W, justify=tk.LEFT)
folder_label.pack(pady=10, fill=tk.X)

select_button = ttk.Button(main_frame, text=_main_gettext("1. Sélectionner Dossier Images", "1. Select Image Folder"), command=select_folder)
select_button.pack(pady=5, fill=tk.X)

analyze_button = ttk.Button(main_frame, text=_main_gettext("2. Pré-filtrer / Analyser Images", "2. Pre-filter / Analyze Images"), command=run_analyzer, state=tk.DISABLED)
analyze_button.pack(pady=5, fill=tk.X)

stack_button = ttk.Button(main_frame, text=_main_gettext("3. Lancer le Stacking", "3. Launch Stacking"), command=run_stacking, state=tk.DISABLED)
stack_button.pack(pady=15, fill=tk.X)

# --- Boucle Principale ---
try:
    root.mainloop()
except KeyboardInterrupt:
    print("\nProgramme interrompu par l'utilisateur.")

# --- FIN DU FICHIER main_stacking_script.py ---