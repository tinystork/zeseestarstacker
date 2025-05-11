

if __name__ == "__main__":
    # --- MODIFIÉ: Parsing des arguments de ligne de commande ---
    print("DEBUG (analyse_gui main): Parsing des arguments...") # <-- AJOUTÉ DEBUG
    parser = argparse.ArgumentParser(description="Astro Image Analyzer GUI")
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Optional: Pre-fill the input directory path."
    )
    # --- NOUVEL ARGUMENT AJOUTÉ ICI ---
    parser.add_argument(
        "--command-file", # <-- AJOUTÉ : Définition de l'argument
        type=str,
        metavar="CMD_FILE_PATH",
        help="Internal: Path to the command file for communicating with the main stacker GUI."
    )
    # --- FIN NOUVEL ARGUMENT ---
    args = parser.parse_args()
    print(f"DEBUG (analyse_gui main): Arguments parsés: {args}") # <-- AJOUTÉ DEBUG
    # --- FIN MODIFICATION ---

    root = None # Initialiser la variable racine
    try:
        # Vérifier si les modules essentiels sont importables
        # (Logique inchangée)
        if 'analyse_logic' not in sys.modules: raise ImportError("analyse_logic.py could not be imported.")
        if 'translations' not in globals() or not translations: raise ImportError("zone.py is empty or could not be imported.")

        # Créer la fenêtre racine Tkinter mais la cacher initialement
        root = tk.Tk(); root.withdraw()

        # Vérifier les dépendances externes
        check_dependencies()

        # Afficher la fenêtre principale
        root.deiconify()

        # --- MODIFIÉ: Passer command_file_path au constructeur ---
        print(f"DEBUG (analyse_gui main): Instanciation AstroImageAnalyzerGUI avec command_file='{args.command_file}'") # <-- AJOUTÉ DEBUG
        # Passer le chemin du fichier de commande (qui sera None s'il n'est pas fourni)
        app = AstroImageAnalyzerGUI(root, command_file_path=args.command_file, main_app_callback=None) # <-- MODIFIÉ
        # --- FIN MODIFICATION ---

        # --- Pré-remplissage dossier d'entrée (Logique inchangée) ---
        if args.input_dir:
            input_path_from_arg = os.path.abspath(args.input_dir)
            if os.path.isdir(input_path_from_arg):
                print(f"INFO (analyse_gui main): Pré-remplissage dossier entrée depuis argument: {input_path_from_arg}")
                app.input_dir.set(input_path_from_arg)
                if not app.output_log.get(): app.output_log.set(os.path.join(input_path_from_arg, "analyse_resultats.log"))
                if not app.snr_reject_dir.get(): app.snr_reject_dir.set(os.path.join(input_path_from_arg, "rejected_low_snr"))
                if not app.trail_reject_dir.get(): app.trail_reject_dir.set(os.path.join(input_path_from_arg, "rejected_satellite_trails"))
            else:
                print(f"AVERTISSEMENT (analyse_gui main): Dossier d'entrée via argument invalide: {args.input_dir}")

        # Lancer la boucle principale de Tkinter
        print("DEBUG (analyse_gui main): Entrée dans root.mainloop().") # <-- AJOUTÉ DEBUG
        root.mainloop()
        print("DEBUG (analyse_gui main): Sortie de root.mainloop().") # <-- AJOUTÉ DEBUG

    # --- Gestion des Erreurs au Démarrage (Inchangée) ---
    except ImportError as e:
        print(f"ERREUR CRITIQUE: Échec import module au démarrage: {e}", file=sys.stderr); traceback.print_exc()
        try:
            if root is None: root = tk.Tk(); root.withdraw(); messagebox.showerror("Erreur Fichier Manquant", f"Impossible de charger un module essentiel ({e}).\nVérifiez que analyse_logic.py et zone.py sont présents et valides."); root.destroy()
        except Exception as msg_e: print(f" -> Erreur affichage message: {msg_e}", file=sys.stderr); sys.exit(1)
    except SystemExit as e: # <-- AJOUTÉ: Gérer SystemExit de argparse
        print(f"DEBUG (analyse_gui main): Argparse a quitté (probablement '-h' ou erreur argument). Code: {e.code}")
        # Ne rien faire de plus, le message d'erreur d'argparse est déjà affiché.
        pass
    except tk.TclError as e:
        print(f"Erreur Tcl/Tk: Impossible d'initialiser l'interface graphique. {e}", file=sys.stderr); print("Assurez-vous d'exécuter ce script dans un environnement graphique.", file=sys.stderr); sys.exit(1)
    except Exception as e_main:
        print(f"Erreur inattendue au démarrage: {e_main}", file=sys.stderr); traceback.print_exc()
        try:
            if root is None: root = tk.Tk(); root.withdraw(); messagebox.showerror("Erreur Inattendue", f"Une erreur s'est produite au démarrage:\n{e_main}"); root.destroy()
        except Exception as msg_e: print(f" -> Erreur affichage message: {msg_e}", file=sys.stderr); sys.exit(1)
    # --- MODIFIÉ: Parsing des arguments de ligne de commande ---
    print("DEBUG (analyse_gui main): Parsing des arguments...") # <-- AJOUTÉ DEBUG
    parser = argparse.ArgumentParser(description="Astro Image Analyzer GUI")
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Optional: Pre-fill the input directory path."
    )
      
# --- NOUVEL ARGUMENT AJOUTÉ ICI ---
    parser.add_argument(
        "--command-file", # <-- Vérifiez l'orthographe EXACTE et les DEUX tirets
        type=str,
        metavar="CMD_FILE_PATH",
        help="Internal: Path to the command file for communicating with the main stacker GUI."
    )
    # --- FIN NOUVEL ARGUMENT ---
    args = parser.parse_args() # <--- Cette ligne doit venir APRES l'ajout de l'argument

    
    print(f"DEBUG (analyse_gui main): Arguments parsés: {args}") # <-- AJOUTÉ DEBUG
    # --- FIN MODIFICATION ---

    root = None # Initialiser la variable racine
    try:
        # Vérifier si les modules essentiels sont importables
        # (Logique inchangée)
        if 'analyse_logic' not in sys.modules: raise ImportError("analyse_logic.py could not be imported.")
        if 'translations' not in globals() or not translations: raise ImportError("zone.py is empty or could not be imported.")

        # Créer la fenêtre racine Tkinter mais la cacher initialement
        root = tk.Tk(); root.withdraw()

        # Vérifier les dépendances externes
        check_dependencies()

        # Afficher la fenêtre principale
        root.deiconify()

        # --- MODIFIÉ: Passer command_file_path au constructeur ---
        print(f"DEBUG (analyse_gui main): Instanciation AstroImageAnalyzerGUI avec command_file='{args.command_file}'") # <-- AJOUTÉ DEBUG
        # Passer le chemin du fichier de commande (qui sera None s'il n'est pas fourni)
        app = AstroImageAnalyzerGUI(root, command_file_path=args.command_file, main_app_callback=None) # <-- MODIFIÉ
        # --- FIN MODIFICATION ---

        # --- Pré-remplissage dossier d'entrée (Logique inchangée) ---
        if args.input_dir:
            input_path_from_arg = os.path.abspath(args.input_dir)
            if os.path.isdir(input_path_from_arg):
                print(f"INFO (analyse_gui main): Pré-remplissage dossier entrée depuis argument: {input_path_from_arg}")
                app.input_dir.set(input_path_from_arg)
                if not app.output_log.get(): app.output_log.set(os.path.join(input_path_from_arg, "analyse_resultats.log"))
                if not app.snr_reject_dir.get(): app.snr_reject_dir.set(os.path.join(input_path_from_arg, "rejected_low_snr"))
                if not app.trail_reject_dir.get(): app.trail_reject_dir.set(os.path.join(input_path_from_arg, "rejected_satellite_trails"))
            else:
                print(f"AVERTISSEMENT (analyse_gui main): Dossier d'entrée via argument invalide: {args.input_dir}")

        # Lancer la boucle principale de Tkinter
        print("DEBUG (analyse_gui main): Entrée dans root.mainloop().") # <-- AJOUTÉ DEBUG
        root.mainloop()
        print("DEBUG (analyse_gui main): Sortie de root.mainloop().") # <-- AJOUTÉ DEBUG

    # --- Gestion des Erreurs au Démarrage (Inchangée) ---
    except ImportError as e:
        print(f"ERREUR CRITIQUE: Échec import module au démarrage: {e}", file=sys.stderr); traceback.print_exc()
        try: 
            if root is None: root = tk.Tk(); root.withdraw()
            messagebox.showerror("Erreur Fichier Manquant", f"Impossible de charger un module essentiel ({e}).\nVérifiez que analyse_logic.py et zone.py sont présents et valides."); root.destroy()
        except Exception as msg_e: print(f" -> Erreur affichage message: {msg_e}", file=sys.stderr); sys.exit(1)
    except tk.TclError as e:
        print(f"Erreur Tcl/Tk: Impossible d'initialiser l'interface graphique. {e}", file=sys.stderr); print("Assurez-vous d'exécuter ce script dans un environnement graphique.", file=sys.stderr); sys.exit(1)
    except Exception as e_main:
        print(f"Erreur inattendue au démarrage: {e_main}", file=sys.stderr); traceback.print_exc()
        try: 
            if root is None: root = tk.Tk(); root.withdraw()
            messagebox.showerror("Erreur Inattendue", f"Une erreur s'est produite au démarrage:\n{e_main}"); root.destroy()
        except Exception as msg_e: print(f" -> Erreur affichage message: {msg_e}", file=sys.stderr); sys.exit(1)
