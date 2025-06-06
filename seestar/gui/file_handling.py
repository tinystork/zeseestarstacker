"""
Module pour la gestion des fichiers et dossiers dans l'interface GSeestar.
(Version Révisée: Simplification add_folder - délégation au GUI)
"""
import os
from tkinter import filedialog, messagebox

class FileHandlingManager:
    """
    Gestionnaire pour les opérations liées aux fichiers et dossiers.
    """
    def __init__(self, gui_instance):
        """
        Initialise le gestionnaire de fichiers.

        Args:
            gui_instance: Instance de l'interface graphique principale (SeestarStackerGUI)
        """
        self.gui = gui_instance # gui_instance is the main SeestarStackerGUI object

    def browse_input(self):
        """Ouvre une boîte de dialogue pour sélectionner le dossier d'entrée."""
        last_path = self.gui.settings.input_folder if hasattr(self.gui.settings, 'input_folder') and self.gui.settings.input_folder else None
        folder = filedialog.askdirectory(
            title=self.gui.tr('Select Input Folder'),
            initialdir=last_path
            )
        if folder:
            abs_folder = os.path.abspath(folder)
            self.gui.input_path.set(abs_folder)
            self.gui.settings.input_folder = abs_folder # Save absolute path
            self.gui.settings.save_settings() # Save settings after modification
            if hasattr(self.gui, '_update_show_folders_button_state'):
                self.gui.root.after_idle(self.gui._update_show_folders_button_state) # Use after_idle
            # Try to display the first image from the newly selected folder
            self.gui._try_show_first_input_image()

    def browse_output(self):
        """Ouvre une boîte de dialogue pour sélectionner le dossier de sortie."""
        last_path = self.gui.settings.output_folder if hasattr(self.gui.settings, 'output_folder') and self.gui.settings.output_folder else None
        folder = filedialog.askdirectory(
            title=self.gui.tr('Select Output Folder'),
            initialdir=last_path
            )
        if folder:
            abs_folder = os.path.abspath(folder)
            self.gui.output_path.set(abs_folder)
            self.gui.settings.output_folder = abs_folder # Save absolute path
            self.gui.settings.save_settings() # Save settings

    def browse_reference(self):
        """Ouvre une boîte de dialogue pour sélectionner l'image de référence."""
        initial_dir = None
        current_ref = self.gui.reference_image_path.get()
        if current_ref and os.path.isfile(current_ref):
             initial_dir = os.path.dirname(current_ref)
        elif hasattr(self.gui.settings, 'input_folder') and self.gui.settings.input_folder and os.path.isdir(self.gui.settings.input_folder):
             initial_dir = self.gui.settings.input_folder
        else:
             initial_dir = "."

        file = filedialog.askopenfilename(
            title=self.gui.tr('Select Reference Image (Optional)'),
            filetypes=[("FITS files", "*.fit *.fits")],
            initialdir=initial_dir
        )
        if file:
            abs_file = os.path.abspath(file)
            self.gui.reference_image_path.set(abs_file)
            self.gui.settings.reference_image_path = abs_file # Save setting
            self.gui.settings.save_settings()
            # Optional: Try to show the reference image preview
            # if hasattr(self.gui, '_try_show_reference_image'):
            #     self.gui._try_show_reference_image(abs_file)

    def add_folder(self):
        """
        Demande un dossier à l'utilisateur et transmet la requête au GUI principal
        pour décider où l'ajouter (liste pré-démarrage ou backend).
        """
        # Suggerer répertoire initial basé sur dossier input
        last_path = self.gui.settings.input_folder if hasattr(self.gui.settings, 'input_folder') and self.gui.settings.input_folder else None
        folder = filedialog.askdirectory(
            title=self.gui.tr('Select Additional Images Folder'),
            initialdir=last_path
        )

        if not folder: return # Annulé par l'utilisateur

        abs_folder = os.path.abspath(folder)

        # --- Validation minimale dans le GUI avant de passer au handle ---
        if not os.path.isdir(abs_folder):
            messagebox.showerror(self.gui.tr('error'), self.gui.tr('Folder not found'))
            return

        abs_input = os.path.abspath(self.gui.input_path.get()) if self.gui.input_path.get() else None
        abs_output = os.path.abspath(self.gui.output_path.get()) if self.gui.output_path.get() else None

        if abs_input and os.path.normcase(abs_folder) == os.path.normcase(abs_input):
            messagebox.showwarning(self.gui.tr('warning'), self.gui.tr('Input folder cannot be added'))
            return
        if abs_output:
            if os.path.normcase(abs_folder) == os.path.normcase(abs_output):
                 messagebox.showwarning(self.gui.tr('warning'), self.gui.tr('Output folder cannot be added'))
                 return
            if os.path.normcase(abs_folder).startswith(os.path.normcase(abs_output) + os.sep):
                 messagebox.showwarning(self.gui.tr('warning'), self.gui.tr('Cannot add subfolder of output folder'))
                 return
        # --- Fin Validation ---

        # --- Transmettre la requête au GUI principal ---
        # Il est préférable que SeestarStackerGUI gère l'état (processing ou non)
        if hasattr(self.gui, 'handle_add_folder_request'):
            self.gui.handle_add_folder_request(abs_folder)
        else:
             # Fallback ou erreur si la méthode n'existe pas dans le GUI
             print("Erreur: Méthode handle_add_folder_request non trouvée dans le GUI.")
             messagebox.showerror(self.gui.tr("error"), "Erreur interne lors de l'ajout du dossier.")

    # La méthode update_additional_folders_display est maintenant gérée dans main_window.py

# --- END OF FILE seestar/gui/file_handling.py ---