# --- START OF FILE seestar/gui/file_handling.py ---
"""
Module pour la gestion des fichiers et dossiers dans l'interface GSeestar.
"""
import os
import tkinter as tk
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
        # Use last path from settings if available
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
        # Try getting dir from current reference path
        current_ref = self.gui.reference_image_path.get()
        if current_ref and os.path.isfile(current_ref):
             initial_dir = os.path.dirname(current_ref)
        # Fallback to input dir if no ref set or invalid
        if not initial_dir:
             initial_dir = self.gui.settings.input_folder if hasattr(self.gui.settings, 'input_folder') and self.gui.settings.input_folder else None

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
            # self.gui._try_show_reference_image(abs_file) # Implement in main_window if desired

    def add_folder(self):
        """
        Ajoute un dossier supplémentaire à traiter pendant le traitement.
        """
        # Check if processing is conceptually "started" from the UI perspective
        if not self.gui.processing:
             messagebox.showinfo(
                 self.gui.tr('info'),
                 self.gui.tr('Start processing to add folders')
             )
             return

        # Check if the backend worker thread is actually running
        if not hasattr(self.gui, 'queued_stacker') or not self.gui.queued_stacker.is_running():
            messagebox.showinfo(
                self.gui.tr('info'),
                self.gui.tr('Processing not active or finished.', default='Processing not active or finished.')
            )
            return

        # Suggest initial directory based on input folder
        last_path = self.gui.settings.input_folder if hasattr(self.gui.settings, 'input_folder') and self.gui.settings.input_folder else None
        folder = filedialog.askdirectory(
            title=self.gui.tr('Select Additional Images Folder'),
            initialdir=last_path
        )

        if not folder: return # User cancelled

        abs_folder = os.path.abspath(folder)

        if not os.path.isdir(abs_folder):
            messagebox.showerror(self.gui.tr('error'), self.gui.tr('Folder not found'))
            return

        # Prevent adding input/output folders or subfolders of output
        abs_input = os.path.abspath(self.gui.input_path.get()) if self.gui.input_path.get() else None
        abs_output = os.path.abspath(self.gui.output_path.get()) if self.gui.output_path.get() else None

        if abs_input and abs_folder == abs_input:
            messagebox.showwarning(self.gui.tr('warning'), self.gui.tr('Input folder cannot be added'))
            return
        if abs_output:
            if abs_folder == abs_output:
                 messagebox.showwarning(self.gui.tr('warning'), self.gui.tr('Output folder cannot be added'))
                 return
            # Use normcase for case-insensitive comparison on Windows
            if os.path.normcase(abs_folder).startswith(os.path.normcase(abs_output) + os.sep):
                 messagebox.showwarning(self.gui.tr('warning'), self.gui.tr('Cannot add subfolder of output folder'))
                 return

        # Attempt to add the folder via the queued stacker
        # The stacker now handles checks for duplicates, empty folders, and provides feedback via progress callback
        add_success = self.gui.queued_stacker.add_folder(abs_folder)

        if not add_success:
             # Stacker should have sent a progress message explaining why,
             # but show a generic warning here just in case.
             messagebox.showwarning(
                 self.gui.tr('warning'),
                 self.gui.tr('Folder not added (already present, empty, or error)', default='Folder not added (already present, empty, or error)')
             )

    # This function might become redundant if queue_manager updates the UI var directly via callback
    def update_additional_folders_display(self):
        """Met à jour l'affichage du nombre de dossiers supplémentaires."""
        if hasattr(self.gui, 'queued_stacker'):
            count = len(self.gui.queued_stacker.additional_folders)
            if count == 0:
                self.gui.additional_folders_var.set(self.gui.tr('no_additional_folders'))
            elif count == 1:
                self.gui.additional_folders_var.set(self.gui.tr('1 additional folder'))
            else:
                # Use format for robustness against missing translations
                self.gui.additional_folders_var.set(
                     self.gui.tr('{count} additional folders', default="{count} additional folders").format(count=count)
                )
        else:
            # Set default if stacker isn't available
            self.gui.additional_folders_var.set(self.gui.tr('no_additional_folders'))
# --- END OF FILE seestar/gui/file_handling.py ---