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
            gui_instance: Instance de l'interface graphique principale
        """
        self.gui = gui_instance

    def browse_input(self):
        """Ouvre une boîte de dialogue pour sélectionner le dossier d'entrée."""
        folder = filedialog.askdirectory(title=self.gui.tr('Select Input Folder'))
        if folder:
            self.gui.input_path.set(folder)
            # Essayer d'afficher la première image
            self.gui._try_show_first_input_image()

    def browse_output(self):
        """Ouvre une boîte de dialogue pour sélectionner le dossier de sortie."""
        folder = filedialog.askdirectory(title=self.gui.tr('Select Output Folder'))
        if folder:
            self.gui.output_path.set(folder)

    def browse_reference(self):
        """Ouvre une boîte de dialogue pour sélectionner l'image de référence."""
        file = filedialog.askopenfilename(
            title=self.gui.tr('Select Reference Image (Optional)'),
            filetypes=[("FITS files", "*.fit;*.fits")]
        )
        if file:
            self.gui.reference_image_path.set(file)

    def add_folder(self):
        """
        Ajoute un dossier supplémentaire à traiter pendant le traitement.
        Cette fonction est maintenant activée pendant le traitement.
        """
        if not hasattr(self.gui, 'queued_stacker'):
            messagebox.showinfo(
                self.gui.tr('info'),
                self.gui.tr('Start processing to add folders')
            )
            return

        folder = filedialog.askdirectory(
            title=self.gui.tr('Select Additional Images Folder')
        )
        
        if not folder:
            return
            
        if not os.path.isdir(folder):
            messagebox.showerror(
                self.gui.tr('error'),
                self.gui.tr('Folder not found')
            )
            return
            
        # Vérifier si c'est le dossier d'entrée
        if folder == self.gui.input_path.get():
            messagebox.showwarning(
                self.gui.tr('warning'),
                self.gui.tr('Input folder cannot be added')
            )
            return
            
        # Vérifier si déjà dans la liste
        if folder in self.gui.queued_stacker.additional_folders:
            messagebox.showinfo(
                self.gui.tr('info'),
                self.gui.tr('Folder already added')
            )
            return
            
        # Vérifier s'il contient des fichiers FITS
        try:
            fits_files = [f for f in os.listdir(folder) if f.lower().endswith(('.fit', '.fits'))]
            if not fits_files:
                messagebox.showwarning(
                    self.gui.tr('warning'),
                    self.gui.tr('Folder contains no FITS')
                )
                return
        except Exception as e:
            messagebox.showerror(
                self.gui.tr('error'),
                f"{self.gui.tr('Error reading folder')}: {e}"
            )
            return
            
        # Ajouter le dossier à la file d'attente
        if self.gui.queued_stacker.add_folder(folder):
            self.update_additional_folders_display()
            messagebox.showinfo(
                self.gui.tr('info'),
                f"{self.gui.tr('Folder added')}: {os.path.basename(folder)} ({len(fits_files)} {self.gui.tr('files')})"
            )
            
    def update_additional_folders_display(self):
        """Met à jour l'affichage du nombre de dossiers supplémentaires."""
        if not hasattr(self.gui, 'queued_stacker'):
            self.gui.additional_folders_var.set(self.gui.tr('no_additional_folders'))
            return
            
        count = len(self.gui.queued_stacker.additional_folders)
        if count == 0:
            self.gui.additional_folders_var.set(self.gui.tr('no_additional_folders'))
        elif count == 1:
            self.gui.additional_folders_var.set(self.gui.tr('1 additional folder'))
        else:
            self.gui.additional_folders_var.set(
                self.gui.tr('{count} additional folders').format(count=count)
            )