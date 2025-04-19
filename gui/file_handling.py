"""
File handling module for Seestar.
Contains functions for folder selection, file manipulation, and additional folder management.
"""
import os
from tkinter import filedialog, messagebox

class FileHandlingManager:
    """
    Manages file operations for Seestar.
    """
    
    def __init__(self, main_window):
        """
        Initialize the file handling manager.
        
        Args:
            main_window: The main SeestarStackerGUI instance
        """
        self.main = main_window

    def browse_input(self):
        """Open a dialog to select the input folder."""
        folder = filedialog.askdirectory()
        if folder:
            self.main.input_path.set(folder)
            # Try to display the first image or the last stack
            self.main.refresh_preview()

    def browse_output(self):
        """Open a dialog to select the output folder."""
        folder = filedialog.askdirectory()
        if folder:
            self.main.output_path.set(folder)

    def browse_reference(self):
        """Open a dialog to select the reference image."""
        file = filedialog.askopenfilename(
            filetypes=[("FITS files", "*.fit;*.fits")])
        if file:
            self.main.reference_image_path.set(file)
    
    def add_folder(self):
        """Open a dialog to add a folder to the queue."""
        if not self.main.processing:
            return
        
        # Open the dialog to select a folder
        folder = filedialog.askdirectory(
            title="Select folder with additional images"
        )
        
        if not folder:
            return  # No folder selected
        
        # Check if the folder contains FITS files
        fits_files = [f for f in os.listdir(folder) if f.lower().endswith(('.fit', '.fits'))]
        if not fits_files:
            messagebox.showwarning(
                "Empty folder",
                "The selected folder does not contain any FITS files (.fit or .fits)."
            )
            return
        
        # Add the folder to the list of additional folders
        self.main.additional_folders.append(folder)
        
        # Update the total number of images to process
        file_count = len(fits_files)
        self.main.total_images_count += file_count
        
        # Record this folder as already counted
        if not hasattr(self.main, 'total_additional_counted'):
            self.main.total_additional_counted = set()
        self.main.total_additional_counted.add(folder)
        
        # Update display
        self.update_additional_folders_display()
        
        # Recalculate and update remaining time display
        remaining_time = self.main.processing_manager.calculate_remaining_time()
        self.main.remaining_time_var.set(remaining_time)
        
        self.main.update_progress(f"📂 Folder added to queue: {folder} ({file_count} FITS files)")
    
    def update_additional_folders_display(self):
        """Update the display of additional folders count."""
        if not self.main.additional_folders:
            self.main.additional_folders_var.set("No additional folders")
        elif len(self.main.additional_folders) == 1:
            self.main.additional_folders_var.set("1 additional folder")
        else:
            self.main.additional_folders_var.set(f"{len(self.main.additional_folders)} additional folders")