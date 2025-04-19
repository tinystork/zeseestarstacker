"""
Main module for the Seestar graphical interface.
Contains the main window class with initialization and core functionality.
"""
import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import time
from seestar.core.alignment import SeestarAligner
from seestar.core.stacking import ProgressiveStacker
from seestar.localization.localization import Localization
from .preview import PreviewManager
from .progress import ProgressManager
from .settings import SettingsManager
from .ui_components import UIComponentsManager
from .processing import ProcessingManager
from .stacking import StackingManager
from .file_handling import FileHandlingManager

class SeestarStackerGUI:
    """
    GUI for Seestar Stacker application with queue system.
    """
    
    def __init__(self):
        """Initialize the Seestar Stacker graphical interface."""
        self.root = tk.Tk()

        # In the __init__ method
        self.total_additional_counted = set()  # To avoid counting the same folders twice

        # Initialize localization (English by default)
        self.localization = Localization('en')

        # Initialize processing classes
        self.aligner = SeestarAligner()
        self.stacker = ProgressiveStacker()
        
        # Initialize settings manager
        self.settings = SettingsManager()

        # Variables for widgets
        self.init_variables()

        # Create interface
        self.ui_components = UIComponentsManager(self)
        self.ui_components.create_layout()

        # Initialize managers
        self.init_managers()
    
        # Processing state
        self.processing = False
        self.thread = None
        
        # Variables for storing intermediate and final stacks
        self.current_stack_data = None
        self.current_stack_header = None
        
        # List of additional folders to process
        self.additional_folders = []
        self.processing_additional = False

        # Variables for global time estimation
        self.total_images_count = 0
        self.processed_images_count = 0
        self.time_per_image = 0
        self.global_start_time = None
        
        # Set the window title
        self.root.title(self.tr('title'))
        
        # Bind window resize event
        self.root.bind("<Configure>", self.on_window_resize)

    def init_variables(self):
        """Initialize variables for widgets."""
        # Folder paths
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.reference_image_path = tk.StringVar()

        # Variable to display the number of remaining files
        self.remaining_files_var = tk.StringVar(value="No files waiting")        

        # Folder monitoring
        #self.watch_mode = tk.BooleanVar(value=False)

        # Stacking options
        self.stacking_mode = tk.StringVar(value="kappa-sigma")
        self.kappa = tk.DoubleVar(value=2.5)
        self.batch_size = tk.IntVar(value=0)
        self.remove_aligned = tk.BooleanVar(value=False)
        self.apply_denoise = tk.BooleanVar(value=False)

        # Hot pixel options
        self.correct_hot_pixels = tk.BooleanVar(value=True)
        self.hot_pixel_threshold = tk.DoubleVar(value=3.0)
        self.neighborhood_size = tk.IntVar(value=5)

        # Processing mode
        self.use_queue = tk.BooleanVar(value=True)
        self.use_traditional = tk.BooleanVar(value=True)

        # Preview options
        self.apply_stretch = tk.BooleanVar(value=True)

        # Time variables
        self.remaining_time_var = tk.StringVar(value="--:--:--")
        self.elapsed_time_var = tk.StringVar(value="00:00:00")

        # Language variable
        self.language_var = tk.StringVar(value='en')
        
        # Additional folders display
        self.additional_folders_var = tk.StringVar(value="No additional folders")

    def init_managers(self):
        """Initialize preview and progress managers."""
        # Initialize preview manager
        self.preview_manager = PreviewManager(
            self.ui_components.preview_canvas,
            self.ui_components.current_stack_label,
            self.ui_components.image_info_text
        )

        # Initialize progress manager
        self.progress_manager = ProgressManager(
            self.ui_components.progress_bar,
            self.ui_components.status_text,
            self.remaining_time_var,
            self.elapsed_time_var
        )
        
        # Initialize processing manager
        self.processing_manager = ProcessingManager(self)
        
        # Initialize stacking manager
        self.stacking_manager = StackingManager(self)
        
        # Initialize file handling manager
        self.file_handling = FileHandlingManager(self)

        # Configure callbacks
        self.aligner.set_progress_callback(self.update_progress)
        self.stacker.set_progress_callback(self.update_progress)
        #self.queued_stacker.set_progress_callback(self.update_progress)
        #self.queued_stacker.set_preview_callback(self.update_preview)

    def tr(self, key):
        """
        Shortcut for localization lookup.

        Args:
            key (str): Translation key

        Returns:
            str: Translated text
        """
        return self.localization.get(key)

    def update_progress(self, message, progress=None):
        """
        Update progress bar and status text.

        Args:
            message (str): Message to display in the status area
            progress (float, optional): Progress value (0-100)
        """
        # Use the progress manager
        self.progress_manager.update_progress(message, progress)

    def on_window_resize(self, event):
        """Handle window resizing."""
        # Process only events from the main window
        if event.widget == self.root:
            self.refresh_preview()

    def refresh_preview(self):
        """Refresh the preview with the last stored image."""
        self.preview_manager.refresh_preview(
#            self.queued_stacker, 
            self.current_stack_data, 
            self.input_path.get(), 
            self.output_path.get(), 
            self.apply_stretch.get()
        )

    def update_preview(self, image_data, stack_name=None, apply_stretch=None):
        """
        Update preview with provided image and detailed progress information.

        Args:
            image_data (numpy.ndarray): Image data
            stack_name (str, optional): Stack name
            apply_stretch (bool, optional): Apply automatic stretching to the image
        """
        if apply_stretch is None:
            apply_stretch = self.apply_stretch.get()
        
        # Enhance stack name with progress information
        stack_name = self.processing_manager.enhance_stack_name(
            stack_name, 
            self.total_images_count, 
            self.processed_images_count,
            self.additional_folders,
            self.total_additional_counted,
            self.processing_additional,
            self.batch_size.get()
        )
        
        # Use the preview manager to update the display
        self.preview_manager.update_preview(image_data, stack_name, apply_stretch)
        
        # Update the estimated remaining time with each preview update
        if self.processed_images_count > 0 and self.global_start_time:
            elapsed_time = time.time() - self.global_start_time
            self.time_per_image = elapsed_time / self.processed_images_count
            remaining_time = self.processing_manager.calculate_remaining_time()
            self.remaining_time_var.set(remaining_time)

    def update_additional_folders_display(self):
        """Met à jour l'affichage du nombre de dossiers additionnels."""
        if not self.additional_folders:
            self.additional_folders_var.set("Aucun dossier additionnel")
        elif len(self.additional_folders) == 1:
            self.additional_folders_var.set("1 dossier additionnel")
        else:
            self.additional_folders_var.set(f"{len(self.additional_folders)} dossiers additionnels")

    def change_language(self, event=None):
        """Change interface to selected language."""
        selected = self.language_var.get()

        if selected == 'English':
            self.localization.set_language('en')
        elif selected == 'Français':
            self.localization.set_language('fr')

        # Update all interface elements with the new language
        self.ui_components.update_ui_language()

    def start_processing(self):
        """Start image processing."""
        self.processing_manager.start_processing()

    def stop_processing(self):
        """Stop current processing."""
        self.processing_manager.stop_processing()

    def add_folder(self):
        """Open a dialog to add a folder to the queue."""
        self.file_handling.add_folder()

    def browse_input(self):
        """Open a dialog to select the input folder."""
        self.file_handling.browse_input()

    def browse_output(self):
        """Open a dialog to select the output folder."""
        self.file_handling.browse_output()

    def browse_reference(self):
        """Open a dialog to select the reference image."""
        self.file_handling.browse_reference()
    
    def run(self):
        """Launch the graphical interface."""
        self.root.mainloop()