"""
Processing management module for Seestar.
Contains functions for starting, stopping and running image processing.
"""
import os
import time
import threading
import tkinter as tk
from tkinter import messagebox

class ProcessingManager:
    """
    Manages the image processing operations for Seestar.
    """
    
    def __init__(self, main_window):
        """
        Initialize the processing manager.
        
        Args:
            main_window: The main SeestarStackerGUI instance
        """
        self.main = main_window
    
    def start_processing(self):
        """Start image processing."""
        # Check that input and output paths are specified
        input_folder = self.main.input_path.get()
        output_folder = self.main.output_path.get()
        
        if not input_folder or not output_folder:
            messagebox.showerror(self.main.tr('error'), self.main.tr('select_folders'))
            return
        
        # Disable start button and enable stop button
        self.main.ui_components.start_button.config(state=tk.DISABLED)
        self.main.ui_components.stop_button.config(state=tk.NORMAL)
        self.main.ui_components.add_files_button.config(state=tk.NORMAL)
        
        # Reset additional folders
        self.main.additional_folders = []
        self.main.update_additional_folders_display()
        
        # Reset counters
        self.main.total_images_count = 0
        self.main.processed_images_count = 0
        self.main.global_start_time = time.time()
        
        # Mark the beginning of processing
        self.main.processing = True
        
        # Configure processing objects with current parameters
        # Alignment parameters
        self.main.aligner.bayer_pattern = "GRBG"  # Default
        self.main.aligner.batch_size = self.main.batch_size.get()
        self.main.aligner.reference_image_path = self.main.reference_image_path.get() or None
        self.main.aligner.correct_hot_pixels = self.main.correct_hot_pixels.get()
        self.main.aligner.hot_pixel_threshold = self.main.hot_pixel_threshold.get()
        self.main.aligner.neighborhood_size = self.main.neighborhood_size.get()
        
        # Stacker parameters
        self.main.stacker.stacking_mode = self.main.stacking_mode.get()
        self.main.stacker.kappa = self.main.kappa.get()
        self.main.stacker.batch_size = self.main.batch_size.get()
        self.main.stacker.denoise = self.main.apply_denoise.get()
        
       
        # Start the progress manager
        self.main.progress_manager.reset()
        self.main.progress_manager.start_timer()
        
        # Start processing in a separate thread
        self.main.update_progress(self.main.tr('stacking_start'))
        self.main.thread = threading.Thread(target=self.run_processing, args=(input_folder, output_folder))
        self.main.thread.daemon = True
        self.main.thread.start()
        
        # Start periodic update
        self.start_periodic_update()
    
    def stop_processing(self):
        """Stop current processing."""

        if self.main.processing:
            self.main.aligner.stop_processing = True
            self.main.stacker.stop_processing = True
            self.main.update_progress(self.main.tr('stop_requested'))
    
    def run_processing(self, input_folder, output_folder):
        """
        Run the stacking process with traditional stacking mode.
        
        Args:
            input_folder (str): Input folder containing raw images
            output_folder (str): Output folder for stacked images
        """
        try:
            # Enhanced traditional mode with real-time visualization
            self.main.stacking_manager.run_enhanced_traditional_stacking(input_folder, output_folder)
            
            # Process additional folders
            self.main.stacking_manager.process_additional_folders(output_folder)

        except Exception as e:
            self.main.update_progress(f"❌ {self.main.tr('error')}: {e}")
        finally:
            self.main.processing = False
            self.main.ui_components.start_button.config(state=tk.NORMAL)
            self.main.ui_components.stop_button.config(state=tk.DISABLED)

            # Stop timer
            self.main.progress_manager.stop_timer()
    
    def start_periodic_update(self):
        """Start periodic update of remaining files count."""
        if self.main.processing:
            self.update_remaining_files()
            # Schedule next update in 1 second
            self.main.root.after(1000, self.start_periodic_update)
    
    def update_remaining_files(self):
        """
        Update the display of remaining files to process.
        In traditional mode, this will always show 'No files waiting' 
        since there's no active queue mechanism.
        """
        # For traditional stacking mode, we don't track files in a queue
        # If you want to add file tracking in the future, add it here
        self.main.remaining_files_var.set("No files waiting")
    
    def calculate_remaining_time(self):
        """
        Calculate total remaining time based on number of images and average time per image.
        
        Returns:
            str: Formatted remaining time (HH:MM:SS)
        """
        if self.main.time_per_image <= 0 or self.main.processed_images_count == 0:
            return "--:--:--"
        
        # Calculate total remaining images (main folder + already counted additional folders)
        remaining_images = self.main.total_images_count - self.main.processed_images_count
        
        # Add images from additional folders that have NOT yet been counted in total_images_count
        if not self.main.processing_additional:  # Don't count during additional folder processing
            for folder in self.main.additional_folders:
                # Check if this folder has already been counted in total_images_count
                if folder not in self.main.total_additional_counted:
                    try:
                        fits_files = [f for f in os.listdir(folder) if f.lower().endswith(('.fit', '.fits'))]
                        remaining_images += len(fits_files)
                        # We don't update total_additional_counted here as it would modify the state
                    except Exception as e:
                        print(f"Error counting files in {folder}: {e}")
        
        if remaining_images <= 0:
            return "00:00:00"
        
        # Calculate remaining time based on average time per image
        estimated_time_remaining = remaining_images * self.main.time_per_image
        hours, remainder = divmod(int(estimated_time_remaining), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"
    
    def enhance_stack_name(self, stack_name, total_images, processed_images, 
                           additional_folders, total_additional_counted, 
                           processing_additional, batch_size):
        """
        Enhance stack name with progress information.
        
        Args:
            stack_name (str): Original stack name
            total_images (int): Total number of images to process
            processed_images (int): Number of processed images
            additional_folders (list): List of additional folders
            total_additional_counted (set): Set of already counted folders
            processing_additional (bool): Whether processing additional folders
            batch_size (int): Batch size
            
        Returns:
            str: Enhanced stack name with progress information
        """
        if stack_name is None:
            return None
            
        # Calculate global progress
        if total_images > 0:
            progress_percent = (processed_images * 100 / total_images)
            
            # Basic progress information
            progress_info = f" - Progress: {progress_percent:.1f}% ({processed_images}/{total_images} images)"
            
            # Calculate remaining batches in additional folders
            remaining_batches_info = ""
            if len(additional_folders) > 0:
                total_remaining_batches = 0
                batch_size = batch_size or 10  # Use 10 as default size if not specified
                
                for folder in additional_folders:
                    if folder not in total_additional_counted or processing_additional:
                        try:
                            fits_files = [f for f in os.listdir(folder) if f.lower().endswith(('.fit', '.fits'))]
                            folder_batches = (len(fits_files) + batch_size - 1) // batch_size  # Ceiling division
                            total_remaining_batches += folder_batches
                        except Exception as e:
                            print(f"Error calculating batches for {folder}: {e}")
                
                if total_remaining_batches > 0:
                    remaining_batches_info = f" - {total_remaining_batches} additional batches to process"
            
            # Add additional information depending on context
            if "Cumulative stack" in stack_name or "Stack cumulatif" in stack_name:
                # For a cumulative stack, highlight progress data
                if not processing_additional and len(additional_folders) > 0:
                    # If additional folders are waiting
                    pending_folders = len(additional_folders)
                    progress_info += f" - {pending_folders} additional folder(s) waiting{remaining_batches_info}"
                elif processing_additional:
                    # If processing additional folders
                    current_folder = len(additional_folders) - len([f for f in additional_folders if f not in total_additional_counted])
                    progress_info += f" - Processing additional folder {current_folder}/{len(additional_folders)}{remaining_batches_info}"
            
            # Update stack name for display
            if ("batch" in stack_name.lower() or "lot" in stack_name.lower()) and "(" in stack_name and ")" in stack_name:
                # Keep batch information but add progress
                batch_info = stack_name[stack_name.find("("):stack_name.find(")")+1]
                stack_name = f"{stack_name.split('(')[0]}{batch_info}{progress_info}"
            else:
                # Simply add progress information
                stack_name = f"{stack_name}{progress_info}"
                
        return stack_name