"""
Module pour la gestion de la prévisualisation des images astronomiques.
"""

import os
import numpy as np
import cv2
from PIL import Image, ImageTk # Requires Pillow
import tkinter as tk
from tkinter import ttk  # <--- Added this import
import traceback # Added for better error reporting in update_preview

# Import stretch tool using the package structure
try:
    from seestar.tools.stretch import Stretch
except ImportError:
    # Fallback or raise error if tools are essential
    print("Warning: Could not import Stretch tool from seestar.tools")
    # Define a dummy class if needed, or let it fail later if essential
    class Stretch:
        def stretch(self, data): return data # Dummy implementation

class PreviewManager:
    """
    Classe pour gérer la prévisualisation des images astronomiques dans le Canvas Tkinter.
    """

    def __init__(self, canvas, stack_label, info_text_widget):
        """
        Initialise le gestionnaire de prévisualisation.

        Args:
            canvas (tk.Canvas): Canvas Tkinter pour l'affichage des images.
            stack_label (ttk.Label): Label pour afficher le nom du stack/image.
            info_text_widget (tk.Text): Zone de texte pour les informations sur l'image.
        """
        # Input validation
        if not isinstance(canvas, tk.Canvas):
            raise TypeError("canvas must be a tkinter Canvas widget")
        # Corrected: Check for both tk.Label and ttk.Label
        if not isinstance(stack_label, (tk.Label, ttk.Label)):
             raise TypeError("stack_label must be a tkinter Label or ttk.Label widget")
        if not isinstance(info_text_widget, tk.Text):
             raise TypeError("info_text_widget must be a tkinter Text widget")

        self.canvas = canvas
        self.current_stack_label = stack_label
        self.image_info_text = info_text_widget # Store reference to the text widget
        self.tk_img = None # Holds the PhotoImage object to prevent garbage collection
        try:
            self.stretch_tool = Stretch() # Instance of the stretch algorithm
        except NameError: # Handle case where Stretch couldn't be imported
            print("Error: Stretch tool not available.")
            self.stretch_tool = None


        # Variables for zoom and pan (pan not implemented here yet)
        self.zoom_level = 1.0
        self.MAX_ZOOM = 10.0 # Increased max zoom
        self.MIN_ZOOM = 0.1 # Decreased min zoom
        self.original_image_data = None # Store the raw (or debayered) full-res data
        self.current_stack_name = "No Stack" # Name displayed in the label
        self.last_displayed_pil_image = None # Store the last PIL image used for display

        # Configurer les événements pour le zoom sur le canvas
        self.canvas.bind("<MouseWheel>", self._zoom_on_scroll)  # Windows/macOS
        self.canvas.bind("<Button-4>", self._zoom_on_scroll)   # Linux (scroll up)
        self.canvas.bind("<Button-5>", self._zoom_on_scroll)   # Linux (scroll down)
        # Add bindings for panning if needed (e.g., <B1-Motion>, <ButtonPress-1>, <ButtonRelease-1>)


    def create_test_image(self, width=300, height=200):
        """
        Crée une image numpy de test colorée (RGB).

        Args:
            width (int): Largeur de l'image.
            height (int): Hauteur de l'image.

        Returns:
            numpy.ndarray: Image de test (uint8 HxWx3 RGB).
        """
        img = np.zeros((height, width, 3), dtype=np.uint8)
        # Gradient rouge horizontal
        img[:, :, 0] = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
        # Gradient vert vertical
        img[:, :, 1] = np.tile(np.linspace(0, 255, height, dtype=np.uint8)[:, np.newaxis], (1, width))
        # Motif de grille bleue
        img[::20, :, 2] = 255
        img[:, ::20, 2] = 255
        return img

    def update_preview(self, image_data, stack_name=None, apply_stretch=False, info_text=None, force_redraw=False):
        """
        Met à jour l'aperçu avec les données d'image fournies.

        Args:
            image_data (numpy.ndarray): Données de l'image (HxW ou HxWx3, numeric type).
            stack_name (str, optional): Nom du stack/image à afficher.
            apply_stretch (bool): Appliquer un étirement automatique à l'image.
            info_text (str, optional): Texte d'information à afficher dans la zone dédiée.
            force_redraw (bool): Forcer le redessinage même si les données n'ont pas changé.
        """
        # Basic validation
        if image_data is None:
            self.clear_preview(self.tr("No Image Data", default="No Image Data")) # Use localization if available
            return

        if not isinstance(image_data, np.ndarray):
             print("Error: update_preview received non-numpy array data.")
             self.clear_preview(self.tr("Preview Error", default="Preview Error"))
             return

        # --- Data Preparation ---
        try:
            # Store original data if it's different or forced
            if force_redraw or self.original_image_data is None or not np.array_equal(image_data, self.original_image_data):
                 self.original_image_data = image_data.copy() # Store a copy

            # Update stack name
            self.current_stack_name = stack_name or self.tr("Stack", default="Stack")

            # Process the data for display (use the stored original data)
            display_data = self.original_image_data.copy()

            # Apply stretch if requested
            if apply_stretch and self.stretch_tool:
                display_data = self.stretch_tool.stretch(display_data) # Stretch returns 0-1 float
                # Scale stretched data (0-1) to 0-255 for uint8 conversion
                display_uint8 = (np.clip(display_data, 0, 1) * 255).astype(np.uint8)
            else:
                 # Normalize directly to 0-255 uint8
                 min_val, max_val = np.nanmin(display_data), np.nanmax(display_data)
                 if max_val > min_val:
                      display_norm = (display_data.astype(np.float32) - min_val) / (max_val - min_val)
                      display_uint8 = (display_norm * 255.0).astype(np.uint8)
                 elif max_val == min_val: # Handle constant image
                      display_uint8 = np.full_like(display_data, 128, dtype=np.uint8) # Mid-gray
                 else: # Handle all NaN or other weird cases
                      display_uint8 = np.zeros_like(display_data, dtype=np.uint8)

            # Ensure image is in RGB uint8 format (HxWx3) for PIL
            if display_uint8.ndim == 2:
                 # Convert grayscale to RGB
                 pil_img = Image.fromarray(display_uint8).convert('RGB')
            elif display_uint8.ndim == 3 and display_uint8.shape[-1] == 3:
                 # Assume it's already RGB
                 pil_img = Image.fromarray(display_uint8)
            else:
                 print(f"Error: Cannot display image with shape {display_uint8.shape}")
                 self.clear_preview(self.tr("Preview Error", default="Preview Error"))
                 return

            # Store this PIL image before resizing for redraws
            self.last_displayed_pil_image = pil_img

            # Redraw the canvas with the processed PIL image
            self._redraw_canvas(self.last_displayed_pil_image)

            # --- Update UI Elements ---
            # Update stack name label
            if self.current_stack_label and self.current_stack_label.winfo_exists():
                 self.current_stack_label.config(text=self.current_stack_name)

            # Update info text area if text is provided
            if info_text and self.image_info_text and self.image_info_text.winfo_exists():
                try:
                    self.image_info_text.config(state=tk.NORMAL)
                    self.image_info_text.delete(1.0, tk.END)
                    self.image_info_text.insert(tk.END, info_text)
                    self.image_info_text.config(state=tk.DISABLED)
                except tk.TclError: pass # Ignore if widget destroyed

        except Exception as e:
            print(f"Error during preview update: {e}")
            traceback.print_exc()
            self.clear_preview(self.tr("Preview Update Error", default="Preview Update Error"))

    def _redraw_canvas(self, pil_image_to_draw):
        """Redessine l'image PIL fournie sur le canvas, en tenant compte du zoom et de la taille du canvas."""
        if not pil_image_to_draw or not hasattr(self.canvas, 'winfo_exists') or not self.canvas.winfo_exists():
            return

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Prevent division by zero if canvas is not yet sized
        if canvas_width <= 1 or canvas_height <= 1:
            # print("Warning: Canvas not properly sized yet for redraw.")
            # Schedule redraw later?
            self.canvas.after(100, lambda: self._redraw_canvas(pil_image_to_draw))
            return

        img_width, img_height = pil_image_to_draw.size
        if img_width <= 0 or img_height <= 0:
             print(f"Warning: Invalid image dimensions for redraw ({img_width}x{img_height}).")
             return


        # Calculate the display size based on zoom and canvas fit
        # Fit the base image (zoom=1.0) within the canvas
        try:
             base_ratio = min(canvas_width / img_width, canvas_height / img_height)
        except ZeroDivisionError:
             print("Error: Image dimensions are zero, cannot calculate ratio.")
             return

        # Apply current zoom level
        display_ratio = base_ratio * self.zoom_level

        new_width = max(1, int(img_width * display_ratio))
        new_height = max(1, int(img_height * display_ratio))

        # Resize the PIL image using LANCZOS for better quality
        try:
             # Check Pillow version for resampling attribute
             if hasattr(Image, 'Resampling') and hasattr(Image.Resampling, 'LANCZOS'):
                 resample_method = Image.Resampling.LANCZOS
             else: # Fallback for older Pillow versions
                 resample_method = Image.LANCZOS
             resized_img = pil_image_to_draw.resize((new_width, new_height), resample_method)
        except Exception as resize_err:
             print(f"Error resizing preview image: {resize_err}")
             return # Cannot proceed if resize fails


        # Convert PIL image to Tkinter PhotoImage
        # Keep a reference to prevent garbage collection!
        try:
            self.tk_img = ImageTk.PhotoImage(image=resized_img)
        except Exception as photoimg_err:
            print(f"Error creating PhotoImage: {photoimg_err}")
            return


        # Clear previous drawings and display the new image centered
        try:
            self.canvas.delete("all")
            # Place image at the center of the canvas
            self.canvas.create_image(canvas_width / 2, canvas_height / 2, anchor="center", image=self.tk_img)
        except tk.TclError as draw_err:
             # Catch errors if canvas is destroyed during redraw attempt
             print(f"Error drawing image on canvas: {draw_err}")


    def _update_info_text_area(self):
        """Met à jour la zone d'information (e.g., avec dimensions/zoom)."""
        # This is better handled by update_image_info based on FITS header.
        pass # No action needed here now


    def clear_preview(self, message=None):
        """Efface la prévisualisation et affiche un message optionnel."""
        if not hasattr(self.canvas, 'winfo_exists') or not self.canvas.winfo_exists(): return

        self.canvas.delete("all")
        self.original_image_data = None
        self.tk_img = None # Clear PhotoImage reference
        self.last_displayed_pil_image = None
        self.current_stack_name = self.tr("no_current_stack", default="No Stack") # Reset name

        if message:
            # Display message centered on canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            if canvas_width > 1 and canvas_height > 1:
                 try:
                      self.canvas.create_text(
                           canvas_width / 2, canvas_height / 2,
                           text=message, fill="white", font=("Arial", 12), anchor="center", justify="center"
                      )
                 except tk.TclError as text_err:
                       print(f"Error drawing clear message on canvas: {text_err}")


        # Update UI labels/text areas
        if self.current_stack_label and self.current_stack_label.winfo_exists():
            self.current_stack_label.config(text=self.current_stack_name)
        if self.image_info_text and self.image_info_text.winfo_exists():
            try:
                self.image_info_text.config(state=tk.NORMAL)
                self.image_info_text.delete(1.0, tk.END)
                self.image_info_text.insert(tk.END, self.tr("image_info_waiting", default="Image info: waiting..."))
                self.image_info_text.config(state=tk.DISABLED)
            except tk.TclError: pass


    def refresh_preview_with_current_data(self, apply_stretch_override=None):
        """
        Actualise l'aperçu en utilisant les données `original_image_data` stockées.
        Utile après un zoom, un redimensionnement, ou un changement de paramètre de stretch.

        Args:
            apply_stretch_override (bool, optional): Si spécifié, utilise cette valeur pour l'étirement,
                                                      sinon utilise la valeur actuelle de l'interface.
        """
        if self.original_image_data is None:
            # print("Refresh preview called but no original data stored.")
            return # Nothing to refresh

        # How to get the UI stretch value? Assume it's accessible via a main GUI instance passed somewhere
        # For now, let's assume apply_stretch_override is passed correctly or handled externally
        current_stretch_setting = apply_stretch_override if apply_stretch_override is not None else False # Default to False if no override


        # Call update_preview with stored data and current settings
        self.update_preview(
            image_data=self.original_image_data, # Use the stored data
            stack_name=self.current_stack_name, # Use the stored name
            apply_stretch=current_stretch_setting, # Use determined stretch value
            force_redraw=True # Force redraw as parameters might have changed
        )

    def _zoom_on_scroll(self, event):
        """Gère le zoom avec la molette de la souris (multi-plateforme)."""
        if self.original_image_data is None: return # No image to zoom

        # Determine zoom direction
        zoom_factor = 1.1 # Zoom in factor
        if event.num == 5 or event.delta < 0: # Scroll down/away
            self.zoom_level /= zoom_factor
        elif event.num == 4 or event.delta > 0: # Scroll up/towards
            self.zoom_level *= zoom_factor
        else: # Should not happen with bound events
            return

        # Clamp zoom level within min/max bounds
        self.zoom_level = np.clip(self.zoom_level, self.MIN_ZOOM, self.MAX_ZOOM)

        # Refresh the preview with the new zoom level
        if self.last_displayed_pil_image: # Redraw directly if we have the PIL image
            self._redraw_canvas(self.last_displayed_pil_image)
        # else: # Fallback to full update if PIL image isn't stored (might be slow)
        #      self.refresh_preview_with_current_data() # Need to implement access to stretch var

    def reset_zoom(self):
        """Réinitialise le niveau de zoom à 1.0."""
        if self.original_image_data is not None and self.zoom_level != 1.0:
            self.zoom_level = 1.0
            # Refresh the preview
            if self.last_displayed_pil_image:
                self._redraw_canvas(self.last_displayed_pil_image)
            # else: # Fallback (might be slow)
            #      self.refresh_preview_with_current_data() # Need to implement access to stretch var

    # Helper to get localized text if needed within this class
    def tr(self, key, default=None):
         # Placeholder: Assumes access to a localization instance if needed directly
         # In practice, text comes from the main GUI via update_preview arguments
         return default or key