# --- START OF FILE seestar/gui/preview.py ---
"""
Module pour la gestion de la prévisualisation des images astronomiques (Canvas + Panning).
Le traitement de l'image pour l'affichage (WB, Stretch, Gamma) est effectué ici.
"""
import tkinter as tk
# from tkinter import ttk # No longer needed here
import numpy as np
from PIL import Image, ImageTk
import traceback

# Import stretch/color tools using the package structure
try:
    from seestar.tools.stretch import StretchPresets, ColorCorrection
except ImportError:
    print("Warning: Could not import StretchPresets/ColorCorrection from seestar.tools")
    # Define dummy classes if needed for basic functionality
    class StretchPresets:
        @staticmethod
        def linear(data, bp=0., wp=1.): wp=max(wp,bp+1e-6); return np.clip((data-bp)/(wp-bp), 0, 1)
        @staticmethod
        def asinh(data, scale=1., bp=0.): data_s=data-bp; data_c=np.maximum(data_s,0.); max_v=np.nanmax(data_c); den=np.arcsinh(scale*max_v); return np.arcsinh(scale*data_c)/den if den>1e-6 else np.zeros_like(data)
        @staticmethod
        def logarithmic(data, scale=1., bp=0.): data_s=data-bp; data_c=np.maximum(data_s,1e-10); max_v=np.nanmax(data_c); den=np.log1p(scale*max_v); return np.log1p(scale*data_c)/den if den>1e-6 else np.zeros_like(data)
        @staticmethod
        def gamma(data, gamma=1.0): return np.power(np.maximum(data, 1e-10), gamma)
    class ColorCorrection:
        @staticmethod
        def white_balance(data, r=1., g=1., b=1.):
            if data is None or data.ndim != 3: return data
            corr=data.astype(np.float32).copy(); corr[...,0]*=r; corr[...,1]*=g; corr[...,2]*=b; return np.clip(corr,0,1)


class PreviewManager:
    """
    Gère l'affichage et l'interaction (zoom, pan) avec l'image dans un Canvas Tkinter.
    Applique la balance des blancs, l'étirement et le gamma pour l'affichage.
    """
    def __init__(self, canvas):
        """
        Initialise le gestionnaire de prévisualisation.

        Args:
            canvas (tk.Canvas): Canvas Tkinter pour l'affichage des images.
        """
        if not isinstance(canvas, tk.Canvas):
            raise TypeError("canvas must be a tkinter Canvas widget")

        self.canvas = canvas
        self.tk_img = None  # Holds the PhotoImage object to prevent garbage collection
        self.stretch_presets = StretchPresets()
        self.color_correction = ColorCorrection()

        # State Variables
        self.zoom_level = 1.0
        self.MAX_ZOOM = 15.0 # Increased max zoom
        self.MIN_ZOOM = 0.05 # Allow zooming out more
        self.image_data_raw = None     # The initial data (0-1 float, potentially debayered)
        self.image_data_wb = None      # Data after white balance (for histogram)
        self.last_displayed_pil_image = None # Stretched PIL image for redraws
        self.current_display_params = {} # Store params used for last display

        # Pan State
        self._is_panning = False
        self._pan_start_x = 0
        self._pan_start_y = 0
        self._view_offset_x = 0 # Current pan offset X relative to center
        self._view_offset_y = 0 # Current pan offset Y relative to center
        self._img_id_on_canvas = None # Store the ID of the image item on canvas

        # Bind events
        self.canvas.bind("<MouseWheel>", self._zoom_on_scroll)  # Windows/macOS
        self.canvas.bind("<Button-4>", self._zoom_on_scroll)  # Linux (scroll up)
        self.canvas.bind("<Button-5>", self._zoom_on_scroll)  # Linux (scroll down)
        self.canvas.bind("<ButtonPress-1>", self._start_pan)   # Left-click press
        self.canvas.bind("<B1-Motion>", self._pan_image)       # Left-click drag
        self.canvas.bind("<ButtonRelease-1>", self._stop_pan)    # Left-click release
        self.canvas.bind("<Configure>", self._on_canvas_resize) # Handle resize

    def update_preview(self, raw_image_data, params):
        """
        Met à jour l'aperçu avec les données brutes et les paramètres d'affichage.
        Applique WB, Stretch, Gamma pour l'affichage.

        Args:
            raw_image_data (np.ndarray): Données brutes (HxW ou HxWx3, float32, 0-1).
                                         Doit être l'image après chargement/débayering.
            params (dict): Dictionnaire des paramètres d'affichage:
                           {'r_gain', 'g_gain', 'b_gain', 'stretch_method',
                            'black_point', 'white_point', 'gamma'}

        Returns:
            tuple: (processed_pil_image, data_for_histogram)
                   processed_pil_image: PIL Image prête pour affichage (après stretch/gamma) ou None on error.
                   data_for_histogram: NumPy array après WB, avant stretch/gamma (0-1 float) or None on error.
        """
        if raw_image_data is None:
            self.clear_preview("No Image Data")
            return None, None

        if not isinstance(raw_image_data, np.ndarray):
            print("Error: update_preview received non-numpy array data.")
            self.clear_preview("Preview Error")
            return None, None

        # Check if data or parameters have actually changed significantly
        data_changed = self.image_data_raw is None or not np.array_equal(self.image_data_raw, raw_image_data)
        params_changed = self.current_display_params != params

        if not data_changed and not params_changed:
             # print("Skipping preview update - no change") # Debug
             # Return previous results if no change
             return self.last_displayed_pil_image, self.image_data_wb

        # Store new data and params
        if data_changed:
            self.image_data_raw = raw_image_data.copy()
            # Reset zoom/pan only if image data itself changes, not just params
            self.reset_zoom_and_pan()

        self.current_display_params = params.copy()

        try:
            # --- 1. Apply White Balance (if color) ---
            if self.image_data_raw.ndim == 3 and self.image_data_raw.shape[2] == 3:
                self.image_data_wb = self.color_correction.white_balance(
                    self.image_data_raw,
                    r=params.get('r_gain', 1.0),
                    g=params.get('g_gain', 1.0),
                    b=params.get('b_gain', 1.0)
                )
            else: # Grayscale
                self.image_data_wb = self.image_data_raw.copy() # Use a copy

            # This data_for_histogram is what the histogram should display
            data_for_histogram = self.image_data_wb.copy()

            # --- 2. Apply Stretch ---
            stretch_method = params.get('stretch_method', 'Linear')
            bp = params.get('black_point', 0.0)
            wp = params.get('white_point', 1.0)
            # Start stretch from the white-balanced data
            data_stretched = self.image_data_wb

            if stretch_method == "Linear":
                data_stretched = self.stretch_presets.linear(data_stretched, bp, wp)
            elif stretch_method == "Asinh":
                # Simple asinh scale guess - adjust as needed
                asinh_scale = 10.0 / max(0.01, wp - bp) if wp > bp else 10.0
                data_stretched = self.stretch_presets.asinh(data_stretched, scale=asinh_scale, bp=bp)
            elif stretch_method == "Log":
                log_scale = 10.0 # Fixed scale, could be parameter later
                data_stretched = self.stretch_presets.logarithmic(data_stretched, scale=log_scale, bp=bp)
            # Ensure stretch result is clipped 0-1
            data_stretched = np.clip(data_stretched, 0.0, 1.0)

            # --- 3. Apply Gamma ---
            gamma = params.get('gamma', 1.0)
            data_final = self.stretch_presets.gamma(data_stretched, gamma) # Apply gamma
            data_final = np.clip(data_final, 0.0, 1.0) # Clip again after gamma

            # --- 4. Convert to Display Format (uint8 PIL) ---
            # Use np.nan_to_num before converting to uint8
            display_uint8 = (np.nan_to_num(data_final) * 255).astype(np.uint8)

            if display_uint8.ndim == 2:
                pil_img = Image.fromarray(display_uint8, mode='L').convert("RGB") # Ensure RGB for display consistency
            elif display_uint8.ndim == 3:
                pil_img = Image.fromarray(display_uint8, mode='RGB')
            else:
                raise ValueError(f"Cannot display image with processed shape {display_uint8.shape}")

            self.last_displayed_pil_image = pil_img # Store for redraws

            # --- 5. Redraw Canvas ---
            self._redraw_canvas()

            return pil_img, data_for_histogram # Return images

        except Exception as e:
            print(f"Error during preview processing: {e}")
            traceback.print_exc(limit=2)
            self.clear_preview("Preview Processing Error")
            return None, None

    def _redraw_canvas(self):
        """Redessine l'image PIL stockée sur le canvas, gérant zoom et pan."""
        pil_image_to_draw = self.last_displayed_pil_image
        if pil_image_to_draw is None or not self.canvas.winfo_exists():
            return

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1: return # Canvas not ready

        img_width, img_height = pil_image_to_draw.size
        if img_width <= 0 or img_height <= 0: return # Invalid image

        # Calculate display size based on zoom level
        display_width = int(img_width * self.zoom_level)
        display_height = int(img_height * self.zoom_level)

        # Ensure minimum size for display
        display_width = max(1, display_width)
        display_height = max(1, display_height)

        # Resize using appropriate method based on zoom level
        # Use NEAREST for large zoom-in to preserve pixels, LANCZOS otherwise
        try:
            if self.zoom_level > 2.0: # Threshold for using NEAREST
                 resample_method = Image.NEAREST
            elif hasattr(Image, "Resampling") and hasattr(Image.Resampling, "LANCZOS"):
                 resample_method = Image.Resampling.LANCZOS
            else: # Fallback for older Pillow versions
                 resample_method = Image.LANCZOS

            resized_img = pil_image_to_draw.resize((display_width, display_height), resample_method)
        except Exception as resize_err:
            print(f"Error resizing preview image: {resize_err}")
            # Optionally clear preview or show error?
            # self.clear_preview("Resize Error")
            return

        # Convert PIL image to Tkinter PhotoImage (keep reference!)
        try:
            self.tk_img = ImageTk.PhotoImage(image=resized_img)
        except Exception as photoimg_err:
            print(f"Error creating PhotoImage: {photoimg_err}")
            self.tk_img = None # Ensure tk_img is cleared on error
            return

        # Calculate image position on canvas including pan offset
        # The position is the coordinate of the *center* of the image
        display_x = canvas_width / 2 + self._view_offset_x
        display_y = canvas_height / 2 + self._view_offset_y

        # --- Draw image on canvas ---
        try:
            # Delete previous image item if it exists
            if self._img_id_on_canvas:
                self.canvas.delete(self._img_id_on_canvas)
                self._img_id_on_canvas = None

            # Create the new image item
            if self.tk_img: # Ensure PhotoImage was created successfully
                 self._img_id_on_canvas = self.canvas.create_image(
                     display_x, display_y, anchor="center", image=self.tk_img
                 )
                 # Ensure message is below image if both exist
                 self.canvas.tag_lower(self._img_id_on_canvas, "message")

        except tk.TclError as draw_err:
            # Catch errors if canvas is destroyed during redraw attempt
            print(f"Error drawing image on canvas: {draw_err}")
            self._img_id_on_canvas = None


    def clear_preview(self, message=None):
        """Efface la prévisualisation et affiche un message optionnel."""
        if not hasattr(self.canvas, "winfo_exists") or not self.canvas.winfo_exists(): return

        self.canvas.delete("all") # Clear everything including potential messages
        self._img_id_on_canvas = None
        self.image_data_raw = None
        self.image_data_wb = None
        self.tk_img = None
        self.last_displayed_pil_image = None
        self.reset_zoom_and_pan() # Reset view state

        if message:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            if canvas_width > 1 and canvas_height > 1:
                try:
                    # Draw message centered, use a specific tag
                    self.canvas.create_text(
                        canvas_width / 2, canvas_height / 2, text=message,
                        fill="gray", font=("Arial", 11), anchor="center", justify="center",
                        tags="message" # Add tag for potential later deletion/management
                    )
                except tk.TclError: pass # Ignore if canvas destroyed


    def trigger_redraw(self):
        """Forces a redraw using the last displayed PIL image."""
        # Used primarily for resize events where data/params haven't changed
        self._redraw_canvas()

    # --- Zoom Logic ---
    def _zoom_on_scroll(self, event):
        """Gère le zoom avec la molette."""
        if self.last_displayed_pil_image is None: return

        # Determine zoom factor
        zoom_factor = 1.15 # Speed of zoom
        if event.num == 5 or event.delta < 0: # Scroll down/away (Zoom Out)
            new_zoom = self.zoom_level / zoom_factor
        elif event.num == 4 or event.delta > 0: # Scroll up/towards (Zoom In)
            new_zoom = self.zoom_level * zoom_factor
        else: return # Should not happen

        # Clamp zoom level
        new_zoom = np.clip(new_zoom, self.MIN_ZOOM, self.MAX_ZOOM)

        # If zoom didn't change significantly, do nothing
        if abs(new_zoom - self.zoom_level) < 1e-6: return

        # --- Calculate Zoom Anchor ---
        # Get mouse coordinates relative to the canvas
        canvas_x = event.x
        canvas_y = event.y

        # Get current image center coordinates on canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_center_x = canvas_width / 2 + self._view_offset_x
        img_center_y = canvas_height / 2 + self._view_offset_y

        # Calculate mouse position relative to the *current* image center
        mouse_rel_img_center_x = canvas_x - img_center_x
        mouse_rel_img_center_y = canvas_y - img_center_y

        # --- Adjust Pan Offset ---
        # Calculate how much the image center needs to shift to keep the point
        # under the mouse stationary relative to the canvas viewport.
        zoom_ratio = new_zoom / self.zoom_level
        # The distance from the new center to the mouse point should be zoom_ratio times the old distance
        # new_center_x = canvas_x - mouse_rel_img_center_x * zoom_ratio
        # new_center_y = canvas_y - mouse_rel_img_center_y * zoom_ratio
        # The pan offset is the difference between the new center and the canvas center
        # self._view_offset_x = new_center_x - canvas_width / 2
        # self._view_offset_y = new_center_y - canvas_height / 2
        # Simplified: Keep the point under the mouse fixed relative to the canvas origin.
        # Find the data coordinate under the mouse before zoom.
        # Find where that data coordinate will be after zoom.
        # Adjust offset so the new position matches the original mouse position.
        # This involves mapping canvas <-> image coords, which is complex.
        # Let's use the simpler approach: adjust offset based on mouse relative to *canvas center*
        mouse_rel_canvas_center_x = canvas_x - canvas_width / 2
        mouse_rel_canvas_center_y = canvas_y - canvas_height / 2

        self._view_offset_x = mouse_rel_canvas_center_x - (mouse_rel_canvas_center_x - self._view_offset_x) * zoom_ratio
        self._view_offset_y = mouse_rel_canvas_center_y - (mouse_rel_canvas_center_y - self._view_offset_y) * zoom_ratio


        # Update zoom level and redraw
        self.zoom_level = new_zoom
        self._redraw_canvas()

    def reset_zoom_and_pan(self):
        """Réinitialise le zoom et le panoramique."""
        needs_redraw = False
        if abs(self.zoom_level - 1.0) > 1e-6:
            self.zoom_level = 1.0
            needs_redraw = True
        if abs(self._view_offset_x) > 1e-6 or abs(self._view_offset_y) > 1e-6:
            self._view_offset_x = 0.0
            self._view_offset_y = 0.0
            needs_redraw = True

        if needs_redraw and self.last_displayed_pil_image:
             self._redraw_canvas()

    # --- Pan Logic ---
    def _start_pan(self, event):
        """Démarre le panoramique."""
        if self.last_displayed_pil_image is None: return
        # Check if click is actually on the image (optional, prevents panning background)
        # This requires getting image bounds on canvas, which can be complex with zoom/pan.
        # For simplicity, allow panning anywhere on canvas if an image exists.
        self._is_panning = True
        self._pan_start_x = event.x
        self._pan_start_y = event.y
        self.canvas.config(cursor="fleur") # Change cursor

    def _stop_pan(self, event):
        """Arrête le panoramique."""
        if self._is_panning:
             self._is_panning = False
             self.canvas.config(cursor="") # Reset cursor

    def _pan_image(self, event):
        """Déplace l'image pendant le panoramique."""
        if not self._is_panning or self.last_displayed_pil_image is None: return

        dx = event.x - self._pan_start_x
        dy = event.y - self._pan_start_y

        # Update view offset immediately
        self._view_offset_x += dx
        self._view_offset_y += dy

        # Update start position for the next motion event delta calculation
        self._pan_start_x = event.x
        self._pan_start_y = event.y

        # Move the existing image item on canvas directly for smoother panning
        if self._img_id_on_canvas:
            try:
                 # Calculate new center coordinates based on updated offset
                 canvas_width = self.canvas.winfo_width()
                 canvas_height = self.canvas.winfo_height()
                 display_x = canvas_width / 2 + self._view_offset_x
                 display_y = canvas_height / 2 + self._view_offset_y
                 # Move the image item instead of full redraw
                 self.canvas.coords(self._img_id_on_canvas, display_x, display_y)
            except tk.TclError: # Handle if canvas/item destroyed
                 self._img_id_on_canvas = None # Clear invalid ID
                 self._redraw_canvas() # Fallback to full redraw if move fails
            except Exception as e:
                 print(f"Error moving image during pan: {e}")
                 self._redraw_canvas() # Fallback
        else:
             self._redraw_canvas() # Fallback if no image ID


    # --- Canvas Resize Logic ---
    def _on_canvas_resize(self, event=None): # Add event=None for direct calls
        """Redraws the image when the canvas size changes."""
        # A small delay might prevent excessive redraws during rapid interactive resizing
        # Could implement debounce here if needed, but direct redraw is often acceptable.
        if self.last_displayed_pil_image:
             self._redraw_canvas()
        elif self.image_data_raw is None: # Redraw welcome message if empty
             self.clear_preview(self.tr("Select input/output folders.", default="Select input/output folders."))

    # --- Helper Functions ---
    def tr(self, key, default=None):
        # Placeholder for localization if needed directly
        return default or key
# --- END OF FILE seestar/gui/preview.py ---