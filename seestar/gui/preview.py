"""
Module pour la gestion de la prévisualisation des images astronomiques (Canvas + Panning).
Le traitement de l'image pour l'affichage (WB, Stretch, Gamma, B/C/S) est effectué ici.
(Version Révisée 2: Accepte stack_count et l'affiche)
"""
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk, ImageEnhance
import traceback
import platform # For finding fonts
import os 

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
    Applique la balance des blancs, l'étirement, le gamma et B/C/S pour l'affichage.
    Affiche le nombre d'images stackées dans le coin supérieur droit.
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
        self.MAX_ZOOM = 15.0
        self.MIN_ZOOM = 0.05
        self.image_data_raw = None     # The initial data (0-1 float, potentially debayered)
        self.image_data_raw_shape = None # Store shape to detect dimension changes
        self.image_data_wb = None      # Data after white balance (for histogram)
        self.last_displayed_pil_image = None # Stretched PIL image for redraws
        self.current_display_params = {} # Store params used for last display
        self.image_info_text_content = "" # Store content for info text area
        self.current_stack_count = 0 # Add variable to store stack count

        # --- Store info for text overlay ---
        self.display_img_count = 0
        self.display_total_imgs = 0
        self.display_current_batch = 0
        self.display_total_batches = 0

        # Pan State
        self._is_panning = False
        self._pan_start_x = 0
        self._pan_start_y = 0
        self._view_offset_x = 0 # Current pan offset X relative to center
        self._view_offset_y = 0 # Current pan offset Y relative to center
        self._img_id_on_canvas = None # Store the ID of the image item on canvas
        self._text_id_on_canvas = None # ID for the stack count text
        # --- Load Background Image ---
        self.bg_pil_image = None # Holds the PIL Image for the background
        self.tk_bg_img = None    # Holds the Tkinter PhotoImage for the background
        try:
            # IMPORTANT: Replace this with the ACTUAL path to YOUR background image file!
            # Example: bg_image_path = "assets/background.png"
            bg_image_path = 'icon/back.png'

            if os.path.exists(bg_image_path):
                # Load the background image using Pillow
                self.bg_pil_image = Image.open(bg_image_path)
                # Convert to RGBA to handle potential transparency (optional but safer)
                # self.bg_pil_image = self.bg_pil_image.convert("RGBA")
                print(f"DEBUG: Background image loaded: {bg_image_path} (Size: {self.bg_pil_image.size})")
                # We will create the Tk PhotoImage later, when needed for drawing
            else:
                print(f"Warning: Background image file not found at: {bg_image_path}")
        except FileNotFoundError:
             print(f"Error: Background image file not found at path: {bg_image_path}")
        except Exception as e:
            print(f"Error loading background image: {e}")
        # --- End Background Image Loading ---

        # Load a suitable font
        self._load_font()

        # Bind events
        self.canvas.bind("<MouseWheel>", self._zoom_on_scroll)  # Windows/macOS
        self.canvas.bind("<Button-4>", self._zoom_on_scroll)  # Linux (scroll up)
        self.canvas.bind("<Button-5>", self._zoom_on_scroll)  # Linux (scroll down)
        self.canvas.bind("<ButtonPress-1>", self._start_pan)   # Left-click press
        self.canvas.bind("<B1-Motion>", self._pan_image)       # Left-click drag
        self.canvas.bind("<ButtonRelease-1>", self._stop_pan)    # Left-click release
        self.canvas.bind("<Configure>", self._on_canvas_resize) # Handle resize

        
    def _load_font(self):
        """ Tries to load a suitable small font. """
        # Define preferred fonts
        fonts = ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'] # Common sans-serif
        font_size = 10 # Small size

        # Select based on platform for better defaults
        os_name = platform.system()
        if os_name == "Windows":
            fonts.insert(0, 'Segoe UI') # Good default on Windows
            fonts.insert(1, 'Calibri')
        elif os_name == "Darwin": # macOS
            fonts.insert(0, 'Helvetica Neue')
            fonts.insert(0, 'San Francisco') # Newer default
        # Linux will likely pick up DejaVu or Liberation

        # Try loading the font
        self.display_font = None
        for font_name in fonts:
            try:
                # Try loading with PIL first (might fail if font not in standard paths)
                # self.display_font = ImageFont.truetype(f"{font_name}.ttf", font_size)
                # Let's rely on Tkinter finding the font by name primarily
                self.tk_font_tuple = (font_name, font_size)
                # Test if Tkinter recognizes it (crude test)
                _ = tk.font.Font(family=font_name, size=font_size)
                print(f"Using Tkinter font: {font_name} {font_size}")
                self.display_font = None # Indicate we should use Tkinter font tuple
                break
            except Exception:
                 continue # Try next font

        # Fallback if no preferred font found (use Tkinter default)
        if not hasattr(self, 'tk_font_tuple'):
             self.tk_font_tuple = ('TkDefaultFont', font_size)
             print(f"Warning: Could not load preferred font. Using fallback: {self.tk_font_tuple}")
             self.display_font = None # Ensure PIL font isn't used




    def process_image(self, raw_image_data, params):
        """Return processed PIL image and histogram data without GUI operations."""

        if raw_image_data is None or not isinstance(raw_image_data, np.ndarray):
            return None, None

        new_shape = raw_image_data.shape
        if self.image_data_raw_shape is None or new_shape != self.image_data_raw_shape:
            self.reset_zoom_and_pan()
            self.image_data_raw_shape = new_shape

        self.image_data_raw = raw_image_data
        self.current_display_params = params.copy()

        try:
            data = self.image_data_raw.copy()
            if data.ndim == 3 and data.shape[2] == 3:
                self.image_data_wb = self.color_correction.white_balance(
                    data,
                    r=params.get("r_gain", 1.0),
                    g=params.get("g_gain", 1.0),
                    b=params.get("b_gain", 1.0),
                )
            else:
                self.image_data_wb = data

            hist_data = self.image_data_wb.copy()

            stretch_method = params.get("stretch_method", "Linear")
            bp = params.get("black_point", 0.0)
            wp = params.get("white_point", 1.0)
            if stretch_method == "Linear":
                data_stretched = self.stretch_presets.linear(self.image_data_wb, bp, wp)
            elif stretch_method == "Asinh":
                scale = 10.0 / max(0.01, wp - bp) if wp > bp else 10.0
                data_stretched = self.stretch_presets.asinh(self.image_data_wb, scale=scale, bp=bp)
            elif stretch_method == "Log":
                data_stretched = self.stretch_presets.logarithmic(self.image_data_wb, scale=10.0, bp=bp)
            else:
                data_stretched = self.image_data_wb

            data_stretched = np.clip(data_stretched, 0.0, 1.0)
            gamma = params.get("gamma", 1.0)
            data_gamma = self.stretch_presets.gamma(data_stretched, gamma)
            data_gamma = np.clip(data_gamma, 0.0, 1.0)

            disp_uint8 = (np.nan_to_num(data_gamma) * 255).astype(np.uint8)
            pil_img = Image.fromarray(disp_uint8, mode="RGB" if disp_uint8.ndim == 3 else "L")

            brightness = params.get("brightness", 1.0)
            contrast = params.get("contrast", 1.0)
            saturation = params.get("saturation", 1.0)
            if abs(brightness - 1.0) > 1e-3:
                pil_img = ImageEnhance.Brightness(pil_img).enhance(brightness)
            if abs(contrast - 1.0) > 1e-3:
                pil_img = ImageEnhance.Contrast(pil_img).enhance(contrast)
            if pil_img.mode == "RGB" and abs(saturation - 1.0) > 1e-3:
                pil_img = ImageEnhance.Color(pil_img).enhance(saturation)

            return pil_img, hist_data
        except Exception:
            traceback.print_exc(limit=2)
            return None, None

    def display_processed_image(self, pil_img):
        """Display PIL image on the canvas. Must run on main thread."""

        if pil_img is None:
            self.clear_preview("Preview Processing Error")
            return

        self.last_displayed_pil_image = pil_img
        self._redraw_canvas()




# --- DANS seestar/gui/preview.py ---
# --- DANS la classe PreviewManager ---

    def update_preview(self, raw_image_data, params, stack_count=None, total_images=None, current_batch=None, total_batches=None):
        """Updates the preview with raw data. Heavy processing is offloaded."""

        if stack_count is not None:
            self.display_img_count = stack_count
        if total_images is not None:
            self.display_total_imgs = total_images
        if current_batch is not None:
            self.display_current_batch = current_batch
        if total_batches is not None:
            self.display_total_batches = total_batches

        if raw_image_data is None or not isinstance(raw_image_data, np.ndarray):
            if self.image_data_raw is not None:
                self.clear_preview("No Image Data")
            return None, None

        pil_img, hist_data = self.process_image(raw_image_data, params)
        if pil_img is not None:
            self.display_processed_image(pil_img)
        return pil_img, hist_data





    def _redraw_canvas(self):
        """Redessine l'image de fond (si chargée), l'image PIL astro et le texte d'info."""
        pil_image_to_draw = self.last_displayed_pil_image # The astronomical image

        if not self.canvas.winfo_exists(): return
        # print("DEBUG: _redraw_canvas called.")

        # --- Clear previous drawn items ---
        # Clear specific items instead of "all" to preserve bindings etc.
        self.canvas.delete("message") # Delete any messages like "No Image Data"
        if hasattr(self, '_bg_img_id_on_canvas') and self._bg_img_id_on_canvas:
            self.canvas.delete(self._bg_img_id_on_canvas)
            self._bg_img_id_on_canvas = None
        if self._img_id_on_canvas:
            self.canvas.delete(self._img_id_on_canvas)
            self._img_id_on_canvas = None
        if self._text_id_on_canvas:
            self.canvas.delete(self._text_id_on_canvas)
            self._text_id_on_canvas = None
        # We clear self.tk_bg_img reference here, it will be recreated if needed
        self.tk_bg_img = None

        # --- Get Canvas Size ---
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1: return # Avoid drawing on tiny canvas

        # --- 1. Draw Background Image (if available) ---
        bg_drawn = False
        if self.bg_pil_image: # Check if the background PIL image was loaded in __init__
            try:
                # Create the Tkinter PhotoImage for the background *now*
                self.tk_bg_img = ImageTk.PhotoImage(self.bg_pil_image)
                # Draw it centered
                self._bg_img_id_on_canvas = self.canvas.create_image(
                    canvas_width / 2,
                    canvas_height / 2,
                    anchor="center",
                    image=self.tk_bg_img,
                    tags="background" # Add a tag for potential layering
                )
                bg_drawn = True
                # print("DEBUG: Background image drawn.") # Optional debug
            except Exception as bg_draw_err:
                print(f"Error creating/drawing background PhotoImage: {bg_draw_err}")
                self.tk_bg_img = None # Clear reference if creation failed
                self._bg_img_id_on_canvas = None

        # --- 2. Draw Astronomical Preview Image (if available) ---
        if pil_image_to_draw:
            img_width, img_height = pil_image_to_draw.size
            if img_width > 0 and img_height > 0:
                # --- Resize and create Tk PhotoImage for astro image ---
                display_width = max(1, int(img_width * self.zoom_level))
                display_height = max(1, int(img_height * self.zoom_level))
                try:
                    resample_filter = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
                    if self.zoom_level > 2.0: resample_filter = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST
                    resized_img = pil_image_to_draw.resize((display_width, display_height), resample_filter)
                except Exception as resize_err: print(f"Error resizing preview image: {resize_err}"); return
                try:
                    self.tk_img = ImageTk.PhotoImage(image=resized_img) # Store main image reference
                except Exception as photoimg_err: print(f"Error creating PhotoImage: {photoimg_err}"); self.tk_img = None; return

                # --- Calculate position ---
                display_x = canvas_width / 2 + self._view_offset_x
                display_y = canvas_height / 2 + self._view_offset_y

                # --- Draw astro image ---
                try:
                    if self.tk_img:
                        self._img_id_on_canvas = self.canvas.create_image(
                            display_x, display_y,
                            anchor="center",
                            image=self.tk_img,
                            tags="foreground" # Add a tag
                        )
                except tk.TclError as draw_err: print(f"Error drawing image: {draw_err}"); self._img_id_on_canvas = None

        # --- 3. Draw Stack/Batch Info Text (if applicable) ---
        # (This part is largely the same as before)
        if self.display_img_count > 0:
            try:
                total_imgs_str = str(self.display_total_imgs) if self.display_total_imgs > 0 else "?"
                total_batches_str = str(self.display_total_batches) if self.display_total_batches > 0 else "?"
                text_content = f"Img: #{self.display_img_count}/{total_imgs_str}\nBatch: {self.display_current_batch}/{total_batches_str}"
                text_x = canvas_width - 10; text_y = 10
                font_to_use = self.tk_font_tuple if hasattr(self, 'tk_font_tuple') else ('TkDefaultFont', 10)
                self._text_id_on_canvas = self.canvas.create_text(
                    text_x, text_y,
                    text=text_content,
                    anchor="ne", fill="yellow", font=font_to_use, justify=tk.RIGHT,
                    tags="textoverlay" # Add a tag
                )
            except tk.TclError as text_err:
                print(f"Error drawing stack count text: {text_err}")
                self._text_id_on_canvas = None

        # --- 4. Ensure Correct Layering ---
        # Lower the background, raise the foreground and text
        if hasattr(self, '_bg_img_id_on_canvas') and self._bg_img_id_on_canvas:
            self.canvas.tag_lower(self._bg_img_id_on_canvas)
        if self._img_id_on_canvas:
            self.canvas.tag_raise(self._img_id_on_canvas)
        if self._text_id_on_canvas:
            self.canvas.tag_raise(self._text_id_on_canvas)

    def clear_preview(self, message=None):
        """
        Efface l'image astro, le texte, et les messages, puis redessine le fond (si chargé).
        Affiche un message optionnel par-dessus le fond.
        """
        if not hasattr(self.canvas, "winfo_exists") or not self.canvas.winfo_exists(): return

        # --- Clear specific items, keep background image object ---
        # Delete the astro image, text overlay, and any previous message text
        if self._img_id_on_canvas: self.canvas.delete(self._img_id_on_canvas); self._img_id_on_canvas = None
        if self._text_id_on_canvas: self.canvas.delete(self._text_id_on_canvas); self._text_id_on_canvas = None
        self.canvas.delete("message") # Delete previous status messages

        # Reset astro image data references
        self.image_data_raw = None; self.image_data_raw_shape = None
        self.image_data_wb = None; self.tk_img = None; self.last_displayed_pil_image = None
        # Reset display info counts
        self.display_img_count = 0; self.display_total_imgs = 0
        self.display_current_batch = 0; self.display_total_batches = 0
        # Reset zoom/pan state
        self.reset_zoom_and_pan()

        # --- Redraw the background image (if it exists) ---
        # This reuses the logic now embedded in _redraw_canvas
        # We call _redraw_canvas which will now only draw the background
        # because self.last_displayed_pil_image is None.
        self._redraw_canvas()

        # --- Display the message (if provided) on top of the background ---
        if message:
            canvas_width = self.canvas.winfo_width(); canvas_height = self.canvas.winfo_height()
            if canvas_width > 1 and canvas_height > 1:
                try:
                    # Create the message text item
                    msg_id = self.canvas.create_text(
                        canvas_width / 2, canvas_height / 2,
                        text=message, fill="gray", font=("Arial", 11),
                        anchor="center", justify="center", tags="message"
                    )
                    # Ensure it's on top
                    self.canvas.tag_raise(msg_id)
                except tk.TclError:
                    pass # Ignore if canvas is destroyed during message creation

    def update_info_text(self, text_content): self.image_info_text_content = text_content

    def _on_canvas_resize(self, event=None): # Add event=None for direct calls
        """Redraws the image when the canvas size changes."""
        # Redraw if we have an image, otherwise show the default message
        if self.last_displayed_pil_image:
            self._redraw_canvas()
        elif self.image_data_raw is None:
            # Use self.tr() if localization is needed here, otherwise hardcode
            self.clear_preview("Select input/output folders.")    
    def trigger_redraw(self): self._redraw_canvas()

    def _zoom_on_scroll(self, event):
        if self.last_displayed_pil_image is None: return
        zoom_factor = 1.15
        if event.num == 5 or event.delta < 0: new_zoom = self.zoom_level / zoom_factor
        elif event.num == 4 or event.delta > 0: new_zoom = self.zoom_level * zoom_factor
        else: return
        new_zoom = np.clip(new_zoom, self.MIN_ZOOM, self.MAX_ZOOM)
        if abs(new_zoom - self.zoom_level) < 1e-6: return
        canvas_x = event.x; canvas_y = event.y
        canvas_width = self.canvas.winfo_width(); canvas_height = self.canvas.winfo_height()
        zoom_ratio = new_zoom / self.zoom_level
        mouse_rel_view_center_x = canvas_x - (canvas_width / 2 + self._view_offset_x)
        mouse_rel_view_center_y = canvas_y - (canvas_height / 2 + self._view_offset_y)
        self._view_offset_x += mouse_rel_view_center_x * (1 - zoom_ratio)
        self._view_offset_y += mouse_rel_view_center_y * (1 - zoom_ratio)
        self.zoom_level = new_zoom
        self._redraw_canvas()

    def reset_zoom_and_pan(self):
        needs_redraw = False
        if abs(self.zoom_level - 1.0) > 1e-6: self.zoom_level = 1.0; needs_redraw = True
        if abs(self._view_offset_x) > 1e-6 or abs(self._view_offset_y) > 1e-6: self._view_offset_x = 0.0; self._view_offset_y = 0.0; needs_redraw = True
        if needs_redraw and self.last_displayed_pil_image: self._redraw_canvas()

    def zoom_full_size(self):
        """Display the current image at 100% with top-left anchoring."""
        if self.last_displayed_pil_image is None:
            return
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w <= 0 or canvas_h <= 0:
            return
        img_w, img_h = self.last_displayed_pil_image.size
        self.zoom_level = 1.0
        self._view_offset_x = img_w / 2 - canvas_w / 2
        self._view_offset_y = img_h / 2 - canvas_h / 2
        self._redraw_canvas()

    def zoom_fit(self):
        """Fit the current image within the canvas while preserving aspect ratio."""
        if self.last_displayed_pil_image is None:
            return
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w <= 0 or canvas_h <= 0:
            return
        img_w, img_h = self.last_displayed_pil_image.size
        if img_w == 0 or img_h == 0:
            return
        self.zoom_level = min(canvas_w / img_w, canvas_h / img_h)
        self._view_offset_x = 0
        self._view_offset_y = 0
        self._redraw_canvas()

    def _start_pan(self, event):
        if self.last_displayed_pil_image is None: return
        self._is_panning = True; self._pan_start_x = event.x; self._pan_start_y = event.y
        self.canvas.config(cursor="fleur")

    def _stop_pan(self, event):
        if self._is_panning: self._is_panning = False; self.canvas.config(cursor="")

    def _pan_image(self, event):
        """Déplace l'image et le texte pendant le panoramique.""" # Docstring updated
        if not self._is_panning or self.last_displayed_pil_image is None: return
        dx = event.x - self._pan_start_x; dy = event.y - self._pan_start_y
        self._view_offset_x += dx; self._view_offset_y += dy
        self._pan_start_x = event.x; self._pan_start_y = event.y
        if self._img_id_on_canvas:
            try:
                 self.canvas.move(self._img_id_on_canvas, dx, dy)
                 # --- ADD move text item ---
                 if self._text_id_on_canvas:
                     self.canvas.move(self._text_id_on_canvas, dx, dy)
                 # --- END ADD ---
            except tk.TclError: self._img_id_on_canvas = None; self._text_id_on_canvas = None; self._redraw_canvas()
            except Exception as e: print(f"Error moving image/text during pan: {e}"); self._redraw_canvas()
        else: self._redraw_canvas()
# --- END OF FILE seestar/gui/preview.py ---
