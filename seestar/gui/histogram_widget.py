# --- START OF FILE seestar/gui/histogram_widget.py ---
"""
Widget Tkinter intégrant Matplotlib pour afficher un histogramme interactif
des données d'image astronomique.
"""
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Explicitly use TkAgg backend for compatibility
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# Removed NavigationToolbar2Tk import as it's not used by default now
from matplotlib.figure import Figure
import traceback

class HistogramWidget(ttk.Frame):
    """
    Widget d'histogramme interactif utilisant Matplotlib dans Tkinter.
    Permet le déplacement des lignes de point noir/blanc et le zoom/pan.
    """
    def __init__(self, master=None, range_change_callback=None, **kwargs):
        """
        Initialise le widget histogramme.

        Args:
            master: Widget parent Tkinter.
            range_change_callback: Fonction à appeler quand les lignes BP/WP sont modifiées
                                   par l'utilisateur (prend min_val, max_val en argument).
            **kwargs: Arguments pour ttk.Frame.
        """
        super().__init__(master, **kwargs)
        self.range_change_callback = range_change_callback

        # Configuration Matplotlib
        # Adjusted figsize/dpi for better embedding
        self.figure = Figure(figsize=(5, 2.2), dpi=80, facecolor='#353535')
        self.ax = self.figure.add_subplot(111)
        self._configure_plot_style()

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Variables internes
        self._current_hist_data = None # Stocke les données (bins, hists, colors)
        self._min_val = 0.0
        self._max_val = 1.0
        self.line_min = None
        self.line_max = None
        self.dragging_line = None # 'min', 'max', or None
        self._is_panning = False
        self._pan_start_x = None
        self._pan_start_xlim = None

        # Connexion des événements Matplotlib
        self.canvas.mpl_connect('button_press_event', self._on_press)
        self.canvas.mpl_connect('button_release_event', self._on_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('scroll_event', self._on_scroll)

        self.plot_histogram(None) # Afficher un histogramme vide initial

    def _configure_plot_style(self):
        """Configure l'apparence du graphique Matplotlib."""
        self.ax.clear() # Ensure axis is clear before styling
        self.ax.set_facecolor('#2E2E2E')
        # self.figure.set_facecolor('#353535') # Set on figure creation

        self.ax.tick_params(axis='x', colors='lightgray', labelsize=8)
        self.ax.tick_params(axis='y', colors='lightgray', labelsize=8)
        self.ax.tick_params(axis='both', which='major', length=3, width=0.5, pad=2) # Smaller ticks

        for spine in ['bottom', 'top', 'left', 'right']:
            self.ax.spines[spine].set_color('darkgray') # Slightly darker spines
            self.ax.spines[spine].set_linewidth(0.6)

        self.ax.yaxis.label.set_color('lightgray')
        self.ax.xaxis.label.set_color('lightgray')
        self.ax.yaxis.label.set_fontsize(8)
        self.ax.xaxis.label.set_fontsize(8)
        # Adjust margins for better fit
        self.figure.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.95)

    def update_histogram(self, data):
        """
        Met à jour l'histogramme avec de nouvelles données.

        Args:
            data (np.ndarray): Données d'image (HxW ou HxWx3, float32, 0-1 range).
                               Ces données doivent être celles *avant* l'application
                               du stretch final, mais *après* la balance des blancs.
        """
        self._current_hist_data = self._calculate_hist_data(data)
        self.plot_histogram(self._current_hist_data)

    def _calculate_hist_data(self, data):
        """Calcule les données de l'histogramme (bins, comptes)."""
        if data is None or data.size == 0:
            return None

        num_bins = 256
        hist_range = (0.0, 1.0001) # Include 1.0 in the last bin slightly better
        hist_data = {'bins': None, 'hists': [], 'colors': []}

        try:
            # Ensure data is float for calculations
            data_float = data.astype(np.float32)

            if data_float.ndim == 3 and data_float.shape[2] == 3: # Color
                colors = ['#FF4444', '#44FF44', '#4466FF'] # R, G, B
                hist_data['colors'] = colors
                valid_bins = None
                for i in range(3):
                    ch_data = data_float[..., i].ravel()
                    # Filter out non-finite values BEFORE histogram calculation
                    fin_data = ch_data[np.isfinite(ch_data)]
                    if fin_data.size > 0:
                        # Clip data *before* histogram to ensure it's within range
                        hist, bins = np.histogram(np.clip(fin_data, 0.0, 1.0), bins=num_bins, range=hist_range)
                        if valid_bins is None: valid_bins = bins
                        hist_data['hists'].append(hist)
                    else:
                        hist_data['hists'].append(np.zeros(num_bins))
                hist_data['bins'] = valid_bins if valid_bins is not None else np.linspace(hist_range[0], hist_range[1], num_bins + 1)

            elif data_float.ndim == 2: # Grayscale
                colors = ['lightgray']
                hist_data['colors'] = colors
                fin_data = data_float.ravel()[np.isfinite(data_float.ravel())]
                if fin_data.size > 0:
                     hist, bins = np.histogram(np.clip(fin_data, 0.0, 1.0), bins=num_bins, range=hist_range)
                     hist_data['hists'].append(hist)
                     hist_data['bins'] = bins
                else:
                     hist_data['hists'].append(np.zeros(num_bins))
                     hist_data['bins'] = np.linspace(hist_range[0], hist_range[1], num_bins + 1)
            else:
                print(f"Warning: Unsupported data shape for histogram: {data_float.shape}")
                return None # Format non supporté

            return hist_data

        except Exception as e:
            print(f"Erreur calcul histogramme: {e}")
            traceback.print_exc(limit=2)
            return None

    def plot_histogram(self, hist_data):
        """Affiche les données d'histogramme calculées."""
        # Store current view limits to restore after replotting
        # Only store if they are valid numbers and range makes sense
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        restore_lim = (all(np.isfinite(current_xlim)) and
                       all(np.isfinite(current_ylim)) and
                       current_ylim[1] > current_ylim[0] and
                       current_xlim[1] > current_xlim[0])

        self._configure_plot_style() # Clears and resets style
        self.line_min = None # Reset line references
        self.line_max = None

        # Handle case where no data is provided
        if hist_data is None or not hist_data['hists'] or hist_data['bins'] is None:
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(1, 10) # Default y range for log scale
            self.ax.set_yscale('log')
            self.ax.set_xlabel("Niveau (0-1)"); self.ax.set_ylabel("Nbre Pixels (log)")
            self.ax.text(0.5, 0.5, "Aucune donnée", color="gray", ha='center', va='center', transform=self.ax.transAxes)
            self.canvas.draw_idle()
            return

        try:
            bins = hist_data['bins']
            bin_centers = (bins[:-1] + bins[1:]) / 2

            max_count_overall = 0
            all_valid_counts = []
            for i, hist in enumerate(hist_data['hists']):
                # Add 1 to counts for log scale plotting (avoids log(0))
                counts_for_plot = hist + 1
                if counts_for_plot.size == bin_centers.size: # Ensure sizes match
                    self.ax.plot(bin_centers, counts_for_plot, color=hist_data['colors'][i], alpha=0.85, drawstyle='steps-mid', linewidth=1.0)
                    # Collect non-zero original counts for Y limit calculation
                    valid_original_counts = hist[hist > 0]
                    if valid_original_counts.size > 0:
                        all_valid_counts.extend(valid_original_counts)
                        current_max = np.max(hist)
                        if current_max > max_count_overall: max_count_overall = current_max
                else:
                     print(f"Warn: Histogram size mismatch for channel {i}")

            # Dynamic Y limits based on actual counts
            if all_valid_counts:
                # Use 99.5th percentile of non-zero counts for the top limit
                p995_max = np.percentile(all_valid_counts, 99.5)
                # Ensure top_y is significantly larger than 1 for log scale
                top_y = max(10, p995_max * 1.5) # Add some headroom
            else:
                top_y = 100 # Fallback if no counts > 0

            self.ax.set_ylim(bottom=0.8, top=top_y) # Start slightly below 1, ensure top_y > bottom
            self.ax.set_yscale('log')

            # Draw BP/WP Lines after setting scales
            line_alpha = 0.8
            self.line_min = self.ax.axvline(self._min_val, color='#FFAAAA', linestyle='--', linewidth=1.2, alpha=line_alpha, picker=5) # Enable picking
            self.line_max = self.ax.axvline(self._max_val, color='#AAAAFF', linestyle='--', linewidth=1.2, alpha=line_alpha, picker=5) # Enable picking

            self.ax.set_xlabel("Niveau (0-1)"); self.ax.set_ylabel("Nbre Pixels (log)")

            # Restore X Limits if they were valid, otherwise set default
            if restore_lim:
                self.ax.set_xlim(current_xlim)
                # Also restore Y limits if restoring X? Or always recalculate Y? Recalculating Y is safer.
                # self.ax.set_ylim(current_ylim)
            else:
                self.ax.set_xlim(0, 1)

            # self.figure.tight_layout(pad=0.1) # May interfere with fixed margins
            self.canvas.draw_idle()

        except Exception as e:
            print(f"Erreur affichage histogramme: {e}")
            traceback.print_exc(limit=2)
            # Attempt to draw a clear error state
            try:
                self._configure_plot_style()
                self.ax.set_xlim(0, 1); self.ax.set_ylim(1, 10); self.ax.set_yscale('log')
                self.ax.text(0.5, 0.5, "Erreur Histogramme", color="red", ha='center', va='center', transform=self.ax.transAxes)
                self.canvas.draw_idle()
            except Exception: pass # Ignore errors during error display

    def set_range(self, min_val, max_val):
        """Met à jour la position des lignes BP/WP depuis l'extérieur."""
        # Clip and ensure min < max
        self._min_val = np.clip(min_val, 0.0, 1.0)
        self._max_val = np.clip(max_val, 0.0, 1.0)
        min_separation = 1e-4 # Ensure minimum separation
        if self._min_val >= self._max_val - min_separation:
             self._min_val = max(0.0, self._max_val - min_separation)

        # Update lines if they exist
        if self.line_min:
            self.line_min.set_xdata([self._min_val, self._min_val])
        if self.line_max:
            self.line_max.set_xdata([self._max_val, self._max_val])

        # Redraw the canvas only if the lines were actually updated
        if self.line_min or self.line_max:
             self.canvas.draw_idle()

    def _on_press(self, event):
        """Gère le clic souris sur le canvas Matplotlib."""
        if event.inaxes != self.ax or event.xdata is None: return
        self.dragging_line = None; self._is_panning = False

        # Check if clicking near a line (Button 1 - Left Click)
        if event.button == 1 and self.line_min and self.line_max:
            # Calculate tolerance dynamically based on current view width
            x_display_range = self.ax.get_xlim()[1] - self.ax.get_xlim()[0]
            pick_radius_data = max(0.005, x_display_range * 0.02) # ~2% of view width, min 0.005

            d_min = abs(event.xdata - self._min_val)
            d_max = abs(event.xdata - self._max_val)

            # Prioritize dragging min line if clicks are very close to both
            if d_min <= pick_radius_data and d_min <= d_max:
                self.dragging_line = 'min'
                self.canvas_widget.config(cursor="sb_h_double_arrow")
            elif d_max <= pick_radius_data:
                self.dragging_line = 'max'
                self.canvas_widget.config(cursor="sb_h_double_arrow")

        # Check for Pan (Button 3 - Right click)
        elif event.button == 3:
             self._is_panning = True
             self._pan_start_x = event.xdata
             self._pan_start_xlim = self.ax.get_xlim()
             self.canvas_widget.config(cursor="fleur") # Use "fleur" for panning cursor

    def _on_release(self, event):
        """Gère le relâchement du bouton souris."""
        if self.dragging_line:
            self.canvas_widget.config(cursor="") # Reset cursor
            # Callback AFTER releasing the drag
            if self.range_change_callback:
                try: self.range_change_callback(self._min_val, self._max_val)
                except Exception as cb_err: print(f"Error in range_change_callback: {cb_err}")
            self.dragging_line = None # Clear flag after callback

        if self._is_panning:
             self._is_panning = False
             self.canvas_widget.config(cursor="") # Reset cursor

    def _on_motion(self, event):
        """Gère le mouvement de la souris."""
        # Reset cursor if moved outside axes while not dragging/panning
        if event.inaxes != self.ax:
             if not self._is_panning and not self.dragging_line:
                  self.canvas_widget.config(cursor="")
             return
        # Ensure xdata is valid within the axes
        if event.xdata is None: return

        # --- Dragging BP/WP lines ---
        if self.dragging_line:
            x_val = np.clip(event.xdata, 0, 1)
            min_separation = 1e-4 # Minimum separation

            if self.dragging_line == 'min':
                new_min = min(x_val, self._max_val - min_separation)
                if abs(new_min - self._min_val) > 1e-9: # Avoid unnecessary updates
                     self._min_val = new_min
                     if self.line_min: self.line_min.set_xdata([self._min_val] * 2)
                     self.canvas.draw_idle() # Redraw needed only if value changed

            elif self.dragging_line == 'max':
                new_max = max(x_val, self._min_val + min_separation)
                if abs(new_max - self._max_val) > 1e-9:
                     self._max_val = new_max
                     if self.line_max: self.line_max.set_xdata([self._max_val] * 2)
                     self.canvas.draw_idle()

        # --- Panning ---
        elif self._is_panning:
            if self._pan_start_x is None: return # Should not happen if _is_panning is true
            dx = event.xdata - self._pan_start_x
            start_lim = self._pan_start_xlim
            width = start_lim[1] - start_lim[0]

            # Calculate new limits based on drag
            new_min = start_lim[0] - dx
            new_max = start_lim[1] - dx

            # Constrain panning within 0-1 range without changing width
            if new_min < 0: new_min = 0; new_max = width
            if new_max > 1: new_max = 1; new_min = 1 - width
            new_min = max(0.0, min(1.0 - width, new_min)) # Ensure min doesn't push max > 1
            new_max = new_min + width

            # Apply new limits if they changed significantly
            current_lim = self.ax.get_xlim()
            if abs(new_min - current_lim[0]) > 1e-9 or abs(new_max - current_lim[1]) > 1e-9:
                self.ax.set_xlim(new_min, new_max)
                self.canvas.draw_idle()

    def _on_scroll(self, event):
        """Gère le zoom avec la molette."""
        if event.inaxes != self.ax or event.xdata is None: return

        xlim = self.ax.get_xlim()
        x_center = event.xdata # Zoom centered on mouse pointer

        # Determine zoom factor
        factor = 1.25 if event.step > 0 else 1/1.25 # Zoom in or out

        # Calculate new limits
        width = xlim[1] - xlim[0]
        new_width = width / factor
        # Prevent zooming too far in or out
        min_allowed_width = 1e-4
        max_allowed_width = 1.0
        new_width = np.clip(new_width, min_allowed_width, max_allowed_width)

        # Calculate new min/max based on centered zoom
        new_min = x_center - (x_center - xlim[0]) * (new_width / width)
        new_max = new_min + new_width

        # Clip limits to 0-1 range and adjust if necessary
        if new_min < 0: new_min = 0; new_max = new_width
        if new_max > 1: new_max = 1; new_min = 1 - new_width
        new_min = max(0.0, new_min) # Final safety clip
        new_max = min(1.0, new_max)

        # Apply new limits if they changed significantly
        if abs(new_min - xlim[0]) > 1e-9 or abs(new_max - xlim[1]) > 1e-9:
            self.ax.set_xlim(new_min, new_max)
            self.canvas.draw_idle()

    def reset_zoom(self):
        """Réinitialise le zoom et le pan de l'histogramme."""
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        needs_redraw = False

        # Reset X axis to full range
        if abs(xlim[0] - 0.0) > 1e-6 or abs(xlim[1] - 1.0) > 1e-6:
            self.ax.set_xlim(0, 1)
            needs_redraw = True

        # Reset Y axis limits based on current data
        if self._current_hist_data and self._current_hist_data['hists']:
             all_valid_counts = []
             for hist_channel in self._current_hist_data['hists']:
                 all_valid_counts.extend(hist_channel[hist_channel > 0])
             if all_valid_counts:
                 p995_max = np.percentile(all_valid_counts, 99.5)
                 target_top_y = max(10, p995_max * 1.5)
             else: target_top_y = 100
             target_ylim = (0.8, target_top_y) # Match Y limits from plot_histogram

             # Check if Y limits significantly differ from target
             if abs(ylim[0] - target_ylim[0]) > 1e-1 or abs(ylim[1] - target_ylim[1]) > 1e-1:
                 self.ax.set_ylim(target_ylim)
                 needs_redraw = True
        elif abs(ylim[0]-0.8)>1e-6 or abs(ylim[1]-100)>1e-6: # Reset to default if no data
             self.ax.set_ylim(0.8, 100)
             needs_redraw = True


        if needs_redraw:
             print("Histogram zoom/pan reset.")
             self.canvas.draw_idle()
# --- END OF FILE seestar/gui/histogram_widget.py ---