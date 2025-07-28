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
from matplotlib.figure import Figure
import traceback
import concurrent.futures, time
import threading

_HIST_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1)

class HistogramWidget(ttk.Frame):
    """
    Widget d'histogramme interactif utilisant Matplotlib dans Tkinter.
    Permet le déplacement des lignes de point noir/blanc et le zoom/pan.
    Version: HistoFix_YScaleFor01_1
    """
    def __init__(self, master=None, range_change_callback=None, **kwargs):
        super().__init__(master, **kwargs)
        self.range_change_callback = range_change_callback

        self.figure = Figure(figsize=(5, 2.2), dpi=80, facecolor='#353535')
        self.ax = self.figure.add_subplot(111)
        self._configure_plot_style()

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        self._current_hist_data_details = None
        self._current_data = None
        self.auto_zoom_enabled = False
        # When True, the X axis range is preserved across batches
        self.freeze_x_range = False
        # When True, the Y axis range is preserved across batches
        self.freeze_y_range = True
        self._stored_ylim = None

        # When True, the X axis scale is preserved across batches
        self._stored_xlim = None

        self._min_line_val_data_scale = 0.0
        self._max_line_val_data_scale = 1.0
        self.data_min_for_current_plot = 0.0
        self.data_max_for_current_plot = 1.0

        self.line_min_obj = None 
        self.line_max_obj = None 
        self.dragging_line_type = None 
        self._is_panning_active = False
        self._pan_start_coord_x = None
        self._pan_initial_xlim = None

        self._last_hist_req = 0.0
        self._hist_future = None

        self.canvas.mpl_connect('button_press_event', self._on_press)
        self.canvas.mpl_connect('button_release_event', self._on_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('scroll_event', self._on_scroll)

        self.plot_histogram(None) 
        print("DEBUG HistogramWidget (HistoFix_YScaleFor01_1): __init__ terminé.")

    def _configure_plot_style(self):
        self.ax.clear()
        # Clear any references to the BP/WP lines since clearing the axis
        # removes them from the canvas. Mark them as None so that they are
        # recreated on the next draw call.
        self.line_min_obj = None
        self.line_max_obj = None
        self.ax.set_facecolor('#2E2E2E')
        self.ax.tick_params(axis='x', colors='lightgray', labelsize=8)
        self.ax.tick_params(axis='y', colors='lightgray', labelsize=8)
        self.ax.tick_params(axis='both', which='major', length=3, width=0.5, pad=2)
        for spine in ['bottom', 'top', 'left', 'right']:
            self.ax.spines[spine].set_color('darkgray')
            self.ax.spines[spine].set_linewidth(0.6)
        self.ax.yaxis.label.set_color('lightgray'); self.ax.xaxis.label.set_color('lightgray')
        self.ax.yaxis.label.set_fontsize(8); self.ax.xaxis.label.set_fontsize(8)
        self.figure.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.95)

    def update_histogram(self, data_for_analysis):
        """Version async : lance le calcul dans le pool, puis revient
        via Tk `after` pour tracer. 100 % thread-safe."""
        if threading.current_thread() is not threading.main_thread():
            print(
                "WARNING: Accès widget Tkinter hors du main thread dans update_histogram"
            )
            # SAFE: Tkinter widget update via after_idle
            self.after_idle(lambda d=data_for_analysis: self.update_histogram(d))
            return
        print("DEBUG HistoWidget.update_histogram: Reçu data_for_analysis.")
        if data_for_analysis is not None:
            print(
                f"  -> data_for_analysis - Shape: {data_for_analysis.shape}, Dtype: {data_for_analysis.dtype}, Range: [{np.nanmin(data_for_analysis):.4g} - {np.nanmax(data_for_analysis):.4g}]"
            )
        else:
            print("  -> data_for_analysis est None.")

        if data_for_analysis is None or data_for_analysis.size == 0:
            # Keep current histogram if no new data is provided
            return

        self._current_data = data_for_analysis
        now = time.monotonic()
        if now - self._last_hist_req < 0.40 and self._hist_future:
            return
        self._last_hist_req = now

        small = None
        if data_for_analysis is not None and data_for_analysis.size:
            small = (
                data_for_analysis[::4, ::4]
                if data_for_analysis.ndim == 2
                else data_for_analysis[::4, ::4, :]
            )

        if self._hist_future and not self._hist_future.done():
            self._hist_future.cancel()

        self._hist_future = _HIST_EXECUTOR.submit(self._calculate_hist_data, small)
        self._hist_future.add_done_callback(
            lambda fut: self.after(0, self._apply_histogram, fut)
        )

    def _compute_histogram(self, img):
        """Worker thread: compute histogram details using _calculate_hist_data."""
        return self._calculate_hist_data(img)

    def _apply_histogram(self, fut):
        """Thread GUI : reçoit le résultat, trace sans recalculer les bins."""
        if threading.current_thread() is not threading.main_thread():
            print(
                "WARNING: Accès widget Tkinter hors du main thread dans _apply_histogram"
            )
            # SAFE: Tkinter widget update via after_idle
            self.after_idle(lambda f=fut: self._apply_histogram(f))
            return
        if fut.cancelled() or fut.exception():
            return
        res = fut.result()
        if res is None:
            # Keep previous display if worker returned nothing
            if self._current_hist_data_details:
                self.plot_histogram(self._current_hist_data_details)
            return
        self._current_hist_data_details = res
        self.plot_histogram(res)




    def _calculate_hist_data(self, data):
        """
        Calcule les données de l'histogramme (bins, comptes).
        S'adapte à la plage des données (0-1 ou ADU).
        Met à jour self.data_min_for_current_plot et self.data_max_for_current_plot.
        MODIFIED: Ajoute original_input_shape au dictionnaire retourné.
        Version: HistoCalc_AddInputShape_1
        """
        original_input_shape = data.shape if data is not None else (0,0,0) # Stocker pour le calcul de ylim, assurer 3D pour HWC
        if data is None or data.size == 0:
            self.data_min_for_current_plot = 0.0 
            self.data_max_for_current_plot = 1.0
            print(f"DEBUG HistoWidget._calculate_hist_data: Données None/vides. Plage plot réinitialisée à [{self.data_min_for_current_plot}, {self.data_max_for_current_plot}]")
            return None

        num_bins = 256
        hist_data_dict = {'bins': None, 'hists': [], 'colors': []}

        try:
            data_float = data.astype(np.float32)
            finite_data_for_range = data_float[np.isfinite(data_float)]

            if finite_data_for_range.size > 0:
                calculated_min = np.min(finite_data_for_range)
                calculated_max = np.max(finite_data_for_range)
                if calculated_min < 0 and np.all(finite_data_for_range[finite_data_for_range < 0] > -1e-5):
                    calculated_min = 0.0
                calculated_min = max(0.0, calculated_min) 
            else: 
                calculated_min, calculated_max = 0.0, 1.0 
            
            if calculated_max <= calculated_min + 1e-7:
                if calculated_max < 1.5 and calculated_min > -0.5:
                     current_plot_min, current_plot_max = 0.0, 1.0001 
                else: 
                     current_plot_min = max(0, calculated_min - 0.5) if calculated_min > 0 else 0.0
                     current_plot_max = current_plot_min + 1.0
            else:
                current_plot_min = calculated_min
                current_plot_max = calculated_max
            
            if not self.freeze_x_range or self._current_hist_data_details is None:
                self.data_min_for_current_plot = current_plot_min
                self.data_max_for_current_plot = (
                    current_plot_max * 1.001
                    if (current_plot_max - current_plot_min) > 1e-9
                    else current_plot_max + 1e-5
                )
            
            print(f"DEBUG HistoWidget._calculate_hist_data (V_HistoCalc_AddInputShape_1): Plage données pour histo (self.data_min/max_for_current_plot): [{self.data_min_for_current_plot:.4g}, {self.data_max_for_current_plot:.4g}]")
            
            hist_range_for_np = (self.data_min_for_current_plot, self.data_max_for_current_plot)

            if data_float.ndim == 3 and data_float.shape[2] == 3: 
                hist_data_dict['colors'] = ['#FF4444', '#44FF44', '#4466FF'] 
                valid_bins = None
                for i in range(3):
                    ch_data = data_float[..., i].ravel()
                    fin_data_ch = ch_data[np.isfinite(ch_data)]
                    if fin_data_ch.size > 0:
                        hist_counts, bins_edges = np.histogram(np.clip(fin_data_ch, hist_range_for_np[0], hist_range_for_np[1]), bins=num_bins, range=hist_range_for_np)
                        if valid_bins is None: valid_bins = bins_edges
                        hist_data_dict['hists'].append(hist_counts)
                    else: hist_data_dict['hists'].append(np.zeros(num_bins))
                hist_data_dict['bins'] = valid_bins if valid_bins is not None else np.linspace(hist_range_for_np[0], hist_range_for_np[1], num_bins + 1)
            
            elif data_float.ndim == 2: 
                hist_data_dict['colors'] = ['lightgray']
                fin_data_flat = data_float.ravel()[np.isfinite(data_float.ravel())]
                if fin_data_flat.size > 0:
                     hist_counts, bins_edges = np.histogram(np.clip(fin_data_flat, hist_range_for_np[0], hist_range_for_np[1]), bins=num_bins, range=hist_range_for_np)
                     hist_data_dict['hists'].append(hist_counts)
                     hist_data_dict['bins'] = bins_edges
                else:
                     hist_data_dict['hists'].append(np.zeros(num_bins))
                     hist_data_dict['bins'] = np.linspace(hist_range_for_np[0], hist_range_for_np[1], num_bins + 1)
            else:
                print(f"Warning HistoWidget._calculate_hist_data: Unsupported data shape: {data_float.shape}"); return None

            # AJOUT: Stocker la shape originale des données pour aider au calcul de Ylim
            if hist_data_dict is not None:
                hist_data_dict['input_shape_for_ylim_calc'] = original_input_shape

            return hist_data_dict
        except Exception as e:
            print(f"ERREUR HistoWidget._calculate_hist_data: {e}"); traceback.print_exc(limit=2); return None




    def plot_histogram(self, hist_data_details_to_plot):
        """
        Redessine UNIQUEMENT les barres de l'histogramme et configure les axes.
        Ne dessine PAS les lignes BP/WP ici.
        MODIFIED: Amélioration du calcul de target_top_y_limit pour données [0,1].
        Version: HistoFix_YScaleFor01_4
        """
        if threading.current_thread() is not threading.main_thread():
            print(
                "WARNING: Accès widget Tkinter hors du main thread dans plot_histogram"
            )
            # SAFE: Tkinter widget update via after_idle
            self.after_idle(lambda h=hist_data_details_to_plot: self.plot_histogram(h))
            return
        print(f"DEBUG HistoWidget.plot_histogram (V_HistoFix_YScaleFor01_4): Tentative de plot des barres.") # Version Log
        current_plot_min_x = self.data_min_for_current_plot
        current_plot_max_x = self.data_max_for_current_plot
        print(f"  -> Utilisant plage X stockée: [{current_plot_min_x:.4g}, {current_plot_max_x:.4g}]")

        xlim_before_plot = self.ax.get_xlim()
        if self.freeze_x_range and self._stored_xlim is not None:
            xlim_before_plot = self._stored_xlim
        
        was_x_zoomed_and_relevant = (
            abs(xlim_before_plot[0] - current_plot_min_x) > 1e-6 * abs(current_plot_max_x - current_plot_min_x) or \
            abs(xlim_before_plot[1] - current_plot_max_x) > 1e-6 * abs(current_plot_max_x - current_plot_min_x)
        )
        if was_x_zoomed_and_relevant:
            if not (xlim_before_plot[0] >= current_plot_min_x and xlim_before_plot[1] <= current_plot_max_x and xlim_before_plot[0] < xlim_before_plot[1]):
                was_x_zoomed_and_relevant = False

        self._configure_plot_style() 

        if hist_data_details_to_plot is None or not hist_data_details_to_plot.get('hists') or hist_data_details_to_plot.get('bins') is None:

            if self.freeze_x_range and self._stored_xlim is not None:
                self.ax.set_xlim(self._stored_xlim)
            else:
                self.ax.set_xlim(current_plot_min_x, current_plot_max_x)

            self.ax.set_ylim(1, 10)
            self.ax.set_yscale('log')
            if self.freeze_y_range and self._stored_ylim is None:
                self._stored_ylim = self.ax.get_ylim()
            self.ax.set_xlabel(f"Niveau ({current_plot_min_x:.1f}-{current_plot_max_x:.1f})"); self.ax.set_ylabel("Nbre Pixels (log)")
            self.ax.text(0.5, 0.5, "Aucune donnée", color="gray", ha='center', va='center', transform=self.ax.transAxes)
            print(f"  -> Affichage 'Aucune donnée'. Xlim réglé sur [{current_plot_min_x:.4g}, {current_plot_max_x:.4g}].")

            if self.freeze_x_range:
                self._stored_xlim = self.ax.get_xlim()
            self.canvas.draw_idle();
            return


        try:
            bins = hist_data_details_to_plot['bins']; bin_centers = (bins[:-1] + bins[1:]) / 2
            
            all_pixel_counts_for_ylim = [] 
            for i, hist_counts in enumerate(hist_data_details_to_plot['hists']):
                counts_for_plotting = hist_counts + 1 
                if counts_for_plotting.size == bin_centers.size:
                    self.ax.plot(bin_centers, counts_for_plotting, color=hist_data_details_to_plot['colors'][i], alpha=0.85, drawstyle='steps-mid', linewidth=1.0)
                    all_pixel_counts_for_ylim.extend(hist_counts) 
                else: print(f"Warn HistoWidget.plot_histogram: Discrépance taille histo pour canal {i}")
            
            target_top_y_limit = 100 
            if all_pixel_counts_for_ylim:
                 all_pixel_counts_np = np.array(all_pixel_counts_for_ylim)
                 counts_greater_than_zero = all_pixel_counts_np[all_pixel_counts_np > 0]

                 if counts_greater_than_zero.size > 0:
                    is_data_01_like_range = (self.data_max_for_current_plot - self.data_min_for_current_plot) <= 2.0 and \
                                           self.data_min_for_current_plot >= -0.1 and \
                                           self.data_max_for_current_plot <= 2.015 

                    if is_data_01_like_range:
                        print(f"  -> Données [0,1]-like détectées pour calcul Ylim (plage X: {self.data_max_for_current_plot - self.data_min_for_current_plot:.3f})")
                        
                        p98_counts = np.percentile(counts_greater_than_zero, 98)
                        max_actual_count = np.max(counts_greater_than_zero)
                        
                        target_top_y_limit = p98_counts * 3.0 
                        target_top_y_limit = max(500, target_top_y_limit) 
                        target_top_y_limit = min(target_top_y_limit, max_actual_count * 1.2) 
                        
                        if p98_counts < 100 and max_actual_count > p98_counts : 
                            target_top_y_limit = max(target_top_y_limit, max_actual_count * 0.5, 500)
                        
                        target_top_y_limit = max(target_top_y_limit, 100) 

                        print(f"    -> Ylim (0-1 data): p98_counts={p98_counts:.0f}, max_actual_count={max_actual_count:.0f}, target_top_y={target_top_y_limit:.0f}")
                    else: # Données ADU ou plage plus large
                        percentile_99_5_y = np.percentile(counts_greater_than_zero, 99.5)
                        target_top_y_limit = max(10, percentile_99_5_y * 1.5)
                        print(f"    -> Ylim (ADU data): percentile_99_5_y={percentile_99_5_y:.0f}, target_top_y={target_top_y_limit:.0f}")
                 else: 
                    print(f"    -> Aucun compte > 0 pour Ylim, utilisation défaut.")
            
            current_ylim_bottom = self.ax.get_ylim()[0] if self.ax.get_ylim() else 0.8 # Fallback si ylim non défini
            target_top_y_limit = max(target_top_y_limit, current_ylim_bottom + 10)
            if not self.freeze_y_range or self._stored_ylim is None:
                self.ax.set_ylim(bottom=0.8, top=target_top_y_limit)
                self._stored_ylim = self.ax.get_ylim()
            else:
                self.ax.set_ylim(self._stored_ylim)
            self.ax.set_yscale('log')
            print(f"  -> Ylim recalculé et appliqué: (0.8, {target_top_y_limit:.2f})")

            self.ax.set_xlabel(f"Niveau ({current_plot_min_x:.2f}-{current_plot_max_x:.2f})"); self.ax.set_ylabel("Nbre Pixels (log)")

            if was_x_zoomed_and_relevant:
                self.ax.set_xlim(xlim_before_plot)
                print(f"  -> Zoom X utilisateur restauré. Xlim: {xlim_before_plot}")
            elif self.freeze_x_range and self._stored_xlim is not None:
                self.ax.set_xlim(self._stored_xlim)
                print(f"  -> Xlim gelé restauré: {self._stored_xlim}")
            else:
                self.ax.set_xlim(current_plot_min_x, current_plot_max_x)
                print(f"  -> Xlim initialisé à la plage des données: [{current_plot_min_x:.4g}, {current_plot_max_x:.4g}]")
            if self.freeze_x_range:
                self._stored_xlim = self.ax.get_xlim()

            self.canvas.draw_idle()
            # --- NEW: restore BP/WP lines if they existed ---
            try:
                plot_span = self.data_max_for_current_plot - self.data_min_for_current_plot
                if plot_span < 1e-9:
                    plot_span = 1.0

                if self.line_min_obj is None:
                    self.line_min_obj = self.ax.axvline(
                        self._min_line_val_data_scale,
                        color="#FFAAAA",
                        linestyle="--",
                        linewidth=1.2,
                        alpha=0.8,
                        picker=5,
                    )
                else:
                    self.line_min_obj.set_xdata([self._min_line_val_data_scale] * 2)

                if self.line_max_obj is None:
                    self.line_max_obj = self.ax.axvline(
                        self._max_line_val_data_scale,
                        color="#AAAAFF",
                        linestyle="--",
                        linewidth=1.2,
                        alpha=0.8,
                        picker=5,
                    )
                else:
                    self.line_max_obj.set_xdata([self._max_line_val_data_scale] * 2)

                self.canvas.draw_idle()
            except Exception as e_redraw:
                print(f"[Histo] Erreur redraw lignes : {e_redraw}")
            if self.auto_zoom_enabled:
                try:
                    self.zoom_histogram()
                except Exception as auto_e:
                    print(f"ERREUR HistoWidget.auto_zoom: {auto_e}")
        except Exception as e:
            print(f"ERREUR HistoWidget.plot_histogram: {e}"); traceback.print_exc(limit=2)
            try: 
                self._configure_plot_style()
                self.ax.set_xlim(0,1); self.ax.set_ylim(1,10); self.ax.set_yscale('log')
                if self.freeze_x_range:
                    self._stored_xlim = self.ax.get_xlim()
                self.ax.text(0.5, 0.5, "Erreur Histogramme", color="red", ha='center', va='center', transform=self.ax.transAxes)
                self.canvas.draw_idle()
            except Exception: pass





    def set_range(self, min_val_from_ui, max_val_from_ui):
        """
        Met à jour la position des lignes BP/WP et les DESSINE.
        Les valeurs reçues (min_val_from_ui, max_val_from_ui) sont DANS L'ÉCHELLE 0-1.
        Version: HistoFix_YScaleFor01_1 (Intègre la logique de HistoFix_YZoomReset_LinesSeparate_1)
        """
        if threading.current_thread() is not threading.main_thread():
            print(
                "WARNING: Accès widget Tkinter hors du main thread dans set_range"
            )
            # SAFE: Tkinter widget update via after_idle
            self.after_idle(lambda mi=min_val_from_ui, ma=max_val_from_ui: self.set_range(mi, ma))
            return
        plot_data_range = self.data_max_for_current_plot - self.data_min_for_current_plot
        if plot_data_range < 1e-9: plot_data_range = 1.0

        new_min_line_val = self.data_min_for_current_plot + min_val_from_ui * plot_data_range
        new_max_line_val = self.data_min_for_current_plot + max_val_from_ui * plot_data_range
        
        print(f"DEBUG HistoWidget.set_range (V_HistoFix_YScaleFor01_1): Reçu UI BP={min_val_from_ui:.4f}, WP={max_val_from_ui:.4f}")
        print(f"  -> Plage histo actuelle: [{self.data_min_for_current_plot:.4g}, {self.data_max_for_current_plot:.4g}]")
        print(f"  -> Lignes BP/WP converties à l'échelle des données: [{new_min_line_val:.4g}, {new_max_line_val:.4g}]")

        self._min_line_val_data_scale = np.clip(new_min_line_val, self.data_min_for_current_plot, self.data_max_for_current_plot)
        self._max_line_val_data_scale = np.clip(new_max_line_val, self.data_min_for_current_plot, self.data_max_for_current_plot)
        
        min_sep_abs = plot_data_range * 0.001
        min_sep_abs = max(min_sep_abs, 1e-7 * (self.data_max_for_current_plot if self.data_max_for_current_plot > 0 else 1.0) )

        if self._min_line_val_data_scale >= self._max_line_val_data_scale - min_sep_abs:
             self._min_line_val_data_scale = max(self.data_min_for_current_plot, self._max_line_val_data_scale - min_sep_abs)
        self._max_line_val_data_scale = np.clip(self._max_line_val_data_scale, self._min_line_val_data_scale + min_sep_abs, self.data_max_for_current_plot)
        self._min_line_val_data_scale = np.clip(self._min_line_val_data_scale, self.data_min_for_current_plot, self._max_line_val_data_scale - min_sep_abs)

        if self.line_min_obj:
            try: self.line_min_obj.remove()
            except: pass 
            self.line_min_obj = None
        if self.line_max_obj:
            try: self.line_max_obj.remove()
            except: pass
            self.line_max_obj = None

        if self.ax and hasattr(self.ax, 'axvline'): 
            self.line_min_obj = self.ax.axvline(self._min_line_val_data_scale, color='#FFAAAA', linestyle='--', linewidth=1.2, alpha=0.8, picker=5) 
            self.line_max_obj = self.ax.axvline(self._max_line_val_data_scale, color='#AAAAFF', linestyle='--', linewidth=1.2, alpha=0.8, picker=5) 
            self.canvas.draw_idle()
            print(f"  -> Lignes BP/WP (échelle données) DESSINÉES à : [{self._min_line_val_data_scale:.4g}, {self._max_line_val_data_scale:.4g}]")

    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None: return
        self.dragging_line_type = None; self._is_panning_active = False
        if event.button == 1 and self.line_min_obj and self.line_max_obj:
            x_display_range = self.ax.get_xlim()[1] - self.ax.get_xlim()[0]
            pick_radius_data_scale = max(0.005 * (self.data_max_for_current_plot - self.data_min_for_current_plot), x_display_range * 0.02)
            pick_radius_data_scale = max(pick_radius_data_scale, 1e-6 * (self.data_max_for_current_plot if self.data_max_for_current_plot > 0 else 1.0))
            d_min = abs(event.xdata - self._min_line_val_data_scale)
            d_max = abs(event.xdata - self._max_line_val_data_scale)
            if d_min <= pick_radius_data_scale and d_min <= d_max:
                self.dragging_line_type = 'min'
            elif d_max <= pick_radius_data_scale:
                self.dragging_line_type = 'max'
            if self.dragging_line_type: self.canvas_widget.config(cursor="sb_h_double_arrow")
        elif event.button == 3:
             self._is_panning_active = True
             self._pan_start_coord_x = event.xdata
             self._pan_initial_xlim = self.ax.get_xlim()
             self.canvas_widget.config(cursor="fleur")

    def _on_release(self, event):
        if self.dragging_line_type:
            self.canvas_widget.config(cursor="")
            if self.range_change_callback:
                try: 
                    bp_to_send = self._min_line_val_data_scale
                    wp_to_send = self._max_line_val_data_scale
                    print(f"DEBUG HistoWidget._on_release: Appel callback avec BP_histo={bp_to_send:.4g}, WP_histo={wp_to_send:.4g} (échelle données)")
                    self.range_change_callback(bp_to_send, wp_to_send)
                except Exception as cb_err: print(f"Erreur HistoWidget.range_change_callback: {cb_err}")
            self.dragging_line_type = None
        if self._is_panning_active:
             self._is_panning_active = False
             self.canvas_widget.config(cursor="")

    def _on_motion(self, event):
        if event.inaxes != self.ax:
             if not self._is_panning_active and not self.dragging_line_type: self.canvas_widget.config(cursor="")
             return
        if event.xdata is None: return
        min_separation_abs = (self.data_max_for_current_plot - self.data_min_for_current_plot) * 0.001
        min_separation_abs = max(min_separation_abs, 1e-7 * (self.data_max_for_current_plot if self.data_max_for_current_plot > 0 else 1.0))
        if self.dragging_line_type:
            x_val_in_data_scale = np.clip(event.xdata, self.data_min_for_current_plot, self.data_max_for_current_plot)
            if self.dragging_line_type == 'min':
                new_min_candidate = min(x_val_in_data_scale, self._max_line_val_data_scale - min_separation_abs)
                if abs(new_min_candidate - self._min_line_val_data_scale) > 1e-9 * (self.data_max_for_current_plot - self.data_min_for_current_plot): 
                     self._min_line_val_data_scale = new_min_candidate
                     if self.line_min_obj: self.line_min_obj.set_xdata([self._min_line_val_data_scale] * 2)
                     self.canvas.draw_idle() 
            elif self.dragging_line_type == 'max':
                new_max_candidate = max(x_val_in_data_scale, self._min_line_val_data_scale + min_separation_abs)
                if abs(new_max_candidate - self._max_line_val_data_scale) > 1e-9 * (self.data_max_for_current_plot - self.data_min_for_current_plot):
                     self._max_line_val_data_scale = new_max_candidate
                     if self.line_max_obj: self.line_max_obj.set_xdata([self._max_line_val_data_scale] * 2)
                     self.canvas.draw_idle()
        elif self._is_panning_active:
            if self._pan_start_coord_x is None: return
            dx_pan = event.xdata - self._pan_start_coord_x
            start_lim_pan = self._pan_initial_xlim; current_width_pan = start_lim_pan[1] - start_lim_pan[0]
            new_min_panned = start_lim_pan[0] - dx_pan; new_max_panned = start_lim_pan[1] - dx_pan
            if new_min_panned < self.data_min_for_current_plot:
                 new_min_panned = self.data_min_for_current_plot
                 new_max_panned = self.data_min_for_current_plot + current_width_pan
            if new_max_panned > self.data_max_for_current_plot:
                 new_max_panned = self.data_max_for_current_plot
                 new_min_panned = self.data_max_for_current_plot - current_width_pan
            new_min_panned = max(self.data_min_for_current_plot, min(self.data_max_for_current_plot - current_width_pan, new_min_panned)) 
            new_max_panned = new_min_panned + current_width_pan
            current_ax_lim = self.ax.get_xlim()
            if abs(new_min_panned - current_ax_lim[0]) > 1e-9 or abs(new_max_panned - current_ax_lim[1]) > 1e-9:
                self.ax.set_xlim(new_min_panned, new_max_panned)
                if self.freeze_x_range:
                    self._stored_xlim = self.ax.get_xlim()
                self.canvas.draw_idle()

    def _on_scroll(self, event):
        if event.inaxes != self.ax or event.xdata is None: return
        current_xlim_scroll = self.ax.get_xlim(); x_center_scroll = event.xdata
        zoom_factor = 1.25 if event.step > 0 else 1/1.25
        current_width_scroll = current_xlim_scroll[1] - current_xlim_scroll[0]
        new_zoomed_width = current_width_scroll / zoom_factor
        min_allowed_plot_width = (self.data_max_for_current_plot - self.data_min_for_current_plot) * 0.001 
        min_allowed_plot_width = max(min_allowed_plot_width, 1e-7 * (self.data_max_for_current_plot if self.data_max_for_current_plot > 0 else 1.0))
        max_allowed_plot_width = self.data_max_for_current_plot - self.data_min_for_current_plot
        new_zoomed_width = np.clip(new_zoomed_width, min_allowed_plot_width, max_allowed_plot_width if max_allowed_plot_width > 0 else 1.0)
        new_min_zoomed = x_center_scroll - (x_center_scroll - current_xlim_scroll[0]) * (new_zoomed_width / current_width_scroll)
        new_max_zoomed = new_min_zoomed + new_zoomed_width
        if new_min_zoomed < self.data_min_for_current_plot:
             new_min_zoomed = self.data_min_for_current_plot
             new_max_zoomed = self.data_min_for_current_plot + new_zoomed_width
        if new_max_zoomed > self.data_max_for_current_plot:
             new_max_zoomed = self.data_max_for_current_plot
             new_min_zoomed = self.data_max_for_current_plot - new_zoomed_width
        new_min_zoomed = max(self.data_min_for_current_plot, new_min_zoomed)
        new_max_zoomed = min(self.data_max_for_current_plot, new_max_zoomed)
        if new_zoomed_width < 1e-7: 
            if (self.data_max_for_current_plot - self.data_min_for_current_plot) > 1e-6:
                 self.ax.set_xlim(self.data_min_for_current_plot, self.data_max_for_current_plot)
            else:
                 self.ax.set_xlim(0,1)
            if self.freeze_x_range:
                self._stored_xlim = self.ax.get_xlim()
            self.canvas.draw_idle(); return
        if abs(new_min_zoomed - current_xlim_scroll[0]) > 1e-9 or abs(new_max_zoomed - current_xlim_scroll[1]) > 1e-9:
            self.ax.set_xlim(new_min_zoomed, new_max_zoomed)
            if self.freeze_x_range:
                self._stored_xlim = self.ax.get_xlim()
            self.canvas.draw_idle()

    def zoom_histogram(self, percentile_max=99.5):
        try:
            if self._current_data is None:
                return
            data_flat = self._current_data[np.isfinite(self._current_data)].ravel()
            if data_flat.size == 0:
                return
            x_max = np.percentile(data_flat, percentile_max)
            if not np.isfinite(x_max):
                return
            self.ax.set_xlim(0.0, max(0.02, float(x_max)))
            if self.freeze_x_range:
                self._stored_xlim = self.ax.get_xlim()
            self.canvas.draw()
        except Exception as e:
            print(f"ERREUR HistoWidget.zoom_histogram: {e}")

    def reset_histogram_view(self):
        try:
            self.ax.set_xlim(0.0, 1.0)
            if self.freeze_x_range:
                self._stored_xlim = self.ax.get_xlim()
            self.canvas.draw()
        except Exception as e:
            print(f"ERREUR HistoWidget.reset_histogram_view: {e}")






    def reset_zoom(self):
        """
        Réinitialise le zoom X à la plage complète des données actuelles et
        réinitialise le zoom Y à une échelle calculée sur les données actuelles.
        Version: HistoFix_ResetZoom_FixNameError_1
        """
        print(f"DEBUG HistoWidget.reset_zoom (V_HistoFix_ResetZoom_FixNameError_1): Réinitialisation zoom. Plage X données: [{self.data_min_for_current_plot:.4g}, {self.data_max_for_current_plot:.4g}]") # Version Log
        
        xlim_current_state = self.ax.get_xlim()
        ylim_current_state = self.ax.get_ylim()
        needs_redraw_flag = False
        
        # Réinitialiser le zoom X à la plage complète des données actuelles
        if abs(xlim_current_state[0] - self.data_min_for_current_plot) > 1e-6 or \
           abs(xlim_current_state[1] - self.data_max_for_current_plot) > 1e-6:
            self.ax.set_xlim(self.data_min_for_current_plot, self.data_max_for_current_plot)
            if self.freeze_x_range:
                self._stored_xlim = self.ax.get_xlim()
            needs_redraw_flag = True
            print(f"  -> Xlim réinitialisé à [{self.data_min_for_current_plot:.4g}, {self.data_max_for_current_plot:.4g}]")
        
        # Recalculer le Ylim basé sur les données actuelles de l'histogramme
        # (Utilise la même logique que plot_histogram pour target_top_y_limit)
        default_top_y_reset = 100 # Fallback si pas de données d'histogramme
        
        if self._current_hist_data_details and self._current_hist_data_details.get('hists'):
            all_pixel_counts_for_ylim_reset = []
            for hist_ch_reset in self._current_hist_data_details['hists']:
                if isinstance(hist_ch_reset, np.ndarray) and hist_ch_reset.size > 0:
                     all_pixel_counts_for_ylim_reset.extend(hist_ch_reset) 
            
            if all_pixel_counts_for_ylim_reset:
                 all_pixel_counts_np_reset = np.array(all_pixel_counts_for_ylim_reset)
                 counts_greater_than_zero_reset = all_pixel_counts_np_reset[all_pixel_counts_np_reset > 0]

                 if counts_greater_than_zero_reset.size > 0:
                    is_data_01_like_range_reset = (self.data_max_for_current_plot - self.data_min_for_current_plot) <= 2.0 and \
                                                 self.data_min_for_current_plot >= -0.1 and \
                                                 self.data_max_for_current_plot <= 2.015

                    if is_data_01_like_range_reset:
                        p98_counts_reset = np.percentile(counts_greater_than_zero_reset, 98)
                        max_actual_count_reset = np.max(counts_greater_than_zero_reset)
                        default_top_y_reset = p98_counts_reset * 3.0
                        default_top_y_reset = max(500, default_top_y_reset)
                        default_top_y_reset = min(default_top_y_reset, max_actual_count_reset * 1.5)
                        default_top_y_reset = max(default_top_y_reset, 100)
                    else: # Données ADU
                        percentile_99_5_y_reset = np.percentile(counts_greater_than_zero_reset, 99.5)
                        default_top_y_reset = max(10, percentile_99_5_y_reset * 1.5)
        
        default_top_y_reset = max(default_top_y_reset, ylim_current_state[0] + 10) # Assurer Ymax > Ymin
        target_ylim_reset = (0.8, default_top_y_reset) 
        
        if abs(ylim_current_state[0] - target_ylim_reset[0]) > 1e-1 or \
           abs(ylim_current_state[1] - target_ylim_reset[1]) > 1e-1:
            self.ax.set_ylim(target_ylim_reset)
            if self.freeze_y_range:
                self._stored_ylim = self.ax.get_ylim()
            needs_redraw_flag = True
            print(f"  -> Ylim réinitialisé à {target_ylim_reset}")
            
        if needs_redraw_flag:
            self.canvas.draw_idle()
            print("  -> Histogramme redessiné après reset_zoom.")

    def destroy(self):
        if _HIST_EXECUTOR:
            _HIST_EXECUTOR.shutdown(wait=False)
        super().destroy()









# --- END OF FILE seestar/gui/histogram_widget.py ---
