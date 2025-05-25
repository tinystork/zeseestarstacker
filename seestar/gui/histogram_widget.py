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
from matplotlib.figure import Figure
import traceback

class HistogramWidget(ttk.Frame):
    """
    Widget d'histogramme interactif utilisant Matplotlib dans Tkinter.
    Permet le déplacement des lignes de point noir/blanc et le zoom/pan.
    Version: HistoFix_DataRange_1
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

        self._current_hist_data_details = None # Stocke le retour de _calculate_hist_data
        
        # Valeurs des lignes BP/WP DANS L'ÉCHELLE DES DONNÉES ACTUELLES DE L'HISTOGRAMME
        self._min_line_val_data_scale = 0.0
        self._max_line_val_data_scale = 1.0
        
        # Plage effective des données affichées par l'histogramme (sera mis à jour par _calculate_hist_data)
        self.data_min_for_current_plot = 0.0
        self.data_max_for_current_plot = 1.0

        self.line_min_obj = None # Référence à l'objet ligne Matplotlib
        self.line_max_obj = None # Référence à l'objet ligne Matplotlib
        self.dragging_line_type = None # 'min', 'max', or None
        self._is_panning_active = False
        self._pan_start_coord_x = None
        self._pan_initial_xlim = None

        self.canvas.mpl_connect('button_press_event', self._on_press)
        self.canvas.mpl_connect('button_release_event', self._on_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('scroll_event', self._on_scroll)

        self.plot_histogram(None) # Afficher un histogramme vide initial
        print("DEBUG HistogramWidget (HistoFix_DataRange_1): __init__ terminé.")

    def _configure_plot_style(self):
        self.ax.clear()
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
        """
        Calcule et affiche l'histogramme pour les nouvelles données.
        data_for_analysis: Peut être en ADU ou 0-1.
        """
        # ---- HISTOFIX 1: Log entrée ----
        print(f"DEBUG HistoWidget.update_histogram: Reçu data_for_analysis.")
        if data_for_analysis is not None:
            print(f"  -> data_for_analysis - Shape: {data_for_analysis.shape}, Dtype: {data_for_analysis.dtype}, Range: [{np.nanmin(data_for_analysis):.4g} - {np.nanmax(data_for_analysis):.4g}]")
        else:
            print(f"  -> data_for_analysis est None.")
        # ---- FIN LOG ----
        self._current_hist_data_details = self._calculate_hist_data(data_for_analysis)
        # _calculate_hist_data met à jour self.data_min_for_current_plot et self.data_max_for_current_plot
        self.plot_histogram(self._current_hist_data_details)


    def _calculate_hist_data(self, data):
        """
        Calcule les données de l'histogramme (bins, comptes).
        S'adapte à la plage des données (0-1 ou ADU).
        Met à jour self.data_min_for_current_plot et self.data_max_for_current_plot.
        """
        if data is None or data.size == 0:
            self.data_min_for_current_plot = 0.0 # Réinitialiser la plage stockée
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
                # Si les valeurs sont très proches de 0 mais légèrement négatives (artefacts float?)
                if calculated_min < 0 and np.all(finite_data_for_range[finite_data_for_range < 0] > -1e-5):
                    calculated_min = 0.0
                calculated_min = max(0.0, calculated_min) # Assurer >= 0 pour données astro typiques
            else: # Pas de données valides pour déterminer la plage
                calculated_min, calculated_max = 0.0, 1.0 # Fallback
            
            # Gérer le cas d'une image plate
            if calculated_max <= calculated_min + 1e-7:
                # Si la plage est quasi-nulle et proche de 0-1, utiliser 0-1
                if calculated_max < 1.5 and calculated_min > -0.5:
                     current_plot_min, current_plot_max = 0.0, 1.0001 # Petit offset pour range
                else: # Sinon, centrer une petite plage autour de la valeur constante
                     current_plot_min = max(0, calculated_min - 0.5) if calculated_min > 0 else 0.0
                     current_plot_max = current_plot_min + 1.0
            else:
                current_plot_min = calculated_min
                current_plot_max = calculated_max
            
            # Stocker la plage déterminée pour l'utilisation dans plot_histogram et les contrôles
            self.data_min_for_current_plot = current_plot_min
            self.data_max_for_current_plot = current_plot_max + (current_plot_max - current_plot_min) * 0.001 if (current_plot_max - current_plot_min) > 1e-9 else current_plot_max + 1e-5 # Pour inclure la valeur max
            
            # ---- HISTOFIX 1: Log de la plage déterminée ----
            print(f"DEBUG HistoWidget._calculate_hist_data: Plage données pour histo (self.data_min/max_for_current_plot): [{self.data_min_for_current_plot:.4g}, {self.data_max_for_current_plot:.4g}]")
            # ---- FIN LOG ----
            
            hist_range_for_np = (self.data_min_for_current_plot, self.data_max_for_current_plot)

            if data_float.ndim == 3 and data_float.shape[2] == 3: # Color
                hist_data_dict['colors'] = ['#FF4444', '#44FF44', '#4466FF'] # R, G, B
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
            
            elif data_float.ndim == 2: # Grayscale
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
            return hist_data_dict
        except Exception as e:
            print(f"ERREUR HistoWidget._calculate_hist_data: {e}"); traceback.print_exc(limit=2); return None

    def plot_histogram(self, hist_data_details_to_plot):
        """Redessine l'histogramme."""
        # ---- HISTOFIX 1: Log entrée et utilisation de la plage stockée ----
        print(f"DEBUG HistoWidget.plot_histogram: Tentative de plot.")
        # Utiliser self.data_min/max_for_current_plot qui ont été mis à jour par _calculate_hist_data
        current_plot_min_x = self.data_min_for_current_plot
        current_plot_max_x = self.data_max_for_current_plot
        print(f"  -> Utilisant plage X stockée: [{current_plot_min_x:.4g}, {current_plot_max_x:.4g}]")
        # ---- FIN LOG ----

        xlim_before_plot = self.ax.get_xlim(); ylim_before_plot = self.ax.get_ylim()
        was_zoomed_or_panned = (abs(xlim_before_plot[0] - current_plot_min_x) > 1e-6 * abs(current_plot_max_x - current_plot_min_x) or \
                                abs(xlim_before_plot[1] - current_plot_max_x) > 1e-6 * abs(current_plot_max_x - current_plot_min_x))
        
        self._configure_plot_style()
        self.line_min_obj = None; self.line_max_obj = None # Réinitialiser les objets lignes

        if hist_data_details_to_plot is None or not hist_data_details_to_plot['hists'] or hist_data_details_to_plot['bins'] is None:
            self.ax.set_xlim(current_plot_min_x, current_plot_max_x) # Utiliser la plage même si vide
            self.ax.set_ylim(1, 10); self.ax.set_yscale('log')
            self.ax.set_xlabel(f"Niveau ({current_plot_min_x:.1f}-{current_plot_max_x:.1f})"); self.ax.set_ylabel("Nbre Pixels (log)")
            self.ax.text(0.5, 0.5, "Aucune donnée", color="gray", ha='center', va='center', transform=self.ax.transAxes)
            print(f"  -> Affichage 'Aucune donnée'. Xlim réglé sur [{current_plot_min_x:.4g}, {current_plot_max_x:.4g}].")
            self.canvas.draw_idle(); return

        try:
            bins = hist_data_details_to_plot['bins']; bin_centers = (bins[:-1] + bins[1:]) / 2
            
            # Les lignes BP/WP (_min_line_val_data_scale, _max_line_val_data_scale) doivent être dans la plage [current_plot_min_x, current_plot_max_x]
            # On les clippe ici si elles sont en dehors, ce qui peut arriver si les données changent radicalement (0-1 vs ADU)
            self._min_line_val_data_scale = np.clip(self._min_line_val_data_scale, current_plot_min_x, current_plot_max_x)
            self._max_line_val_data_scale = np.clip(self._max_line_val_data_scale, current_plot_min_x, current_plot_max_x)
            
            min_sep = (current_plot_max_x - current_plot_min_x) * 0.001
            min_sep = max(min_sep, 1e-7) 
            if self._max_line_val_data_scale <= self._min_line_val_data_scale + min_sep:
                 self._max_line_val_data_scale = self._min_line_val_data_scale + min_sep
                 if self._max_line_val_data_scale > current_plot_max_x:
                     self._max_line_val_data_scale = current_plot_max_x
                     self._min_line_val_data_scale = current_plot_max_x - min_sep
            self._min_line_val_data_scale = np.clip(self._min_line_val_data_scale, current_plot_min_x, current_plot_max_x - min_sep) # Re-clip min

            max_y_count_overall = 0; all_valid_pixel_counts = []
            for i, hist_counts in enumerate(hist_data_details_to_plot['hists']):
                counts_for_plotting = hist_counts + 1 # Pour échelle log
                if counts_for_plotting.size == bin_centers.size:
                    self.ax.plot(bin_centers, counts_for_plotting, color=hist_data_details_to_plot['colors'][i], alpha=0.85, drawstyle='steps-mid', linewidth=1.0)
                    valid_original_pixel_counts = hist_counts[hist_counts > 0]
                    if valid_original_pixel_counts.size > 0:
                        all_valid_pixel_counts.extend(valid_original_pixel_counts)
                        current_ch_max_y = np.max(hist_counts)
                        if current_ch_max_y > max_y_count_overall: max_y_count_overall = current_ch_max_y
                else: print(f"Warn HistoWidget.plot_histogram: Discrépance taille histo pour canal {i}")
            
            target_top_y_limit = 100 
            if all_valid_pixel_counts:
                 percentile_99_5_y = np.percentile(all_valid_pixel_counts, 99.5)
                 target_top_y_limit = max(10, percentile_99_5_y * 1.5)
            self.ax.set_ylim(bottom=0.8, top=target_top_y_limit); self.ax.set_yscale('log')

            self.line_min_obj = self.ax.axvline(self._min_line_val_data_scale, color='#FFAAAA', linestyle='--', linewidth=1.2, alpha=0.8, picker=5) 
            self.line_max_obj = self.ax.axvline(self._max_line_val_data_scale, color='#AAAAFF', linestyle='--', linewidth=1.2, alpha=0.8, picker=5) 
            self.ax.set_xlabel(f"Niveau ({current_plot_min_x:.2f}-{current_plot_max_x:.2f})"); self.ax.set_ylabel("Nbre Pixels (log)")
            # ---- HISTOFIX 1: Log des lignes BP/WP positionnées ----
            print(f"  -> Lignes BP/WP positionnées à [{self._min_line_val_data_scale:.4g}, {self._max_line_val_data_scale:.4g}] sur l'échelle des données.")
            # ---- FIN LOG ----

            if was_zoomed_or_panned and \
               xlim_before_plot[0] >= current_plot_min_x and xlim_before_plot[1] <= current_plot_max_x and \
               abs(ylim_before_plot[1] - target_top_y_limit) / max(1.0, target_top_y_limit) < 2.0 : # Si le zoom Y n'est pas trop différent
                self.ax.set_xlim(xlim_before_plot)
                self.ax.set_ylim(ylim_before_plot)
                print(f"  -> Zoom/Pan utilisateur restauré. Xlim: {xlim_before_plot}, Ylim: {ylim_before_plot}")
            else:
                self.ax.set_xlim(current_plot_min_x, current_plot_max_x)
                print(f"  -> Xlim initialisé à la plage des données: [{current_plot_min_x:.4g}, {current_plot_max_x:.4g}]")
            
            self.canvas.draw_idle()
        except Exception as e:
            print(f"ERREUR HistoWidget.plot_histogram: {e}"); traceback.print_exc(limit=2)
            try: # Fallback error display
                self._configure_plot_style()
                self.ax.set_xlim(0,1); self.ax.set_ylim(1,10); self.ax.set_yscale('log')
                self.ax.text(0.5, 0.5, "Erreur Histogramme", color="red", ha='center', va='center', transform=self.ax.transAxes)
                self.canvas.draw_idle()
            except Exception: pass

    def set_range(self, min_val_from_ui, max_val_from_ui):
        """
        Met à jour la position des lignes BP/WP.
        Les valeurs reçues (min_val_from_ui, max_val_from_ui) sont DANS L'ÉCHELLE 0-1 (venant des sliders UI).
        Elles doivent être converties à l'échelle des données actuelles de l'histogramme.
        """
        # ---- HISTOFIX 1: Conversion des BP/WP UI (0-1) vers l'échelle des données de l'histogramme ----
        # self.data_min_for_current_plot et self.data_max_for_current_plot reflètent la plage des données
        # sur laquelle l'histogramme est actuellement tracé.
        plot_data_range = self.data_max_for_current_plot - self.data_min_for_current_plot
        if plot_data_range < 1e-9: plot_data_range = 1.0 # Éviter division par zéro

        # Convertir les valeurs 0-1 des sliders vers l'échelle des données de l'histogramme
        new_min_line_val = self.data_min_for_current_plot + min_val_from_ui * plot_data_range
        new_max_line_val = self.data_min_for_current_plot + max_val_from_ui * plot_data_range
        
        print(f"DEBUG HistoWidget.set_range: Reçu UI BP={min_val_from_ui:.4f}, WP={max_val_from_ui:.4f}")
        print(f"  -> Plage histo actuelle: [{self.data_min_for_current_plot:.4g}, {self.data_max_for_current_plot:.4g}]")
        print(f"  -> Lignes BP/WP converties à l'échelle des données: [{new_min_line_val:.4g}, {new_max_line_val:.4g}]")
        # ---- FIN CONVERSION ----

        # Clipper et assurer la séparation dans l'échelle des données
        self._min_line_val_data_scale = np.clip(new_min_line_val, self.data_min_for_current_plot, self.data_max_for_current_plot)
        self._max_line_val_data_scale = np.clip(new_max_line_val, self.data_min_for_current_plot, self.data_max_for_current_plot)
        
        min_sep_abs = plot_data_range * 0.001
        min_sep_abs = max(min_sep_abs, 1e-7 * (self.data_max_for_current_plot if self.data_max_for_current_plot > 0 else 1.0) ) # Sécurité absolue relative à l'échelle

        if self._min_line_val_data_scale >= self._max_line_val_data_scale - min_sep_abs:
             self._min_line_val_data_scale = max(self.data_min_for_current_plot, self._max_line_val_data_scale - min_sep_abs)
        self._max_line_val_data_scale = np.clip(self._max_line_val_data_scale, self._min_line_val_data_scale + min_sep_abs, self.data_max_for_current_plot)
        self._min_line_val_data_scale = np.clip(self._min_line_val_data_scale, self.data_min_for_current_plot, self._max_line_val_data_scale - min_sep_abs) # Re-clip


        if self.line_min_obj: self.line_min_obj.set_xdata([self._min_line_val_data_scale] * 2)
        if self.line_max_obj: self.line_max_obj.set_xdata([self._max_line_val_data_scale] * 2)
        if self.line_min_obj or self.line_max_obj: self.canvas.draw_idle()
        # ---- HISTOFIX 1: Log des valeurs finales des lignes ----
        print(f"  -> Lignes BP/WP (échelle données) mises à jour à : [{self._min_line_val_data_scale:.4g}, {self._max_line_val_data_scale:.4g}]")
        # ---- FIN LOG ----

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
                    # Le callback attend des valeurs normalisées 0-1 par rapport à la plage des données de l'histogramme
                    data_range_for_cb = self.data_max_for_current_plot - self.data_min_for_current_plot
                    if data_range_for_cb < 1e-9 : data_range_for_cb = 1.0
                    
                    # Les valeurs _min/_max_line_val_data_scale SONT DÉJÀ DANS L'ÉCHELLE DES DONNÉES
                    # Le callback UI doit recevoir ces valeurs telles quelles.
                    # C'est la responsabilité de update_stretch_from_histogram dans main_window de les normaliser pour les sliders 0-1.
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
            
            # Contraindre le pan pour que la fenêtre de vue reste dans la plage des données
            if new_min_panned < self.data_min_for_current_plot:
                 new_min_panned = self.data_min_for_current_plot
                 new_max_panned = self.data_min_for_current_plot + current_width_pan
            if new_max_panned > self.data_max_for_current_plot:
                 new_max_panned = self.data_max_for_current_plot
                 new_min_panned = self.data_max_for_current_plot - current_width_pan
            
            # Assurer que min < max après contraintes
            new_min_panned = max(self.data_min_for_current_plot, min(self.data_max_for_current_plot - current_width_pan, new_min_panned)) 
            new_max_panned = new_min_panned + current_width_pan
            
            current_ax_lim = self.ax.get_xlim()
            if abs(new_min_panned - current_ax_lim[0]) > 1e-9 or abs(new_max_panned - current_ax_lim[1]) > 1e-9:
                self.ax.set_xlim(new_min_panned, new_max_panned); self.canvas.draw_idle()

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
            else: self.ax.set_xlim(0,1) 
            self.canvas.draw_idle(); return
            
        if abs(new_min_zoomed - current_xlim_scroll[0]) > 1e-9 or abs(new_max_zoomed - current_xlim_scroll[1]) > 1e-9:
            self.ax.set_xlim(new_min_zoomed, new_max_zoomed); self.canvas.draw_idle()

    def reset_zoom(self):
        # ---- HISTOFIX 1: Utiliser la plage stockée pour le reset ----
        print(f"DEBUG HistoWidget.reset_zoom: Réinitialisation du zoom à la plage des données actuelles: [{self.data_min_for_current_plot:.4g}, {self.data_max_for_current_plot:.4g}]")
        # ---- FIN LOG ----
        xlim_current_state = self.ax.get_xlim(); ylim_current_state = self.ax.get_ylim()
        needs_redraw_flag = False
        
        if abs(xlim_current_state[0] - self.data_min_for_current_plot) > 1e-6 or \
           abs(xlim_current_state[1] - self.data_max_for_current_plot) > 1e-6:
            self.ax.set_xlim(self.data_min_for_current_plot, self.data_max_for_current_plot)
            needs_redraw_flag = True
        
        if self._current_hist_data_details and self._current_hist_data_details['hists']:
            all_valid_counts_reset = []
            for hist_ch_reset in self._current_hist_data_details['hists']:
                if isinstance(hist_ch_reset, np.ndarray) and hist_ch_reset.size > 0:
                     all_valid_counts_reset.extend(hist_ch_reset[hist_ch_reset > 0])
            default_top_y_reset = 100 
            if all_valid_counts_reset:
                 p995_max_reset = np.percentile(all_valid_counts_reset, 99.5)
                 default_top_y_reset = max(10, p995_max_reset * 1.5)
            
            default_ylim_reset = (0.8, default_top_y_reset) 
            if abs(ylim_current_state[0] - default_ylim_reset[0]) > 1e-1 or \
               abs(ylim_current_state[1] - default_ylim_reset[1]) > 1e-1:
                self.ax.set_ylim(default_ylim_reset)
                needs_redraw_flag = True
        elif abs(ylim_current_state[0] - 0.8) > 1e-6 or abs(ylim_current_state[1] - 100) > 1e-6: # Fallback si pas de données
            self.ax.set_ylim(0.8, 100)
            needs_redraw_flag = True
            
        if needs_redraw_flag: self.canvas.draw_idle()