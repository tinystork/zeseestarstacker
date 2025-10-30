import sys
import numpy as np
from astropy.io import fits
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QSlider, QComboBox, QGraphicsView, QGraphicsScene,
                             QGroupBox, QToolBar, QAction,
                             QDoubleSpinBox, QTabWidget, QSizePolicy, QMessageBox,
                             QSplitter, QScrollArea)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPalette
from PyQt5.QtCore import Qt, QRectF, QSettings, pyqtSignal, QTimer, pyqtSlot, QFileInfo, QDir
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import cv2
import time # For performance checking if needed
import traceback # For detailed error printing

# --- DebayerProcessor ---
class DebayerProcessor:
    @staticmethod
    def _normalize_to_uint16(data): data_c=np.clip(data,0.,1.); return (data_c*65535).astype(np.uint16)
    @staticmethod
    def _convert_back_to_float32(data_u16): return data_u16.astype(np.float32)/65535.
    @staticmethod
    def debayer_rggb_bilinear(data_f32):
        if data_f32.ndim != 2: return data_f32
        data_u16 = DebayerProcessor._normalize_to_uint16(data_f32)
        try:
            bgr = cv2.cvtColor(data_u16, cv2.COLOR_BayerRG2RGB)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); return DebayerProcessor._convert_back_to_float32(rgb)
        except cv2.error as e: print(f"Err Bilin Debayer: {e}"); h,w=data_f32.shape; return np.zeros((h,w,3),dtype=np.float32)

# --- ColorCorrection ---
class ColorCorrection:
    @staticmethod
    def white_balance(data, r=1., g=1., b=1.):
        if data.ndim != 3 or data.shape[2] != 3: return data
        corr=data.copy(); corr[...,0]*=r; corr[...,1]*=g; corr[...,2]*=b
        return corr
    @staticmethod
    def color_matrix_transform(data, matrix):
        if data.ndim != 3: return data
        h,w,_=data.shape; resh=data.reshape(-1,3); transf=np.dot(resh,matrix.T)
        return transf.reshape(h,w,3)

# --- StretchPresets ---
class StretchPresets:
    @staticmethod
    def linear(data, bp=0., wp=1.): wp=max(wp,bp+1e-6); return (data-bp)/(wp-bp)
    @staticmethod
    def logarithmic(data, scale=1., bp=0.):
        data_s=data-bp; data_c=np.maximum(data_s,1e-10); max_v=np.max(data_c)
        if max_v<=0: return np.zeros_like(data)
        den=np.log1p(scale*max_v)
        if den<1e-10: return np.zeros_like(data)
        return np.log1p(scale*data_c)/den
    @staticmethod
    def asinh(data, scale=1., bp=0.):
        data_s=data-bp; data_c=np.maximum(data_s,0.); max_v=np.max(data_c)
        if max_v<=0: return np.zeros_like(data) # Check after calculating max_v
        den=np.arcsinh(scale*max_v)
        if den<1e-10: return np.zeros_like(data)
        return np.arcsinh(scale*data_c)/den

# --- HistogramWidget (avec Zoom/Pan manuel + Y-scale amélioré) ---
class HistogramWidget(QWidget):
    rangeChanged = pyqtSignal(float, float)
    histogramUpdated = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(4, 3), dpi=72); self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#2E2E2E'); self.figure.set_facecolor('#353535')
        self.ax.tick_params(axis='x', colors='lightgray'); self.ax.tick_params(axis='y', colors='lightgray')
        for spine in ['bottom', 'top', 'left', 'right']: self.ax.spines[spine].set_color('lightgray')
        self.ax.yaxis.label.set_color('lightgray'); self.ax.xaxis.label.set_color('lightgray')
        self.line_min=None; self.line_max=None; self.dragging=None
        self._min_val=0.; self._max_val=1.; self._current_data=None; self._hist_data_cache=None
        layout = QVBoxLayout(); layout.setContentsMargins(0, 0, 0, 0); layout.addWidget(self.canvas); self.setLayout(layout)
        self._is_panning = False; self._pan_start_x = None; self._pan_start_xlim = None
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)

    def update_histogram(self, data):
        self._current_data = data; self._hist_data_cache = None; self.plot_histogram()

    def plot_histogram(self):
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        is_valid_lim = all(np.isfinite(current_xlim)) and all(np.isfinite(current_ylim)) and current_ylim[1] > current_ylim[0]

        self.ax.clear()
        if self._current_data is None:
            self.ax.set_xlim(0, 1); self.ax.set_ylim(0, 1); self.canvas.draw(); self.histogramUpdated.emit(); return
        try:
            # --- 1. Calculate histogram data (if needed) ---
            if self._hist_data_cache is None:
                data = self._current_data; num_bins = 256; hist_range = (0, 1)
                if data.ndim == 3 and data.shape[2] == 3: # Color
                    colors = ['#FF4444', '#44FF44', '#4466FF']; self._hist_data_cache = {'type': 'color', 'bins': None, 'hists': [], 'colors': colors}
                    valid_bins = None
                    for i in range(3):
                        ch_data = data[..., i].ravel(); fin_data = ch_data[np.isfinite(ch_data)]
                        if fin_data.size > 0:
                            hist, bins = np.histogram(np.clip(fin_data, 0, 1), bins=num_bins, range=hist_range)
                            if valid_bins is None: valid_bins = bins
                            self._hist_data_cache['hists'].append(hist)
                        else: self._hist_data_cache['hists'].append(np.zeros(num_bins))
                    self._hist_data_cache['bins'] = valid_bins if valid_bins is not None else np.linspace(hist_range[0], hist_range[1], num_bins + 1)
                elif data.ndim == 2: # Grayscale
                    colors = ['lightgray']; self._hist_data_cache = {'type': 'gray', 'bins': None, 'hists': [], 'colors': colors}
                    fin_data = data.ravel()[np.isfinite(data.ravel())]
                    if fin_data.size > 0:
                         hist, bins = np.histogram(np.clip(fin_data, 0, 1), bins=num_bins, range=hist_range)
                         self._hist_data_cache['hists'].append(hist); self._hist_data_cache['bins'] = bins
                    else: self._hist_data_cache['hists'].append(np.zeros(num_bins)); self._hist_data_cache['bins'] = np.linspace(hist_range[0], hist_range[1], num_bins + 1)
                else: self._hist_data_cache = None; self.ax.set_xlim(0, 1); self.ax.set_ylim(0, 1); self.canvas.draw(); self.histogramUpdated.emit(); return

            # --- 2. Plot Histogram Bars ---
            if self._hist_data_cache:
                bins = self._hist_data_cache['bins']; bin_centers = (bins[:-1] + bins[1:]) / 2
                for i, hist in enumerate(self._hist_data_cache['hists']):
                    self.ax.plot(bin_centers, hist, color=self._hist_data_cache['colors'][i], alpha=0.8, drawstyle='steps-mid')

                # --- Calcul amélioré des limites Y ---
                all_counts = []
                for hist_channel in self._hist_data_cache['hists']:
                    all_counts.extend(hist_channel[hist_channel > 0]) # Prendre seulement les comptes > 0

                if all_counts: # S'il y a des comptes non nuls
                    p995_max = np.percentile(all_counts, 99.5)
                    p995_max = max(10, p995_max) # Assurer un max minimum
                else: # Fallback si image noire
                    p995_max = 10

                min_y = max(1, p995_max * 0.001) # Basé sur le percentile max
                # Appliquer nouvelles limites Y avec moins de marge
                # --- Fin calcul limites Y ---

                self.ax.set_yscale('log') # Appliquer l'échelle log

            # --- 3. Plot Min/Max Lines ---
            self.line_min = self.ax.axvline(self._min_val, color='#FF8888', linestyle='--', linewidth=1.5)
            self.line_max = self.ax.axvline(self._max_val, color='#8888FF', linestyle='--', linewidth=1.5)
            self.ax.set_xlabel("Niveau normalisé", color='lightgray'); self.ax.set_ylabel("Nbre Pixels (log)", color='lightgray')

            # --- 4. Restore/Set X Limits ---
            if is_valid_lim: self.ax.set_xlim(current_xlim) # Restaurer zoom/pan précédent
            else: self.ax.set_xlim(0, 1) # Afficher plage complète par défaut

            # --- 5. Ensure X/Y axes fit data on each update ---
            try:
                if self._hist_data_cache and 'hists' in self._hist_data_cache:
                    # --- Y autoscale with robust headroom ---
                    all_counts = []
                    display_max = 0.0
                    for hist_channel in self._hist_data_cache['hists']:
                        try:
                            all_counts.extend(hist_channel[hist_channel > 0])
                            display_max = max(display_max, float(np.max(hist_channel) + 1.0))
                        except Exception:
                            pass
                    if all_counts:
                        counts = np.asarray(all_counts)
                        max_actual = float(np.max(counts))
                        p99 = float(np.percentile(counts, 99))
                        desired_top = max(100.0, p99 * 2.5, max_actual * 1.40, display_max * 1.40)
                    else:
                        desired_top = max(100.0, display_max * 1.40)
                    bottom = max(0.8, self.ax.get_ylim()[0] if self.ax.get_ylim() else 0.8)
                    if desired_top <= bottom + 1.0:
                        desired_top = bottom + 10.0
                    self.ax.set_yscale('log')
                    self.ax.set_ylim(bottom, desired_top)

                    # --- X autoscale: expand to include new data range ---
                    try:
                        bins = self._hist_data_cache.get('bins', None)
                        if bins is not None and len(bins) > 1:
                            data_min_x = float(bins[0])
                            data_max_x = float(bins[-1])
                            cur_xlim = self.ax.get_xlim()
                            x0, x1 = cur_xlim[0], cur_xlim[1]
                            span = max(data_max_x - data_min_x, 1e-12)
                            eps = 1e-9 * span
                            if not np.isfinite(x0) or not np.isfinite(x1) or x1 <= x0:
                                self.ax.set_xlim(data_min_x, data_max_x)
                            else:
                                if x0 > data_min_x + eps:
                                    x0 = data_min_x
                                if x1 < data_max_x - eps:
                                    x1 = data_max_x
                                self.ax.set_xlim(x0, x1)
                    except Exception:
                        pass
            except Exception:
                pass

            # --- 6. Final Draw ---
            self.figure.tight_layout(pad=0.5); self.canvas.draw()
        except Exception as e: print(f"Erreur plot_histogram: {e}"); traceback.print_exc(); self.ax.clear(); self.ax.set_xlim(0, 1); self.ax.set_ylim(0, 1); self.canvas.draw()
        finally: self.histogramUpdated.emit()

    def set_range(self, min_val, max_val):
        self._min_val = np.clip(min_val, 0.0, 1.0); self._max_val = np.clip(max_val, 0.0, 1.0)
        if self._min_val >= self._max_val: self._min_val = max(0.0, self._max_val - 0.001)
        if self.line_min: self.line_min.set_xdata([self._min_val, self._min_val])
        if self.line_max: self.line_max.set_xdata([self._max_val, self._max_val])
        self.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None: return
        self.dragging = None; self._is_panning = False
        if event.button == 1: # Gauche: Drag lignes
            tol=0.02; d_min=abs(event.xdata-self._min_val); d_max=abs(event.xdata-self._max_val)
            if d_min < tol and d_min <= d_max: self.dragging='min'
            elif d_max < tol: self.dragging='max'
        elif event.button == 3: # Droit: Pan
            self._is_panning = True; self._pan_start_x = event.xdata; self._pan_start_xlim = self.ax.get_xlim()
            self.canvas.setCursor(Qt.ClosedHandCursor)

    def on_release(self, event):
        if self.dragging: self.dragging = None
        if self._is_panning: self._is_panning = False; self.canvas.setCursor(Qt.ArrowCursor)

    def on_motion(self, event):
        if event.inaxes != self.ax or event.xdata is None: return
        if self.dragging:
            x=np.clip(event.xdata, 0, 1); sep=0.005
            if self.dragging == 'min': self._min_val=min(x,self._max_val-sep); self.line_min.set_xdata([self._min_val]*2) if self.line_min else None
            elif self.dragging == 'max': self._max_val=max(x,self._min_val+sep); self.line_max.set_xdata([self._max_val]*2) if self.line_max else None
            self.canvas.draw_idle(); self.rangeChanged.emit(self._min_val, self._max_val)
        elif self._is_panning:
            dx = event.xdata - self._pan_start_x; start_lim = self._pan_start_xlim
            new_min = start_lim[0] - dx; new_max = start_lim[1] - dx
            width = start_lim[1] - start_lim[0]
            if new_min < 0: new_min = 0; new_max = new_min + width
            if new_max > 1: new_max = 1; new_min = new_max - width
            new_min = max(0.0, new_min); new_max = min(1.0, new_max)
            if new_max <= new_min : new_max = min(1.0, new_min + 1e-4)
            self.ax.set_xlim(new_min, new_max); self.canvas.draw_idle()

    def on_scroll(self, event):
        if event.inaxes != self.ax or event.xdata is None: return
        xlim = self.ax.get_xlim(); x = event.xdata
        factor = 1.2 if event.step > 0 else 1/1.2
        new_min = x - (x - xlim[0]) / factor; new_max = x + (xlim[1] - x) / factor
        new_min = max(0.0, new_min); new_max = min(1.0, new_max)
        min_width = 1e-3
        if (new_max - new_min) < min_width:
            center=(new_min+new_max)/2; new_min=center-min_width/2; new_max=center+min_width/2
            new_min = max(0.0, new_min); new_max = min(1.0, new_max)
        if new_max > new_min: self.ax.set_xlim(new_min, new_max); self.canvas.draw_idle()

    def reset_zoom(self):
        xlim = self.ax.get_xlim(); needs_redraw = False
        if abs(xlim[0] - 0.0) > 1e-6 or abs(xlim[1] - 1.0) > 1e-6:
            self.ax.set_xlim(0, 1); needs_redraw = True; print("Histogram X zoom reset")
        # Reset Y zoom based on current histogram data if possible
        if self._hist_data_cache:
             all_counts = []
             for hist_channel in self._hist_data_cache['hists']: all_counts.extend(hist_channel[hist_channel > 0])
             if all_counts: p995_max = max(10, np.percentile(all_counts, 99.5))
             else: p995_max = 10
             min_y = max(1, p995_max * 0.001); target_ylim = (min_y, p995_max * 1.1)
             current_ylim = self.ax.get_ylim()
             if abs(current_ylim[0]-target_ylim[0])>1e-6 or abs(current_ylim[1]-target_ylim[1])>1e-6:
                 self.ax.set_ylim(target_ylim); needs_redraw = True
        if needs_redraw: self.canvas.draw_idle()

# --- ZoomGraphicsView ---
# (Pas de changements)
class ZoomGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent); self.scene=QGraphicsScene(self); self.setScene(self.scene)
        self.setDragMode(QGraphicsView.ScrollHandDrag); self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse); self.setRenderHint(QPainter.Antialiasing, True); self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self._scale_factor = 1.0; self._pixmap_item = None
    def wheelEvent(self, event):
        zin=1.25; zout=1/zin; factor = zin if event.angleDelta().y() > 0 else zout
        self._scale_factor *= factor; self.scale(factor, factor)
    def fit_view(self):
        if self._pixmap_item: self.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio); self._scale_factor = self.transform().m11()
    def set_pixmap(self, pixmap):
        self.scene.clear();
        if not pixmap.isNull(): self._pixmap_item=self.scene.addPixmap(pixmap); self.setSceneRect(QRectF(pixmap.rect()))
        else: self._pixmap_item = None; self.setSceneRect(QRectF())


# --- TelescopeImageViewer ---
class TelescopeImageViewer(QMainWindow):
    # ... (__init__ reste identique) ...
    def __init__(self):
        super().__init__(); self.setWindowTitle("Advanced Seestar S50 Viewer"); self.setGeometry(100, 100, 1500, 950)
        self.raw_data=None; self.debayered_data=None; self.white_balanced_data=None; self.stretched_data=None; self.display_data=None
        self._needs_debayer_update=True; self._needs_wb_update=True; self._needs_stretch_update=True; self._needs_display_update=True; self._is_processing=False
        self.settings = QSettings("AstroImaging", "AdvancedSeestarViewer")
        self.update_timer = QTimer(self); self.update_timer.setSingleShot(True); self.update_timer.setInterval(100); self.update_timer.timeout.connect(self.process_image_updates)
        self.init_ui(); self.load_settings()

    # ... (init_ui : création bouton R identique) ...
    def init_ui(self):
        self.view = ZoomGraphicsView(); self.histogram = HistogramWidget(); self.init_toolbar()
        control_panel=QWidget(); control_layout=QVBoxLayout();
        control_panel.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred); control_panel.setMaximumWidth(450)
        control_panel.setMinimumWidth(320)
        scroll_control = QScrollArea(); scroll_control.setWidgetResizable(True); scroll_control.setFrameShape(QScrollArea.NoFrame); scroll_control.setWidget(control_panel)
        tabs=QTabWidget(); file_tab=QWidget(); self.init_file_tab(file_tab); tabs.addTab(file_tab, "Fichier")
        debayer_tab=QWidget(); self.init_debayer_tab(debayer_tab); tabs.addTab(debayer_tab, "Débayering")
        color_tab=QWidget(); self.init_color_tab(color_tab); tabs.addTab(color_tab, "Couleur")
        stretch_tab=QWidget(); self.init_stretch_tab(stretch_tab); tabs.addTab(stretch_tab, "Étirement")
        control_layout.addWidget(tabs); control_panel.setLayout(control_layout)
        display_panel=QWidget(); display_layout=QVBoxLayout()
        display_layout.addWidget(self.view, 5)
        hist_container_layout = QHBoxLayout(); hist_container_layout.addWidget(self.histogram, 10)
        reset_zoom_btn = QPushButton("R"); reset_zoom_btn.setToolTip("Réinitialiser zoom histogramme"); reset_zoom_btn.setFixedWidth(35); reset_zoom_btn.setFixedHeight(35)
        reset_zoom_btn.clicked.connect(self.histogram.reset_zoom); hist_container_layout.addWidget(reset_zoom_btn, 0, Qt.AlignTop)
        display_layout.addLayout(hist_container_layout, 2); display_panel.setLayout(display_layout)
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(scroll_control)
        self.splitter.addWidget(display_panel)
        self.splitter.setSizes([350, 900])
        self.splitter.setCollapsible(0, False)
        self.splitter.setStretchFactor(1, 1)
        self.setCentralWidget(self.splitter)
        sizes = QSettings().value("viewer/splitter", None)
        if sizes: self.splitter.setSizes([int(s) for s in sizes])
        self.histogram.rangeChanged.connect(self.update_stretch_sliders_from_histogram); self.histogram.histogramUpdated.connect(self.on_histogram_ready)
        self.statusBar().showMessage("Prêt. Chargez une image.")

    # ... (init_toolbar, init_file_tab, init_debayer_tab, init_color_tab, init_stretch_tab, create_slider_with_spinbox identiques) ...
    def init_toolbar(self):
        toolbar=QToolBar("Outils principaux"); toolbar.setObjectName("MainToolBar"); self.addToolBar(Qt.TopToolBarArea, toolbar)
        oa=QAction("Ouvrir", self); oa.setShortcut("Ctrl+O"); oa.triggered.connect(self.load_image); toolbar.addAction(oa)
        sa=QAction("Sauvegarder", self); sa.setShortcut("Ctrl+S"); sa.triggered.connect(self.save_image); toolbar.addAction(sa)
        toolbar.addSeparator()
        zi=QAction("Zoom +", self); zi.setShortcut("Ctrl++"); zi.triggered.connect(lambda: self.view.scale(1.25, 1.25)); toolbar.addAction(zi)
        zo=QAction("Zoom -", self); zo.setShortcut("Ctrl+-"); zo.triggered.connect(lambda: self.view.scale(0.8, 0.8)); toolbar.addAction(zo)
        fa=QAction("Ajuster", self); fa.setShortcut("Ctrl+F"); fa.triggered.connect(self.view.fit_view); toolbar.addAction(fa)
    def init_file_tab(self, tab):
        layout=QVBoxLayout(); fg=QGroupBox("Format d'image"); fl=QVBoxLayout()
        self.fc=QComboBox(); self.fc.addItems(["FITS (*.fit *.fits)", "TIFF (*.tif *.tiff)", "PNG (*.png)", "JPEG (*.jpg *.jpeg)"])
        fl.addWidget(self.fc); fg.setLayout(fl); bl=QHBoxLayout()
        self.lb=QPushButton("Charger"); self.lb.clicked.connect(self.load_image)
        self.sb=QPushButton("Sauvegarder"); self.sb.clicked.connect(self.save_image)
        bl.addWidget(self.lb); bl.addWidget(self.sb); layout.addWidget(fg); layout.addLayout(bl); layout.addStretch(); tab.setLayout(layout)
    def init_debayer_tab(self, tab):
        layout=QVBoxLayout(); mg=QGroupBox("Méthode de débayering"); ml=QVBoxLayout()
        self.dc=QComboBox(); self.dc.addItems(["Bilinéaire"]); self.dc.setCurrentIndex(0); self.dc.currentIndexChanged.connect(self.request_debayer_update)
        ml.addWidget(QLabel("Algorithme (Bilinéaire):")); ml.addWidget(self.dc); mg.setLayout(ml); layout.addWidget(mg); layout.addStretch(); tab.setLayout(layout)
    def init_color_tab(self, tab):
        layout=QVBoxLayout(); wg=QGroupBox("Balance des blancs"); wl=QVBoxLayout()
        self.rs=self.create_slider_with_spinbox("R:", 0.1, 5.0, 1.0, 0.01, self.request_wb_update)
        self.gs=self.create_slider_with_spinbox("G:", 0.1, 5.0, 1.0, 0.01, self.request_wb_update)
        self.bs=self.create_slider_with_spinbox("B:", 0.1, 5.0, 1.0, 0.01, self.request_wb_update)
        wl.addLayout(self.rs['layout']); wl.addLayout(self.gs['layout']); wl.addLayout(self.bs['layout'])
        auto_wb_btn = QPushButton("Auto Balance Blancs"); auto_wb_btn.clicked.connect(self.apply_auto_white_balance); wl.addWidget(auto_wb_btn)
        reset_wb_btn = QPushButton("Réinitialiser Gains"); reset_wb_btn.clicked.connect(self.reset_white_balance); wl.addWidget(reset_wb_btn)
        wg.setLayout(wl); layout.addWidget(wg); layout.addStretch(); tab.setLayout(layout)
    def init_stretch_tab(self, tab):
        layout=QVBoxLayout(); sg=QGroupBox("Fonction d'étirement"); sl=QVBoxLayout()
        self.sc=QComboBox(); self.sc.addItems(["Linéaire", "Arcsinh", "Logarithmique"]); self.sc.currentIndexChanged.connect(self.request_stretch_update)
        sl.addWidget(QLabel("Méthode:")); sl.addWidget(self.sc); sg.setLayout(sl)
        pg=QGroupBox("Paramètres d'étirement"); pl=QVBoxLayout()
        self.bps=self.create_slider_with_spinbox("Noir:", 0.0, 1.0, 0.0, 0.001, self.request_stretch_update_from_sliders)
        self.wps=self.create_slider_with_spinbox("Blanc:", 0.0, 1.0, 1.0, 0.001, self.request_stretch_update_from_sliders)
        self.gamma_ctrls=self.create_slider_with_spinbox("Gamma:", 0.1, 3.0, 1.0, 0.01, self.request_stretch_update)
        pl.addLayout(self.bps['layout']); pl.addLayout(self.wps['layout']); pl.addLayout(self.gamma_ctrls['layout'])
        auto_stretch_btn = QPushButton("Auto Étirement"); auto_stretch_btn.clicked.connect(self.apply_auto_stretch); pl.addWidget(auto_stretch_btn)
        reset_stretch_btn = QPushButton("Réinitialiser Étirement"); reset_stretch_btn.clicked.connect(self.reset_stretch); pl.addWidget(reset_stretch_btn)
        pg.setLayout(pl); layout.addWidget(sg); layout.addWidget(pg); layout.addStretch(); tab.setLayout(layout)
    def create_slider_with_spinbox(self, label, min_val, max_val, default_val, step, change_callback):
        wd={}; lo=QHBoxLayout(); lbl=QLabel(label); lo.addWidget(lbl)
        sl=QSlider(Qt.Horizontal); sl.setRange(0, int((max_val-min_val)/step) if step > 0 else 0); sl.setValue(int((default_val-min_val)/step) if step > 0 else 0); wd['slider']=sl
        sb=QDoubleSpinBox(); sb.setRange(min_val, max_val); sb.setValue(default_val); sb.setSingleStep(step)
        sb.setDecimals(max(1, -int(np.log10(step)) if step>0 else 3)); sb.setFixedWidth(80); wd['spinbox']=sb
        sl.valueChanged.connect(lambda val, s=sb, m=min_val, st=step: s.setValue(m+val*st));
        sb.valueChanged.connect(lambda val, s=sl, m=min_val, st=step: s.setValue(int((val-m)/st)) if st>0 else 0)
        sb.valueChanged.connect(change_callback); lo.addWidget(sl); lo.addWidget(sb); wd['layout']=lo
        return wd

    # ... (load_settings, save_settings identiques) ...
    def load_settings(self):
        self.settings.beginGroup("MainWindow"); geo=self.settings.value("geometry", self.saveGeometry()); state=self.settings.value("state", self.saveState()); self.restoreGeometry(geo); self.restoreState(state); self.settings.endGroup()
        self.settings.beginGroup("Controls"); self.dc.setCurrentIndex(0) # Force Bilinear
        self.sc.setCurrentText(self.settings.value("stretchMethod", "Linéaire"))
        self.rs['spinbox'].setValue(float(self.settings.value("rGain", 1.0))); self.gs['spinbox'].setValue(float(self.settings.value("gGain", 1.0))); self.bs['spinbox'].setValue(float(self.settings.value("bGain", 1.0)))
        self.bps['spinbox'].setValue(float(self.settings.value("blackPoint", 0.0))); self.wps['spinbox'].setValue(float(self.settings.value("whitePoint", 1.0))); self.gamma_ctrls['spinbox'].setValue(float(self.settings.value("gamma", 1.0))) # Use gamma_ctrls
        self.settings.endGroup()
    def save_settings(self):
        self.settings.beginGroup("MainWindow"); self.settings.setValue("geometry", self.saveGeometry()); self.settings.setValue("state", self.saveState()); self.settings.endGroup()
        self.settings.beginGroup("Controls"); self.settings.setValue("debayerMethod", self.dc.currentText()); self.settings.setValue("stretchMethod", self.sc.currentText())
        self.settings.setValue("rGain", self.rs['spinbox'].value()); self.settings.setValue("gGain", self.gs['spinbox'].value()); self.settings.setValue("bGain", self.bs['spinbox'].value())
        self.settings.setValue("blackPoint", self.bps['spinbox'].value()); self.settings.setValue("whitePoint", self.wps['spinbox'].value()); self.settings.setValue("gamma", self.gamma_ctrls['spinbox'].value()) # Use gamma_ctrls
        self.settings.endGroup()

    # --- load_image : Réinitialise le zoom ---
    def load_image(self):
        ldir = self.settings.value("LastOpenDir", ".")
        path, _ = QFileDialog.getOpenFileName(self, "Ouvrir", ldir, self.fc.currentText()+";;All Files (*)")
        if not path: return
        self.settings.setValue("LastOpenDir", QFileInfo(path).absolutePath())
        self.statusBar().showMessage(f"Chargement {QFileInfo(path).fileName()}..."); QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            start = time.time()
            if path.lower().endswith(('.fit', '.fits')):
                with fits.open(path) as hdul:
                    if not hdul: raise ValueError("FITS invalide.")
                    hdu = None;
                    for h in hdul:
                        if h.data is not None and h.is_image: hdu = h; break
                    if hdu is None or hdu.data is None: raise ValueError("Aucune donnée image trouvée.")
                    self.raw_data = hdu.data.astype(np.float32)
                    if self.raw_data.ndim == 3:
                        shape = self.raw_data.shape
                        if shape[0] == 3 and shape[1] > 3 and shape[2] > 3: print(f"FITS shape {shape}. Transpose (C,H,W)->(H,W,C)..."); self.raw_data = np.transpose(self.raw_data, (1, 2, 0)); print(f"New shape: {self.raw_data.shape}")
                        elif shape[2] == 3 and shape[0] > 3 and shape[1] > 3:
                             h_hdr=hdu.header.get('NAXIS2',-1); w_hdr=hdu.header.get('NAXIS1',-1)
                             if shape[0]==w_hdr and shape[1]==h_hdr: print(f"FITS shape {shape}. Transpose (W,H,C)->(H,W,C)..."); self.raw_data = np.transpose(self.raw_data, (1, 0, 2)); print(f"New shape: {self.raw_data.shape}")
            elif path.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
                img=Image.open(path); arr=np.array(img)
                if arr.dtype == np.uint8: self.raw_data = arr.astype(np.float32)/255.0
                elif arr.dtype == np.uint16: self.raw_data = arr.astype(np.float32)/65535.0
                else: self.raw_data = arr.astype(np.float32)
                if self.raw_data.ndim == 3 and self.raw_data.shape[2] == 4: self.raw_data = self.raw_data[..., :3]
            else: raise ValueError(f"Format non supporté: {path}")
            if self.raw_data is None: raise ValueError("Echec chargement.")
            self.raw_data = np.nan_to_num(self.raw_data, nan=0.0, posinf=0.0, neginf=0.0)
            if self.raw_data.size == 0: raise ValueError("Image vide.")
            min_v=np.min(self.raw_data); max_v=np.max(self.raw_data); print(f"Data Range: {min_v}, {max_v}")
            if max_v > min_v: self.raw_data = (self.raw_data - min_v) / (max_v - min_v)
            elif max_v > 0: self.raw_data = self.raw_data / max_v
            else: self.raw_data = np.zeros_like(self.raw_data)
            print(f"Data chargé et normalisé: {self.raw_data.shape}, {self.raw_data.dtype}, range=[{np.min(self.raw_data):.3f}, {np.max(self.raw_data):.3f}]")
            self.invalidate_caches(debayer=True, wb=True, stretch=True)
            self.request_update()
            QTimer.singleShot(100, self.view.fit_view); end=time.time()
            self.statusBar().showMessage(f"Chargé en {end-start:.2f}s. Traitement...", 5000)
        except Exception as e:
            self.raw_data=None; self.invalidate_caches(True,True,True); self.update_image_display(); self.histogram.update_histogram(None); self.histogram.reset_zoom()
            print(f"Erreur chargement: {e}"); traceback.print_exc(); QMessageBox.critical(self, "Erreur", f"Chargement impossible:\n{e}")
            self.statusBar().showMessage("Erreur chargement.", 5000)
        finally: QApplication.restoreOverrideCursor()

    # ... (invalidate_caches, request_update*, process_image_updates identiques) ...
    def invalidate_caches(self, debayer=False, wb=False, stretch=False):
        if debayer: self.debayered_data=None; self._needs_debayer_update=True
        if wb or debayer: self.white_balanced_data=None; self._needs_wb_update=True
        if stretch or wb or debayer: self.stretched_data=None; self._needs_stretch_update=True
        self._needs_display_update=True
    def request_update(self):
        if self.raw_data is None: return
        self.update_timer.start();
    def request_debayer_update(self): self.invalidate_caches(debayer=True); self.request_update()
    def request_wb_update(self): self.invalidate_caches(wb=True); self.request_update()
    def request_stretch_update(self): self.invalidate_caches(stretch=True); self.request_update()
    def request_stretch_update_from_sliders(self):
        self.invalidate_caches(stretch=True); self.request_update()
        bp=self.bps['spinbox'].value(); wp=self.wps['spinbox'].value()
        self.histogram.set_range(bp, wp) # Met à jour seulement les lignes
    def process_image_updates(self):
        if self.raw_data is None or self._is_processing: return
        self._is_processing=True; self.statusBar().showMessage("Traitement..."); QApplication.processEvents(); start=time.time()
        try:
            if self._needs_debayer_update:
                if self.raw_data.ndim == 2: self.debayered_data = DebayerProcessor.debayer_rggb_bilinear(self.raw_data)
                else: self.debayered_data = self.raw_data.copy()
                self._needs_debayer_update = False; self._needs_wb_update = True
            current_data_for_wb = self.debayered_data
            if self._needs_wb_update and current_data_for_wb is not None:
                 if current_data_for_wb.ndim == 3:
                     r=self.rs['spinbox'].value(); g=self.gs['spinbox'].value(); b=self.bs['spinbox'].value()
                     self.white_balanced_data = ColorCorrection.white_balance(current_data_for_wb, r, g, b)
                 else: self.white_balanced_data = current_data_for_wb
                 self._needs_wb_update = False; self._needs_stretch_update = True
            elif self._needs_wb_update: self.white_balanced_data = None; self._needs_wb_update = False
            current_data_for_stretch = self.white_balanced_data
            if self._needs_stretch_update and current_data_for_stretch is not None:
                bp=self.bps['spinbox'].value(); wp=self.wps['spinbox'].value(); method = self.sc.currentText()
                if method == "Linéaire": self.stretched_data = StretchPresets.linear(current_data_for_stretch, bp, wp)
                elif method=="Logarithmique": self.stretched_data = StretchPresets.logarithmic(current_data_for_stretch, scale=10.0, black_point=bp)
                elif method == "Arcsinh": self.stretched_data = StretchPresets.asinh(current_data_for_stretch, scale=10.0, black_point=bp)
                else: self.stretched_data = StretchPresets.linear(current_data_for_stretch, bp, wp)
                self.stretched_data = np.maximum(self.stretched_data, 0.0)
                self._needs_stretch_update = False; self._needs_display_update = True
            elif self._needs_stretch_update: self.stretched_data = None; self._needs_stretch_update = False
            current_data_for_display = self.stretched_data
            if self._needs_display_update and current_data_for_display is not None:
                gamma = self.gamma_ctrls['spinbox'].value()
                if gamma == 1.0: gamma_corrected = current_data_for_display
                else: gamma_corrected = np.power(np.maximum(current_data_for_display, 0.0), gamma)
                self.display_data = (np.clip(gamma_corrected, 0.0, 1.0) * 255).astype(np.uint8)
                self.update_image_display()
                self.histogram.update_histogram(current_data_for_display) # Update histogram
                self._needs_display_update = False
        except Exception as e: print(f"Erreur traitement: {e}"); traceback.print_exc(); self.statusBar().showMessage("Erreur traitement.", 5000)
        finally: self._is_processing=False;

    # --- update_stretch_sliders_from_histogram : Met à jour seulement les sliders ---
    def update_stretch_sliders_from_histogram(self, min_val, max_val):
        self.bps['spinbox'].blockSignals(True); self.wps['spinbox'].blockSignals(True); self.bps['slider'].blockSignals(True); self.wps['slider'].blockSignals(True)
        self.bps['spinbox'].setValue(min_val); self.wps['spinbox'].setValue(max_val)
        self.bps['spinbox'].blockSignals(False); self.wps['spinbox'].blockSignals(False); self.bps['slider'].blockSignals(False); self.wps['slider'].blockSignals(False)
        self.invalidate_caches(stretch=True); self.request_update()

    @pyqtSlot()
    def on_histogram_ready(self):
        if not self._is_processing: self.statusBar().showMessage("Prêt.", 2000)

    def reset_white_balance(self):
        self.rs['spinbox'].setValue(1.0); self.gs['spinbox'].setValue(1.0); self.bs['spinbox'].setValue(1.0)

    # --- reset_stretch : S'assure de reset le zoom ---
    def reset_stretch(self):
        self.bps['spinbox'].setValue(0.0); self.wps['spinbox'].setValue(1.0); self.gamma_ctrls['spinbox'].setValue(1.0)
        self.histogram.set_range(0.0, 1.0)

    # --- apply_auto_stretch : N'active PLUS le zoom (géré par l'interaction user) ---
    def apply_auto_stretch(self):
        data_to_analyze = self.white_balanced_data
        if data_to_analyze is None: self.statusBar().showMessage("Auto Étirement: Aucune donnée.", 3000); return
        self.statusBar().showMessage("Calcul Auto Étirement..."); QApplication.processEvents()
        try:
            # ... (Calcul de luminance identique) ...
            if data_to_analyze.ndim == 3:
                finite_mask = np.all(np.isfinite(data_to_analyze), axis=-1);
                if not np.any(finite_mask): raise ValueError("Pas de data.")
                luminance = np.mean(data_to_analyze[finite_mask], axis=-1)
            else: finite_mask = np.isfinite(data_to_analyze);
            if not np.any(finite_mask): raise ValueError("Pas de data.")
            luminance = data_to_analyze[finite_mask]
            if luminance.size < 10: raise ValueError("Pas assez pixels.")

            black_percentile_value = np.percentile(luminance, 1.0); black_point = black_percentile_value + 0.001
            pixels_above_black = luminance[luminance > black_point]
            if pixels_above_black.size > 0: white_point = np.percentile(pixels_above_black, 99.0)
            else: print("Warn AutoStretch: Pas de pixels > BP. WP=1.0."); white_point = 1.0
            black_point = np.clip(black_point, 0.0, 0.99); white_point = np.clip(white_point, black_point + 0.01, 1.0)
            print(f"Auto Étirement: BP={black_point:.4f}, WP={white_point:.4f}")

            self.bps['spinbox'].setValue(black_point); self.wps['spinbox'].setValue(white_point)
            self.histogram.set_range(black_point, white_point) # Met à jour les lignes
            # PAS d'activation de zoom ici, l'utilisateur le fera s'il veut
            self.statusBar().showMessage("Auto Étirement appliqué.", 3000)
        except Exception as e: print(f"Erreur Auto Étirement: {e}"); traceback.print_exc(); self.statusBar().showMessage("Erreur Auto Étirement.", 3000)

    # ... (apply_auto_white_balance, update_image_display, save_image, closeEvent identiques) ...
    def apply_auto_white_balance(self):
        data_to_analyze = self.debayered_data
        if data_to_analyze is None or data_to_analyze.ndim != 3: self.statusBar().showMessage("Auto WB: Données couleur requises.", 3000); return
        self.statusBar().showMessage("Calcul Auto WB..."); QApplication.processEvents()
        try:
            modes = []
            num_bins = 256
            for i in range(3):
                channel_data = data_to_analyze[..., i].ravel(); finite_data = channel_data[np.isfinite(channel_data)]
                if finite_data.size == 0: raise ValueError(f"Canal {i} vide.")
                min_r, max_r = np.percentile(finite_data, [0.1, 99.0]);
                if max_r <= min_r: max_r = min_r + 1e-5
                hist, bin_edges = np.histogram(finite_data, bins=num_bins, range=(min_r, max_r))
                mode_index = np.argmax(hist); channel_mode = (bin_edges[mode_index] + bin_edges[mode_index+1]) / 2
                if channel_mode <= 1e-5: print(f"Warn AutoWB: Mode canal {i} bas ({channel_mode:.4f}). Use 1e-5."); channel_mode = 1e-5
                modes.append(channel_mode)
            mode_r, mode_g, mode_b = modes; print(f"Auto WB: Modes R={mode_r:.4f}, G={mode_g:.4f}, B={mode_b:.4f}")
            gain_r = mode_g / mode_r; gain_g = 1.0; gain_b = mode_g / mode_b
            max_gain = 5.0; min_gain = 0.2
            gain_r = np.clip(gain_r, min_gain, max_gain); gain_b = np.clip(gain_b, min_gain, max_gain)
            print(f"Auto WB: Gains: R={gain_r:.3f}, G={gain_g:.3f}, B={gain_b:.3f}")
            self.rs['spinbox'].setValue(gain_r); self.gs['spinbox'].setValue(gain_g); self.bs['spinbox'].setValue(gain_b)
            self.statusBar().showMessage("Auto Balance Blancs appliquée.", 3000)
        except Exception as e: print(f"Erreur Auto WB: {e}"); traceback.print_exc(); self.statusBar().showMessage("Erreur Auto WB.", 3000)

    def update_image_display(self):
        qimg=None
        if self.display_data is not None and self.display_data.size > 0:
            try:
                data=self.display_data; req='C_CONTIGUOUS'
                if not data.flags[req]: data=np.ascontiguousarray(data)
                if data.ndim==3 and data.shape[2]==3: h,w,c=data.shape; bpl=3*w; qimg=QImage(data.data, w, h, bpl, QImage.Format_RGB888)
                elif data.ndim==2: h,w=data.shape; bpl=w; qimg=QImage(data.data, w, h, bpl, QImage.Format_Grayscale8)
                if qimg and qimg.isNull(): print("Warn: QImage null."); qimg=None
            except Exception as e: print(f"Erreur QImage: {e}"); qimg=None
        pixmap=QPixmap.fromImage(qimg) if qimg else QPixmap()
        self.view.set_pixmap(pixmap)

    def save_image(self):
        if self.display_data is None: QMessageBox.warning(self, "Erreur", "Aucune image à sauvegarder."); return
        ldir = self.settings.value("LastSaveDir", ".")
        dfn="processed_image.png"; path, sel = QFileDialog.getSaveFileName(self, "Sauvegarder", QDir(ldir).filePath(dfn), "PNG (*.png);;TIFF (*.tif *.tiff);;JPEG (*.jpg *.jpeg)")
        if not path: return
        self.settings.setValue("LastSaveDir", QFileInfo(path).absolutePath())
        QApplication.setOverrideCursor(Qt.WaitCursor); self.statusBar().showMessage(f"Sauvegarde {QFileInfo(path).fileName()}...")
        try:
            data=self.display_data;
            if data.ndim == 3: img=Image.fromarray(data, 'RGB')
            elif data.ndim == 2: img=Image.fromarray(data, 'L')
            else: raise ValueError("Format invalide.")
            ext=QFileInfo(path).suffix().lower()
            if ext in ('tif','tiff'): img.save(path, compression=None)
            elif ext in ('jpg','jpeg'): img.save(path, quality=95)
            else: img.save(path) # PNG default
            print(f"Image sauvegardée: {path}"); self.statusBar().showMessage("Sauvegardée.", 5000)
        except Exception as e: print(f"Erreur sauvegarde: {e}"); QMessageBox.critical(self, "Erreur", f"Sauvegarde impossible:\n{e}"); self.statusBar().showMessage("Erreur sauvegarde.", 5000)
        finally: QApplication.restoreOverrideCursor()

    def closeEvent(self, event):
        QSettings().setValue("viewer/splitter", self.splitter.sizes())
        self.save_settings(); print("Fermeture, paramètres sauvegardés."); super().closeEvent(event)

# --- Main Execution ---
if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True); QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app=QApplication(sys.argv); app.setStyle('Fusion')
    pal=QPalette(); pal.setColor(QPalette.Window, QColor(53,53,53)); pal.setColor(QPalette.WindowText, Qt.white); pal.setColor(QPalette.Base, QColor(35,35,35)); pal.setColor(QPalette.AlternateBase, QColor(53,53,53))
    pal.setColor(QPalette.ToolTipBase, QColor(53,53,53)); pal.setColor(QPalette.ToolTipText, Qt.white); pal.setColor(QPalette.Text, Qt.white); pal.setColor(QPalette.Button, QColor(53,53,53)); pal.setColor(QPalette.ButtonText, Qt.white)
    pal.setColor(QPalette.BrightText, Qt.red); pal.setColor(QPalette.Link, QColor(42,130,218)); pal.setColor(QPalette.Highlight, QColor(42,130,218)); pal.setColor(QPalette.HighlightedText, QColor(240,240,240))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor(127,127,127)); pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(127,127,127)); pal.setColor(QPalette.Disabled, QPalette.WindowText, QColor(127,127,127))
    app.setPalette(pal)
    window=TelescopeImageViewer(); window.show(); sys.exit(app.exec_())
