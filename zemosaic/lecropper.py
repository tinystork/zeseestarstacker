# zemosaic_autocrop_gui.py
# ZeMosaic Auto-Crop (signal-based) – version corrigée
# Dépendances: numpy, matplotlib, astropy, tkinter
# (pas de SciPy / skimage requis)
"""
╔══════════════════════════════════════════════════════════════════════╗
║ ZeMosaic / ZeSeestarStacker Project                                  ║
║                                                                      ║
║ Auteur  : Tinystork, seigneur des couteaux à beurre (aka Tristan Nauleau)  
║ Partenaire : J.A.R.V.I.S. (/ˈdʒɑːrvɪs/) — Just a Rather Very Intelligent System  
║              (aka ChatGPT, Grand Maître du ciselage de code)         ║
║                                                                      ║
║ Licence : GNU General Public License v3.0 (GPL-3.0)                  ║
║                                                                      ║
║ Description :                                                        ║
║   Ce programme a été forgé à la lueur des pixels et de la caféine,   ║
║   dans le but noble de transformer des nuages de photons en art      ║
║   astronomique. Si vous l’utilisez, pensez à dire “merci”,           ║
║   à lever les yeux vers le ciel, ou à citer Tinystork et J.A.R.V.I.S.║
║   (le karma des développeurs en dépend).                             ║
║                                                                      ║
║ Avertissement :                                                      ║
║   Aucune IA ni aucun couteau à beurre n’a été blessé durant le       ║
║   développement de ce code.                                          ║
╚══════════════════════════════════════════════════════════════════════╝


╔══════════════════════════════════════════════════════════════════════╗
║ ZeMosaic / ZeSeestarStacker Project                                  ║
║                                                                      ║
║ Author  : Tinystork, Lord of the Butter Knives (aka Tristan Nauleau) ║
║ Partner : J.A.R.V.I.S. (/ˈdʒɑːrvɪs/) — Just a Rather Very Intelligent System  
║           (aka ChatGPT, Grand Master of Code Chiseling)              ║
║                                                                      ║
║ License : GNU General Public License v3.0 (GPL-3.0)                  ║
║                                                                      ║
║ Description:                                                         ║
║   This program was forged under the sacred light of pixels and       ║
║   caffeine, with the noble intent of turning clouds of photons into  ║
║   astronomical art. If you use it, please consider saying “thanks,”  ║
║   gazing at the stars, or crediting Tinystork and J.A.R.V.I.S. —     ║
║   developer karma depends on it.                                     ║
║                                                                      ║
║ Disclaimer:                                                          ║
║   No AIs or butter knives were harmed in the making of this code.    ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import glob
import csv
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox

import logging

import numpy as np
from astropy.io import fits

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ----------------------------- utilitaires numpy -----------------------------

def _safe_minmax(a):
    a = np.asarray(a)
    finite = np.isfinite(a)
    if not finite.any():
        return 0.0, 0.0
    return float(a[finite].min()), float(a[finite].max())

def _normalize01(a):
    a = np.asarray(a, dtype=np.float32)
    a = np.clip(a, 0, None)
    mn, mx = _safe_minmax(a)
    if mx <= 0 or mx - mn <= 0:
        return np.zeros_like(a, dtype=np.float32)
    a = (a - mn) / (mx - mn)
    return a.astype(np.float32)

def pool2(a, mode="mean", band=32):
    """Réduction en grille par blocs band x band (mode: mean ou var)."""
    H, W = a.shape
    by = max(1, band)
    bx = max(1, band)
    sy = H // by
    sx = W // bx
    if sy <= 0 or sx <= 0:
        return _normalize01(a)  # fallback
    a = a[:sy*by, :sx*bx]
    if mode == "mean":
        out = a.reshape(sy, by, sx, bx).mean(axis=(1,3))
    elif mode == "var":
        out = a.reshape(sy, by, sx, bx).var(axis=(1,3))
    else:
        out = a.reshape(sy, by, sx, bx).mean(axis=(1,3))
    return out.astype(np.float32)

def binary_erosion1(m):
    """Érosion binaire 8-connexe (1 itération) sans SciPy."""
    m = m.astype(bool)
    n = (
        m &
        np.roll(m,  1, 0) & np.roll(m, -1, 0) &
        np.roll(m,  1, 1) & np.roll(m, -1, 1) &
        np.roll(np.roll(m,  1, 0),  1, 1) &
        np.roll(np.roll(m,  1, 0), -1, 1) &
        np.roll(np.roll(m, -1, 0),  1, 1) &
        np.roll(np.roll(m, -1, 0), -1, 1)
    )
    return n


def _binary_erosion(mask, iterations=1):
    """Binary erosion with zero padding to avoid wrap-around effects."""
    mask = mask.astype(bool)
    iters = max(0, int(iterations))
    if iters == 0:
        return mask
    for _ in range(iters):
        padded = np.pad(mask, 1, mode="constant", constant_values=False)
        mask = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1] & padded[2:, 1:-1]
            & padded[1:-1, :-2] & padded[1:-1, 2:]
            & padded[:-2, :-2] & padded[:-2, 2:]
            & padded[2:, :-2] & padded[2:, 2:]
        )
    return mask

def bbox_from_mask(m):
    """BBox (y0,x0,y1,x1) du masque True. Si vide -> None."""
    ys, xs = np.where(m)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return (y0, x0, y1, x1)


# -------------------------- coeur: détection auto-crop -----------------------

logger = logging.getLogger(__name__)


def detect_autocrop_rgb(lum2d, R, G, B, band_px=32, k_sigma=2.0, margin_px=8):
    """
    Renvoie (y0, x0, y1, x1) en pleine résolution.
    Corrigé: normalisation, non-noir, planchers, double érosion, trim full-res.
    """

    H, W = lum2d.shape

    # --- 1) Normalisation préalable (0..1) ---
    lum2d = _normalize01(lum2d)
    R = _normalize01(R)
    G = _normalize01(G)
    B = _normalize01(B)

    # --- 2) Pooling (grille réduite) ---
    Ls = pool2(lum2d, "mean", band=band_px)
    Vs = pool2(lum2d, "var",  band=band_px)
    Rs = pool2(R, "mean", band=band_px)
    Gs = pool2(G, "mean", band=band_px)
    Bs = pool2(B, "mean", band=band_px)

    Hs, Ws = Ls.shape
    if Hs < 3 or Ws < 3:
        # grille trop petite -> garde tout (avec marge)
        return max(0, 0 + margin_px), max(0, 0 + margin_px), min(H, H - margin_px), min(W, W - margin_px)

    # --- 3) Seuils robustes + non-noir ---
    medL = float(np.nanmedian(Ls))
    sigL = float(np.nanstd(Ls))
    eps_small = max(1e-6, float(np.nanpercentile(Ls, 0.5)))

    thrL = max(medL - k_sigma * max(sigL, 1e-6), float(np.nanpercentile(Ls, 5.0)), eps_small)
    thrV = max(float(np.nanpercentile(Vs, 15.0)), float(np.nanpercentile(Vs, 1.0)) or 1e-6)

    nonblack_small = (Rs > eps_small) | (Gs > eps_small) | (Bs > eps_small)
    base = np.isfinite(Ls) & nonblack_small & (Ls >= thrL) & (Vs >= thrV)

    # --- 4) Rejet chroma générique en bordure ---
    Cmax = np.maximum(np.maximum(Rs, Gs), Bs)
    Cmin = np.minimum(np.minimum(Rs, Gs), Bs)
    sat = Cmax - Cmin
    edge_band_px = max(24, margin_px)
    edge_band_small = max(1, int(np.ceil(edge_band_px / float(max(1, band_px)))))
    edge = np.zeros_like(base, dtype=bool)
    edge[:edge_band_small, :] = True
    edge[-edge_band_small:, :] = True
    edge[:, :edge_band_small] = True
    edge[:, -edge_band_small:] = True
    t_chroma = 0.18
    mask_chroma = edge & (sat > t_chroma)
    mask_chroma = _binary_erosion(mask_chroma, iterations=2)
    logger.debug("QBC edge-chroma: t=%.2f, erode=%d", t_chroma, 2)
    if mask_chroma.any():
        base[mask_chroma] = False

    # --- 5) Érosion (deux passes) pour casser les ponts fins ---
    m_small = binary_erosion1(base)
    m_small = binary_erosion1(m_small)

    # --- 6) BBox sur grille réduite ---
    box_small = bbox_from_mask(m_small)
    if box_small is None:
        # rien de fiable détecté -> garde tout
        return 0, 0, H, W

    y0s, x0s, y1s, x1s = box_small

    # --- 7) Reprojection plein format + marges ---
    fy = H / float(Hs)
    fx = W / float(Ws)
    y0 = max(0, int(y0s * fy) + margin_px)
    x0 = max(0, int(x0s * fx) + margin_px)
    y1 = min(H, int(y1s * fy) - margin_px)
    x1 = min(W, int(x1s * fx) - margin_px)

    if y1 <= y0 + 8 or x1 <= x0 + 8:
        # rectangle dégénéré -> garde tout
        return 0, 0, H, W

    # --- 8) Trim final en pleine résolution (retire 3 px si trop sombre) ---
    def _trim_dark_edges_fullres(lum, rect, step=3, minsize=32):
        y0, x0, y1, x1 = rect
        eps_fr = max(1e-6, float(np.nanpercentile(lum, 0.5)))
        # bas
        while (y1 - y0) > minsize and np.nanmean(lum[y1-step:y1, x0:x1]) < eps_fr:
            y1 -= step
        # droite
        while (x1 - x0) > minsize and np.nanmean(lum[y0:y1, x1-step:x1]) < eps_fr:
            x1 -= step
        return y0, x0, y1, x1

    y0, x0, y1, x1 = _trim_dark_edges_fullres(lum2d, (y0, x0, y1, x1))
    before_color_trim = (int(y0), int(x0), int(y1), int(x1))
    after_color_trim = _trim_color_edges_fullres(R, G, B, before_color_trim, step=3, minsize=32, k=k_sigma)
    logger.debug("QBC color-trim: before=%s after=%s", before_color_trim, after_color_trim)

    y0, x0, y1, x1 = after_color_trim
    touches = (
        y0 <= margin_px or x0 <= margin_px or y1 >= (H - margin_px) or x1 >= (W - margin_px)
    )
    if touches and band_px > 32:
        band_px2 = max(16, band_px // 2)
        logger.debug("QBC second-pass triggered: band %d -> %d", band_px, band_px2)
        refined = detect_autocrop_rgb(lum2d, R, G, B, band_px=band_px2, k_sigma=k_sigma, margin_px=margin_px)
        y0 = max(y0, refined[0])
        x0 = max(x0, refined[1])
        y1 = min(y1, refined[2])
        x1 = min(x1, refined[3])
        if y1 <= y0 or x1 <= x0:
            return refined

    return int(y0), int(x0), int(y1), int(x1)


def _trim_color_edges_fullres(R, G, B, rect, step=3, minsize=32, k=3.0):
    y0, x0, y1, x1 = rect
    if (y1 - y0) <= minsize * 2 or (x1 - x0) <= minsize * 2:
        return rect

    inner = (slice(y0 + minsize, y1 - minsize), slice(x0 + minsize, x1 - minsize))
    if (inner[0].start >= inner[0].stop) or (inner[1].start >= inner[1].stop):
        return rect

    r_inner = R[inner]
    g_inner = G[inner]
    b_inner = B[inner]

    r_med = np.nanmedian(r_inner)
    g_med = np.nanmedian(g_inner)
    b_med = np.nanmedian(b_inner)

    r_mad = np.nanmedian(np.abs(r_inner - r_med)) + 1e-6
    g_mad = np.nanmedian(np.abs(g_inner - g_med)) + 1e-6
    b_mad = np.nanmedian(np.abs(b_inner - b_med)) + 1e-6

    def over(chan, med, mad, sl):
        return np.nanmean(chan[sl]) > med + k * mad

    while (x1 - x0) > minsize and (
        over(R, r_med, r_mad, (slice(y0, y1), slice(x0, min(x0 + step, x1))))
        or over(G, g_med, g_mad, (slice(y0, y1), slice(x0, min(x0 + step, x1))))
        or over(B, b_med, b_mad, (slice(y0, y1), slice(x0, min(x0 + step, x1))))
    ):
        x0 += step

    while (x1 - x0) > minsize and (
        over(R, r_med, r_mad, (slice(y0, y1), slice(max(x1 - step, x0), x1)))
        or over(G, g_med, g_mad, (slice(y0, y1), slice(max(x1 - step, x0), x1)))
        or over(B, b_med, b_mad, (slice(y0, y1), slice(max(x1 - step, x0), x1)))
    ):
        x1 -= step

    while (y1 - y0) > minsize and (
        over(R, r_med, r_mad, (slice(y0, min(y0 + step, y1)), slice(x0, x1)))
        or over(G, g_med, g_mad, (slice(y0, min(y0 + step, y1)), slice(x0, x1)))
        or over(B, b_med, b_mad, (slice(y0, min(y0 + step, y1)), slice(x0, x1)))
    ):
        y0 += step

    while (y1 - y0) > minsize and (
        over(R, r_med, r_mad, (slice(max(y1 - step, y0), y1), slice(x0, x1)))
        or over(G, g_med, g_mad, (slice(max(y1 - step, y0), y1), slice(x0, x1)))
        or over(B, b_med, b_mad, (slice(max(y1 - step, y0), y1), slice(x0, x1)))
    ):
        y1 -= step

    return int(y0), int(x0), int(y1), int(x1)


# ------------------------------ I/O FITS & preview ---------------------------

def load_fits_rgb(path):
    """Retourne (lum2d, R, G, B) en float32 normalisés 0..1 (NaN -> 0)."""
    with fits.open(path, memmap=True) as hdul:
        data = hdul[0].data

    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        mono = _normalize01(data)
        R = G = B = mono
    elif data.ndim == 3:
        # on accepte (H,W,3) ou (3,H,W)
        if data.shape[-1] == 3:
            R, G, B = data[..., 0], data[..., 1], data[..., 2]
        elif data.shape[0] == 3:
            R, G, B = data[0], data[1], data[2]
            R = np.ascontiguousarray(R.transpose(1, 2, 0))[:, :, 0] if R.ndim == 3 else R
        else:
            # fallback: moyenne
            mono = _normalize01(data.mean(axis=0))
            R = G = B = mono
        R = _normalize01(R)
        G = _normalize01(G)
        B = _normalize01(B)
    else:
        mono = _normalize01(data.reshape(data.shape[-2], data.shape[-1]))
        R = G = B = mono

    lum = np.nanmean(np.stack([R, G, B], axis=-1), axis=-1).astype(np.float32)
    lum[np.isnan(lum)] = 0.0
    R[np.isnan(R)] = 0.0
    G[np.isnan(G)] = 0.0
    B[np.isnan(B)] = 0.0
    return lum, R, G, B


def save_cropped_fits(in_path, rect, out_suffix="_cropped"):
    y0, x0, y1, x1 = rect
    # Use memmap=False so the FITS file handle is fully released before
    # potentially overwriting the source file (Windows needs the file closed).
    with fits.open(in_path, mode="readonly", memmap=False) as hdul:
        data = np.asarray(hdul[0].data)
        header = hdul[0].header.copy()

    if data.ndim == 2:
        cropped = data[y0:y1, x0:x1]
    elif data.ndim == 3:
        if data.shape[-1] == 3:
            cropped = data[y0:y1, x0:x1, :]
        elif data.shape[0] == 3:
            cropped = data[:, y0:y1, x0:x1]
        else:
            cropped = data[:, y0:y1, x0:x1]
    else:
        cropped = data

    # Mise à jour rapide du CRPIX si présent (optionnel)
    if "CRPIX1" in header and "CRPIX2" in header:
        header["CRPIX1"] = header.get("CRPIX1", 0.0) - x0
        header["CRPIX2"] = header.get("CRPIX2", 0.0) - y0

    base, ext = os.path.splitext(in_path)
    out_path = f"{base}{out_suffix}{ext}"
    fits.writeto(out_path, cropped, header, overwrite=True)
    return out_path


# ---------------------------------- GUI --------------------------------------

class AutoCropApp:
    def __init__(self, root):
        self.root = root
        root.title("ZeMosaic Auto-Crop (signal-based)")
        root.geometry("1200x720")

        top = tk.Frame(root)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        tk.Label(top, text="Input directory:").pack(side=tk.LEFT)
        self.dir_var = tk.StringVar(value=os.getcwd())
        self.dir_entry = tk.Entry(top, textvariable=self.dir_var, width=65)
        self.dir_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(top, text="Browse...", command=self.browse_dir).pack(side=tk.LEFT, padx=4)

        tk.Label(top, text="Band px").pack(side=tk.LEFT, padx=(10, 2))
        self.band_var = tk.StringVar(value="32")
        tk.Entry(top, textvariable=self.band_var, width=5).pack(side=tk.LEFT)

        tk.Label(top, text="k:sigma").pack(side=tk.LEFT, padx=(10, 2))
        self.ks_var = tk.StringVar(value="2.0")
        tk.Entry(top, textvariable=self.ks_var, width=5).pack(side=tk.LEFT)

        tk.Label(top, text="min run").pack(side=tk.LEFT, padx=(10, 2))
        self.minrun_var = tk.StringVar(value="2")
        tk.Entry(top, textvariable=self.minrun_var, width=5).pack(side=tk.LEFT)

        tk.Label(top, text="margin px").pack(side=tk.LEFT, padx=(10, 2))
        self.margin_var = tk.StringVar(value="8")
        tk.Entry(top, textvariable=self.margin_var, width=5).pack(side=tk.LEFT)

        tk.Button(top, text="Analyze", command=self.analyze).pack(side=tk.LEFT, padx=8)
        tk.Button(top, text="Export CSV", command=self.export_csv).pack(side=tk.LEFT, padx=4)

        self.replace_var = tk.BooleanVar(value=False)
        tk.Checkbutton(top, text="Replace files", variable=self.replace_var).pack(side=tk.LEFT, padx=8)

        tk.Button(top, text="Apply Crop", command=self.apply_crop).pack(side=tk.LEFT, padx=4)

        mid = tk.Frame(root)
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Liste des fichiers
        left = tk.Frame(mid, width=300)
        left.pack(side=tk.LEFT, fill=tk.Y)
        self.listbox = tk.Listbox(left, width=40)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.on_select)
        self.scroll = tk.Scrollbar(left, orient=tk.VERTICAL, command=self.listbox.yview)
        self.scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(yscrollcommand=self.scroll.set)

        # Figure Matplotlib pour l'aperçu
        right = tk.Frame(mid)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.results = {}   # path -> (y0,x0,y1,x1) ou None
        self.files = []

        self.status = tk.StringVar(value="Ready.")
        tk.Label(root, textvariable=self.status, anchor="w").pack(side=tk.BOTTOM, fill=tk.X)

        self.refresh_files()

    # ------------------ actions GUI ------------------

    def browse_dir(self):
        d = filedialog.askdirectory(initialdir=self.dir_var.get() or os.getcwd())
        if d:
            self.dir_var.set(d)
            self.refresh_files()

    def refresh_files(self):
        d = self.dir_var.get()
        pats = ["*.fits", "*.fit", "*.fts"]
        files = []
        for p in pats:
            files += sorted(glob.glob(os.path.join(d, p)))
        self.files = files
        self.listbox.delete(0, tk.END)
        for f in files:
            self.listbox.insert(tk.END, os.path.basename(f))
        self.status.set(f"Found {len(files)} FITS files.")

    def analyze(self):
        if not self.files:
            self.status.set("No FITS files.")
            return

        try:
            band = int(float(self.band_var.get()))
        except:
            band = 32
        try:
            ks = float(self.ks_var.get())
        except:
            ks = 2.0
        try:
            margin = int(float(self.margin_var.get()))
        except:
            margin = 8

        analyzed = 0
        self.results.clear()
        self.listbox.delete(0, tk.END)
        for path in self.files:
            try:
                lum, R, G, B = load_fits_rgb(path)
                rect = detect_autocrop_rgb(lum, R, G, B, band_px=band, k_sigma=ks, margin_px=margin)
                self.results[path] = rect
                label = f"OK  {os.path.basename(path)}"
                self.listbox.insert(tk.END, label)
            except Exception as e:
                print(f"[ERROR] {path}: {e}\n{traceback.format_exc()}", file=sys.stderr)
                self.results[path] = None
                label = f"ERR {os.path.basename(path)}"
                self.listbox.insert(tk.END, label)
            analyzed += 1

        self.status.set(f"Analyzed {analyzed} files.")
        if analyzed:
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(0)
            self.on_select(None)

    def on_select(self, _evt):
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx >= len(self.files):
            return
        path = self.files[idx]
        rect = self.results.get(path)
        self.show_preview(path, rect)

    def show_preview(self, path, rect):
        self.ax.clear()
        self.ax.set_title(os.path.basename(path))
        try:
            lum, R, G, B = load_fits_rgb(path)
            # on affiche la luminance (low-stretch pour voir les bords sombres)
            disp = np.power(np.clip(lum, 0, 1), 0.5)
            self.ax.imshow(disp, origin="upper", cmap="gray")
            if rect is not None:
                y0, x0, y1, x1 = rect
                self.ax.plot([x0, x1, x1, x0, x0],
                             [y0, y0, y1, y1, y0],
                             linewidth=1.5)
        except Exception as e:
            self.ax.text(0.5, 0.5, f"Preview error:\n{e}", ha="center", va="center", transform=self.ax.transAxes)
        self.ax.set_axis_off()
        self.canvas.draw_idle()

    def export_csv(self):
        if not self.results:
            messagebox.showinfo("Export CSV", "No results to export.")
            return
        out = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="autocrop_rects.csv")
        if not out:
            return
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "y0", "x0", "y1", "x1"])
            for p in self.files:
                r = self.results.get(p)
                if r is None:
                    w.writerow([p, "", "", "", ""])
                else:
                    y0, x0, y1, x1 = r
                    w.writerow([p, y0, x0, y1, x1])
        self.status.set(f"CSV exported to {out}")

    def apply_crop(self):
        if not self.results:
            messagebox.showinfo("Apply Crop", "No results to apply.")
            return
        ok, err = 0, 0
        for p in self.files:
            rect = self.results.get(p)
            if not rect:
                continue
            try:
                suffix = "" if self.replace_var.get() else "_cropped"
                outp = save_cropped_fits(p, rect, out_suffix=suffix)
                ok += 1
            except Exception as e:
                print(f"[CROP ERROR] {p}: {e}\n{traceback.format_exc()}", file=sys.stderr)
                err += 1
        mode = "replaced" if self.replace_var.get() else "saved"
        self.status.set(f"Cropped: {ok} files ({mode}), errors: {err}")
        messagebox.showinfo(
            "Apply Crop",
            f"Cropped: {ok} files ({mode})\nErrors: {err}"
        )


# --------------------------------- main --------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = AutoCropApp(root)
    root.mainloop()
