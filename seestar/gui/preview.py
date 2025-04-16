"""
Module pour la gestion de la prévisualisation des images astronomiques.
"""

import os
import numpy as np
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from seestar.tools.stretch import Stretch

class PreviewManager:
    """
    Classe pour gérer la prévisualisation des images astronomiques.
    """

    def __init__(self, canvas, stack_label, info_text):
        """
        Initialise le gestionnaire de prévisualisation.

        Args:
            canvas (tk.Canvas): Canvas Tkinter pour l'affichage des images
            stack_label (ttk.Label): Label pour afficher le nom du stack
            info_text (tk.Text): Zone de texte pour les informations sur l'image
        """
        self.canvas = canvas
        self.current_stack_label = stack_label
        self.image_info_text = info_text
        self.tk_img = None
        self.stretch_tool = Stretch()
        
        # Variables pour le zoom
        self.zoom_level = 1.0
        self.MAX_ZOOM = 5.0
        self.MIN_ZOOM = 0.2
        self.original_image_data = None
        self.current_stack_name = "Aucun stack"
        
        # Configurer les événements pour le zoom
        self.canvas.bind("<MouseWheel>", self.zoom_preview)  # Windows
        self.canvas.bind("<Button-4>", self.zoom_in_linux)   # Linux (molette vers le haut)
        self.canvas.bind("<Button-5>", self.zoom_out_linux)  # Linux (molette vers le bas)

    def create_test_image(self, width=300, height=200):
        """
        Crée une image de test colorée avec des motifs visibles.

        Args:
            width (int): Largeur de l'image
            height (int): Hauteur de l'image

        Returns:
            numpy.ndarray: Image de test
        """
        # Créer un gradient coloré
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Gradient rouge horizontal
        for x in range(width):
            img[:, x, 0] = int(255 * x / width)

        # Gradient vert vertical
        for y in range(height):
            img[y, :, 1] = int(255 * y / height)

        # Motif de grille bleue
        for y in range(0, height, 20):
            img[y:y+2, :, 2] = 255
        for x in range(0, width, 20):
            img[:, x:x+2, 2] = 255

        return img

    def update_preview(self, image_data, stack_name=None, apply_stretch=False):
        """
        Met à jour l'aperçu avec l'image fournie.

        Args:
            image_data (numpy.ndarray): Données de l'image
            stack_name (str, optional): Nom du stack
            apply_stretch (bool): Appliquer un étirement automatique à l'image
        """
        try:
            if image_data is None:
                print("Erreur: image_data est None")
                return

            # Stocker les données d'origine
            self.original_image_data = image_data
            self.current_stack_name = stack_name or "Aucun stack"

            # Convertir (3, H, W) en (H, W, 3) si nécessaire
            if image_data.ndim == 3 and image_data.shape[0] == 3:
                image_data = np.transpose(image_data, (1, 2, 0))
                print(f"Format converti de (3,H,W) à (H,W,3): {image_data.shape}")

            # Appliquer le stretch si demandé
            if apply_stretch:
                display_img = self.stretch_tool.stretch(np.copy(image_data))
                display_img = (display_img * 255).astype(np.uint8)
            else:
                display_img = cv2.normalize(image_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Assurer que l'image est en RGB
            if display_img.ndim == 2:
                display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB)

            # Obtenir les dimensions du canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width, canvas_height = 400, 300

            # Redimensionner l'image en fonction du zoom
            img_height, img_width = display_img.shape[:2]
            base_ratio = min(canvas_width / img_width, canvas_height / img_height)
            zoom_ratio = base_ratio * self.zoom_level
            new_width = int(img_width * zoom_ratio)
            new_height = int(img_height * zoom_ratio)

            # Convertir en image PIL pour l'affichage
            pil_img = Image.fromarray(display_img)
            pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
            self.tk_img = ImageTk.PhotoImage(image=pil_img)

            # Afficher l'image
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width // 2, canvas_height // 2, anchor="center", image=self.tk_img)

            # Mettre à jour les informations
            if stack_name:
                self.current_stack_label.config(text=f"Stack actuel: {stack_name}")

            # Mettre à jour les informations de l'image
            image_info = f"Dimensions: {img_width}x{img_height}\n"
            image_info += f"Type: {'Couleur' if display_img.ndim == 3 else 'Monochrome'}\n"
            image_info += f"Dimension affichée: {new_width}x{new_height}\n"
            image_info += f"Zoom: {self.zoom_level:.1f}x"

            self.image_info_text.config(state="normal")
            self.image_info_text.delete(1.0, "end")
            self.image_info_text.insert("end", image_info)
            self.image_info_text.config(state="disabled")

        except Exception as e:
            print(f"Erreur lors de la mise à jour de l'aperçu: {e}")
            import traceback
            traceback.print_exc()

    def refresh_preview(self, apply_stretch=None):
        """
        Actualise l'aperçu avec les paramètres actuels (p.ex. après un changement de zoom).
        
        Args:
            apply_stretch (bool, optional): Appliquer l'étirement (utilise la valeur précédente si None)
        """
        if self.original_image_data is not None:
            self.update_preview(self.original_image_data, self.current_stack_name, 
                               apply_stretch if apply_stretch is not None else True)

    def zoom_preview(self, event):
        """Gère le zoom avec la molette de la souris (Windows)."""
        if self.original_image_data is not None:
            if event.delta > 0:
                self.zoom_level = min(self.MAX_ZOOM, self.zoom_level * 1.1)
            else:
                self.zoom_level = max(self.MIN_ZOOM, self.zoom_level / 1.1)
            self.refresh_preview()

    def zoom_in_linux(self, event):
        """Gère le zoom avant (Linux)."""
        if self.original_image_data is not None:
            self.zoom_level = min(self.MAX_ZOOM, self.zoom_level * 1.1)
            self.refresh_preview()

    def zoom_out_linux(self, event):
        """Gère le zoom arrière (Linux)."""
        if self.original_image_data is not None:
            self.zoom_level = max(self.MIN_ZOOM, self.zoom_level / 1.1)
            self.refresh_preview()