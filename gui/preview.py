"""
Module pour la gestion de la prévisualisation des images astronomiques.
"""

import os
import numpy as np
import cv2
from PIL import Image, ImageTk
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
        self.apply_stretch = True

    def create_test_image(self, width=300, height=200):
        """
        Crée une image de test colorée avec des motifs bien visibles.

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

        print(f"update_preview appelé, image shape: {image_data.shape}")

        # Convertir (3, H, W) en (H, W, 3)
        if image_data.ndim == 3 and image_data.shape[0] == 3:
            image_data = np.transpose(image_data, (1, 2, 0))
            print(f"Format converti de (3,H,W) à (H,W,3): {image_data.shape}")

        # Appliquer le stretch si demandé
        if apply_stretch:
            image_data = stretch_image(image_data)
            img = (image_data * 255).astype(np.uint8)
        else:
            img = cv2.normalize(image_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Gérer les dimensions du canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = 400, 300

        img_height, img_width = img.shape[:2]
        ratio = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)

        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(resized)
        self.tk_img = ImageTk.PhotoImage(image=pil_img)

        self.canvas.delete("all")
        self.canvas.create_image(canvas_width // 2, canvas_height // 2, anchor="center", image=self.tk_img)

        if stack_name:
            self.current_stack_label.config(text=f"Stack actuel: {stack_name}")

        image_info = f"Dimensions: {img_width}x{img_height}\n"
        image_info += f"Type: {'Couleur' if img.ndim == 3 else 'Monochrome'}\n"
        image_info += f"Dimension affichée: {new_width}x{new_height}"

        self.image_info_text.config(state="normal")
        self.image_info_text.delete(1.0, "end")
        self.image_info_text.insert("end", image_info)
        self.image_info_text.config(state="disabled")

        print(f"Image mise à jour avec succès: {new_width}x{new_height}")

    except Exception as e:
        print(f"Erreur lors de la mise à jour de l'aperçu: {e}")
        import traceback
        traceback.print_exc()
        
    def _apply_stretch_mono(self, img_data):
        """
        Applique un étirement avancé à une image monochrome.

        Args:
            img_data (numpy.ndarray): Données de l'image

        Returns:
            numpy.ndarray: Image étirée
        """
        try:
            # S'assurer que l'image est en flottant pour les calculs
            if img_data.dtype != np.float32 and img_data.dtype != np.float64:
                img_data = img_data.astype(np.float32)

            # Normaliser entre 0 et 1
            min_val = np.min(img_data)
            max_val = np.max(img_data)
            if max_val > min_val:  # Éviter la division par zéro
                norm_img = (img_data - min_val) / (max_val - min_val)
            else:
                norm_img = np.zeros_like(img_data)

            # Convertir en uint8 pour CLAHE
            img_norm = (norm_img * 255).astype(np.uint8)

            # Appliquer CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            img_stretched = clahe.apply(img_norm)

            # Reconvertir en flottant normalisé
            return img_stretched.astype(np.float32) / 255.0
        except Exception as e:
            print(f"Erreur dans _apply_stretch_mono: {e}")
            return img_data  # Retourner l'image d'origine en cas d'erreur

    def _apply_stretch_color(self, img_data):
        """
        Applique un étirement avancé à une image couleur.

        Args:
            img_data (numpy.ndarray): Données de l'image

        Returns:
            numpy.ndarray: Image étirée
        """
        try:
            # S'assurer que l'image est en format HxWx3
            if img_data.shape[0] == 3 and len(img_data.shape) == 3:
                img_data = np.transpose(img_data, (1, 2, 0))

            # S'assurer que l'image est en flottant
            if img_data.dtype != np.float32 and img_data.dtype != np.float64:
                img_data = img_data.astype(np.float32)

            # Normaliser entre 0 et 1
            min_val = np.min(img_data)
            max_val = np.max(img_data)
            if max_val > min_val:
                norm_img = (img_data - min_val) / (max_val - min_val)
            else:
                norm_img = np.zeros_like(img_data)

            # Convertir en uint8 pour traitement
            img_norm = (norm_img * 255).astype(np.uint8)

            # Convertir en espace de couleur LAB
            lab = cv2.cvtColor(img_norm, cv2.COLOR_RGB2LAB)

            # Appliquer CLAHE uniquement sur le canal L
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_stretched = clahe.apply(l)

            # Fusionner les canaux
            lab_stretched = cv2.merge((l_stretched, a, b))
            img_stretched = cv2.cvtColor(lab_stretched, cv2.COLOR_LAB2RGB)

            # Reconvertir en flottant normalisé
            return img_stretched.astype(np.float32) / 255.0
        except Exception as e:
            print(f"Erreur dans _apply_stretch_color: {e}")
            return img_data  # Retourner l'image d'origine en cas d'erreur
