import os
import sys
import pathlib
import argparse
import tqdm
import numpy as np
import pyvips
import matplotlib.pyplot as plt
import cv2
import copy
import json

from shapely.geometry import LineString, Point
from debug import *
# from shredder.debug import display_fragments

# ---------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------

def interpolate_contour(contour, num_points=2000):
    """
    Utility function that linearly interpolates the (x,y) points
    along a contour (contorno) to increase the point density.
    """
    line = LineString(contour)
    length = line.length
    distances = np.linspace(0, length, num_points)

    coords = []
    for dist in distances:
        point = line.interpolate(dist)
        coords.append((int(point.x), int(point.y)))
    return coords

def apply_im_tform_to_coords(coords, pyvips_image, downscale_factor, rot_k):
    """
    NOT UTILIZED IN THIS SCRIPT. 
    In this example, it only offsets coords if needed.
    The `rot_k` determines the 90-degree rotation increments.
    We assume a top-left origin and standard image coordinate system.
    """
    w = pyvips_image.width
    h = pyvips_image.height

    coords_out = []
    if rot_k == 0:
        # No rotation
        coords_out = coords
    elif rot_k == 1:
        # 90 deg: (x, y) -> (y, W - x - 1)
        coords_out = [(c[1], w - c[0] - 1) for c in coords]
    elif rot_k == 2:
        # 180 deg: (x, y) -> (W - x - 1, H - y - 1)
        coords_out = [(w - c[0] - 1, h - c[1] - 1) for c in coords]
    elif rot_k == 3:
        # 270 deg: (x, y) -> (H - y - 1, c[0])
        coords_out = [(h - c[1] - 1, c[0]) for c in coords]

    return np.array(coords_out, dtype=int)

# ---------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------

def collect_arguments():
    """
    Parse arguments, omitting any reference to a mask directory.
    """
    parser = argparse.ArgumentParser(
        description="Shred slides into fragments by random lines (no external mask)."
    )
    parser.add_argument(
        "--datadir", required=True, type=pathlib.Path, help="Path with the TIFFs to shred"
    )
    parser.add_argument(
        "--savedir", required=True, type=pathlib.Path, help="Path to save the shreds"
    )
    parser.add_argument(
        "--rotation", required=False, type=int, default=0,
        help="Random rotation range (−rotation to +rotation) of the slide before shredding"
    )
    parser.add_argument(
        "--fragments", required=False, type=int, default=4,
        help="Number of fragments to shred to (2 or 4)"
    )
    args = parser.parse_args()

    data_dir = pathlib.Path(args.datadir)
    save_dir = pathlib.Path(args.savedir)
    rotation = args.rotation
    n_fragments = args.fragments

    assert data_dir.exists() and data_dir.is_dir(), "provided data location doesn't exist"
    assert rotation in np.arange(0, 26), "rotation must be in range [0, 25]"
    assert n_fragments in [2, 4], "number of fragments must be either 2 or 4"

    if not save_dir.is_dir():
        save_dir.mkdir(parents=True)

    print(
        f"\nRunning job with parameters:"
        f"\n - Data dir: {data_dir}"
        f"\n - Save dir: {save_dir}"
        f"\n - Rotation: {rotation}"
        f"\n - Number of fragments: {n_fragments}\n"
    )

    return data_dir, save_dir, rotation, n_fragments

# ---------------------------------------------------------------
# Shredder class (no external mask usage, with DEBUG prints)
# ---------------------------------------------------------------

class Shredder:
    def __init__(self, case, save_dir=None, rotation=0, n_fragments=4):
        self.case = case
        if save_dir is not None:
            self.savedir = save_dir.joinpath(self.case.stem)
        self.rotation = rotation
        self.n_fragments = n_fragments

        # Downscale factor for quick computations
        self.lowres_downscale = 32
        # Extra padding around the slide before rotation
        self.pad_factor = 0.3
        # Number of contour samples to define the 'zig-zag' lines
        self.n_samples = 10
        # Maximum random displacement of the lines
        self.noise = 2
        # Base step between points in the line
        self.step = 50

        self.parameters = {
            "rotation": self.rotation,
            "n_fragments": self.n_fragments
        }

        # Make sure output directories exist
        # if savedir is not None and not self.savedir.is_dir():
        #    self.savedir.mkdir(parents=True)

        self.process_case()
            

    def load_images(self):
        """
        Load the full-resolution image with pyvips and a low-resolution version for line computations.
        """
        print(f"[DEBUG] Loading slide: {self.case}")
        self.full_image = pyvips.Image.new_from_file(str(self.case))

        print(f"[DEBUG] Full-resolution image size: {self.full_image.width} x {self.full_image.height} bands={self.full_image.bands}")

        # Create a low-resolution image
        lowres = self.full_image.resize(1 / self.lowres_downscale)

        self.lowres_image = np.ndarray(
            buffer=lowres.write_to_memory(),
            dtype=np.uint8,
            shape=[lowres.height, lowres.width, lowres.bands]
        )
        print(f"[DEBUG] Low-res image shape: {self.lowres_image.shape}")

    def process(self):
        """
        Determine a random rotation (within ± self.rotation) and store it.
        """
        angle_noise = np.random.randint(-self.rotation, self.rotation + 1)
        self.angle = angle_noise

        self.height, self.width = self.lowres_image.shape[:2]

        print(f"[DEBUG] Chosen rotation angle for this slide = {self.angle} degrees")
        print(f"[DEBUG] Low-res dimensions: width={self.width}, height={self.height}")

    def get_shred_parameters(self, debug=False):
        """
        Create random 'zig-zag' lines to define fragment boundaries.
        """
        self.offset = 5

        # 1) Vertical line from top to bottom (roughly at half the width)
        # v_start = [int(0.5 * self.width), -self.offset]
        # v_end   = [int(0.5 * self.width), self.height + self.offset - 1]
        v_start = [int(0.5 * self.width), 0]
        v_end   = [int(0.5 * self.width), self.height - 1]

        print(f"[DEBUG] Vertical line start={v_start} end={v_end}")

        self.v_line_y = np.arange(v_start[1], v_end[1] + self.step, step=self.step)
        self.v_line_x = [v_start[0]]
        while len(self.v_line_x) < len(self.v_line_y):
            # Random horizontal wiggle
            self.v_line_x.append(self.v_line_x[-1] + np.random.randint(-self.noise, self.noise))

        # 2) (Optional) Horizontal line from left to right (roughly at half the height)
        # h_start = [-self.offset, int(0.5 * self.height)]
        # h_end   = [self.width + self.offset - 1, int(0.5 * self.height)]
        h_start = [0, int(0.5 * self.height)]
        h_end   = [self.width - 1, int(0.5 * self.height)]

        print(f"[DEBUG] Horizontal line start={h_start} end={h_end}")

        self.h_line_x = np.arange(h_start[0], h_end[0] + self.step, step=self.step)
        self.h_line_y = [h_start[1]]
        while len(self.h_line_y) < len(self.h_line_x):
            # Random vertical wiggle
            self.h_line_y.append(self.h_line_y[-1] + np.random.randint(-self.noise, self.noise))

        # Convert to shapely to find intersection
        v_line = LineString([(x, y) for x, y in zip(self.v_line_x, self.v_line_y)])
        h_line = LineString([(x, y) for x, y in zip(self.h_line_x, self.h_line_y)])
        inter = v_line.intersection(h_line)

        # Intersection might be empty or a point
        if inter.is_empty:
            self.intersection = [int(self.width * 0.5), int(self.height * 0.5)]
            print("[DEBUG] Intersection is empty, forcing midpoint intersection.")
        else:
            self.intersection = [int(inter.x), int(inter.y)]
            print(f"[DEBUG] Intersection found at {self.intersection}")

        # Interpolate lines and keep points inside the image
        v_raw = np.column_stack([self.v_line_x, self.v_line_y])
        h_raw = np.column_stack([self.h_line_x, self.h_line_y])

        self.h_line_temp = interpolate_contour(h_raw)
        self.h_line_temp = [
            pt for pt in self.h_line_temp
            if 0 <= pt[0] < self.width and 0 <= pt[1] < self.height
        ]
        self.h_line_interp = np.array(self.h_line_temp, dtype=int)

        self.v_line_temp = interpolate_contour(v_raw)
        self.v_line_temp = [
            pt for pt in self.v_line_temp
            if 0 <= pt[0] < self.width and 0 <= pt[1] < self.height
        ]
        self.v_line_interp = np.array(self.v_line_temp, dtype=int)

        print(f"[DEBUG] Interpolated line lengths => horizontal={len(self.h_line_interp)} vertical={len(self.v_line_interp)}")

        self.parameters["step_size"] = self.step
        self.parameters["edge_curvature"] = self.noise

        if debug: 
            debug_plot_lines(self, self.savedir.joinpath("zigzag_lines.png"))

    def apply_shred(self, debug=False):
        """
        Costruisce la maschera di 'shredding' (senza maschere esterne) e 
        definisce le regioni corrispondenti a ciascun frammento.
        Visualizza la maschera a ogni passaggio se debug=False.

        Passaggi:
        1) Crea una canvas bianca e traccia linee nere (0) per separare i frammenti.
        2) Esegue un'erosione leggera per evitare punti di contatto indesiderati.
        3) Converte i pixel in {0, 255} per ottenere la maschera binaria.
        4) Definisce seed points (offset) per n_fragments.
        5) Esegue connectedComponents per etichettare le regioni.
        6) Ricava la maschera di ciascun frammento.
        7) Mostra l'etichetta finale con un grafico a colori.
        """

        # 1) Crea una canvas bianca di dimensioni (height, width)
        self.shredded_mask = np.ones((self.height, self.width), dtype=np.uint8) * 255
        
        if debug:
            plt.figure()
            plt.title("Passo 1: Canvas iniziale (bianca)")
            plt.imshow(self.shredded_mask, cmap="gray")
            plt.show()

        # 2) Disegna le linee in nero (0) per separare le regioni.
        if self.n_fragments == 4:
            # Linea orizzontale
            self.shredded_mask[self.h_line_interp[:, 1], self.h_line_interp[:, 0]] = 0
            # Linea verticale
            self.shredded_mask[self.v_line_interp[:, 1], self.v_line_interp[:, 0]] = 0
            print("[DEBUG] Drawing both horizontal & vertical lines for 4 fragments.")
        elif self.n_fragments == 2:
            # Solo la linea verticale
            self.shredded_mask[self.v_line_interp[:, 1], self.v_line_interp[:, 0]] = 0
            print("[DEBUG] Drawing only vertical line for 2 fragments.")

        if debug:
            plt.figure()
            plt.title("Passo 2: Dopo disegno delle linee nere")
            plt.imshow(self.shredded_mask, cmap="gray")
            plt.show()

        # 3) Erosione leggera: riduce di 1 pixel la larghezza delle linee nere,
        #    così da assicurarsi che le regioni siano effettivamente scollegate.
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.shredded_mask = cv2.erode(self.shredded_mask, strel, iterations=1)

        if debug:
            plt.figure()
            plt.title("Passo 3: Dopo erosione")
            plt.imshow(self.shredded_mask, cmap="gray")
            plt.show()

        # 4) Converte i pixel in {0, 255} in base a una soglia, per evitare valori intermedi.
        self.shredded_mask = ((self.shredded_mask > 128) * 255).astype(np.uint8)

        if debug:
            plt.figure()
            plt.title("Passo 4: Dopo threshold (binaria)")
            plt.imshow(self.shredded_mask, cmap="gray")
            plt.show()

        # 5) Definisce i punti-seme (seed_points) per ciascun frammento
        seed_offset = int(min(self.width, self.height) / 4)
        if self.n_fragments == 2:
            seed_points = np.array([
                [self.intersection[0] - seed_offset, self.intersection[1]],
                [self.intersection[0] + seed_offset, self.intersection[1]]
            ])
        else:
            seed_points = np.array([
                [self.intersection[0] - seed_offset, self.intersection[1] - seed_offset],
                [self.intersection[0] - seed_offset, self.intersection[1] + seed_offset],
                [self.intersection[0] + seed_offset, self.intersection[1] - seed_offset],
                [self.intersection[0] + seed_offset, self.intersection[1] + seed_offset],
            ])

        print(f"[DEBUG] Seed points => {seed_points}")

        if debug:
            # Mostriamo i seed su un'immagine grigia
            temp_show = cv2.cvtColor(self.shredded_mask, cv2.COLOR_GRAY2RGB)
            for (sx, sy) in seed_points:
                if 0 <= sx < self.width and 0 <= sy < self.height:
                    cv2.drawMarker(temp_show, (sx, sy), (255, 0, 0), markerType=cv2.MARKER_CROSS, thickness=2)
            plt.figure()
            plt.title("Passo 5: Seed Points")
            plt.imshow(temp_show)
            plt.show()

        # 6) Etichetta le componenti connesse:
        num_labels, labeled_mask, stats, centroids = cv2.connectedComponentsWithStats(
            self.shredded_mask, connectivity=8
        )

        num_labels = 4
        print(f"[DEBUG] Labeled mask => num_labels={num_labels}")
        for i in range(num_labels):
            x, y, w, h, area = stats[i]
            print(f"   Label {i}: bounding box=({x},{y},{w},{h}) "
                f"area={area} centroid={centroids[i]}")

        # 7) Per ognuno dei seed_points, recupera l'etichetta e crea la maschera specifica
        self.mask_fragments = []
        for i, seed in enumerate(seed_points):
            sx, sy = seed
            if 0 <= sx < self.width and 0 <= sy < self.height:
                label_value = labeled_mask[sy, sx]
            else:
                print(f"[DEBUG] Seed {seed} is out of bounds, ignoring.")
                label_value = 0

            fragment = ((labeled_mask == label_value) * 255).astype(np.uint8)
            self.mask_fragments.append(fragment)
            print(f"[DEBUG] Fragment {i+1} => label={label_value}, seed={seed}")

        # Visualizza i frammenti
        if debug:
            display_fragments(self.mask_fragments)

        # 8) (Opzionale) Visualizzazione rapida dell'etichettatura finale
        if debug:
            plt.figure()
            plt.title(f"Passo 8: Labeled regions (step={self.step}, noise={self.noise})")
            plt.imshow(labeled_mask, cmap="tab20")  # 'tab20' per colori diversi
            # Visualizziamo i seed in rosso
            plt.scatter(seed_points[:, 0], seed_points[:, 1], marker='x', c='red')
            plt.show()

    def get_shredded_images(self, debug=False):
        print("[DEBUG] Step a) Starting high-res transformations.")
        
        # 1) Padding
        # (Commentato per ora, ma aggiungiamo dei print e grafici per il debug)
        # output_padding = int(self.pad_factor * min(self.full_image.width, self.full_image.height))
        # output_width  = self.full_image.width  + 2 * output_padding
        # output_height = self.full_image.height + 2 * output_padding
        # self.full_image = self.full_image.gravity("centre", output_width, output_height, background=255)
        # print(f"[DEBUG] After padding => new size = {padded_image.width} x {padded_image.height}")

        # (Opzionale) Visualizzo subito la maschera fragment (low-res)
        # if debug:
            # print(f"[DEBUG] Step 1) applied padding")
            # plt.figure()
            # plt.imshow(self.full_image)
            # plt.title(f"Step 1) applied padding")
            # plt.show()


        # 2) Rotazione (Opzionale, da aggiungere se necessario)
        # rot_mat = cv2.getRotationMatrix2D(center=(0, 0), angle=self.angle, scale=1)
        # rot_coeffs = [rot_mat[0, 0], rot_mat[0, 1], rot_mat[1, 0], rot_mat[1, 1]]
        # padded_image = padded_image.affine(
        #     rot_coeffs,
        #     area=[0, 0, padded_image.width, padded_image.height]
        # )
        # print(f"[DEBUG] Applied rotation to padded_image (angle={self.angle}).")
        # print(f"[DEBUG] Padded_image size => {padded_image.width} x {padded_image.height}")

        self.final_fragments = []
        # 3) Per ogni frammento
        for count, fragment in enumerate(self.mask_fragments, start=1):
            print(f"\n[DEBUG] Step Extracting fragment #{count} ...")

            # 'fragment' è un array numpy 2D binario [0..255], shape (h_lr, w_lr)
            h_lr, w_lr = fragment.shape
            print(f"[DEBUG] Step Low-res mask shape: ({h_lr}, {w_lr})")

            # (Opzionale) Visualizzo subito la maschera fragment (low-res)
            if debug:
                print(f"[DEBUG] Step Visualizing low-res mask for fragment {count} ...")
                plt.figure()
                plt.imshow(fragment, cmap="gray")
                plt.title(f"Fragment {count}: step 0) Low-res mask (NumPy)")
                plt.show()

            # a) Convertiamo la maschera in PyVips: float32 [0..1]
            print(f"[DEBUG] Step a) Converting fragment to PyVips format ...")
            fragment_py = pyvips.Image.new_from_memory(
                (fragment / 255.0).astype(np.float32).ravel(),
                w_lr, h_lr, 1, "float"
            )

            fragment_py = fragment_py.cast("uchar")

            # (Opzionale) Visualizzo la maschera come PyVips (prima di resize/embed)
            if debug:
                print(f"[DEBUG] Step a) Visualizing PyVips mask before resize for fragment {count} ...")
                frag_py_np = np.ndarray(
                    buffer=fragment_py.write_to_memory(),
                    dtype=np.uint8,
                    shape=[h_lr, w_lr, 1]
                )
                plt.figure()
                plt.imshow(frag_py_np[:, :, 0], cmap="gray")
                plt.title(f"Fragment {count} step a): PyVips mask (before resize)")
                plt.show()

            # b) Riscalare la maschera dalla dimensione low-res a full-res
            print(f"[DEBUG] Step b) Resizing the mask to full-res ...")
            fragment_py = fragment_py.resize(self.lowres_downscale)

            # c) "Embed" la maschera nello stesso canvas dell'immagine pad/ruotata
            print(f"[DEBUG] Step c) Embedding the mask to full canvas ...")
            fragment_py = fragment_py.embed(
                0, 0,
                self.full_image.width,
                self.full_image.height,
                background=0.0
            )
            print(f"[DEBUG] Step c) Upsampled fragment mask => {fragment_py.width} x {fragment_py.height}")

            # (Opzionale) Visualizzo la maschera dopo l'embed
            if debug:
                print(f"[DEBUG] Step c) Visualizing embedded mask for fragment {count} ...")
                frag_py_embed_np = np.ndarray(
                    buffer=fragment_py.write_to_memory(),
                    dtype=np.uint8,
                    shape=[self.full_image.height, self.full_image.width, 1]
                )
                plt.figure()
                plt.imshow(frag_py_embed_np[:, :, 0], cmap="gray")
                plt.title(f"Fragment {count}: Step c)  PyVips mask (after embed)")
                plt.show()

            # d) Moltiplica pixel a pixel
            test = True
            if test:
                print(f"[DEBUG] Step d) Applying the mask with condition (1 for image, 0 for white)...")
                print(type(self.full_image))

                # Creiamo una maschera binaria da fragment_py (1 per vero, 0 per falso)
                mask_binary = fragment_py > 0  # Maschera booleana (True = 1, False = 0)
                # Convertiamo la maschera binaria in un array numpy
                mask_binary_np = mask_binary.numpy()

                # print(mask_binary_np)
                # print(mask_binary_np.shape)

                if debug:
                    # Convertiamo la maschera binaria in un array numpy
                    mask_binary_np = mask_binary.numpy()

                    # Visualizzazione della maschera binaria
                    print(f"[DEBUG] Step d) Visualizing binary mask ...")

                    plt.figure()
                    plt.imshow(mask_binary_np, cmap="gray")
                    plt.title(f"Binary Mask")
                    plt.show()

                # Applichiamo la maschera direttamente all'immagine
                full_image_np = self.full_image.numpy()  # Convertiamo l'immagine in un array NumPy

                # Dove la maschera è False (0), mettiamo il valore bianco (255)
                # Dove la maschera è True (1), manteniamo il valore dell'immagine originale
                full_image_masked = np.where(mask_binary_np[:, :, None] == 255, full_image_np, 255)  # Mantieni i valori dell'immagine, altrimenti bianco

                # Convertiamo l'immagine mascherata di nuovo in un oggetto PyVips
                frag_fullres = pyvips.Image.new_from_array(full_image_masked.astype(np.uint8))

                # print(frag_fullres.numpy())

                print("[DEBUG] Step d) Created full-res masked fragment with condition.")

                frag_fullres = frag_fullres.cast("uchar")

                print("[DEBUG] Step d) Created full-res masked fragment with condition.")
            else:
                print(f"[DEBUG] Step d) Multiplying mask with image bands ...")
                channels = []
                for b in range(self.full_image.bands):
                    band = self.full_image.extract_band(b)
                    band_masked = band.multiply(fragment_py)
                    channels.append(band_masked)
                
                frag_fullres = channels[0].bandjoin(channels[1:]) if len(channels) > 1 else channels[0]
                frag_fullres = frag_fullres.cast("uchar")
            
            
            print("[DEBUG] Step d) Created full-res masked fragment.")
            print(type(frag_fullres))

            if debug:
                print(f"[DEBUG] Step e) Visualizing full-res masked fragment ...")
                # print(frag_fullres.bands)
                
                # Scrivi l'immagine su memoria e convertila in numpy array
                frag_np = np.ndarray(
                    buffer=frag_fullres.write_to_memory(),
                    dtype=np.uint8,
                    shape=[frag_fullres.height, frag_fullres.width, frag_fullres.bands]
                )

                # Visualizza la forma dell'immagine
                print(f"[DEBUG] frag_np shape: {frag_np.shape}")

                # Se l'immagine ha 1 band, visualizzarla in scala di grigi
                # Se ha >=3 band, prenderne i primi 3 come RGB
                plt.figure()
                if frag_np.shape[2] == 1:
                    plt.imshow(frag_np[:, :, 0], cmap="gray")
                    plt.title(f"Fragment {count} step d) (final) - Full-res (Grayscale)")
                else:
                    plt.imshow(frag_np[:, :, :3])
                    plt.title(f"Fragment {count} step d) (final) - Full-res (RGB)")
                plt.show()



            # e) Trovo bounding box
            print(f"[DEBUG] Step e) Finding bounding box for fragment {count} ...")
            mask_np = np.ndarray(
                buffer=fragment_py.write_to_memory(),
                dtype=np.uint8,
                shape=[self.full_image.height, self.full_image.width, 1]
            )
            threshold = 0.01
            ys, xs = np.where(mask_np[:, :, 0] > threshold)
            if len(xs) == 0 or len(ys) == 0:
                print(" - [DEBUG] Step n) No content found in this fragment. Skipping.")
                continue

            xmin, xmax = xs.min(), xs.max()
            ymin, ymax = ys.min(), ys.max()
            width_f  = xmax - xmin + 1
            height_f = ymax - ymin + 1
            print(f"[DEBUG] Step e) Fragment bounding box => (xmin={xmin}, ymin={ymin}, width={width_f}, height={height_f})")

            # Visualizza la bounding box sulla maschera
            if debug:
                print(f"[DEBUG] Step f) Visualizing bounding box for fragment {count} ...")
                plt.figure()
                plt.imshow(mask_np[:, :, 0], cmap="gray")
                plt.title(f"Fragment {count} Step e) - Bounding Box")
                
                # Disegna il rettangolo della bounding box
                plt.gca().add_patch(plt.Rectangle(
                    (xmin, ymin), width_f, height_f,
                    edgecolor='red', facecolor='none', linewidth=2
                ))

                plt.show()

            # f) Crop
            # (Opzionale) Visualizzo il frammento prima di cropprare
            if debug:
                print(f"[DEBUG] Step f) Visualizing fragment before cropping")
                frag_np = np.ndarray(
                    buffer=frag_fullres.write_to_memory(),
                    dtype=np.uint8,
                    shape=[frag_fullres.height, frag_fullres.width, frag_fullres.bands]
                )
                if frag_np.shape[2] == 1:
                    plt.figure()
                    plt.imshow(frag_np[:, :, 0], cmap="gray")
                    plt.title(f"Fragment {count} (pre crop) - Grayscale")
                    plt.show()
                else:
                    plt.figure()
                    plt.imshow(frag_np[:, :, :3])
                    plt.title(f"Fragment {count} (pre crop) - RGB")
                    plt.show()

            print(f"[DEBUG] Step f) Cropping the fragment ...")
            frag_fullres = frag_fullres.crop(xmin, ymin, width_f, height_f)
            print("[DEBUG] Step f) Cropped final fragment to bounding box.")
            print(type(frag_fullres))

            # Visualizza la maschera full-res dopo il crop
            if debug:
                print(f"[DEBUG] Step f) Visualizing full-res mask after crop for fragment {count} ...")
                print(f"[DEBUG] frag_fullres.bands")
                frag_np = np.ndarray(
                    buffer=frag_fullres.write_to_memory(),
                    dtype=np.uint8,
                    shape=[frag_fullres.height, frag_fullres.width, frag_fullres.bands]
                )
                
                plt.figure()
                if frag_np.shape[2] == 1:
                    plt.imshow(frag_np[:, :, 0], cmap="gray")
                    plt.title(f"Fragment {count} step f) (final full-res mask after crop) - Cropped Grayscale")
                else:
                    plt.imshow(frag_np[:, :, :3])
                    plt.title(f"Fragment {count} step f) (final full-res mask after crop) - Cropped RGB")
                plt.show()

            # g) Risoluzione e cast a 8 bit
            print(f"[DEBUG] Step g) Adjusting resolution and casting to 8-bit ...")
            spacing = 0.25
            xyres = 1000 / spacing
            frag_fullres = frag_fullres.copy(xres=xyres, yres=xyres)

            # Aggiungi padding (bianco, valore 255)
            padding_size = 100  # Esempio di dimensione del padding, puoi cambiarla come desideri
            output_width = frag_fullres.width + 2 * padding_size
            output_height = frag_fullres.height + 2 * padding_size

            # Crea un'immagine di padding bianco e uniscila con l'immagine originale
            padded_image = frag_fullres.gravity("centre", output_width, output_height, background=255)

            # (Opzionale) Visualizza il confronto tra l'immagine originale e l'immagine con padding
            if debug:
                print(f"[DEBUG] Step g) Visualizing final fragment and comparison ...")

                # Crea un subplot con 2 colonne per il confronto
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                
                # Visualizza l'immagine originale
                axes[0].imshow(frag_fullres)
                axes[0].set_title(f"Fragment {count} - Original")

                # Visualizza l'immagine con padding
                axes[1].imshow(padded_image)
                axes[1].set_title(f"Fragment {count} - With Padding")
                
                # Mostra il confronto
                plt.tight_layout()
                plt.show()


            self.final_fragments.append(padded_image)

        for count, fragment in enumerate(self.final_fragments, start=1): 
            save = False
            if save: 
                # i) Salvataggio TIF piramidale
                print(f"[DEBUG] Step i) Saving fragment as TIF ...")
                outpath_tif = self.savedir.joinpath(f"fragment{count}.tif")
                frag_fullres.write_to_file(
                    str(outpath_tif),
                    tile=True,
                    compression="jpeg",
                    bigtiff=True,
                    pyramid=True,
                    Q=80,
                )
            
            if debug:  
                self.savedir.joinpath("raw_images").mkdir()
                self.savedir.joinpath("parameters").mkdir()
                
                # Salvataggio PNG
                print(f"[DEBUG] Step i) Saving fragment as PNG ...")
                outpath_png = self.savedir.joinpath("raw_images", f"fragment{count}.png")
                frag_fullres.write_to_file(str(outpath_png))

                # j) Salvataggio parametri
                print(f"[DEBUG] Step j) Saving parameters for fragment {count} ...")
                with open(self.savedir.joinpath("parameters", f"fragment{count}_shred_parameters.json"), "w") as f:
                    json.dump(self.parameters, f, ensure_ascii=False)

        print(f"[DEBUG] Done with fragment #{count}.\n")

    def process_case(self):
        print("\n[DEBUG] ===========================")
        print(f"[DEBUG] Processing {self.case.name}")
        print("[DEBUG] ===========================")

        self.load_images()
        self.process()
        self.get_shred_parameters()
        self.apply_shred()
        self.get_shredded_images()

        return self.final_fragments

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    data_dir, save_dir, rotation, n_fragments = collect_arguments()

    # Raccogli tutti i file .tif nelle sottocartelle di data_dir
    cases = sorted([case for case in data_dir.rglob('*.tif') if case.is_file()])

    print(f"Found {len(cases)} cases to shred")

    # Filtra i casi già processati
    cases = [
        case
        for case in cases
        if not save_dir.joinpath(case.stem, "fragment4_shred_parameters.json").exists()
    ]
    print(f"Shredding {len(cases)} remaining cases...\n")

    for case in tqdm.tqdm(cases, total=len(cases)):
        print("case:", case)
        shredder = Shredder(case, save_dir, rotation, n_fragments)
        
    

if __name__ == "__main__":
    main()
