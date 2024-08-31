import math
import cv2
import numpy as np
import scipy.ndimage as ndi
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from skimage.transform import rescale
from skimage import color, feature
import skimage.morphology as morphology
from skimage.filters import threshold_otsu


class Preprocessing:
    def __init__(self, image_path):
        self.original_image = imageio.imread(image_path)
        self.processed_image = None

    def show_images(self, *images, figure_title):
        n_img = len(images)
        fig, ax = plt.subplots(math.ceil(n_img / 4), n_img % 4 if n_img < 4 else 4, figsize=(10, 10))
        fig.suptitle(figure_title, fontsize=16)  # Aggiungi il titolo per l'intera figura
        ax = ax.flatten()
        for n, i in enumerate(images):
            ax[n].imshow(i, cmap='gray')
        plt.show()

    def find_best_threshold_otsu(self):
        otsu_threshold = threshold_otsu(self.processed_image)
        return otsu_threshold

    def rescale_img(self, dtype=np.uint8, show=False):
        self.processed_image = rescale(self.original_image, 0.25, channel_axis=2, preserve_range=True).astype(dtype)
        if show:
            self.show_images(self.original_image, self.processed_image, 'Rescaling')
        return self.processed_image

    def grayscale_transform(self, show=False):
        self.processed_image = color.rgb2gray(self.original_image)
        self.processed_image = (np.round(self.processed_image * 255)).astype(np.uint8)
        if show:
            fig, ax = plt.subplots(1, 2)
            fig.suptitle('grayscale transform', fontsize=16)
            ax[0].imshow(self.original_image)
            ax[1].imshow(self.processed_image, cmap='gray')
            plt.show()
        return self.processed_image

    def histogram(self, show=False):
        histo = ndi.histogram(self.processed_image, min=0, max=255, bins=256)
        if show:
            plt.figure()
            plt.suptitle('Histogram', fontsize=16)
            plt.bar(np.arange(len(histo)), height=histo)
            plt.show()
        return histo

    def thresholding(self, threshold, show=False):
        threshold = self.find_best_threshold_otsu() if threshold is None else threshold
        self.processed_image = self.processed_image < threshold
        if show:
            fig, ax = plt.subplots(1, 2)
            fig.suptitle('Thresholding', fontsize=16)
            ax[0].imshow(self.original_image, cmap='gray')
            ax[1].imshow(self.processed_image, cmap='gray')
            plt.show()
        return self.processed_image

    def applying_median_filter(self, size=5, show=False):
        self.processed_image = ndi.median_filter(self.processed_image, size=size)
        if show:
            self.show_images(self.original_image, self.processed_image, figure_title='Median filtering')
        return self.processed_image

    def binary_closing_image(self, footprint_size, show=False):
        self.processed_image = morphology.binary_closing(self.processed_image, footprint=morphology.square(footprint_size))
        if show:
            self.show_images(self.original_image, self.processed_image, figure_title='Closed image')
        return self.processed_image

    def edge_detection(self, sigma=1.0, show=False):
        self.processed_image = feature.canny(self.processed_image, sigma=sigma)
        if show:
            self.show_images(self.original_image, self.processed_image, figure_title='Edge detection')
        return self.processed_image

    def contours_and_hulls(self, show=False):
        x = self.processed_image.copy().astype(np.uint8)
        c, _ = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = np.squeeze(c[0], axis=1)  # 8 is the index of the largest contour of the image
        hull = cv2.convexHull(c)
        hull1 = np.squeeze(hull, axis=1)  # for drawing purposes

        x_circles = x.copy()
        x_hull = x.copy()
        x_full_contours = x.copy()

        for i in c:
            x_circles = cv2.drawMarker(x_circles, tuple(i), color=130, markerType=cv2.MARKER_CROSS)

        for i in hull1:
            x_hull = cv2.drawMarker(x_hull, tuple(i), color=130, markerType=cv2.MARKER_CROSS, thickness=5)

        cv2.drawContours(x_full_contours, [hull], -1, 1, -1)

        if show:
            self.show_images(x_circles, x_hull, x_full_contours, figure_title='Contours and hull')

        return x_full_contours

    def preprocess_image(self, threshold=None, median_filter_size=20, closing_footprint_size=20, edge_sigma=15,
                         apply_grayscale=True, apply_threshold=True, apply_median_filter=True,
                         apply_binary_closing=True, apply_edge_detection=True, show_steps=False):
        if apply_grayscale:
            self.grayscale_transform(show=show_steps)
        if apply_threshold:
            self.histogram(show=show_steps)
            self.thresholding(threshold, show=show_steps)
        if apply_median_filter:
            self.applying_median_filter(size=median_filter_size, show=show_steps)
        if apply_binary_closing:
            self.binary_closing_image(footprint_size=closing_footprint_size, show=show_steps)
        if apply_edge_detection:
            self.edge_detection(sigma=edge_sigma, show=show_steps)
        filled_image = self.contours_and_hulls(show=show_steps)
        return filled_image


