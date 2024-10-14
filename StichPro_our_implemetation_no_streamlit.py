"""
# StitchPro
"""
## Import packages

OPENSLIDE_PATH = r'C:\Program Files\openslide-bin-4.0.0.3-windows-x64\bin'

import os

if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageEnhance
import numpy as np
from skimage import color, data
import scipy.ndimage as ndi
from skimage import morphology
from skimage.feature import canny, corner_harris, corner_subpix, corner_peaks
from itertools import combinations
from scipy.spatial.distance import cdist, pdist, squareform
from skimage.transform import rescale, resize
import argparse
import tifffile
from tiatoolbox.wsicore import WSIReader
import time
import json

import matplotlib.pyplot as plt  # DEBUGGING

import gc

gc.collect()

# IMPORT OUR CLASSES
from utilities import Preprocessing, Line, Image_Lines

# Names
files = ["upper_right", "bottom_right", "bottom_left", "upper_left"]
root_folder = os.getcwd()
print(f"root_folder: {root_folder}")

# DEBUGGING
# create folder
save_dir = os.path.join(root_folder, 'debug')

try:
    os.makedirs(save_dir, exist_ok=True)
    print(f"Cartella '{save_dir}' creata con successo")
except OSError as e:
    print(f"Errore nella creazione della cartella: {e}")

for i in range(len(files)):
    # DEBUGGING
    # create folder
    save_dir_image_i = os.path.join(root_folder, 'debug', files[i])
    try:
        os.makedirs(save_dir_image_i, exist_ok=True)
        print(f"Cartella '{save_dir_image_i}' creata con successo")
    except OSError as e:
        print(f"Errore nella creazione della cartella: {e}")

## App description
start_time = time.time()

## Upload images and rotate them by given angle
if os.path.exists("/mnt/Volume/Mega/LaureaMagistrale/CorsiSemestre/A2S1/MultudisciplinaryProject/StitchPro/test-data/"):
    img_file_buffer_ur = "/mnt/Volume/Mega/LaureaMagistrale/CorsiSemestre/A2S1/MultudisciplinaryProject/data/Dataset_07/upper_right.tif"
    img_file_buffer_lr = "/mnt/Volume/Mega/LaureaMagistrale/CorsiSemestre/A2S1/MultudisciplinaryProject/data/Dataset_07/bottom_right.tif"
    img_file_buffer_ll = "/mnt/Volume/Mega/LaureaMagistrale/CorsiSemestre/A2S1/MultudisciplinaryProject/data/Dataset_07/bottom_left.tif"
    img_file_buffer_ul = "/mnt/Volume/Mega/LaureaMagistrale/CorsiSemestre/A2S1/MultudisciplinaryProject/data/Dataset_07/upper_left.tif"
elif os.path.exists(r"C:\Users\dicia\NL2_project\datasets\test-data-corretto"):
    img_file_buffer_ur = r"C:\Users\dicia\NL2_project\datasets\downsampled\downsampled_2\upper_right.tif"
    img_file_buffer_lr = r"C:\Users\dicia\NL2_project\datasets\downsampled\downsampled_2\bottom_right.tif"
    img_file_buffer_ll = r"C:\Users\dicia\NL2_project\datasets\downsampled\downsampled_2\bottom_left.tif"
    img_file_buffer_ul = r"C:\Users\dicia\NL2_project\datasets\downsampled\downsampled_2\upper_left.tif"
elif os.path.exists("/kaggle/input/"):
    img_file_buffer_ur = '/kaggle/input/dataset/Dataset_07/upper_right.tif'
    img_file_buffer_ul = '/kaggle/input/dataset/Dataset_07/upper_left.tif'
    img_file_buffer_ll = '/kaggle/input/dataset/Dataset_07/bottom_left.tif'
    img_file_buffer_lr = '/kaggle/input/dataset/Dataset_07/bottom_right.tif'




if (img_file_buffer_ur is not None) & (img_file_buffer_lr is not None) & (img_file_buffer_ll is not None) & (
        img_file_buffer_ul is not None):

    # Read the images with name histo_fragment_[pos]
    histo_fragment_ur = imageio.imread(img_file_buffer_ur)

    histo_fragment_lr = imageio.imread(img_file_buffer_lr)

    histo_fragment_ll = imageio.imread(img_file_buffer_ll)

    histo_fragment_ul = imageio.imread(img_file_buffer_ul)


    # print("Dimensioni originali immagini")
    # print(histo_fragment_ur.shape)
    # print(histo_fragment_lr.shape)
    # print(histo_fragment_ll.shape)
    # print(histo_fragment_ul.shape)

    # Rotate images if user use the option
    angle_options = [-90, 0, 90, 180]
    angle_choice_ur = 0
    if int(angle_choice_ur) == 90:
        histo_fragment_ur = cv2.rotate(histo_fragment_ur, cv2.ROTATE_90_CLOCKWISE)
    if int(angle_choice_ur) == -90:
        histo_fragment_ur = cv2.rotate(histo_fragment_ur, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if int(angle_choice_ur) == 180:
        histo_fragment_ur = cv2.rotate(histo_fragment_ur, cv2.ROTATE_180)
    if int(angle_choice_ur) == 0:
        histo_fragment_ur = histo_fragment_ur

    angle_choice_lr = 0
    if int(angle_choice_lr) == 90:
        histo_fragment_lr = cv2.rotate(histo_fragment_lr, cv2.ROTATE_90_CLOCKWISE)
    if int(angle_choice_lr) == -90:
        histo_fragment_lr = cv2.rotate(histo_fragment_lr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if int(angle_choice_lr) == 180:
        histo_fragment_lr = cv2.rotate(histo_fragment_lr, cv2.ROTATE_180)
    if int(angle_choice_lr) == 0:
        histo_fragment_lr = histo_fragment_lr
    angle_choice_ll = 0
    if int(angle_choice_ll) == 90:
        histo_fragment_ll = cv2.rotate(histo_fragment_ll, cv2.ROTATE_90_CLOCKWISE)
    if int(angle_choice_ll) == -90:
        histo_fragment_ll = cv2.rotate(histo_fragment_ll, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if int(angle_choice_ll) == 180:
        histo_fragment_ll = cv2.rotate(histo_fragment_ll, cv2.ROTATE_180)
    if int(angle_choice_ll) == 0:
        histo_fragment_ll = histo_fragment_ll
    angle_choice_ul = 0
    if int(angle_choice_ul) == 90:
        histo_fragment_ul = cv2.rotate(histo_fragment_ul, cv2.ROTATE_90_CLOCKWISE)
    if int(angle_choice_ul) == -90:
        histo_fragment_ul = cv2.rotate(histo_fragment_ul, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if int(angle_choice_ul) == 180:
        histo_fragment_ul = cv2.rotate(histo_fragment_ul, cv2.ROTATE_180)
    if int(angle_choice_ul) == 0:
        histo_fragment_ul = histo_fragment_ul

    images_original = [histo_fragment_ur, histo_fragment_lr, histo_fragment_ll, histo_fragment_ul]

    original_spacing = 0.25
    level = 5
    sub_bound_x = 550
    sub_bound_y = 550

    # Downsample the images
    # ATTENZIONE il livello di downsampling prima era settato a 4
    DOWNSAMPLE_LEVEL = 4  # avoid running the optimization for the entire image
    add = 0
    histo_fragment_lr = rescale(histo_fragment_lr, 1 / DOWNSAMPLE_LEVEL, channel_axis=2,
                                preserve_range=True).astype(np.uint8)
    histo_fragment_ll = rescale(histo_fragment_ll, 1 / DOWNSAMPLE_LEVEL, channel_axis=2,
                                preserve_range=True).astype(np.uint8)
    histo_fragment_ur = rescale(histo_fragment_ur, 1 / DOWNSAMPLE_LEVEL, channel_axis=2,
                                preserve_range=True).astype(np.uint8)
    histo_fragment_ul = rescale(histo_fragment_ul, 1 / DOWNSAMPLE_LEVEL, channel_axis=2,
                                preserve_range=True).astype(np.uint8)

    # print("Dimensioni dopo downsampling:")
    # print(histo_fragment_ur.shape)
    # print(histo_fragment_lr.shape)
    # print(histo_fragment_ll.shape)
    # print(histo_fragment_ul.shape)

    ### Find image contours

    ## Convert from RGB image to grayscale
    # ATTENTIONE QUI HO CAMBIATO CODICE: ERA 256 PRIMA
    histo_fragment_gray_binary_ll = color.rgb2gray(histo_fragment_ll)
    histo_fragment_gray_ll = (histo_fragment_gray_binary_ll * 255).astype('uint8')
    histo_fragment_gray_binary_lr = color.rgb2gray(histo_fragment_lr)
    histo_fragment_gray_lr = (histo_fragment_gray_binary_lr * 255).astype('uint8')
    histo_fragment_gray_binary_ul = color.rgb2gray(histo_fragment_ul)
    histo_fragment_gray_ul = (histo_fragment_gray_binary_ul * 255).astype('uint8')
    histo_fragment_gray_binary_ur = color.rgb2gray(histo_fragment_ur)
    histo_fragment_gray_ur = (histo_fragment_gray_binary_ur * 255).astype('uint8')

    # print("Dimensioni dopo grayscale:")
    # print(histo_fragment_gray_ur.shape)
    # print(histo_fragment_gray_lr.shape)
    # print(histo_fragment_gray_ll.shape)
    # print(histo_fragment_gray_ul.shape)

    plt.imshow(histo_fragment_gray_ll, cmap="gray")
    plt.savefig(os.path.join(save_dir, 'bottom_left', f'lower_left_gray_scale.png'))
    plt.imshow(histo_fragment_gray_lr, cmap="gray")
    plt.savefig(os.path.join(save_dir, 'bottom_right', f'lower_right_gray_scale.png'))
    plt.imshow(histo_fragment_gray_ul, cmap="gray")
    plt.savefig(os.path.join(save_dir, 'upper_left', f'upper_left_gray_scale.png'))
    plt.imshow(histo_fragment_gray_ur, cmap="gray")
    plt.savefig(os.path.join(save_dir, 'upper_right', f'upper_right_gray_scale.png'))

    ## Intensity histogram
    hist_ul = ndi.histogram(histo_fragment_gray_ul, min=0, max=255, bins=256)
    hist_ur = ndi.histogram(histo_fragment_gray_ur, min=0, max=255, bins=256)
    hist_lr = ndi.histogram(histo_fragment_gray_lr, min=0, max=255, bins=256)
    hist_ll = ndi.histogram(histo_fragment_gray_ll, min=0, max=255, bins=256)

    ## Image segmentation based on threshold
    thresh = 220
    # thresholding based --> binary image is obtained
    image_thresholded_ul = histo_fragment_gray_ul < thresh
    image_thresholded_ur = histo_fragment_gray_ur < thresh
    image_thresholded_lr = histo_fragment_gray_lr < thresh
    image_thresholded_ll = histo_fragment_gray_ll < thresh

    plt.imshow(image_thresholded_ul, cmap="gray")
    plt.savefig(os.path.join(save_dir, 'bottom_left', f'upper_left_segmentation.png'))
    plt.imshow(image_thresholded_ur, cmap="gray")
    plt.savefig(os.path.join(save_dir, 'bottom_right', f'upper_right_segmentation.png'))
    plt.imshow(image_thresholded_lr, cmap="gray")
    plt.savefig(os.path.join(save_dir, 'upper_left', f'lower_right_segmentation.png'))
    plt.imshow(image_thresholded_ll, cmap="gray")
    plt.savefig(os.path.join(save_dir, 'upper_right', f'lower_left_segmentation.png'))

    ## Apply median filter to reduce the noise
    median_filter_ur_x = 20
    median_filter_lr_x = 35
    median_filter_ll_x = 20
    median_filter_ul_x = 20
    image_thresholded_filtered_ul = ndi.median_filter(image_thresholded_ul, size=int(median_filter_ul_x))
    image_thresholded_filtered_ur = ndi.median_filter(image_thresholded_ur, size=int(median_filter_ur_x))
    image_thresholded_filtered_ll = ndi.median_filter(image_thresholded_ll, size=int(median_filter_ll_x))
    image_thresholded_filtered_lr = ndi.median_filter(image_thresholded_lr, size=int(median_filter_lr_x))

    ## Erode the image to eliminate holes
    closing_ur_x = 15
    closing_lr_x = 35
    closing_ll_x = 15
    closing_ul_x = 15
    image_thresholded_filtered_closed_ul = morphology.binary_closing(image_thresholded_filtered_ul,
                                                                     footprint=morphology.square(
                                                                         int(closing_ul_x)))  # morphology.square(30) 28 disk
    image_thresholded_filtered_closed_ur = morphology.binary_closing(image_thresholded_filtered_ur,
                                                                     footprint=morphology.square(
                                                                         int(closing_ur_x)))  # morphology.square(84) 26
    image_thresholded_filtered_closed_ll = morphology.binary_closing(image_thresholded_filtered_ll,
                                                                     footprint=morphology.square(
                                                                         int(closing_ll_x)))  # morphology.square(100) 30
    image_thresholded_filtered_closed_lr = morphology.binary_closing(image_thresholded_filtered_lr,
                                                                     footprint=morphology.square(
                                                                         int(closing_lr_x)))  # morphology.square(150) 26

    # Identify the image boundary
    canny_edges_ul = canny(image_thresholded_filtered_closed_ul, sigma=5)
    canny_edges_ur = canny(image_thresholded_filtered_closed_ur, sigma=5)
    canny_edges_ll = canny(image_thresholded_filtered_closed_ll, sigma=5)
    canny_edges_lr = canny(image_thresholded_filtered_closed_lr, sigma=5)

    # if st.button("Start stitching!") == True:
    if True:
        start_stiching_time = time.time()

        data_dict = []

        # Tissue masks are the 'closed-masks' (see the iamge saved for details)
        tissue_masks = [image_thresholded_filtered_closed_ur, image_thresholded_filtered_closed_lr,
                        image_thresholded_filtered_closed_ll, image_thresholded_filtered_closed_ul]

        images = [histo_fragment_ur, histo_fragment_lr, histo_fragment_ll, histo_fragment_ul]

        # Tissue masks closed are the images containing the edges only
        tissue_masks_closed = [canny_edges_ur, canny_edges_lr, canny_edges_ll, canny_edges_ul]

        N = len(tissue_masks)

        for i in range(len(tissue_masks)):
            print(f"Processing {files[i]} fragment ...\n")
            x = tissue_masks[i]
            x_out = tissue_masks_closed[i]

            # DEBUGGING
            # create folder
            save_dir_image_i = os.path.join(root_folder, 'debug', files[i])
            try:
                os.makedirs(save_dir_image_i, exist_ok=True)
                print(f"Cartella '{save_dir_image_i}' creata con successo")
            except OSError as e:
                print(f"Errore nella creazione della cartella: {e}")
            plt.figure(figsize=(50, 50))
            plt.imshow(x, cmap='gray')  # DEBUGGING
            plt.savefig(os.path.join(save_dir_image_i, f'{files[i]}_tissue_mask.png'))

            plt.figure(figsize=(50, 50))
            plt.imshow(x_out, cmap='gray')
            plt.savefig(os.path.join(save_dir_image_i, f'{files[i]}_tissue_mask_closed.png'))

            ######################### INIZIO NOSTRA IMPLEMENTAZIONE ############################################

            # print("Tipi immagine: ", type(images[i][2, 3]), type(tissue_masks_closed[i][2, 3]))
            image_lines = Image_Lines(images[i], tissue_masks_closed[i], save_dir_image_i)
            image_lines.plot_results()
            ant_line = np.array([image_lines.intersection, image_lines.end_ant_point])
            pos_line = np.array([image_lines.intersection, image_lines.end_pos_point])

            data_dict.append({
                "image": image_lines.original_image,
                "tissue_mask": tissue_masks[i],
                "tissue_mask_closed": image_lines.image,
                "quadrant": i,
                "ant_line": ant_line,
                "pos_line": pos_line,
                "ant_points": image_lines.ant_points,
                "pos_points": image_lines.pos_points})
            print("\n")

            # print("Riassunto dati trovati trovati per quadrante:", data_dict[i]["quadrant"])
            # print("image shape:", data_dict[i]["image"].shape)
            # print("tissue_mask:", data_dict[i]["tissue_mask"].shape)
            # print("tissue_mask_closed:", data_dict[i]["tissue_mask_closed"].shape)
            # print("ant_line:", data_dict[i]["ant_line"])
            # print("pos_line", data_dict[i]["pos_line"])

            # print("type data_dict: ", type(data_dict))
            # print(type(data_dict[i]["image"]))
            # print(type(data_dict[i]["tissue_mask"]))
            # print(type(data_dict[i]["tissue_mask_closed"]))
            # print(type(data_dict[i]["quadrant"]))
            # print(type(data_dict[i]["ant_line"]))
            # print(type(data_dict[i]["pos_line"]))
            # print(type(data_dict[i]["ant_points"]))
            # print(type(data_dict[i]["pos_points"]))

            # COnversion needed since you can't save a nparray in a json file
            data_dict[i]['image'] = data_dict[i]['image'].tolist()
            data_dict[i]['tissue_mask'] = data_dict[i]['tissue_mask'].tolist()
            data_dict[i]['tissue_mask_closed'] = data_dict[i]['tissue_mask_closed'].tolist()
            data_dict[i]['ant_line'] = data_dict[i]['ant_line'].tolist()
            data_dict[i]['pos_line'] = data_dict[i]['pos_line'].tolist()
            data_dict[i]['ant_points'] = data_dict[i]['ant_points'].tolist()
            data_dict[i]['pos_points'] = data_dict[i]['pos_points'].tolist()

            # Save as json
            with open(os.path.join(save_dir_image_i, "data_dict.json"), "w") as file:
                # json.dump(data_dict[i], file)
                json.dump(dict((k, data_dict[i][k]) for k in ['ant_line', 'pos_line', 'ant_points', 'pos_points']),
                          file, separators=(',', ': '))

            load_fixed_dicts = True
            if load_fixed_dicts:  # modo diverso di caricare su i json files fixati
                paolo_path = r"C:\Users\dicia\NL2_project\debugging_series\restults_comparison\fixed_dicts\ciao"
                if os.path.exists(paolo_path) and os.path.isdir(paolo_path):
                    filepath = f"data_dict_fixed_{i}.json"
                    try:
                        with open(os.path.join(paolo_path, filepath), "r") as file:
                            data_dict[i] = json.load(file)
                    except FileNotFoundError:
                        print(f"file {os.path.join(paolo_path, filepath)} non trovato")
                    except json.JSONDecodeError:
                        print(f"errore nella decodifica del file json {os.path.join(paolo_path, filepath)}")
                # else:  # To load it back (it will load the array as a list)
                #    with open(os.path.join(save_dir_image_i, "data_dict_fixed.json"), "r") as file:
                #       data_dict[i] = json.load(file)
                # loaded_dict['data'] = np.array(loaded_dict['data'])  # Convert back to ndarray

            # print(f"Keys in data_dict[{i}]: {list(data_dict[i].keys())}")
            # Li faccio tornare nparray perchè serve cosi dopo
            data_dict[i]['image'] = np.array(data_dict[i]['image'])
            data_dict[i]['tissue_mask'] = np.array(data_dict[i]['tissue_mask'])
            data_dict[i]['tissue_mask_closed'] = np.array(data_dict[i]['tissue_mask_closed'])
            data_dict[i]['ant_line'] = np.array(data_dict[i]['ant_line'])
            data_dict[i]['pos_line'] = np.array(data_dict[i]['pos_line'])
            data_dict[i]['ant_points'] = np.array(data_dict[i]['ant_points'])
            data_dict[i]['pos_points'] = np.array(data_dict[i]['pos_points'])

            # Crea una figura con 3 sottotrame
            fig, axs = plt.subplots(2, 3, figsize=(15, 5))  # 1 riga, 3 colonne
            ax = axs.ravel()
            # Visualizza ciascuna immagine in una sottotrama
            ax[0].imshow(data_dict[i]["image"])
            ax[0].axis('off')  # Nasconde gli assi opzionale

            ax[1].imshow(data_dict[i]["tissue_mask"], cmap="gray")
            ax[1].axis('off')

            ax[2].imshow(data_dict[i]["tissue_mask_closed"], cmap="gray")
            ax[2].axis('off')

            # Mostra la visualizzazione con le tre immagini
            # plt.show()

            # Stampo i punti
            # aux_mask = cv2.cvtColor(aux_mask, cv2.COLOR_GRAY2RGB)
            aux_mask = data_dict[i]["tissue_mask_closed"]
            # Convert aux_mask from CV_32S to CV_8U
            aux_mask = aux_mask.astype(np.uint8)
            aux_mask = cv2.cvtColor(aux_mask, cv2.COLOR_GRAY2RGB)
            # disegna i punti per sola visualizzazione
            for point in data_dict[i]['ant_points']:
                cv2.drawMarker(aux_mask, tuple(point), color=(255, 255, 0), markerType=cv2.MARKER_SQUARE, markerSize=3,
                               thickness=2)

            for point in data_dict[i]['pos_points']:
                cv2.drawMarker(aux_mask, tuple(point), color=(255, 0, 0), markerType=cv2.MARKER_SQUARE, markerSize=3,
                               thickness=2)

            ax[3].imshow(aux_mask)
            ax[3].set_title("Contorni")

            aux_mask2 = data_dict[i]["tissue_mask_closed"]
            ax[4].imshow(aux_mask2)
            # Plot the line over the image
            ax[4].axline(data_dict[i]["ant_line"][0], xy2=data_dict[i]["ant_line"][1], color='yellow', linewidth=2,
                         marker='o')
            ax[4].axline(data_dict[i]["pos_line"][0], xy2=data_dict[i]["pos_line"][1], color='red', linewidth=2,
                         marker='o')

            plt.tight_layout()

            plt.savefig(os.path.join(save_dir_image_i, 'data_dict'), dpi=300)

            plt.close("all")

        ############### FINE NOSTRA IMPLEMENTAZIONE #################################

        gc.collect()

        ## Calculate histograms and distances between histograms
        # set up functions to calculate colour histograms
        square_size = 64
        n_bins = 32


        # from here on it starts working on the real image

        def calculate_histogram(image, mask, center, n_bins, size):
            # function that computes the histograms for red green and blue channel of a certain region of the given
            # image. The region position is given by the mask
            x, y = center
            Mx, My = mask.shape
            x1, x2 = np.maximum(x - size // 2, 0), np.minimum(x + size // 2, Mx)
            y1, y2 = np.maximum(y - size // 2, 0), np.minimum(y + size // 2, My)
            mask = mask[x1:x2, y1:y2]
            sub_image = image[x1:x2, y1:y2]
            sub_image = sub_image.reshape([-1, 3])[mask.reshape([-1]) == 1]

            # DEBUGGING
            # plt.imshow(sub_image)
            # plt.savefig('sub_image')
            # plt.imshow(mask)
            # plt.savefig('sub_mask')

            r_hist = np.histogram(sub_image[:, 0], n_bins, range=[0, 256], density=True)[0]
            g_hist = np.histogram(sub_image[:, 1], n_bins, range=[0, 256], density=True)[0]
            b_hist = np.histogram(sub_image[:, 2], n_bins, range=[0, 256], density=True)[0]

            out = np.concatenate([r_hist, g_hist, b_hist])
            return out


        histograms = []
        print("Calculating histograms for all sections along the x and y edges...")
        for i in range(len(data_dict)):
            # data becomes the new parameter dictionary for image i !!!
            data = data_dict[i]
            hx, hy = [], []
            for y, x in data["ant_points"]:
                H = calculate_histogram(
                    data['image'], data['tissue_mask'], [x, y], n_bins, square_size)
                hx.append(H)
            for y, x in data["pos_points"]:
                H = calculate_histogram(
                    data['image'], data['tissue_mask'], [x, y], n_bins, square_size)
                hy.append(H)

            # hx is a 766x96 array where 766 is the dimension of pos_points (for each point a histo is computed)
            # 96 = 32x3 is the total number of bins for the histo related to the 3 channels
            hx = np.array(hx)
            hy = np.array(hy)
            data_dict[i]['histograms_ant'] = hx
            data_dict[i]['histograms_pos'] = hy

        print("Calculating correlation distances between histograms for every tissue section pair...")
        histogram_dists = {}
        for i, j in combinations(range(len(data_dict) + 1), 2):
            i = i % len(data_dict)
            j = j % len(data_dict)
            histogram_dists[i, j] = {
                "ant_pos": cdist(
                    data_dict[i]['histograms_ant'], data_dict[j]['histograms_pos'],
                    metric="correlation"),
                "pos_ant": cdist(
                    data_dict[i]['histograms_pos'], data_dict[j]['histograms_ant'],
                    metric="correlation")}

        # ensuring that max distance == 1
        for i, j in histogram_dists:
            histogram_dists[i, j]['ant_pos'] = np.divide(
                histogram_dists[i, j]['ant_pos'], histogram_dists[i, j]['ant_pos'].max())
            histogram_dists[i, j]['pos_ant'] = np.divide(
                histogram_dists[i, j]['pos_ant'], histogram_dists[i, j]['pos_ant'].max())

        ## Optimization of tissue segment translations and rotations via differential evolution

        POPSIZE = 25  # parameter for the evolutionary optimization (population size)
        MAXITER = 200  # parameter for the evolutionary optimization (population size)

        from skimage.transform import AffineTransform

        from scipy.optimize import NonlinearConstraint
        from scipy import optimize

        np.random.seed(42)

        output_size = max([max(x['image'].shape) for x in data_dict]) * 2
        output_size = [output_size, output_size]
        print('output_size', output_size)


        def par_to_H(theta, tx, ty):
            # converts a set of three parameters to
            # a homography matrix
            H = AffineTransform(
                scale=1, rotation=theta, shear=None, translation=[tx, ty])
            return H.params


        def M_to_quadrant_dict(M, quadrants, anchor):
            # function that generates the transformation matrices for each quadrant and returns a dict
            H_dict = {}
            Q = [q for q in quadrants if q != anchor]
            for i, q in enumerate(Q):
                H_dict[q] = par_to_H(*[M[i] for i in range(i * 3, i * 3 + 3)])
            return H_dict


        def warp(coords, H):
            # function that does the transformation
            out = cv2.perspectiveTransform(
                np.float32(coords[:, np.newaxis, :]), H)[:, 0, :]
            return out


        print("Optimizing tissue mosaic...")


        def loss_fn(M, quadrants, anchor, data_dict, histogram_dists, max_size, alpha=0.1, d=32):
            # M is a list of parameters for homography matrices (every three parameters is
            # converted into a homography matrix). For convenience, I maintain the upper-left
            # quadrant as the fixed image
            # alpha is a scaling factor for the misalignment loss
            hist_loss = 0.
            mis_loss = 0.

            # H_dict contains as a dictionary the transformation matrices for each quadrant excluding the anchor
            H_dict = M_to_quadrant_dict(M, quadrants, anchor)

            for idx in range(len(data_dict)):
                # for each quadrant retrieves tha parameters
                # for each pair of fragment (quadrant) with indces i and j
                i, j = idx, (idx + 1) % len(data_dict)

                data1 = data_dict[i]
                data2 = data_dict[j]
                q1 = data1['quadrant']
                q2 = data2['quadrant']

                axis1 = data1['pos_line']
                axis2 = data2['ant_line']
                points1 = data1['pos_points']
                points2 = data2['ant_points']
                hist_dist = histogram_dists[i, j]["pos_ant"]

                # if index q1 quadrant is not the anchor retrieve the transformed edges of the fragment and the
                # transformed points coordinates
                if q1 in H_dict and q1 != anchor:
                    axis1 = warp(axis1, H_dict[q1])
                    points1 = warp(points1, H_dict[q1])
                if q2 in H_dict and q2 != anchor:
                    axis2 = warp(axis2, H_dict[q2])
                    points2 = warp(points2, H_dict[q2])

                # term of the final loss function related to the misalignment
                mis_loss += np.mean((axis1 / max_size - axis2 / max_size) ** 2)

                # samples some indices (1/4 of the total number of points) on the two fragment edges
                # (note that points1 are the 'pos_points' on fragment1 and points2 are the 'ant_points' on fragment2,
                # so they are the points that should match during the stitching)
                ss1 = np.random.choice(len(points1),
                                       size=len(points1) // 4,
                                       replace=False)
                ss2 = np.random.choice(len(points2),
                                       size=len(points2) // 4,
                                       replace=False)
                point_distance = cdist(points1[ss1], points2[ss2])
                nx, ny = np.where(point_distance < d)
                nx, ny = ss1[nx], ss2[ny]
                hnb = hist_dist[nx, ny]
                if hnb.size > 0:
                    h = np.mean(hnb)
                else:
                    h = 0.
                # penalty if points have no nearby pixels
                max_dist = hist_dist.max()
                no_nearby_x = point_distance.shape[0] - np.unique(nx).size
                no_nearby_x = no_nearby_x / point_distance.shape[0]
                no_nearby_y = point_distance.shape[1] - np.unique(ny).size
                no_nearby_y = no_nearby_y / point_distance.shape[1]
                hist_loss += h + (no_nearby_x + no_nearby_y) * max_dist

            # print("mis_loss: ", mis_loss)
            # print("hist_loss: ", hist_loss)
            loss = (mis_loss) * alpha + (1 - alpha) * hist_loss
            return loss


        quadrant_list = list(range(len(data_dict)))
        anchor = len(data_dict) - 1  # the last (UL) quadrant is used as an anchor
        bounds = []
        # initialize image positions to the edges where they *should* be
        x0 = []
        a, b, _ = [x['image'] for x in data_dict if x['quadrant'] == anchor][0].shape
        for q in quadrant_list:
            if q != anchor:
                bounds.extend(
                    [(-np.pi / 6, np.pi / 6),
                     (-output_size[0], output_size[0]),
                     (-output_size[0], output_size[0])])

                x0.extend([0, a, b])

        # Tracking progress
        progress = []

        # Al momento non utilizzata, ma se ne può parlare
        def reconstruct_mosaic(H_dict, anchor, data_dict):
            # Create an empty canvas large enough to hold the stitched image
            canvas_height = output_size[0] * 2
            canvas_width = output_size[1] * 2
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

            # Loop over all the quadrants
            for idx, data in enumerate(data_dict):
                img = data['image']
                quadrant = data['quadrant']

                # Debug: Print current homography matrix
                print(f"Quadrant: {quadrant}")
                if quadrant in H_dict:
                    print(f"Homography matrix for quadrant {quadrant}:")
                    print(H_dict[quadrant])

                    # Apply homography if it's not the anchor
                    H = np.array(H_dict[quadrant], dtype=np.float32)  # Ensure it's np.float32
                    try:
                        warped_img = cv2.warpPerspective(img, H, (canvas_width, canvas_height))
                    except cv2.error as e:
                        print(f"Error warping quadrant {quadrant}: {e}")
                        continue
                else:
                    # Anchor image doesn't get transformed
                    warped_img = img

                # Calculate the new position of the warped image based on the homography matrix
                h, w = img.shape[:2]
                pts = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype="float32").reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, H) if quadrant in H_dict else pts

                # Get the bounding box of the transformed points
                x_min, y_min = np.int32(dst.min(axis=0).ravel())
                x_max, y_max = np.int32(dst.max(axis=0).ravel())

                # Adjust the bounding box to fit the canvas
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(canvas_width, x_max)
                y_max = min(canvas_height, y_max)

                # Place the warped image onto the canvas
                canvas[y_min:y_max, x_min:x_max] = warped_img[:(y_max - y_min), :(x_max - x_min)]

            return canvas

        # Gran parte della funzione non usata, ma se ne può parlare
        def cb(xk, convergence):
            # Compute the loss as before
            loss_value = loss_fn(xk, quadrant_list, anchor, data_dict, histogram_dists, output_size[0] / 100, 0.1,
                                 square_size // 2)
            progress.append(loss_value)

            # Generate the homography dictionary from the current parameters
            # H_dict = M_to_quadrant_dict(xk, quadrant_list, anchor)

            # Reconstruct the mosaic with current transformations
            # mosaic = reconstruct_mosaic(H_dict, anchor, data_dict)

            # Save the mosaic image for this iteration
            # iteration_num = len(progress)
            # mosaic_filename = os.path.join(root_folder, 'debug', f'mosaic_iter_{iteration_num}.png')
            # cv2.imwrite(mosaic_filename, mosaic)

            # Optionally display the mosaic image (can slow down the optimization if the display is too frequent)
            # plt.imshow(cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))
            # plt.title(f'Iteration {iteration_num} - Loss: {loss_value}')
            # plt.show()

        de_result = optimize.differential_evolution(
            loss_fn, bounds, popsize=POPSIZE, maxiter=MAXITER, disp=True, x0=x0,
            mutation=[0.2, 1.0], seed=42, callback=cb,
            args=[quadrant_list, anchor, data_dict, histogram_dists, output_size[0] / 100, 0.1, square_size // 2])

        print(de_result)

        # Convert progress to numpy array
        progress = np.array(progress)

        # Plot progress
        plt.plot(progress, label='Loss over iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Optimization Progress (Loss)')
        plt.legend()
        plt.grid(True)

        # Save plot
        plt.savefig(os.path.join(root_folder, 'debug', 'Andamento_loss.png'))
        # plt.show()

        output_size = max([max(x.shape) for x in images_original]) * 2
        # output_size = max([max(x.shape) for x in images_original])
        output_size = [output_size, output_size]
        output = np.zeros((*output_size, 3), dtype=np.int32)
        output_d = np.zeros((*output_size, 3), dtype=np.int32)

        sc = np.array(
            [[1, 1, DOWNSAMPLE_LEVEL], [1, 1, DOWNSAMPLE_LEVEL], [0, 0, 1]])

        axis_1_final = []
        axis_2_final = []
        H_dict = M_to_quadrant_dict(
            de_result.x, quadrant_list, anchor)

        for i, data in enumerate(data_dict):
            image = images_original[i][:, :, :3]
            sh = image.shape
            mask = resize(data['tissue_mask'], [sh[0], sh[1]], order=0)
            q = data['quadrant']
            if q in H_dict:
                print(q)
                im = cv2.warpPerspective(
                    np.uint8(image * mask[:, :, np.newaxis]),
                    sc * H_dict[q], output_size)

                axis_1_q = data['pos_line']
                axis_1_q = warp(axis_1_q, H_dict[q])
                print("AXIS1", axis_1_q)
                axis_2_q = data['ant_line']
                axis_2_q = warp(axis_2_q, H_dict[q])
                print("AXIS2", axis_2_q)
                axis_1_final.append(axis_1_q)
                axis_2_final.append(axis_2_q)

                del axis_1_q  # Libera memoria
                del axis_2_q  # Libera memoria
            else:
                print(q)
                im = cv2.warpPerspective(
                    np.uint8(image * mask[:, :, np.newaxis]),
                    par_to_H(0, 0, 0), output_size)
                axis_1_anchor = data['pos_line']
                axis_1_anchor = warp(axis_1_anchor, par_to_H(0, 0, 0))
                print("AXIS1_anchor", axis_1_anchor)
                axis_2_anchor = data['ant_line']
                axis_2_anchor = warp(axis_2_anchor, par_to_H(0, 0, 0))
                print("AXIS2_anchor", axis_2_anchor)
                axis_1_final.append(axis_1_anchor)
                axis_2_final.append(axis_2_anchor)

                del axis_1_anchor  # Libera memoria
                del axis_2_anchor  # Libera memoria

            # this piece of code combines the final fragment images while taking into account the overlap of the
            # valid pixel (pixels that are not just background)
            if q in quadrant_list:
                output[im[:, :, :] > 0] += im[im[:, :, :] > 0]
                output_d[im[:, :, :] > 0] += 1

        # DEBUGGING
        # plt.figure()
        # plt.imshow(output)
        # plt.savefig(os.path.join(root_folder, 'debug', 'pre_output0_custom.png'))
        plt.imsave(os.path.join(root_folder, 'debug', 'pre_output0_custom.png'), output.astype('uint8'))

        # Converti output_d in un'immagine a singolo canale sommando lungo l'asse del colore (RGB)
        output_d_single_channel = np.sum(output_d, axis=2)

        # Visualizza l'immagine con una colormap
        plt.figure()
        plt.imshow(output_d_single_channel, cmap='hot')  # Usa una colormap per evidenziare i valori
        plt.colorbar()  # Aggiungi una barra per visualizzare la scala dei valori
        plt.savefig(os.path.join(root_folder, 'debug', 'output_d_colormap_fixed.png'))
        # plt.close()



        print("Ho finito il ciclo")

        euclidean_distance_0_1_center = np.sqrt(
            (axis_2_final[1][0][0] - axis_1_final[0][0][0]) ** 2 + (axis_2_final[1][0][1] - axis_1_final[0][0][1]) ** 2)
        euclidean_distance_0_1_out = np.sqrt(
            (axis_2_final[1][1][0] - axis_1_final[0][1][0]) ** 2 + (axis_2_final[1][1][1] - axis_1_final[0][1][1]) ** 2)
        euclidean_distance_1_2_center = np.sqrt(
            (axis_2_final[2][0][0] - axis_1_final[1][0][0]) ** 2 + (axis_2_final[2][0][1] - axis_1_final[1][0][1]) ** 2)
        euclidean_distance_1_2_out = np.sqrt(
            (axis_2_final[2][1][0] - axis_1_final[1][1][0]) ** 2 + (axis_2_final[2][1][1] - axis_1_final[1][1][1]) ** 2)
        euclidean_distance_2_3_center = np.sqrt(
            (axis_2_final[3][0][0] - axis_1_final[2][0][0]) ** 2 + (axis_2_final[3][0][1] - axis_1_final[2][0][1]) ** 2)
        euclidean_distance_2_3_out = np.sqrt(
            (axis_2_final[3][1][0] - axis_1_final[2][1][0]) ** 2 + (axis_2_final[3][1][1] - axis_1_final[2][1][1]) ** 2)
        euclidean_distance_3_0_center = np.sqrt(
            (axis_2_final[0][0][0] - axis_1_final[3][0][0]) ** 2 + (axis_2_final[0][0][1] - axis_1_final[3][0][1]) ** 2)
        euclidean_distance_3_0_out = np.sqrt(
            (axis_2_final[0][1][0] - axis_1_final[3][1][0]) ** 2 + (axis_2_final[0][1][1] - axis_1_final[3][1][1]) ** 2)

        average_euclidean_distance_units = (
                                                   euclidean_distance_0_1_center + euclidean_distance_0_1_out + euclidean_distance_1_2_center
                                                   + euclidean_distance_1_2_out + euclidean_distance_2_3_center + euclidean_distance_2_3_out
                                                   + euclidean_distance_3_0_center + euclidean_distance_3_0_out) / 8

        # Cambio sfondo da nero a bianco
        output[output == 0] = 255
        # plt.figure()
        # plt.imshow(output)
        # plt.savefig(os.path.join(root_folder, 'debug', 'pre_output1_0_255_custom.png'))
        plt.imsave(os.path.join(root_folder, 'debug', 'pre_output1_0_255_custom.png'), output.astype('uint8'))

        print(output_d.shape)
        # if output_d > 1 --> output/output_d
        # else: output_d <= 1 --> output
        # Ma siccome output_d è un'immagine nera, output rimane uguale! NO, c'è qualcosa che mi sfugge
        output = np.where(output_d > 1, output / output_d, output)
        # plt.figure()
        # plt.imshow(output)
        plt.savefig(os.path.join(root_folder, 'debug', 'pre_output2_after_output-output_d.png'))

        # output[np.sum(output, axis = -1) > 650] = 0
        output = output.astype(np.uint8)
        plt.imsave(os.path.join(root_folder, 'debug', 'pre_output3_npuint8.png'), output)

        # imageio.imwrite("/Users/anacastroverde/Desktop/output_black.tif", output, format="tif")

        reader = WSIReader.open(output)
        info_dict = reader.info.as_dict()
        bounds = [0, 0, info_dict['level_dimensions'][0][0] - int(sub_bound_x),
                  info_dict['level_dimensions'][0][1] - int(
                      sub_bound_y)]  # y-550 #To remove the excessive white space around the output image
        region = reader.read_bounds(bounds, resolution=0, units="level", coord_space="resolution")
        #
        # original_spacing = (float(args.original_spacing), float(args.original_spacing))
        # # new_spacing_x = original_size[0]*original_spacing[0]/new_size[0]
        # # new_spacing_y = original_size[1]*original_spacing[1]/new_size[1]
        new_spacing = (2 ** int(level)) * float(original_spacing)  # *(10**(-3))
        #
        tifffile.imwrite(os.path.join(save_dir, "output_custom.tif"), np.array(region), photometric='rgb', imagej=True,
                         resolution=(1 / new_spacing, 1 / new_spacing), metadata={'spacing': new_spacing, 'unit': 'um'})
        # # imageio.imwrite(args.output_path+"output.tif", output, format="tif")

        average_euclidean_distance_mm = average_euclidean_distance_units * new_spacing * (10 ** (-3))
        # print('Average Euclidean Distance between corner points:', round(average_euclidean_distance_mm, 2),
        # 'millimeters')

        end_time = time.time()
        elapsed_time = end_time - start_time
        work_time = end_time - start_stiching_time

        print("Oh qui ho modificato la loss!")

        print(f"Execution time: {work_time} seconds")
        # print('Total execution time of algorithm:', round(elapsed_time, 2), 'seconds')
