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
import sys
from pathlib import Path
import tifffile as tiff
import matplotlib.pyplot as plt # DEBUGGING

# IMPORT OUR CLASSES
from utilities import Preprocessing, Line, Image_Lines, saving_functions, cutter
from utilities.optimization_function import *

# IMPORT OUR CLASSES
from utilities import Preprocessing, Line, Image_Lines
from utilities.optimization_function import *

# Names
files = ["upper_right", "bottom_right", "bottom_left", "upper_left"]
root_folder = os.getcwd()
print(f"root_folder: {root_folder}")

########################################
### Leggo percorso delle immagini ######
########################################

# Possibilità 1: Passo il percorso della cartella tramite command line
parser = argparse.ArgumentParser(description="Script per caricare e ruotare immagini")
parser.add_argument(
    '--input_path',
    type=str,
    default=None,  # Se non viene fornito dall'utente, rimane None
    help="Percorso opzionale per le immagini"
)

parser.add_argument(
    '--save_dir',
    type=str,
    default=None,  # Se non viene fornito dall'utente, rimane None
    help="Percorso dove salvare i risultati",
    required=True
)

# Parsing degli argomenti
args = parser.parse_args()

save_dir = args.save_dir

dataset_folder_command_line = args.input_path
dataset_folder_Davide = "/mnt/Volume/Mega/LaureaMagistrale/CorsiSemestre/A2S1/MultudisciplinaryProject/data/Dataset_00"
dataset_folder_Paolo = r"C:\Users\dicia\NL2_project\datasets\test-data-corretto"
dataset_folder_Kaggle = "/kaggle/input"

if os.path.exists(dataset_folder_command_line):
    dataset_folder = dataset_folder_command_line
elif os.path.exists(dataset_folder_Davide):
    dataset_folder = dataset_folder_Davide
elif os.path.exists(dataset_folder_Paolo):
    dataset_folder = dataset_folder_Paolo
elif os.path.exists(dataset_folder_Kaggle):
    dataset_folder = dataset_folder_Kaggle

print(f"Percorso del dataset che analizzo: {dataset_folder}")

# Controllo che i file dei frammenti esistano all'interno della cartella
try:
    # Controlla se la cartella contiene solo un file .tif
    tif_files = [f for f in os.listdir(dataset_folder) if f.endswith('.tif')]
    if len(tif_files) == 1:
        single_file_path = os.path.join(dataset_folder, tif_files[0])
        print("Passata cartella con un singolo file .tif")
        histo_fragment_ur, histo_fragment_lr, histo_fragment_ll, histo_fragment_ul = cutter.cut_image(tiff.imread(single_file_path))
    else:
        # Percorsi dei file
        img_file_buffer_ur = os.path.join(dataset_folder, 'upper_right.tif')
        img_file_buffer_lr = os.path.join(dataset_folder, 'bottom_right.tif')
        img_file_buffer_ll = os.path.join(dataset_folder, 'bottom_left.tif')
        img_file_buffer_ul = os.path.join(dataset_folder, 'upper_left.tif')

        # Lista dei file da controllare
        required_files = [img_file_buffer_ur, img_file_buffer_lr, img_file_buffer_ll, img_file_buffer_ul]

        # Verifica che i file richiesti esistano
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Il file {file_path} non esiste nella cartella {dataset_folder}.")

        # Read the images with name histo_fragment_[pos]
        histo_fragment_ur = imageio.imread(img_file_buffer_ur)
        histo_fragment_lr = imageio.imread(img_file_buffer_lr)
        histo_fragment_ll = imageio.imread(img_file_buffer_ll)
        histo_fragment_ul = imageio.imread(img_file_buffer_ul)

        print(f"Tutti i file richiesti esistono nella cartella {dataset_folder}.")

except FileNotFoundError as e:
    # Gestisce l'eccezione, stampa l'errore e termina il programma
    print(f"Errore: {e}")
    sys.exit(1)  # Esce con un codice di errore

# Ottengo il nome del dataset, cosi da poter successivamente creare una sottocartella in debug con lo stesso nome
folder_name = os.path.basename(os.path.normpath(dataset_folder))

# create folder for debugging
save_dir = os.path.join(save_dir, folder_name)

print("Cartella di salvataggio: ", save_dir)

# Se la cartella di debug del dataset esiste già, non rifaccio il lavoro
if os.path.exists(save_dir):
    print(f"\nLa cartella '{save_dir}' esiste già, non rifaccio l'ottimizzazione.")
    print("Se si vuole rifare l'ottimizzazione, cancellare la cartella")
    sys.exit(0)

try:
    os.makedirs(save_dir, exist_ok=True)
    print(f"Cartella '{save_dir}' creata con successo")
except OSError as e:
    print(f"Errore nella creazione della cartella: {e}")

for i in range(len(files)):
    # DEBUGGING
    # create folder
    save_dir_image_i = os.path.join(save_dir, files[i])
    try:
        os.makedirs(save_dir_image_i, exist_ok=True)
        print(f"Cartella '{save_dir_image_i}' creata con successo")
    except OSError as e:
        print(f"Errore nella creazione della cartella: {e}")


# FLAGS and GLOBALS
show_all_images = False
square_size = 64
n_bins = 32
POPSIZE = 25  # parameter for the evolutionary optimization (population size)
MAXITER = 200  # parameter for the evolutionary optimization (population size)
DOWNSAMPLE_LEVEL = 4  # avoid running the optimization for the entire image

# Set seed
np.random.seed(42)

# App description
start_time = time.time()

# if (img_file_buffer_ur is not None) & (img_file_buffer_lr is not None) & (img_file_buffer_ll is not None) & (img_file_buffer_ul is not None):
if True:
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

    if show_all_images:
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

    if show_all_images:
        plt.imshow(image_thresholded_ul, cmap="gray")
        plt.savefig(os.path.join(save_dir, f'upper_left_segmentation.png'))
        plt.imshow(image_thresholded_ur, cmap="gray")
        plt.savefig(os.path.join(save_dir, f'upper_right_segmentation.png'))
        plt.imshow(image_thresholded_lr, cmap="gray")
        plt.savefig(os.path.join(save_dir, f'lower_right_segmentation.png'))
        plt.imshow(image_thresholded_ll, cmap="gray")
        plt.savefig(os.path.join(save_dir, f'lower_left_segmentation.png'))

    ## Apply median filter to reduce the noise
    median_filter_ur_x = 20
    median_filter_lr_x = 20
    median_filter_ll_x = 20
    median_filter_ul_x = 20
    image_thresholded_filtered_ul = ndi.median_filter(image_thresholded_ul, size=int(median_filter_ul_x))
    image_thresholded_filtered_ur = ndi.median_filter(image_thresholded_ur, size=int(median_filter_ur_x))
    image_thresholded_filtered_ll = ndi.median_filter(image_thresholded_ll, size=int(median_filter_ll_x))
    image_thresholded_filtered_lr = ndi.median_filter(image_thresholded_lr, size=int(median_filter_lr_x))

    ## Erode the image to eliminate holes
    closing_ur_x = 15
    closing_lr_x = 15
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

    # if st.button("Start stitching!")==True:
    if True:
        start_stiching_time = time.time()

       # par is the array of the solution to be optimized
       # HO CAMBIATO DA 20 A 200 IL PAD
       # L'HO CAMBIATO DA 200 a 1000
        def circle_arc_loss_cv(par, mask, pad=20, save=False, name='best_ellipse_solution.png'):
            mask = cv2.copyMakeBorder(
                mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
            cx, cy, r1, r2, theta_1, theta_plus = par
            cx, cy = cx + pad, cy + pad
            theta_2 = theta_1 + theta_plus
            theta_1, theta_2 = np.rad2deg([theta_1, theta_2])
            O = np.zeros_like(mask)
            cv2.ellipse(O, (int(cy), int(cx)), (int(r1), int(r2)), angle=0,
                        startAngle=theta_1, endAngle=theta_2, color=1,
                        thickness=-1)

            # this is just for debugging purposes
            if save:
                plt.figure(figsize=(50, 50))
                plt.imshow(O, cmap='gray')
                plt.savefig(name)

            fn = np.sum((mask == 1) & (O == 0))
            I = np.sum(O * mask)
            U = np.sum(O + mask) - I
            return 1 - I / U + fn / mask.sum()


        def calculate_intersection(m1, b1, m2, b2):
            cx = (b2 - b1) / (m1 - m2)
            cy = m1 * cx + b1
            return cx, cy


        def pol2cart(rho, phi):
            # from https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
            x = rho * np.cos(phi)
            y = rho * np.sin(phi)
            return x, y


        def radius_at_angle(theta, a, b):
            #  formula to find the radius of an ellipse at a certain angle theta
            #  https://www.quora.com/How-do-I-find-the-radius-of-an-ellipse-at-a-given-angle-to-its-axis
            n = a * b
            d = np.sqrt((a * np.sin(theta)) ** 2 + (b * np.cos(theta)) ** 2)
            return n / d


        from scipy.optimize import NonlinearConstraint
        from scipy import optimize
        # Inizialmente POP_SIZE era 10
        POP_SIZE = 50

        data_dict = []

        # Tissue masks are the 'closed-masks' (see the iamge saved for details)
        tissue_masks = [image_thresholded_filtered_closed_ur, image_thresholded_filtered_closed_lr,
                        image_thresholded_filtered_closed_ll, image_thresholded_filtered_closed_ul]

        images = [histo_fragment_ur, histo_fragment_lr, histo_fragment_ll, histo_fragment_ul]

        # Tissue masks closed are the images containing the edges only
        tissue_masks_closed = [canny_edges_ur, canny_edges_lr, canny_edges_ll, canny_edges_ul]

        N = len(tissue_masks)

        curr_theta = -np.pi
        extra_theta = 2 * np.pi

        N_segments = len(images)
        segment_angle = 2 * np.pi / N_segments



        for i in range(len(tissue_masks)):
            x = tissue_masks[i]
            x_out = tissue_masks_closed[i]
            x = x.copy().astype(np.uint8)

            # DEBUGGING
            # create folder
            save_dir_image_i = os.path.join(save_dir, files[i])

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





            # Trova il contorno e lo restituisce sotto forma di tupla in cui ogni elemento contiene il contorno,
            # 'RETR_EXTERNAL' permette di avere il contorno più esterno (quello della figura in teoria) in c[0]
            c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            x = np.zeros_like(x)
            Mx, My = x.shape


            # print('n.ro pixel immagine binaria: ', Mx, My)

            # Trova la 'forma' dell'immagine attraverso i punti sul contorno
            M = np.maximum(Mx, My)
            hull = cv2.convexHull(c[0])

            # print('hull utilizzata: ', hull.shape)

            # x now is a completely filled image (without the holes present in tissue_masks) with a contour given by
            # the convex hull (it is not a precise contour but just a shape)
            cv2.drawContours(x, [hull], -1, 1, -1)

            # DEBUGGING
            plt.figure(figsize=(50, 50))
            plt.imshow(x, cmap='gray')
            plt.savefig(os.path.join(save_dir_image_i, f'{files[i]}_debugging_x_contours.png'))

            # points variable is a nx2 array containing the 2D-coordinates of the n points in the filled-(contour)-image
            # x that hare white (i.e. satisfy the condition x>0)
            points = np.stack(np.where(x > 0), axis=1)
            # print(points.shape)

            # points out is the same as point but for the tissue_mask_closed image (the edges)
            points_out = np.stack(np.where(x_out > 0), axis=1)
            # print(points_out.shape)

            # initially it was int(M*0.1)
            pad = np.maximum(int(M * 0.4), 100)

            # initialization has inside the coordinates of the
            # initializations = []

            # qui ho cambiato le inizializzazioni
            # for init_x in [0, Mx]:
            #     for init_y in [My, 0]:
            #         initializations.append((init_y, init_x))

            initializations = [
                (0, My),
                (0, 0),
                (Mx, 0),
                (Mx, My)
            ]

            solutions = []
            print("Trying initialization for segment {}: {}".format(i, initializations[i]))

            # curr_theta -np.pi/8 = -9/8*pi
            # curr_theta + extra_theta = pi
            # segment angle is the angle of the n quadrants (e.g. if 4 quadrants, segment angle is 2*pi/4)
            # Delta rappresenta quanto vuoi permettere al centro di muoversi rispetto alla posizione iniziale
            delta = 20
            initial_cx, initial_cy = initializations[i]

            bounds = [
                (max(0, initial_cx - delta), min(M, initial_cx + delta)),  # Limita cx vicino a initial_cx
                (max(0, initial_cy - delta), min(M, initial_cy + delta)),  # Limita cy vicino a initial_cy
                (M / 4, 3*M / 4),  # Semiasse r1 (può essere più ristretto se necessario)
                (M / 4, 3*M / 4),  # Semiasse r2 (può essere più ristretto se necessario)
                (curr_theta - np.pi / 8, curr_theta + extra_theta),  # Angolo iniziale
                (segment_angle * 0.8, segment_angle * 1.2)  # Ampiezza dell'arco
            ]

            x0 = [initializations[i][0], initializations[i][1], M / 2, M / 2,
                  curr_theta, segment_angle]

            # differential_evolution optimizes a properly defined loss function (see circle_arc_loss_cv
            # implementation

            # MAXITER WAS SET TO 250 originally and workers was set to 1
            solution = optimize.differential_evolution(
                circle_arc_loss_cv, x0=x0, bounds=bounds,
                args=[x, pad], popsize=POP_SIZE, maxiter=500, workers=1, seed=42)  # updating='immediate'
            solutions.append(solution)
            # print('\n', solution)
            # solution.x is the array of parameter of the optimized solution
            # circle_arc_loss_cv(solution.x, x, pad, show=False)

            # print([s.fun for s in solutions])

            # this is to save the entire series of ellipse solutions generated
            debug_this = True
            if debug_this:
                new_save_dir_image_i = os.path.join(save_dir_image_i, 'solutions_series')
                try:
                    os.makedirs(new_save_dir_image_i, exist_ok=True)
                    print(f"Cartella '{new_save_dir_image_i}' creata con successo")
                except OSError as e:
                    print(f"Errore nella creazione della cartella: {e}")
            os.chdir(new_save_dir_image_i)
            for y, sol in enumerate(solutions):
                circle_arc_loss_cv(sol.x, x, pad, save=True, name=f'img_{y}')


            # looking for the best solution
            solution_idx = np.argmin([s.fun for s in solutions])
            solution = solutions[solution_idx]

            # Mi muovo avanti e indietro nella cartella perché cosi salvo l'immagine nella posizione giusta
            os.chdir(save_dir_image_i)
            circle_arc_loss_cv(solution.x, x, pad, save=True)

            # retrieving the parameters of the best solution
            # theta_1 is the start angle at which the first side of the ellipse portion is found, theta_2 is the final
            # angle at which the other side of the ellipse portion is found, theta_plus is the delta angle
            # (theta_2-theta_1)
            cy, cx, r1, r2, theta_1, theta_plus = solution.x
            theta_2 = theta_1 + theta_plus
            curr_theta = theta_1
            extra_theta = theta_plus
            theta = theta_plus
            theta_1_, theta_2_ = np.rad2deg([theta_1, theta_2])
            O = images[i].copy()

            ##########################
            ####### Punti ant ########


            # retrieve points from extremal landmarks
            r_ant = radius_at_angle(theta_1, r1, r2)

            # extreme point on the theoretical ellipse at angle theta_1
            points_cart_ant = np.array([pol2cart(r_ant, theta_1)]) + [cx, cy]

            # pca_sorted contains the y coordinate first and then the x coordinate
            pca_sorted = np.array([points_cart_ant[0][::-1]])

            # extreme point on the real fragment at angle theta_1 (the nearest point laying on the fragment wrt
            # points_cart_ant
            point_ant = points_out[
                np.argmin(cdist(points_out, pca_sorted), axis=0)]

            ##########################

            ##########################
            ####### Punti pos ########

            # same as points_cart_ant but for angle theta_2 (see saved image for explaination)
            r_pos = radius_at_angle(theta_2, r1, r2)
            points_cart_pos = np.array([pol2cart(r_pos, theta_2)]) + [cx, cy]
            pcp_sorted = np.array([points_cart_pos[0][::-1]])
            point_pos = points_out[
                np.argmin(cdist(points_out, pcp_sorted[::-1]), axis=0)]

            ##########################

            ##########################
            ####### Punti ant ######## sembra, ma nell'ultima iscruzione li chimata pos

            # retrieve points for histogram
            ant_axis_line_mask = np.zeros_like(x, dtype=np.uint8)
            cv2.line(ant_axis_line_mask, [int(cx), int(cy)],
                     np.int32(point_ant[0][::-1]), 1, 80)
            # np.roll with those arguments just inverts x and y coordinates for each coordinate
            pos_points = np.roll(
                np.array(np.where(ant_axis_line_mask * x_out)).T, 1, 1)

            ##########################
            ####### Punti pos ######## sembra, ma nell'ultima iscruzione li chimata ant

            pos_axis_line_mask = np.zeros_like(x, dtype=np.uint8)
            cv2.line(pos_axis_line_mask, [int(cx), int(cy)],
                     np.int32(point_pos[0][::-1]), 1, 80)
            ant_points = np.roll(
                np.array(np.where(pos_axis_line_mask * x_out)).T, 1, 1)

            ### DEBUGGING
            debug = True
            if debug:
                xx = images[i].copy()
                for pointt in pos_points:
                    cv2.drawMarker(xx, pointt, [255, 0, 0], cv2.MARKER_CROSS, thickness=1)

                for pointt in ant_points:
                    cv2.drawMarker(xx, pointt, [255, 255, 0], cv2.MARKER_CROSS, thickness=1)

                cv2.drawMarker(xx, [int(cx), int(cy)], [0, 0, 0], cv2.MARKER_STAR, thickness=4)
                cv2.line(xx, [int(cx), int(cy)], np.int32(point_ant[0][::-1]), [255, 128, 128], 1)
                cv2.line(xx, [int(cx), int(cy)], np.int32(point_pos[0][::-1]), [128, 255, 128], 1)
                plt.imshow(xx)
                plt.savefig(os.path.join(save_dir_image_i, f'{files[i]}pos_points_and_ant_points.png'))


            # draw examples
            cv2.ellipse(O, (int(cx), int(cy)), (int(r1), int(r2)), angle=0,
                        startAngle=theta_1_, endAngle=theta_2_, color=1,
                        thickness=5)
            cv2.drawMarker(O, [int(cx), int(cy)], [0, 255, 0],
                           markerType=cv2.MARKER_CROSS, thickness=10)

            cv2.drawMarker(O, np.int32(points_cart_ant[0]), [0, 0, 255],
                           markerType=cv2.MARKER_CROSS, thickness=10)
            cv2.drawMarker(O, np.int32(points_cart_pos[0]), [0, 0, 255],
                           markerType=cv2.MARKER_TRIANGLE_UP, thickness=10)

            # The third argument is the extreme point with x and y coordinates inverted
            cv2.line(O, [int(cx), int(cy)], np.int32(point_ant[0][::-1]), [255, 128, 128], 1)
            cv2.line(O, [int(cx), int(cy)], np.int32(point_pos[0][::-1]), [128, 255, 128], 1)

            cv2.drawMarker(O, point_ant[0][::-1], [255, 0, 0],
                           markerType=cv2.MARKER_CROSS, thickness=10)
            cv2.drawMarker(O, point_pos[0][::-1], [255, 0, 0],
                           markerType=cv2.MARKER_TRIANGLE_UP, thickness=10)

            plt.figure(figsize=(50, 50))
            plt.imshow(O)
            plt.savefig(os.path.join(save_dir_image_i, f'{files[i]}_semi_output_example.png'))

            data_dict.append({
                "image": images[i],
                "tissue_mask": tissue_masks[i],
                "tissue_mask_closed": tissue_masks_closed[i],
                "quadrant": i,
                "ant_line": np.array([[int(cx), int(cy)], np.int32(point_ant[0][::-1])]),
                "pos_line": np.array([[int(cx), int(cy)], np.int32(point_pos[0][::-1])]),
                "ant_points": ant_points,
                "pos_points": pos_points})

            # print("Riassunto dati trovati trovati per quadrante:", data_dict[i]["quadrant"])
            # print("image shape:", data_dict[i]["image"].shape)
            # print("tissue_mask:", data_dict[i]["tissue_mask"].shape)
            # print("tissue_mask_closed:", data_dict[i]["tissue_mask_closed"].shape)
            # print("ant_line:", data_dict[i]["ant_line"])
            # print("pos_line", data_dict[i]["pos_line"])
            # print("data_dict: ", data_dict)

            # COnversion needed since you can't save a nparray in a json file
            data_dict[i]['image'] = data_dict[i]['image'].tolist()
            data_dict[i]['tissue_mask'] = data_dict[i]['tissue_mask'].tolist()
            data_dict[i]['tissue_mask_closed'] = data_dict[i]['tissue_mask_closed'].tolist()
            data_dict[i]['ant_line'] = data_dict[i]['ant_line'].tolist()
            data_dict[i]['pos_line'] = data_dict[i]['pos_line'].tolist()
            data_dict[i]['ant_points'] = data_dict[i]['ant_points'].tolist()
            data_dict[i]['pos_points'] = data_dict[i]['pos_points'].tolist()

            # Save as json
            with open(os.path.join(save_dir_image_i, "data_dict_vanilla.json"), "w") as file:
                # json.dump(data_dict[i], file)
                json.dump(dict((k, data_dict[i][k]) for k in ['ant_line', 'pos_line', 'ant_points', 'pos_points']),
                          file,
                          separators=(',', ': '))

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
                cv2.drawMarker(aux_mask, tuple(point), color=(255, 255, 0), markerType=cv2.MARKER_SQUARE, markerSize=3, thickness=2)

            for point in data_dict[i]['pos_points']:
                cv2.drawMarker(aux_mask, tuple(point), color=(255, 0, 0), markerType=cv2.MARKER_SQUARE, markerSize=3, thickness=2)

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

            plt.savefig(os.path.join(save_dir_image_i, 'data_dict_vanilla'), dpi=300)

            plt.close('all')

        ###################################################
        ################ Seconda parte ####################
        ###################################################
        ## Calculate histograms and distances between histograms

        histograms = []
        print("Calculating histograms for all sections along the x and y edges...")
        for i in range(len(data_dict)):
            # data becomes the new parameter dictionary for image i !!!
            data = data_dict[i]
            hx, hy = [], []
            for y, x in data["ant_points"]:
                # calculates histograms around ant_points of a given square_size
                H = calculate_histogram(
                    data['image'], data['tissue_mask'], [x, y], n_bins, square_size)
                # ant_points are associated to x
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
        # computes the combinations from 1 to 4
        for i, j in combinations(range(len(data_dict) + 1), 2):
            # combinations returns: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
            # MA PERCHÈ TUTTE TUTTE LE COMBINAZIONI? QUELLE OPPOSTE NON DOVREBBERO ESSERCI
            i = i % len(data_dict)
            j = j % len(data_dict)

            # key of this dictionary is the tuple (i,j) with the indices of the combination
            # cdist = calcola le distanze tra due insiemi di vettori (istogrammi), restituendo una matrice di distanze
            histogram_dists[i, j] = {
                # distances related to ant vs pos histogarms and pos vs ant histograms for the same combination
                "ant_pos": cdist(
                    data_dict[i]['histograms_ant'], data_dict[j]['histograms_pos'],
                    metric="correlation"),
                "pos_ant": cdist(
                    data_dict[i]['histograms_pos'], data_dict[j]['histograms_ant'],
                    metric="correlation")}

        # ensuring that max distance == 1 (normalizzo)
        for i, j in histogram_dists:
            # normalization of distances of paired histograms by each maximum
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

        def cb(xk, convergence):
            # Non ho ben capito come funziona questo callback
            # Computes the loss and keeps track
            loss_value = loss_fn(xk, quadrant_list, anchor, data_dict, histogram_dists, output_size[0] / 100,
                                 0.1,
                                 square_size // 2)
            progress.append(loss_value)

        de_result = optimize.differential_evolution(
            loss_fn, bounds, popsize=POPSIZE, maxiter=MAXITER, disp=True, x0=x0,
            mutation=[0.2, 1.0], seed=42,
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
        plt.savefig(os.path.join(save_dir, 'Andamento_loss.png'))

        output_size = max([max(x.shape) for x in images_original]) * 2
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


            # this piece of code combines the final fragment images while taking into account the overlap of the
            # valid pixel (pixels that are not just background)
            if q in quadrant_list:
                output[im[:, :, :] > 0] += im[im[:, :, :] > 0]
                output_d[im[:, :, :] > 0] += 1

        # DEBUGGING
        # plt.imsave(os.path.join(save_dir, 'pre_output0_vanilla.png'), output.astype('uint8'))

        # Converti output_d in un'immagine a singolo canale sommando lungo l'asse del colore (RGB)
        output_d_single_channel = np.sum(output_d, axis=2)

        # Visualizza l'immagine con una colormap
        # Visualizza l'immagine con una colormap
        from matplotlib.colors import ListedColormap, BoundaryNorm

        # Definizione dei colori per ciascun valore
        colors = [(1, 1, 1),  # 0: Bianco
                  (0, 1, 0),  # 1: Verde (no sovrapposizioni)
                  (1, 1, 0),  # 2: Giallo (due frammenti si sovrappongono)
                  (1, 0.65, 0),  # 3: Arancione (tre frammenti si sovrappongono)
                  (1, 0, 0)]  # 4: Rosso (quattro frammenti si sovrappongono)

        # Creazione di una colormap discreta
        cmap_custom = ListedColormap(colors)
        norm = BoundaryNorm([0, 2, 4, 6, 8, 10], cmap_custom.N)  # Confini per ogni livello di colore
        print(output_d_single_channel)
        print(output_d_single_channel)
        # Visualizzazione dell'immagine con la nuova colormap
        plt.figure()
        plt.imshow(output_d_single_channel, cmap=cmap_custom, norm=norm)  # Applica la colormap discreta
        plt.colorbar(label='Livello di sovrapposizione', ticks=[0, 2, 4, 6, 8],
                     format='%1.0f')  # Barra con tick centrati sui livelli
        plt.title("Sovrapposizione: bianco -> verde -> giallo -> arancione -> rosso")
        plt.axis('off')  # Rimuove gli assi per pulizia visiva
        plt.savefig(os.path.join(save_dir, 'output_d_colormap_discrete.png'))
        plt.show()
        # plt.close()

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

        output[output == 0] = 255
        output = np.where(output_d > 1, output / output_d, output)
        # output[np.sum(output, axis = -1) > 650] = 0
        output = output.astype(np.uint8)
        plt.imsave(os.path.join(save_dir, 'pre_output3_npuint8.png'), output)

        #imageio.imwrite("/Users/anacastroverde/Desktop/output_black.tif", output, format="tif")

        reader = WSIReader.open(output)
        info_dict = reader.info.as_dict()
        bounds = [0, 0, info_dict['level_dimensions'][0][0] - int(sub_bound_x),
                  info_dict['level_dimensions'][0][1] - int(
                       sub_bound_y)]  # y-550 #To remove the excessive white space around the output image
        region = reader.read_bounds(bounds, resolution=0, units="level", coord_space="resolution")
        #
        #original_spacing = (float(args.original_spacing), float(args.original_spacing))
        # # new_spacing_x = original_size[0]*original_spacing[0]/new_size[0]
        # # new_spacing_y = original_size[1]*original_spacing[1]/new_size[1]
        new_spacing = (2 ** int(level)) * float(original_spacing)  # *(10**(-3))
        #
        # Uncomment if you want the tif image
        # tifffile.imwrite(os.path.join(save_dir, "output_vanilla.tif"), np.array(region), photometric='rgb', imagej=True, resolution=(1 / new_spacing, 1 / new_spacing), metadata={'spacing': new_spacing, 'unit': 'um'})
        # # imageio.imwrite(args.output_path+"output.tif", output, format="tif")

        average_euclidean_distance_mm = average_euclidean_distance_units * new_spacing * (10 ** (-3))
        #print('Average Euclidean Distance between corner points:', round(average_euclidean_distance_mm, 2),
              #'millimeters')

        end_time = time.time()
        elapsed_time = end_time - start_time
        work_time = end_time - start_stiching_time

        # Crea un dizionario con i risultati
        result_data = {
            "dataset": folder_name,
            "success": de_result.success,  # Esito dell'ottimizzazione
            "fun": de_result.fun,  # Valore della funzione obiettivo
            "average_euclidean_distance_mm": average_euclidean_distance_mm,
            "work_time": work_time  # Tempo di esecuzione
        }

        saving_functions.salva_in_csv(result_data, Path(save_dir).parent, "results_vanilla.csv")

        saving_functions.salva_in_json(result_data, save_dir, "results_vanilla.json")

        print(result_data)
        # print('Total execution time of algorithm:', round(elapsed_time, 2), 'seconds')

