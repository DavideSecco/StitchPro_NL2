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
from skimage.transform import AffineTransform
from scipy.optimize import NonlinearConstraint
from scipy import optimize
import matplotlib.pyplot as plt
import tifffile as tiff
import subprocess

# IMPORT OUR CLASSES
from utilities import Preprocessing, Line, Image_Lines, saving_functions, cutter
from utilities.optimization_function import *
from shredder.main_wout_multi_and_masks import * 

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
    # Controlla se la cartella contiene un singolo file con estensione .tif, .tiff o .svs
    valid_extensions = ('.tif', '.tiff', '.svs')
    image_files = [f for f in os.listdir(dataset_folder) if f.endswith(valid_extensions)]
    
    real_cuts = True
    if len(image_files) == 1:
        single_file_path = os.path.join(dataset_folder, image_files[0])
        print("Passata cartella con un singolo file .tif/.tiff/.svs")

        if real_cuts == True:
            histo_fragment_ul, histo_fragment_ur, histo_fragment_ll, histo_fragment_lr = Shredder(pathlib.Path(single_file_path), None, 0, 4).final_fragments
        else: 
            histo_fragment_ul, histo_fragment_ur, histo_fragment_ll, histo_fragment_lr = cutter.cut_image(tiff.imread(single_file_path))
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
original_spacing = 0.25
level = 5
sub_bound_x = 550
sub_bound_y = 550


# Set seed
np.random.seed(42)

# App description
start_time = time.time()

# Dovuto cambiare condizione per aggiungere la possibilità di tagliare le'immagine in input
# if (img_file_buffer_ur is not None) & (img_file_buffer_lr is not None) & (img_file_buffer_ll is not None) & (img_file_buffer_ul is not None):
if True:

    images_original = [histo_fragment_ur, histo_fragment_lr, histo_fragment_ll, histo_fragment_ul]



    if True:
        start_stiching_time = time.time()

        tissue_masks = []
        tissue_masks_closed = []
        images = []

        for fragment in images_original:
            processor = Preprocessing(fragment)
            preprocessed = processor.preprocess_image(show_steps=False)
            images.append(preprocessed['histo_fragment'])
            tissue_masks.append(preprocessed['image_thresholded_filtered_closed'])
            tissue_masks_closed.append(preprocessed['canny_edges'])

        data_dict = []
        N = len(tissue_masks)

        for i in range(len(tissue_masks)):
            print(f"Processing {files[i]} fragment ...\n")
            x = tissue_masks[i]
            x_out = tissue_masks_closed[i]

            # DEBUGGING
            # create folder
            save_dir_image_i = os.path.join(save_dir, files[i])
            try:
                os.makedirs(save_dir_image_i, exist_ok=True)
                print(f"Cartella '{save_dir_image_i}' creata con successo")
            except OSError as e:
                print(f"Errore nella creazione della cartella: {e}")

            if show_all_images:
                plt.figure(figsize=(50, 50))
                plt.imshow(x, cmap='gray')  # DEBUGGING
                plt.savefig(os.path.join(save_dir_image_i, f'{files[i]}_tissue_mask.png'))
                plt.figure(figsize=(50, 50))
                plt.imshow(x_out, cmap='gray')
                plt.savefig(os.path.join(save_dir_image_i, f'{files[i]}_tissue_mask_closed.png'))

            ######################### INIZIO NOSTRA IMPLEMENTAZIONE ############################################

            # print("Tipi immagine: ", type(images[i][2, 3]), type(tissue_masks_closed[i][2, 3]))
            try:
                image_lines = Image_Lines(images[i], tissue_masks_closed[i], save_dir_image_i)
            except Exception as e:
                print(e)

                # Crea un dizionario con i risultati
                result_data = {
                    "dataset": folder_name,
                    "success": 'False - Not found perp lines',  # Esito dell'ottimizzazione
                    "fun": 'undefined',  # Valore della funzione obiettivo
                    "average_euclidean_distance_mm": 'undefined',
                    "work_time": 'undefined'  # Tempo di esecuzione
                }

                saving_functions.salva_in_csv(result_data, Path(save_dir).parent, "results_custom.csv")

                saving_functions.salva_in_json(result_data, save_dir, "results_custom.json")

                sys.exit(1)

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

            # saves the data_dict of our implementation in json format
            saving_functions.save_data_dict(data_dict, i, save_dir_image_i)

            # saves important information of data_dict (debugging)
            saving_functions.save_images_data_dict(data_dict, i, save_dir_image_i)

        ###################################################
        ################ Seconda parte ####################
        ###################################################

        # from here on it starts working on the real image
        # calculates the histograms for all the edges (ant and pos) of every fragment
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

        # calculates the correlation distances between all combinations of ant and pos points
        # e.g. distance between ant_points histogram of fragment 1 and pos_points histogram of fragment 2
        # (they are matching edges!)
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

        # ensuring that max distance == 1, i.e. normalization by the maximum value of distance
        for i, j in histogram_dists:
            histogram_dists[i, j]['ant_pos'] = np.divide(
                histogram_dists[i, j]['ant_pos'], histogram_dists[i, j]['ant_pos'].max())
            histogram_dists[i, j]['pos_ant'] = np.divide(
                histogram_dists[i, j]['pos_ant'], histogram_dists[i, j]['pos_ant'].max())

        # Optimization of tissue segment translations and rotations via differential evolution
        output_size = max([max(x['image'].shape) for x in data_dict]) * 2
        output_size = [output_size, output_size]
        print('output_size', output_size)

        print("Optimizing tissue mosaic...")

        # list to keep track of the fragment position
        quadrant_list = list(range(len(data_dict)))
        # defining the anchor fragment (the one that does not move)
        anchor = len(data_dict) - 1  # the last (UL) quadrant is used as an anchor

        # initialize image positions to the edges where they *should* be
        # defines buonds and x0, parameters to be passed to the optimizer
        bounds = []
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
            loss_value = loss_fn(xk, quadrant_list, anchor, data_dict, histogram_dists, output_size[0] / 100, 0.1,
                                 square_size // 2)
            progress.append(loss_value)

        # OPTIMIZATION IS HERE
        de_result = optimize.differential_evolution(
            loss_fn, bounds, popsize=POPSIZE, maxiter=MAXITER, disp=True, x0=x0,
            mutation=[0.2, 1.0], seed=42, callback=cb,
            args=[quadrant_list, anchor, data_dict, histogram_dists, output_size[0] / 100, 0.08, square_size // 2])

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

        # starts to reconstruct the final image based on the optimization
        output_size = max([max(x.shape) for x in images_original]) * 2
        # output_size = max([max(x.shape) for x in images_original])
        output_size = [output_size, output_size]
        output = np.zeros((*output_size, 3), dtype=np.int32)
        output_d = np.zeros((*output_size, 3), dtype=np.int32)

        # sets a scaling array to be multiplied to the transformation matrix to account for original downsampling
        sc = np.array(
            [[1, 1, DOWNSAMPLE_LEVEL], [1, 1, DOWNSAMPLE_LEVEL], [0, 0, 1]])

        axis_1_final = []
        axis_2_final = []

        # creates the dict containing the final transformation matrices for all the fragments
        H_dict = M_to_quadrant_dict(
            de_result.x, quadrant_list, anchor)

        for i, data in enumerate(data_dict):
            image = images_original[i][:, :, :3]
            sh = image.shape
            mask = resize(data['tissue_mask'], [sh[0], sh[1]], order=0)
            q = data['quadrant']

            if q in H_dict:  # if the current quadrant is not the anchor
                # applies the scaled transformation (accounting for downsampling) to the filtered fragment
                # (no background)
                im = cv2.warpPerspective(
                    np.uint8(image * mask[:, :, np.newaxis]),
                    sc * H_dict[q], output_size)

                # applies the transformation to the lines (ant and pos) through the warp custom function
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
            else:  # if it is the anchor
                print('Anchor fragment number: ', q)

                # applies the identity transformation to the filtered fragment
                im = cv2.warpPerspective(
                    np.uint8(image * mask[:, :, np.newaxis]),
                    par_to_H(0, 0, 0), output_size)

                # applies the identity transformation to the lines
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
        # plt.imsave(os.path.join(save_dir, 'pre_output0_custom.png'), output.astype('uint8'))

        # Converti output_d in un'immagine a singolo canale sommando lungo l'asse del colore (RGB)
        output_d_single_channel = np.sum(output_d, axis=2)

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

        # Cambio sfondo da nero a bianco
        output[output == 0] = 255
        # plt.imsave(os.path.join(save_dir, 'pre_output1_0_255_custom.png'), output.astype('uint8'))

        # print(output_d.shape)
        # if output_d > 1 --> output/output_d
        # else: output_d <= 1 --> output
        # Ma siccome output_d è un'immagine nera, output rimane uguale! NO, c'è qualcosa che mi sfugge
        output = np.where(output_d > 1, output / output_d, output)
        # plt.figure()
        # plt.imshow(output)
        # plt.savefig(os.path.join(save_dir, 'pre_output2_after_output-output_d.png'))

        # output[np.sum(output, axis = -1) > 650] = 0
        output = output.astype(np.uint8)
        plt.imsave(os.path.join(save_dir, 'pre_output3_npuint8.png'), output)

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
        # Uncomment is you want the tif image
        # tifffile.imwrite(os.path.join(save_dir, "output_custom.tif"), np.array(region), photometric='rgb', imagej=True, resolution=(1 / new_spacing, 1 / new_spacing), metadata={'spacing': new_spacing, 'unit': 'um'})
        # # imageio.imwrite(args.output_path+"output.tif", output, format="tif")

        average_euclidean_distance_mm = average_euclidean_distance_units * new_spacing * (10 ** (-3))
        # print('Average Euclidean Distance between corner points:', round(average_euclidean_distance_mm, 2),
        # 'millimeters')

        end_time = time.time()
        elapsed_time = end_time - start_time
        work_time = end_time - start_stiching_time

        # Crea un dizionario con i risultati
        result_data = {
            "dataset": folder_name,
            "success": de_result.success,  # Esito dell'ottimizzazione
            "fun": de_result.fun,  # Valore della funzione obiettivo
            "average_euclidean_distance_mm" : average_euclidean_distance_mm,
            "work_time": work_time  # Tempo di esecuzione
        }

        saving_functions.salva_in_csv(result_data, Path(save_dir).parent, "results_custom.csv")

        saving_functions.salva_in_json(result_data, save_dir, "results_custom.json")

        print(result_data)
        # print('Total execution time of algorithm:', round(elapsed_time, 2), 'seconds')

