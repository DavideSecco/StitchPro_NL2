import sys
import math
import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from skimage.draw import line as draw_line
from skimage import data


from utilities import Preprocessing


def hough_transform(img, save_dir, show=False):
    """
    Funzione che applica la trasformata di Hough su un immagine binaria, restituisce le immagini con sopra evidenziate
    le linee dritte trovate con la trasformata di Hough e la trasformata di Hough probabilistica
    le linee dritte
    :param img: immagine
    :param show:
    :param save:
    :return:
    """
    if len(img.shape) == 2:
        cdst = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cdstP = np.copy(cdst)

    lines = cv.HoughLines(img, 1, np.pi / 180, 150, None, 0, 0)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]

            # condizione per considerare unicamente le linee quasi orizzontali e quasi verticali
            if abs(theta) < np.pi / 36 or abs(theta - np.pi / 2) < np.pi / 36 or abs(theta - np.pi) < np.pi / 36:
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 2000 * (a)))  # estremo linea x + k * dx, k=1000 fattore di scala
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 2000 * (a)))
                cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)
                draw_vertex_lines(cdst, x0, y0, a, b)
    else:
        # raise ValueError("Nessuna linea trovata")
        print("Nessuna linea trovata")

    linesP = cv.HoughLinesP(img, 1, np.pi / 180, 50, None, 50, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            # Calcola la pendenza della linea per filtrare le linee diagonali
            dx = l[2] - l[0]
            dy = l[3] - l[1]
            if dx == 0 or dy == 0 or abs(dy / dx) < 0.1:  # linee verticali, orizzontali o quasi orizzontali
                cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

    if show:
        # cv.imshow("Source", img)
        # cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
        # cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

        plt.figure(figsize=(15, 5))  # Imposta la dimensione della figura (opzionale)

        # Prima immagine
        plt.subplot(1, 3, 1)  # (1 riga, 3 colonne, 1ª posizione)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title("Source")

        # Seconda immagine
        plt.subplot(1, 3, 2)  # (1 riga, 3 colonne, 2ª posizione)
        plt.imshow(cdst, cmap='gray')
        plt.axis('off')
        plt.title("Detected Lines (in red) - Standard Hough Line Transform")

        # Terza immagine
        plt.subplot(1, 3, 3)  # (1 riga, 3 colonne, 3ª posizione)
        plt.imshow(cdstP, cmap='gray')
        plt.axis('off')
        plt.title("Detected Lines (in red) - Probabilistic Line Transform")

        plt.suptitle("Final result for hough_transform", fontsize=16)

        # Optimize layout and show the images
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for suptitle

        plt.savefig(save_dir + 'hough_standard.png', dpi=300)

        plt.show()

    # cv.waitKey()
    return 0



def hough_transform_skimage_implementation(image, save_dir):
    # image = np.zeros((200, 200))
    # idx = np.arange(25, 175)
    # image[idx, idx] = 255
    # image[draw_line(45, 25, 25, 175)] = 255
    # image[draw_line(25, 135, 175, 155)] = 255

    # Create of an array of angles from -90° to 90°, divided in 360 steps.
    # These are the tested angles in hough transformation.
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)

    # h: La matrice di accumulo della Trasformata di Hough.
    # theta: Gli angoli testati in radianti.
    # d: Le distanze dalla linea centrale.
    h, theta, d = hough_line(image, theta=tested_angles)

    # Trova i picchi nella trasformata di Hough
    peaks = hough_line_peaks(h, theta, d)
    # print("peaks: ", peaks)

    # Estrai le linee rilevate
    lines = []
    for _, angle, dist in zip(*peaks):
        x0, y0 = dist * np.array([np.cos(angle), np.sin(angle)])
        # Se angle è diverso da 0, calcola la pendenza come np.tan(angle + np.pi / 2)
        # Altrimenti imposta slope a np.inf
        if angle != 0:
            slope = np.tan(angle + np.pi / 2)
        else:
            slope = np.inf
        lines.append((x0, y0, slope))

    debug = False

    if debug:
        # Generating figure 1
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        ax = axes.ravel()

        ax[0].imshow(image, cmap='gray')
        ax[0].set_title('Input image')
        ax[0].set_axis_off()

        angle_step = 0.5 * np.diff(theta).mean()
        d_step = 0.5 * np.diff(d).mean()
        bounds = [
            np.rad2deg(theta[0] - angle_step),
            np.rad2deg(theta[-1] + angle_step),
            d[-1] + d_step,
            d[0] - d_step,
        ]
        ax[1].imshow(np.log(1 + h), extent=bounds, cmap='gray', aspect=1 / 1.5)
        ax[1].set_title('Hough transform')
        ax[1].set_xlabel('Angles (degrees)')
        ax[1].set_ylabel('Distance (pixels)')
        ax[1].axis('image')

        ax[2].imshow(image, cmap='gray')
        ax[2].set_ylim((image.shape[0], 0))
        # ax[2].set_axis_off()
        ax[2].set_title('Detected lines')

        colors = ["red", "green", "purple", "orange", "yellow"]
        # Disegna le linee e aggiungi etichette
        for idx, (x0, y0, slope) in enumerate(lines, start=0):
            ax[2].axline((x0, y0), slope=slope, color=colors[idx % len(colors)], linewidth=5/(1+idx))
            # Aggiungi il numero della linea vicino a (x0, y0)
            ax[2].text(x0, y0, f"Line {idx}", color=colors[idx % len(colors)], fontsize=12, verticalalignment='top', horizontalalignment='right')


        plt.suptitle("Final result for hough_transform_skimage", fontsize=16)

        plt.tight_layout()

        plt.savefig(save_dir + 'hough_skimage.png', dpi=300)
        plt.show()

        return lines,

    return lines


def draw_vertex_lines(img, x0, y0, a, b):
    img = img.copy()

    cv.drawMarker(img, (int(x0), int(y0)), (0, 255, 0), cv.MARKER_CROSS, 20, 2)

    arrow_length = 50
    pt2 = (int(x0 + arrow_length * a), int(y0 + arrow_length * b))
    cv.arrowedLine(img, (int(x0), int(y0)), pt2, (0, 0, 255), 2, tipLength=0.3)

    plt.imshow(img)
    plt.show()


### Non necessaria
def extract_line_pixels(image, lines):
    # Create an empty mask to store the line pixels
    line_pixels = np.zeros(image.shape, dtype=np.uint8)

    print(image.shape)

    for x0, y0, slope in lines:
        # Generate coordinates along the line
        if slope == np.inf or slope > 10^12:  # Vertical line case
            x = np.full(image.shape[0], x0)
            # y = y0
            y = np.arange(image.shape[0])
        else:
            x = np.arange(image.shape[1])
            y = slope * (x - x0) + y0

        # Ensure coordinates are within image bounds
        valid = (x >= 0) & (x < image.shape[1]) & (y >= 0) & (y < image.shape[0])
        x_valid = x[valid].astype(int)
        y_valid = y[valid].astype(int)

        # Update the mask
        line_pixels[y_valid, x_valid] = 1

    return line_pixels


def calculate_extreme_points(lines, image_shape):
    extreme_points = []
    height, width = image_shape

    # Calcola i punti estremi per ogni linea
    for (x0, y0, slope) in lines:
        if slope == np.inf:  # Caso linea verticale
            # La linea è verticale, quindi x è costante (x0) e y varia dall'inizio alla fine dell'immagine
            x1 = int(x0)
            y1 = 0  # Intersezione con il bordo superiore
            x2 = int(x0)
            y2 = height  # Intersezione con il bordo inferiore
        else:
            # Calcola le coordinate dei punti estremi per linee non verticali
            x1 = 0
            y1 = int(y0 - slope * x0)  # Intersezione con y all'inizio dell'immagine (x=0)

            x2 = width
            y2 = int(y0 + slope * (x2 - x0))  # Intersezione con y alla fine dell'immagine (x=width)

        # Aggiungi le coordinate calcolate come punti estremi
        extreme_points.append([(x1, y1), (x2, y2)])

    return extreme_points


def main(argv):
    # Validate the file path
    try:
        # Check id a path has been inserted
        if len(argv) == 0:
            raise IndexError("Must insert a valid file path")

        # Check if the file path exists
        if not os.path.exists(argv[0]):
            raise FileNotFoundError(f"The file at path '{argv.input_path}' does not exist.")

        # Check if it is a file (not a directory)
        if not os.path.isfile(argv[0]):
            raise IsADirectoryError(f"The path '{argv.input_path}' is not a file.")

    except (FileNotFoundError, IsADirectoryError, IndexError) as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    file_path = argv[0]

    # Ottieni il nome del file senza estensione
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    print("Name of the file: ", file_name)

    # Setto il nome della cartella dove salvare i risultati
    save_dir = "./hough_trasform_results/" + file_name + '/'

    # Crea la cartella, inclusi i genitori se non esistono
    os.makedirs(save_dir, exist_ok=True)


    # PreProcessing of the image and visulization
    prep = Preprocessing(file_path)
    plt.imshow(prep.original_image)
    plt.show()

    processed_img = prep.preprocess_image(show_steps=False,
                                          median_filter_size=30,
                                          closing_footprint_size=30,
                                          apply_hull_image=False)

    # serve perché l'edge detection fatta con Canny restituisce un'immagine di tipo bool
    processed_img = processed_img.astype(np.uint8) * 255

    # debugging
    # print(np.unique(processed_img))
    # print(processed_img.shape)
    # print(processed_img.dtype)

    # Trasformation throgh hough

    hough_transform(processed_img, save_dir=save_dir, show=True)
    lines = hough_transform_skimage_implementation(processed_img, save_dir=save_dir)  # funziona molto meglio

    # Linee ritornate in formato: ((x0, y0), slope)
    print("linee trovate: ", len(lines), lines)

    extreme_points = calculate_extreme_points(lines, processed_img.shape)
    print("extreme points: ", extreme_points)

    # qui utilizzo la stessa procedura che usa stichpro, lo faccio solo per visualizzare il risultato
    # disegna una maschera "ausiliaria" che copra il bordo di interesse nell'immagine (v. primo plot)
    aux_mask_ant = np.zeros_like(processed_img, dtype=np.uint8) # Per i punti dell'altro lato
    aux_mask_pos = np.zeros_like(processed_img, dtype=np.uint8) # Per i punti di un lato
    aux_mask = processed_img.copy()     # Per mettere insieme tutti i risultati, per visualizzare

    cv.line(aux_mask_ant, extreme_points[1][0], extreme_points[1][1], 1, 10)
    cv.line(aux_mask_pos, extreme_points[0][0], extreme_points[0][1], 1, 10)


    # Trovo punti sul bordo come sovrapposizione tra la maschera ausiliaria e l'immagine con i bordi (binary_image)
    ant_points = np.roll(np.array(np.where(aux_mask_ant * processed_img)).T, 1, 1)
    pos_points = np.roll(np.array(np.where(aux_mask_pos * processed_img)).T, 1, 1)

    aux_mask = cv.cvtColor(aux_mask, cv.COLOR_GRAY2RGB)
    # disegna i punti per sola visualizzazione
    for point in ant_points:
        cv.drawMarker(aux_mask, (point), color=(255,255,255), markerType=cv.MARKER_SQUARE, markerSize=3, thickness=2)

    for point in pos_points:
        cv.drawMarker(aux_mask, (point), color=(155,155,15), markerType=cv.MARKER_SQUARE, markerSize=3, thickness=2)

    # cv.drawMarker(aux_mask, (cx, cy), (159), markerType=cv.MARKER_CROSS, markerSize=9, thickness=3)  # disegno centro
    # cv.drawMarker(aux_mask, (rx, ry), (150), markerType=cv.MARKER_CROSS, markerSize=5, thickness=3)  # disegno raggio
    # cv.drawMarker(aux_mask, (point_ant[0][::-1]), (100), markerType=cv.MARKER_CROSS, markerSize=9, thickness=3)  # disegno punto piú vicino rispetto al punto (rx, rY)

    plt.imshow(processed_img, cmap='gray')
    plt.title("ant_axis_line_mask")
    plt.show()

    plt.imshow(aux_mask) # cmap='gray'
    plt.show()

    # queste diventano inutile (le uso solo per visualizzare)
    line_pixels = extract_line_pixels(processed_img, lines)
    print(line_pixels)

    # Plot the results
    fig, ax = plt.subplots(1, 3, figsize=(12, 6))

    ax[0].imshow(processed_img, cmap='gray')
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    ax[1].imshow(processed_img, cmap='gray')
    ax[1].set_ylim((processed_img.shape[0], 0))
    ax[1].set_title('Detected lines')

    colors = ["red", "green", "purple", "orange", "yellow"]
    # Disegna le linee e aggiungi etichette
    for idx, (x0, y0, slope) in enumerate(lines, start=0):
        ax[1].axline((x0, y0), slope=slope, color=colors[idx % len(colors)], linewidth=5 / (1 + idx))
        # Aggiungi il numero della linea vicino a (x0, y0)
        ax[1].text(x0, y0, f"Line {idx}", color=colors[idx % len(colors)], fontsize=12, verticalalignment='top',
                   horizontalalignment='right')

    ax[2].imshow(aux_mask) # cmap='gray'
    ax[2].set_title('Found borders')
    ax[2].set_axis_off()

    plt.tight_layout()
    plt.title("Final Results")
    plt.savefig(save_dir + 'Housh_skimage_final_result.png', dpi=300)
    plt.show()

    return 0


if __name__ == "__main__":
    main(sys.argv[1:])