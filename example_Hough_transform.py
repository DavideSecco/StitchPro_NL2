import sys
import argparse
import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from skimage.draw import line as draw_line
from skimage import data


from utilities import Preprocessing

class Line():
    def __init__(self, angle, dist, image):
        self.x0 = dist * np.cos(angle)
        self.y0 = dist * np.sin(angle)
        self.angle = angle

        # Se angle è diverso da 0, calcola la pendenza come np.tan(angle + np.pi / 2)
        # Altrimenti imposta slope a np.inf
        if np.isclose(angle, 0, atol=1e-3):
            self.slope = np.inf
        else:
            self.slope = np.tan(angle + (np.pi / 2))
            # self.slope = np.tan(self.angle)

        # self.valid_points = self.get_cordinates_points(img_width=image.shape[1], img_height=image.shape[0])
        self.extreme_points = self.get_extreme_points(img_width=image.shape[1], img_height=image.shape[0]) # NON FUNZIONA, SI PUÒ ELIMINARE
        self.cordinate_points = self.extract_line_pixels(img_width=image.shape[1], img_height=image.shape[0])

    # Onnstamenta questa non mi piace, penso di aver fatto la stessa funzione qualche commit fa: LENGTH COSA FA?
    # NON FUNZIONA --> da eliminare o da chiedere a Paolo
    def get_cordinates_points(self, img_width, img_height, length = 1000):
        # Definizione della direzione della linea
        dx = np.cos(np.arctan(self.slope))
        dy = np.sin(np.arctan(self.slope))

        # Generazione di punti simmetrici rispetto al punto iniziale (x0, y0)
        t = np.linspace(-length / 2, length / 2, num=1000)  # Intervallo di parametri per ottenere punti lungo la linea

        # Calcolo delle coordinate X e Y lungo la linea
        x_coords = np.round(self.x0 + t * dx)
        y_coords = np.round(self.y0 + t * dy)

        # Verifica che i punti siano all'interno dei limiti dell'immagine
        valid_indices = (x_coords >= 0) & (x_coords < img_width) & (y_coords >= 0) & (y_coords < img_height)

        # Uso column_stack per unire le coordinate valide in un array di forma (n, 2)
        return np.column_stack((x_coords[valid_indices], y_coords[valid_indices]))

    # Anche questa forse non serve, l'informazione può essere presa da extract_line_pixels e prendendo gli ultimi pixels
    def get_extreme_points(self, img_width, img_height):
        if self.slope == np.inf:  # Caso linea verticale
            # La linea è verticale, quindi x è costante (x0) e y varia dall'inizio alla fine dell'immagine
            x1 = int(self.x0)
            y1 = 0  # Intersezione con il bordo superiore
            x2 = int(self.x0)
            y2 = img_height  # Intersezione con il bordo inferiore
        else:
            # Calcola le coordinate dei punti estremi per linee non verticali
            x1 = 0 # MA QUESTO È SBAGLIATO
            y1 = int(self.y0 - self.slope * self.x0)  # Intersezione con y all'inizio dell'immagine (x=0)

            x2 = img_width
            y2 = int(self.y0 + self.slope * (x2 - self.x0))  # Intersezione con y alla fine dell'immagine (x=width)

        # Aggiungi le coordinate calcolate come punti estremi
        return [(x1, y1), (x2, y2)]

    def extract_line_pixels(self, img_width, img_height):
        # Calcola la tangente dell'angolo

        # Inizializza una lista per salvare i punti (x, y)
        punti_linea = []

        # Se la linea è quasi verticale, iteriamo su y e calcoliamo x
        if np.isinf(self.slope):
           for y in range(0, img_height):
               x = int(self.x0)  # x è costante per una linea verticale
               if 0 <= x < img_width:
                    punti_linea.append((x, y))
        else:
            # Itera attraverso i valori di x e calcola i valori di y
            for x in range(0, img_width):
                y = int(self.y0 + self.slope * (x - self.x0))

                # Controlla se il punto (x, y) è dentro i confini dell'immagine
                if 0 <= y < img_height:
                    punti_linea.append((x, y))

        return punti_linea

    def __repr__(self):
        return (f"Linea(x0={self.x0}, y0={self.y0}, slope={self.slope}, angle={self.angle}) \n"
                f"Punti finali di tutti i punti: {self.cordinate_points[0]} {self.cordinate_points[-1]} \n"
                # f"Extreme Points: {self.extreme_points} \n"
                )

class Image_Lines():
    # Si suppone che venga passata già l'immagine preprocessed
    def __init__(self, image):
        self.image = image

        # Create of an array of angles from -90° to 90°, divided in 360 steps: are the tested angles in hough transformation.
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
        # h: La matrice di accumulo della Trasformata di Hough.
        # theta: Gli angoli testati in radianti.
        # d: Le distanze dalla linea centrale.
        self.h, self.theta, self.d = hough_line(self.image, theta=tested_angles)

        self.lines = self.hough_transform_skimage_implementation()
        print(self.lines)

        # Questa poi sarà da perfezionare, perchè dovremo selezionare le linee che vogliamo considerare
        self.ant_points, self.pos_points = self.extract_ant_pos_points()

    def hough_transform_skimage_implementation(self):
        # Trova i picchi nella trasformata di Hough
        peaks = hough_line_peaks(self.h, self.theta, self.d, threshold=0.2 * np.max(self.h))
        print("peaks: ", peaks)

        # Estrai le linee rilevate
        lines = []
        for _, angle, dist in zip(*peaks):
            lines.append(Line(angle, dist, self.image))

        return lines

    def visualize_all_lines(self, save_dir=None):
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        ax = axes.ravel()

        ax[0].imshow(self.image, cmap='gray')
        ax[0].set_title('Input image')
        ax[0].set_axis_off()

        # Io onestamente toglierei questa ax[1], anche se non so se sia rilevante,
        # non mi sono messo capire a cosa serva, ma se lo si toglie si possono togliere anche
        # self.h, self.theta, self.d dal main
        # e probabilmente raggruppare le due funzioni visualize

        angle_step = 0.5 * np.diff(self.theta).mean()
        d_step = 0.5 * np.diff(self.d).mean()
        bounds = [
            np.rad2deg(self.theta[0] - angle_step),
            np.rad2deg(self.theta[-1] + angle_step),
            self.d[-1] + d_step,
            self.d[0] - d_step,
        ]
        ax[1].imshow(np.log(1 + self.h), extent=bounds, cmap='gray', aspect=1 / 1.5)  # plotta la trasformata
        ax[1].set_title('Hough transform')
        ax[1].set_xlabel('Angles (degrees)')
        ax[1].set_ylabel('Distance (pixels)')
        ax[1].axis('image')

        ax[2].imshow(self.image, cmap='gray')
        ax[2].set_ylim((self.image.shape[0], 0))
        # ax[2].set_axis_off()
        ax[2].set_title('Detected lines')

        colors = ["red", "green", "purple", "orange", "yellow"]
        # Disegna le linee e aggiungi etichette
        for idx, line in enumerate(self.lines, start=0):
            ax[2].axline((line.x0, line.y0), slope=line.slope, color=colors[idx % len(colors)], linewidth=5 / (1 + idx))
            # Disegna un marker in (x0,y0)
            ax[2].scatter(line.x0, line.y0, color=colors[idx % len(colors)], marker='x', s=100)
            # Aggiungi il numero della linea vicino a (x0, y0)
            ax[2].text(line.x0, line.y0, f"Line {idx}", color=colors[idx % len(colors)], fontsize=12,
                       verticalalignment='top', horizontalalignment='right')

        plt.suptitle("Final result for hough_transform_skimage", fontsize=16)

        plt.tight_layout()

        if save_dir is not None:
            plt.savefig(save_dir + 'hough_skimage.png', dpi=300)

        plt.show()

    def extract_ant_pos_points(self):
        # qui utilizzo la stessa procedura che usa stichpro, lo faccio solo per visualizzare il risultato
        # disegna una maschera "ausiliaria" che copra il bordo di interesse nell'immagine (v. primo plot)
        aux_mask_ant = np.zeros_like(self.image, dtype=np.uint8)  # Per i punti dell'altro lato
        aux_mask_pos = np.zeros_like(self.image, dtype=np.uint8)  # Per i punti di un lato

        cv.line(aux_mask_pos, self.lines[0].extreme_points[0], self.lines[0].extreme_points[1], 1, 10)
        cv.line(aux_mask_ant, self.lines[1].extreme_points[0], self.lines[1].extreme_points[1], 1, 10)


        # Trovo punti sul bordo come sovrapposizione tra la maschera ausiliaria e l'immagine con i bordi (binary_image)
        ant_points = np.roll(np.array(np.where(aux_mask_ant * self.image)).T, 1, 1)
        pos_points = np.roll(np.array(np.where(aux_mask_pos * self.image)).T, 1, 1)

        return ant_points, pos_points

    def visualize_ant_pos_points(self):
        aux_mask = self.image.copy()

        aux_mask = cv.cvtColor(aux_mask, cv.COLOR_GRAY2RGB)
        # disegna i punti per sola visualizzazione
        for point in self.ant_points:
            cv.drawMarker(aux_mask, tuple(point), color=(255, 255, 255), markerType=cv.MARKER_SQUARE, markerSize=3,
                          thickness=2)

        for point in self.pos_points:
            cv.drawMarker(aux_mask, tuple(point), color=(155, 155, 15), markerType=cv.MARKER_SQUARE, markerSize=3,
                          thickness=2)

        plt.imshow(self.image, cmap='gray')
        plt.title("ant_axis_line_mask")
        plt.show()

        plt.imshow(aux_mask)  # cmap='gray'
        plt.show()


def filter_lines(image, lines):
    """
    Funzione che filtra le linee verticali ed orizzontali che si incrociano in un certo angolo della figura
    :param image: frammento
    :param lines:
    :return:
    """
    filtered_lines = []
    for line in lines:
        x0, y0, slope, angle = line
        # Controllo per le linee verticali (|θ| ≈ π/2) e orizzontali (θ ≈ 0)
        if abs(angle) < np.pi / 36 or abs(angle - np.pi / 2) < np.pi / 36:
            # Aggiunge una condizione per controllare se la linea è nell'area in basso a sinistra
            if x0 < image.shape[1] / 2 and y0 < image.shape[0] / 2:
                filtered_lines.append((x0, y0, slope, angle))

    return filtered_lines

def plot_results(processed_img, lines, save_dir, aux_mask):
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
    for idx, (x0, y0, slope, _) in enumerate(lines, start=0):
        ax[1].axline((x0, y0), slope=slope, color=colors[idx % len(colors)], linewidth=5 / (1 + idx))
        # Disegna un marker in (x0,y0)
        ax[1].scatter(x0, y0, color=colors[idx % len(colors)], marker='x', s=200)
        # Aggiungi il numero della linea vicino a (x0, y0)
        ax[1].text(x0, y0, f"Line {idx}", color=colors[idx % len(colors)], fontsize=12, verticalalignment='top',
                   horizontalalignment='right')

    ax[2].imshow(aux_mask, cmap='gray')  # cmap='gray'
    ax[2].set_title('Found borders')
    ax[2].set_axis_off()

    plt.tight_layout()
    plt.title("Final Results")
    plt.savefig(save_dir + 'Housh_skimage_final_result.png', dpi=300)
    plt.show()


def main(argv):
    parser = argparse.ArgumentParser(description="Script per line detection con trasformata di Hough.")

    # Definizione degli argomenti
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Percorso al file di input da processare."
    )

    parser.add_argument(
        "-o", "--orientation",
        type=str,
        choices=["ur", "ul", "dr", "dl"],  # Posizioni accettate
        required=True,
        help="Posizione del frammento (ur=up-right, ul=up-left, dr=down-right, dl=down-left)."
    )

    # Parsing degli argomenti
    args = parser.parse_args()

    # Validazione del percorso del file
    input_path = args.input
    orientation = args.orientation


    # Validate the file path
    try:
        # Check if the file path exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"The file at path '{input_path}' does not exist.")

        # Check if it is a file (not a directory)
        if not os.path.isfile(input_path):
            raise IsADirectoryError(f"The path '{input_path}' is not a file.")

    except (FileNotFoundError, IsADirectoryError, IndexError) as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # file_path = argv[0]
    file_name = os.path.splitext(os.path.basename(input_path))[0] # Ottieni il nome del file senza estensione
    print("Name of the file: ", file_name)
    save_dir = "./hough_trasform_results/" + file_name + '/' # Setto il nome della cartella dove salvare i risultati
    os.makedirs(save_dir, exist_ok=True)  # Crea la cartella, inclusi i genitori se non esistono


    # PreProcessing of the image and visulization
    prep = Preprocessing(input_path)
    plt.imshow(prep.original_image)
    plt.title("original_image")
    plt.show()
    print(prep.original_image.shape)
    processed_img = prep.preprocess_image(show_steps=True,
                                          apply_padding=True,
                                          median_filter_size=40,
                                          closing_footprint_size=50,
                                          apply_hull_image=False)

    # serve perché l'edge detection fatta con Canny restituisce un'immagine di tipo bool
    processed_img = processed_img.astype(np.uint8) * 255
    print(processed_img.shape)
    # Trasformation throgh hough

    image_lines = Image_Lines(processed_img)
    print(image_lines.lines)
    image_lines.visualize_all_lines()
    image_lines.visualize_ant_pos_points()
    # hough_transform(processed_img, save_dir=save_dir, show=True)
    # lines = hough_transform_skimage_implementation(processed_img, save_dir=save_dir)  # funziona molto meglio
    # Linee ritornate in formato: ((x0, y0), slope, angle)
    # lines_cordinates = get_coordinates_lines(lines[0], 1000, *processed_img.shape[:2])


    # extreme_points = calculate_extreme_points(lines, processed_img.shape)


    # print(extreme_points)

    # for line in lines:
        # print(line.extreme_points)
        # print(line.cordinate_points[0], line.cordinate_points[-1])
        # print(line)
    # print(lines[1].extreme_points)
    # print(lines[1].valid_points[0])
    # print(lines[1].valid_points[-1])

    #print("linee trovate: ", len(lines), lines)
    #print("extreme points: ", extreme_points)

    #aux_mask = processed_img.copy()  # Per mettere insieme tutti i risultati, per visualizzare
    #extract_ant_pos_points(processed_img, extreme_points, aux_mask)
    #plot_results(processed_img, lines, save_dir, aux_mask)


    return 0


if __name__ == "__main__":
    main(sys.argv[1:])