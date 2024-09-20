import sys
import argparse
import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
import math


class Line():
    """
    Class used to represent a Line

    Attributes
    -------------------------
    x0, y0: float/int
        cordinates in which the line pass
    angle: float
        angle in radiant between the orizonatal line and the line (going down)
    dist: float
        distance between the line and the origin
    slope: float
        slope of the line
    cordinates points: vectors of (int, int)
        all the point cordinates of that line
    extreme points: [(int,int),(int,int)]
        the two extreme points of that line

    Methods
    -----------
    extract_line_pixels():
        estrare dati i dati della linea, i pixels che le appartegono
    """
    def __init__(self, angle, dist, image):
        self.x0 = dist * np.cos(angle)
        self.y0 = dist * np.sin(angle)
        self.dist = dist
        self.angle = angle
        self.image = image

        # if angle close 0 --> np.inf
        # otherwise --> calcola la pendenza come np.tan(angle + np.pi / 2)
        self.slope = np.inf if np.isclose(angle, 0, atol=1e-3) else np.tan(angle + (np.pi / 2))

        self.cordinate_points = self.extract_line_pixels(img_width=image.shape[1], img_height=image.shape[0])
        self.extreme_points = [self.cordinate_points[0], self.cordinate_points[-1]]

    def extract_line_pixels(self, img_width, img_height):
        """
        :param img_width:
        :param img_height:
        :return: prunti_linea: vector of (int,int)
            cordinates of the pixel that belongs to the line
        """
        # Inizializza una lista per salvare i punti (x, y)
        punti_linea = []
        # Se la linea è quasi verticale, iteriamo su y e calcoliamo x
        if self.slope > 100:  # 100 scelto dopo aver fatto un po' di prove
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

        # DEBUGGING
        # test_img = np.zeros_like(self.image, dtype=np.uint8)
        # for point in punti_linea:
        #     cv.drawMarker(test_img, point, 255, cv.MARKER_CROSS, 3)

        # plt.imshow(test_img, cmap='gray')
        # plt.show()
        return punti_linea

    def __repr__(self):
        return (f"Linea(x0={self.x0}, y0={self.y0}, slope={self.slope}, angle={self.angle}) \n"
                f"Punti finali di tutti i punti: {self.cordinate_points[0]} {self.cordinate_points[-1]} \n")

class Image_Lines():
    """
    A class to represent the fragment mask (edges), and the lines found on it. Then finds the fragment borders

    Attributes
    ------------
    image: image
        preprocessed mask (fragment edges)
    lines: vector of Line
        straight line founded
    indexes_vert_horiz_lines: vector of [int]
        selection of the indexes of mask that are horizontal or vertical
    self.ant_points, self.pos_points: vector of [int]
        points that overlap with the mask
    self.intersection: coordinates of the point of intersection between the best 2 lines

    Methods
    ------------

    """
    def __init__(self, image, save_dir):
        self.image = image
        self.save_dir = save_dir
        # QUESTO PEZZO SI PUO' SPOSTARE NELLA FUNZIONE NEL CASO SI DECIDA CHE IL GRAFICO DELLA TRASFORMATA NON SERVE
        # Create of an array of angles from -90° to 90°, divided in 360 steps: are the tested angles in hough transformation.
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
        # h: La matrice di accumulo della Trasformata di Hough.
        # theta: Gli angoli testati in radianti.
        # d: Le distanze dalla linea centrale.
        self.h, self.theta, self.d = hough_line(self.image, theta=tested_angles)

        self.lines = self.hough_transform_skimage_implementation()
        # self.indexes_vert_horiz_lines = self.filter_horizontal_vertical_lines()

        first, second = self.best_line_combination()
        self.find_intersection(first, second)

        self.ant_points = self.extract_points_on_border(first)[2]
        self.pos_points = self.extract_points_on_border(second)[2]
        self.intersection = self.find_intersection(first, second)  # returns (x,y) coordinates of the intersection
        # Questo non funziona, non è sempre l'ultimo elemento del vettore, alle volte è anche il primo
        # penso il modo per farlo sia trovare il punto nei ant_point/pos_point più lontano da intersection
        # self.end_ant_point, self.end_pos_point = (self.ant_points[-1], self.pos_points[-1])
        # Questo non è il metodo migliore per trovare la fine dei lati, ma è il più pratico
        self.end_ant_point, self.end_pos_point = self.farthest_point(self.ant_points), self.farthest_point(self.pos_points)
        # DEBUGGING
        print("end_points: ", self.end_ant_point, self.end_pos_point)

    def hough_transform_skimage_implementation(self):
        threshold = 1
        peaks = hough_line_peaks(self.h, self.theta, self.d, threshold=threshold * np.max(self.h))

        while len(peaks[0]) < 4:
            threshold *= 0.9
            peaks = hough_line_peaks(self.h, self.theta, self.d, threshold = threshold * np.max(self.h))

            print(f"Ho trovato {len(peaks[0])} peaks:", peaks)
        return [Line(angle, dist, self.image) for _, angle, dist in zip(*peaks)]

    def extract_points_on_border(self, index):

        aux_mask = np.zeros_like(self.image, dtype=np.uint8)

        # Il paramentro di thickness è fondamentale: capire quale sis il valore migliore è la chiave:
        # Per partire: (img.shape[0]+img.shape[1])/2 * 1/25
        # Media delle dimensioni dell'immagine e poi divisa per 25
        thickness = int((self.image.shape[0] + self.image.shape[1]) / 2 * (1 / 25))
        # print("thickness", thickness)
        cv.line(aux_mask, self.lines[index].extreme_points[0], self.lines[index].extreme_points[1], 1, thickness=thickness)

        points = np.roll(np.array(np.where(aux_mask * self.image)).T, 1, 1)

        return index, len(points), points

    def is_perpendicular(self, angle1, angle2):
        # 1) Calcola la differenza tra il primo angolo e l'angolo corrente ed evita che superi i 180 gradi
        angle_diff = abs(angle1 - angle2) % np.pi
        # 2) Verifica se la differenza è 90 gradi (entro una tolleranza)
        return np.isclose(angle_diff, np.pi / 2, atol=5e-1)

    def best_line_combination(self):
        """
            # TODO: possibile miglioramento: tenere in condiderazione anche il valori di peaks della funzione hough
            Function that returns the best two lines, considering the fact:
                1. the two lines must be orithontal or vertical
                2. the two lines must overlap with the mask (the more the better)
                3. the two lines must be orthogonal each other

            At the moment is not such a clever function, can be imporved.

            :return: [int, int]
                the index of the best promising lines
        """
        # 1) Filtriamo le linee che sono orizonatali e verticali
        filtered_lines_index = []
        for index, line in enumerate(self.lines, start=0):
            # x0, y0, slope, angle = line
            # Controllo per le linee verticali (|θ| ≈ π/2) e orizzontali (θ ≈ 0)
            print("line", index, "angle", self.lines[index].angle)
            if (abs(self.lines[index].angle) < np.pi / 20 or
                abs(self.lines[index].angle - np.pi / 2) < np.pi / 20 or
                abs(self.lines[index].angle + np.pi / 2) < np.pi / 20):
                # Aggiunge una condizione per controllare se la linea è nell'area in basso a sinistra
                # if x0 < self.image.shape[1] / 2 and y0 < self.image.shape[0] / 2:
                filtered_lines_index.append(index)
                print("added")
            else:
                print("Why:")
                print("Angle:", abs(self.lines[index].angle), " > np.pi/20: ", np.pi /20)
                print("Angle - np.pi/2: ", abs(self.lines[index].angle - np.pi / 2), " > np.pi/20: ", np.pi / 20)
                print("Angle + np.pi/2: ", abs(self.lines[index].angle + np.pi / 2), " > np.pi/20: ", np.pi / 20)


        print("filtered_lines_index:", filtered_lines_index)
        # 2) Ordino le linee in base a quanti pixels sono sovrapposti con il contorno
        results = []
        for index in filtered_lines_index:
            results.append(self.extract_points_on_border(index))

        # Ordina sulla base del secondo valore di ciascuna tupla
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        print("results found: ", len(results))
        print("sorted results found: ",len(sorted_results))
        print("Best lines:")
        for result in sorted_results:
            print(result[0], result[1])

        # 3) Contrllo che le linee trovate siano perperndicolari fra loro:
        for index, _, _ in sorted_results:
            if self.is_perpendicular(self.lines[sorted_results[0][0]].angle, self.lines[index].angle):
                print(f"La linea {sorted_results[0][0]} e la linea {index} sono ortogonali")
                return sorted_results[0][0], index
            else:
                print(f"La linea {sorted_results[0][0]} e la linea {index} NON sono ortogonali")

        # Invece di ritornare questo, andrebbe ritornato un messaggio di errore nel caso non si trovi nulla
        return sorted_results[0][0], sorted_results[1][0]

    def find_intersection(self, first, second):
        """
        Function that founds the intersection point between 2 lines
        https://www.geeksforgeeks.org/program-for-point-of-intersection-of-two-lines/

        :param first: first line found
        :param second: second line found (perpendicular to the first)
        :return: (x,y) coordinates of the intersection point if exists, otherwise raises exception
        """
        first_line = self.lines[first]
        second_line = self.lines[second]

        a1x, a1y = first_line.extreme_points[0]
        a2x, a2y = first_line.extreme_points[1]
        b1x, b1y = second_line.extreme_points[0]
        b2x, b2y = second_line.extreme_points[1]

        p1 = a2y - a1y
        p2 = a1x - a2x
        p3 = p1 * a1x + p2 * a1y

        q1 = b2y - b1y
        q2 = b1x - b2x
        q3 = q1 * b1x + q2 * b1y

        det = p1 * q2 - q1 * p2

        if det == 0:
            raise Exception("Linee trovate parallele: no intersezione")
        else:
            x = int((q2 * p3 - p2 * q3)/det)
            y = int((p1 * q3 - q1 * p3)/det)
            print("Intersection point found: ", x, y)

            return x, y




    # Function to find the farthest point
    def farthest_point(self, points_array):
        max_distance = 0
        farthest = None

        for point in points_array:
            distance = math.sqrt((point[0] - self.intersection[0]) ** 2 + (point[1] - self.intersection[1]) ** 2)
            if distance > max_distance:
                max_distance = distance
                farthest = point

        return farthest

    def plot_results(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        ax = axes.ravel()

        # 1 Immagine

        ax[0].imshow(self.image, cmap='gray')
        ax[0].set_title('Input image')
        ax[0].set_axis_off()

        # Io onestamente toglierei questa ax[1], anche se non so se sia rilevante,
        # non mi sono messo capire a cosa serva, ma se lo si toglie si possono togliere anche
        # self.h, self.theta, self.d dal main
        # e probabilmente raggruppare le due funzioni visualize

        # angle_step = 0.5 * np.diff(self.theta).mean()
        # d_step = 0.5 * np.diff(self.d).mean()
        # bounds = [
        #    np.rad2deg(self.theta[0] - angle_step),
        #    np.rad2deg(self.theta[-1] + angle_step),
        #    self.d[-1] + d_step,
        #    self.d[0] - d_step,
        # ]
        # ax[1].imshow(np.log(1 + self.h), extent=bounds, cmap='gray', aspect=1 / 1.5)  # plotta la trasformata
        # ax[1].set_title('Hough transform')
        # ax[1].set_xlabel('Angles (degrees)')
        # ax[1].set_ylabel('Distance (pixels)')
        # ax[1].axis('image')

        # 2 Immagine:
        cv.drawMarker(self.image, self.intersection, (200, 255, 200), cv.MARKER_STAR, 10, 4)
        ax[1].imshow(self.image, cmap='gray')
        ax[1].set_ylim((self.image.shape[0], 0))
        # ax[2].set_axis_off()
        ax[1].set_title('Detected lines')

        colors = ["red", "green", "purple", "orange", "yellow"]
        # Disegna le linee e aggiungi etichette
        for idx, line in enumerate(self.lines, start=0):
            ax[1].axline((line.x0, line.y0), slope=line.slope, color=colors[idx % len(colors)], linewidth=5 / (1 + idx))
            # Disegna un marker in (x0,y0)
            ax[1].scatter(line.x0, line.y0, color=colors[idx % len(colors)], marker='x', s=100)
            # Aggiungi il numero della linea vicino a (x0, y0)
            ax[1].text(line.x0, line.y0, f"Line {idx}", color=colors[idx % len(colors)], fontsize=12,
                       verticalalignment='top', horizontalalignment='right')


        # Immagine 3
        aux_mask = self.image.copy()
        aux_mask = cv.cvtColor(aux_mask, cv.COLOR_GRAY2RGB)
        # disegna i punti per sola visualizzazione
        for point in self.ant_points:
            cv.drawMarker(aux_mask, tuple(point), color=(255, 255, 255), markerType=cv.MARKER_SQUARE, markerSize=3,
                          thickness=2)

        for point in self.pos_points:
            cv.drawMarker(aux_mask, tuple(point), color=(155, 155, 15), markerType=cv.MARKER_SQUARE, markerSize=3,
                          thickness=2)

        ax[2].imshow(aux_mask)
        ax[2].set_title("Contorni")


        plt.suptitle("Final result for hough_transform_skimage", fontsize=16)

        plt.tight_layout()

        if self.save_dir is not None:
            plt.savefig(self.save_dir + 'hough_skimage.png', dpi=300)
        else:
            print("La cartella di salvataggio NON è stata impostata")

        plt.show()


def main():
    from preprocessing import Preprocessing
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

    file_name = os.path.splitext(os.path.basename(input_path))[0] # Ottieni il nome del file senza estensione
    print("Name of the file: ", file_name)
    save_dir = "./hough_trasform_results/" + file_name + '/' # Setto il nome della cartella dove salvare i risultati
    os.makedirs(save_dir, exist_ok=True)  # Crea la cartella, inclusi i genitori se non esistono

    # PreProcessing of the image and visulization
    prep = Preprocessing(input_path)
    plt.imshow(prep.original_image)
    plt.title("original_image")
    plt.show()
    print("original_imag shape: ", prep.original_image.shape)
    processed_img = prep.preprocess_image(show_steps=False,
                                          apply_padding=True,
                                          median_filter_size=40,
                                          closing_footprint_size=50,
                                          apply_hull_image=False)

    # serve perché l'edge detection fatta con Canny restituisce un'immagine di tipo bool
    processed_img = processed_img.astype(np.uint8) * 255
    print("preprocessed_img shape: ", processed_img.shape)

    # Trasformation through hough
    image_lines = Image_Lines(processed_img, save_dir)
    image_lines.plot_results()

    return 0


if __name__ == "__main__":
    main()