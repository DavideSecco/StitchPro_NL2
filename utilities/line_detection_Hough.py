import sys
import argparse
import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
import math
from utilities import Preprocessing
from scipy.ndimage import binary_dilation

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
    def __init__(self, peak, angle, dist, image):
        self.x0 = dist * np.cos(angle)
        self.y0 = dist * np.sin(angle)
        self.peak = peak
        self.dist = dist
        self.angle = angle
        self.image = image

        # if angle close 0 --> np.inf
        # otherwise --> calcola la pendenza come np.tan(angle + np.pi / 2)
        self.slope = np.inf if np.isclose(angle, 0, atol=1e-3) else np.tan(angle + (np.pi / 2))

        self.cordinate_points = self.extract_line_pixels(img_width=image.shape[1], img_height=image.shape[0])
        # print(self.cordinate_points[-20:])
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
    def __init__(self, orig_image, image, save_dir):
        self.original_image = orig_image
        # se l'immagine è booleana convertila in scala di grigi
        self.image = image.astype(np.uint8) * 255 if image.dtype == np.bool_ else image
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
        print("first line:", first)
        print("second line:", second)

        self.ant_points = self.extract_points_on_border(first)[2]
        self.pos_points = self.extract_points_on_border(second)[2]
        self.intersection = self.find_intersection(first, second)  # returns (x,y) coordinates of the intersection

        # Inverti! end_ant_point va con pos_line e end_pos_points va con ant_line
        # Questo non è il metodo migliore per trovare la fine dei lati, ma è il più pratico
        # self.end_pos_point, self.end_ant_point = self.farthest_point(self.ant_points), self.farthest_point(self.pos_points)
        # print(self.end_pos_point, self.end_ant_point)
        self.end_pos_point, self.end_ant_point = self.farthest_middle_point(self.ant_points, first), self.farthest_middle_point(self.pos_points, second)
        print(self.end_pos_point, self.end_ant_point)
        # self.end_pos_point, self.end_ant_point = self.farthest_point_2(self.lines[second]), self.farthest_point_2(self.lines[first])
        # print("New proposal", n1, n2)
        # DEBUGGING
        print("end_points: ", self.end_ant_point, self.end_pos_point)
        # self.plot_results()

        self.quadrant = self.identify_quadrant()
        self.invert_line_and_points()
        # self.plot_results()

    def hough_transform_skimage_implementation(self):
        threshold = 1
        peaks = hough_line_peaks(self.h, self.theta, self.d, threshold=threshold * np.max(self.h))
        print("\nperforming hough transform...")
        while len(peaks[0]) < 4:
            threshold *= 0.9
            peaks = hough_line_peaks(self.h, self.theta, self.d, threshold = threshold * np.max(self.h))

        print(f"Ho trovato {len(peaks[0])} peaks:", peaks)
        return [Line(peak, angle, dist, self.image) for peak, angle, dist in zip(*peaks)]

    def extract_points_on_border(self, index):

        aux_mask = np.zeros_like(self.image, dtype=np.uint8)

        # Il paramentro di thickness è fondamentale: capire quale sis il valore migliore è la chiave:
        # Per partire: (img.shape[0]+img.shape[1])/2 * 1/25
        # Media delle dimensioni dell'immagine e poi divisa per 25
        thickness = int((self.image.shape[0] + self.image.shape[1]) / 2 * (1 / 15))
        # print("thickness", thickness)
        cv.line(aux_mask, self.lines[index].extreme_points[0], self.lines[index].extreme_points[1], 1, thickness=thickness)

        points = np.roll(np.array(np.where(aux_mask * self.image)).T, 1, 1)

        return index, len(points), points

    def best_line_combination(self):
        """
        Function to find the best two lines based on the following criteria:
            1. The two lines must be horizontal or vertical.
            2. The two lines must overlap with the mask as much as possible.
            3. The two lines must be orthogonal to each other.

        :return: tuple (int, int)
            Indices of the two best lines.
        """

        def lines_are_perpendicular(angle1, angle2, tol=5e-1):
            """
            Check if two angles are perpendicular (difference close to 90 degrees).

            :param angle1: float, angle of the first line
            :param angle2: float, angle of the second line
            :param tol: float, tolerance for angle difference
            :return: bool, True if the angles are perpendicular
            """
            angle_diff = abs(angle1 - angle2) % np.pi
            return np.isclose(angle_diff, np.pi / 2, atol=tol)

        def is_horizontal_or_vertical(angle, tolerance):
            """
            Check if an angle corresponds to a horizontal or vertical line.

            :param angle: float, angle of the line
            :param tolerance: float, angular tolerance
            :return: bool, True if the angle is horizontal or vertical
            """
            return (
                    abs(angle) < tolerance or
                    abs(angle - np.pi / 2) < tolerance or
                    abs(angle + np.pi / 2) < tolerance
            )

        def calculate_line_score(index, points_on_border, peak_value, weight_peak=1):
            """
            Calculate a combined score for a line based on points on the border and its peak value.

            :param index: int, index of the line
            :param points_on_border: int, number of points the line overlaps with the mask
            :param peak_value: float, the "straightness" of the line
            :param weight_peak: float, weight given to the peak value in the score
            :return: float, combined score
            """
            score = points_on_border + weight_peak * peak_value
            log_message = (
                f"Line {index} - Score calculation: points_on_border = {points_on_border}, peak_value = {peak_value}, "
                f"weight_peak = {weight_peak} => score = {score:.2f}\n"
            )
            print(log_message)
            with open(os.path.join(self.save_dir,"line_scores.txt"), "a") as log_file:
                log_file.write(log_message)
            return score

        # Step 1: Filter horizontal and vertical lines
        tolerance = np.pi / 12
        print("\nFinding best line combination ...")
        print(f"Using tolerance = {tolerance:.4f} (π/8) and π = {np.pi:.4f} π/2 = {np.pi/2:.4f}")
        filtered_lines_index = []
        for index, line in enumerate(self.lines):
            if is_horizontal_or_vertical(line.angle, tolerance):
                filtered_lines_index.append(index)
                print(f"Line {index} added: angle = {line.angle:.2f}, within tolerance")
            else:
                print(
                    f"Line {index} not added: angle = {line.angle:.2f}, "
                    f"abs(angle) = {abs(line.angle):.2f}, "
                    f"abs(angle - π/2) = {abs(line.angle - np.pi / 2):.2f}, "
                    f"abs(angle + π/2) = {abs(line.angle + np.pi / 2):.2f}, "
                    f"tolerance = {tolerance:.4f}"
                )

        print(f"Filtered {len(filtered_lines_index)} horizontal/vertical lines")

        # Step 2: Rank lines based on combined score
        print("\nEvaluating combined score for lines ...")
        scored_results = []
        for index in filtered_lines_index:
            points_on_border = self.extract_points_on_border(index)[1]
            peak_value = self.lines[index].peak
            score = calculate_line_score(index, points_on_border, peak_value)
            scored_results.append((index, score))

        # Sort lines by the combined score (descending)
        sorted_results = sorted(scored_results, key=lambda x: x[1], reverse=True)
        print(f"Found {len(sorted_results)} sorted lines by score")
        if sorted_results:
            print("Best line candidates by score:")
            for result in sorted_results[:5]:  # Print the top 5 results for inspection
                print(f"Line {result[0]}: score = {result[1]:.2f}")

        # Step 3: Find two orthogonal lines
        if not sorted_results:
            raise ValueError("No lines found after filtering and scoring")

        best_line_index = sorted_results[0][0]
        for index, _ in sorted_results[1:]:
            if lines_are_perpendicular(self.lines[best_line_index].angle, self.lines[index].angle):
                print(f"Lines {best_line_index} and {index} are orthogonal")
                return best_line_index, index

        # Raise an exception if no orthogonal lines are found
        raise ValueError("No orthogonal line pair found")

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
            print("Intersection point found: x = ", x, "y = ", y)

            # Ritorno (y,x) cosi che sia coerente con la funzione shape
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

    def farthest_middle_point(self, points_array, index):
        """
        Finds the farthest point from self.intersection among the given points,
        projects it onto the straight line, and returns the projected point with integer coordinates.

        :param points_array: Array of points representing the non-straight line.
        :param index: Index of the straight line in self.lines.
        :return: The projected point on the straight line farthest from self.intersection.
        """
        # Straight line endpoints
        straight_line = self.lines[index].cordinate_points
        p1, p2 = np.array(straight_line[0]), np.array(straight_line[-1])

        # Intersection point
        intersection = np.array(self.intersection)

        # Function to project a point onto a line segment
        def project_point_on_line(point, p1, p2):
            line_vector = p2 - p1
            point_vector = point - p1
            line_length_squared = np.dot(line_vector, line_vector)
            if line_length_squared == 0:  # Line is just a point
                return p1
            projection_factor = np.dot(point_vector, line_vector) / line_length_squared
            projection_factor = np.clip(projection_factor, 0, 1)  # Clamp to segment bounds
            return p1 + projection_factor * line_vector

        # Step 1: Find the farthest point from intersection in points_array
        farthest_point = max(points_array, key=lambda point: np.linalg.norm(np.array(point) - intersection))
        farthest_point = np.array(farthest_point)

        # Step 2: Project the farthest point onto the straight line
        projected_point = project_point_on_line(farthest_point, p1, p2)

        # Step 3: Round to nearest integers
        projected_point_int = np.rint(projected_point).astype(int)

        return projected_point_int

    def farthest_point_2(self, line):
        """
        Trova quale dei due punti in `line.extreme_points` è più lontano da `self.intersection`.

        Args:
            line: Oggetto con un attributo `extreme_points`, un array contenente due punti [p1, p2].

        Returns:
            Il punto più lontano da `self.intersection`.
        """
        # Recupera i due punti estremi
        p1, p2 = line.extreme_points

        # Calcola la distanza euclidea tra `self.intersection` e i due punti
        dist1 = np.linalg.norm(np.array(self.intersection) - np.array(p1))
        dist2 = np.linalg.norm(np.array(self.intersection) - np.array(p2))

        # Restituisci il punto con la distanza maggiore
        return p1 if dist1 > dist2 else p2

    def plot_results(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 9))
        ax = axes.ravel()


        # Assumendo che `self.image` sia un'immagine binaria con bordi sottili
        dilated_image = binary_dilation(self.image,
                                        structure=np.ones((11, 11)))  # Cambia il kernel (3x3) per controllare lo spessore

        dilated_image = (dilated_image.astype(np.uint8) * 255)

        # Mostra l'immagine dilatata
        ax[0].imshow(dilated_image, cmap='gray')
        ax[0].set_title('Thickened Input Image')
        ax[0].set_axis_off()

        # 2 Immagine:
        # Copia separata per la visualizzazione
        aux_mask = dilated_image.copy()

        cv.drawMarker(dilated_image, self.intersection, (200, 255, 200), cv.MARKER_STAR, 50, 4)
        cv.drawMarker(dilated_image, self.end_ant_point, (200, 255, 200), cv.MARKER_STAR, 50, 4)
        cv.drawMarker(dilated_image, self.end_pos_point, (200, 255, 200), cv.MARKER_STAR, 50, 4)
        ax[1].imshow(aux_mask, cmap='gray')
        ax[1].set_ylim((dilated_image.shape[0], 0))
        # ax[2].set_axis_off()
        ax[1].set_title('Detected lines')

        colors = ["red", "green", "purple", "orange", "yellow"]
        # Disegna le linee e aggiungi etichette
        for idx, line in enumerate(self.lines, start=0):
            ax[1].axline((line.x0, line.y0), slope=line.slope, color=colors[idx % len(colors)], linewidth=15 / (3 + idx))
            # Disegna un marker in (x0,y0)
            ax[1].scatter(line.x0, line.y0, color=colors[idx % len(colors)], marker='x', s=100)
            # Aggiungi il numero della linea vicino a (x0, y0)
            ax[1].text(line.x0, line.y0, f"Line {idx}", color=colors[idx % len(colors)], fontsize=16,
                       verticalalignment='top', horizontalalignment='right')


        # Immagine 3
        aux_mask = dilated_image.copy()
        aux_mask = cv.cvtColor(aux_mask, cv.COLOR_GRAY2RGB)
        # disegna i punti per sola visualizzazione
        for point in self.ant_points:
            cv.drawMarker(aux_mask, tuple(point), color=(255, 255, 0), markerType=cv.MARKER_SQUARE, markerSize=10,
                          thickness=2)

        for point in self.pos_points:
            cv.drawMarker(aux_mask, tuple(point), color=(255, 0, 0), markerType=cv.MARKER_SQUARE, markerSize=10,
                          thickness=2)

        ax[2].imshow(aux_mask)
        ax[2].set_title("Contorni")

        # 4 Immagine

        aux_mask = dilated_image.copy()
        aux_mask = cv.cvtColor(aux_mask, cv.COLOR_GRAY2RGB)
        ax[3].imshow(aux_mask)
        # Plot the line over the image
        ax[3].axline(xy1=self.intersection, xy2=self.end_ant_point, color='yellow', linewidth=5,marker='o')
        ax[3].axline(xy1=self.intersection, xy2=self.end_pos_point, color='red', linewidth=5, marker='o')


        plt.suptitle("Final result for hough_transform_skimage", fontsize=16)

        plt.tight_layout()

        if self.save_dir is not None:
            plt.savefig(os.path.join(self.save_dir, 'hough_skimage.png'), dpi=300)
        else:
            print("La cartella di salvataggio NON è stata impostata")
        if show_in_place:
            plt.show()

    def identify_quadrant(self):
        """
        Identifica il quadrante dell'intersezione e salva il frammento identificato in un file.
        """
        log_message = ""
        if self.intersection[1] >= int(self.image.shape[0] / 2) and self.intersection[0] >= int(
                self.image.shape[1] / 2):
            log_message = "Intersezione trovata in basso a dx - Frammento UL"
            print(log_message)
            fragment = "UL"
        elif self.intersection[1] <= int(self.image.shape[0] / 2) and self.intersection[0] >= int(
                self.image.shape[1] / 2):
            log_message = "Intersezione trovata in alto a dx - Frammento LL"
            print(log_message)
            fragment = "LL"
        elif self.intersection[1] <= int(self.image.shape[0] / 2) and self.intersection[0] <= int(
                self.image.shape[1] / 2):
            log_message = "Intersezione trovata in alto a sx - Frammento LR"
            print(log_message)
            fragment = "LR"
        elif self.intersection[1] >= int(self.image.shape[0] / 2) and self.intersection[0] <= int(
                self.image.shape[1] / 2):
            log_message = "Intersezione trovata in basso a sx - Frammento UR"
            print(log_message)
            fragment = "UR"
        else:
            log_message = "Intersezione non identificata"
            print(log_message)
            fragment = "Unknown"

        # Salva il messaggio nel file
        with open(os.path.join(self.save_dir, "line_scores.txt"), "a") as log_file:
            log_file.write(f"{log_message}\n")

        return fragment

    def invert_line_and_points(self):
        def invert():
            tmp_points = self.ant_points
            self.ant_points = self.pos_points
            self.pos_points = tmp_points

            tmp_end_point = self.end_ant_point
            self.end_ant_point = self.end_pos_point
            self.end_pos_point = tmp_end_point
        """
        input: self.end_ant_point: (x,y)
        :return:
        """
        print("self.end_ant_point", self.end_ant_point)
        print("self.end_pos_point", self.end_pos_point)
        if self.quadrant == "UR":
            if self.end_ant_point[1] < self.end_pos_point[1]:
                print("Da non invertire")
            else:
                print("Da invertire")
                invert()
        elif self.quadrant == "LR":
            if self.end_ant_point[1] < self.end_pos_point[1]:
                print("Da non invertire")
            else:
                print("Da invertire")
                invert()
        elif self.quadrant == "LL":
            if self.end_ant_point[1] > self.end_pos_point[1]:
                print("Da non invertire")
            else:
                print("Da invertire")
                invert()
        elif self.quadrant == "UL":
            if self.end_ant_point[1] > self.end_pos_point[1]:
                print("Da non invertire")
            else:
                print("Da invertire")
                invert()


def main():
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
        required=False,
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

    print("Keys in processed_img:", processed_img.keys())
    for key, value in processed_img.items():
        print(f"Key: {key}, Shape: {value.shape}")
    # serve perché l'edge detection fatta con Canny restituisce un'immagine di tipo bool
    # processed_img = processed_img.astype(np.uint8) * 255
    # print("preprocessed_img shape: ", processed_img.shape)

    # Trasformation through hough
    image_lines = Image_Lines(processed_img["histo_fragment"], processed_img["canny_edges"], save_dir)
    image_lines.plot_results()

    return 0


if __name__ == "__main__":
    show_in_place = 1
    main()
else:
    show_in_place = 0
