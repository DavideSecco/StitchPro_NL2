import sys
import math
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


from utilities import Preprocessing


def hough_transform(img, show=False, save=False):
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

            if abs(theta) < np.pi / 36 or abs(theta - np.pi / 2) < np.pi / 36 or abs(theta - np.pi) < np.pi / 36:
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

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
        cv.imshow("Source", img)
        cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
        cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

    cv.waitKey()
    return 0


def main(argv):
    default_file = "../bottom_left.tif"
    filename = argv[0] if len(argv) > 0 else default_file

    prep = Preprocessing(filename)
    original_img = prep.original_image
    print(original_img.shape)

    plt.imshow(original_img)
    plt.show()



    processed_img = prep.preprocess_image(show_steps=True,
                                          median_filter_size=30,
                                          closing_footprint_size=60,
                                          apply_hull_image=False)

    # serve perché l'edge detection fatta con Canny restituisce un'immagine di tipo bool
    processed_img = processed_img.astype(np.uint8) * 255

    # debugging
    print(np.unique(processed_img))
    print(processed_img.shape)
    print(processed_img.dtype)

    hough_transform(processed_img, show=True)

    return 0


if __name__ == "__main__":
    main(sys.argv[1:])