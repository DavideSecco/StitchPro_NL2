from utilities import Preprocessing

import sys
import math
import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist


def find_points(binary_image):
    '''
    :param x_out: imagine binaria in 8 bit: i valori sono 0 (nero) o 255
    :return:
    '''

    # coordinate di tutti i bordi dell'immagine:
    border_points = np.stack(np.where(binary_image > 0), axis=1)

    # Creo una matrice della stessa forma dell'immagine di partenza
    aux_image = np.zeros_like(binary_image, dtype=np.uint8)     # metto lo sfondo a 0. Cioé nera
    aux_image[np.where(binary_image > 0)] = 255                 # aggiungo il bordo

    plt.imshow(aux_image, cmap='gray')
    plt.title("aux_image")
    plt.show()

    plt.imshow(binary_image, cmap='gray')
    plt.title("Binary image")
    plt.show()

    if np.array_equal(aux_image, binary_image):
        print("I due array sono uguali.")
    else:
        print("I due array non sono uguali.")

    # PAOLO le due immagini binary image (prima x_out) and aux image (prima x) sono esattamente identiche.
    # Perché non hai fatto semplicemente una copia?


    # scelta ancora euristica: Dovrebbero essere le cordinate dei centri
    cx = 430
    cy = 50

    # per ora scelte in modo euristico, sostituirebbero le coordinate del punto finale dell'arco di ellisse
    rx = 145
    ry = 45


    points_cart_ant = np.array([(rx, ry)])
    pca_sorted = np.array([points_cart_ant[0][::-1]]) # ho semplicemente invertito cx con cy

    # trova il punto sul bordo della maschera più vicino a pca_sorted
    point_ant = border_points[np.argmin(cdist(border_points, pca_sorted), axis=0)]

    # disegna una maschera "ausiliaria" che copra il bordo di interesse nell'immagine (v. primo plot)
    ant_axis_line_mask = np.zeros_like(binary_image, dtype=np.uint8)
    cv.line(ant_axis_line_mask, [int(cx), int(cy)], np.int32(point_ant[0][::-1]), 1, 20)

    # Trovo punti sul bordo come sovrapposizione tra la maschera ausiliaria e l'immagine con i bordi (binary_image)
    ant_points = np.roll(np.array(np.where(ant_axis_line_mask * binary_image)).T, 1, 1)

    # STESSA PROCEDURA VA FATTA PER I PUNTI SULL'ALTRO BORDO

    # sfruttando questo principio il problema è molto semplice, basti vedere come viene creata la maschera ausiliaria
    # ossia di fatto partendo dalla linea che si "sovrappone" al bordo semplicemente disegnandola molto spessa
    # (nell'esempio in riga 37 il parametro thickness=20 rende la linea spessa) e poi sovrapponendoci l'immagine
    # contenente i bordi. In questo modo di fatto si 'isolano' i punti sul bordo che sono di interesse per lo stitching


    # Visulizzazione del risultato (quindi punti in input e bordo trovato)

    # disegna i punti per sola visualizzazione
    for point in ant_points:
        cv.drawMarker(aux_image, (point), color=(255), markerType=cv.MARKER_SQUARE, markerSize=3, thickness=2)

    cv.drawMarker(aux_image, (cx, cy), (159), markerType=cv.MARKER_CROSS, markerSize=9, thickness=3)        # disegno centro
    cv.drawMarker(aux_image, (rx, ry), (150), markerType=cv.MARKER_CROSS, markerSize=5, thickness=3)        # disegno raggio
    cv.drawMarker(aux_image, (point_ant[0][::-1]), (100), markerType=cv.MARKER_CROSS, markerSize=9, thickness=3)    # disegno punto piú vicino rispetto al punto (rx, rY)

    plt.imshow(ant_axis_line_mask, cmap='gray')
    plt.title("ant_axis_line_mask")
    plt.show()

    plt.imshow(aux_image, cmap='gray')
    plt.show()



    print('point_ant: ', point_ant)
    print(border_points.shape)


def main(args):

    # PER QUESTO ESEMPIO viene utilizzata un'imamgine in particolare, te la mando. Il codice in questo momento non
    # funziona su un frammento qualsiasi

    default_path = './bottom_left.tif'
    filepath = args[0] if len(args) > 0 else default_path

    prep = Preprocessing(filepath)
    processed_img = prep.preprocess_image(show_steps=False,
                                          median_filter_size=30,
                                          closing_footprint_size=20,
                                          apply_hull_image=False)
    # immagine binaria ad 8 bit
    processed_img = processed_img.astype(np.uint8) * 255

    find_points(processed_img)


if __name__ == "__main__":
    main(sys.argv[1:])
