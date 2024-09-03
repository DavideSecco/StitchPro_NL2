import sys
import math
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from skimage.draw import line as draw_line
from skimage import data


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
        raise ValueError("Nessuna linea trovata")


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



def hough_transform_skimage_implementation(image):
    # image = np.zeros((200, 200))
    # idx = np.arange(25, 175)
    # image[idx, idx] = 255
    # image[draw_line(45, 25, 25, 175)] = 255
    # image[draw_line(25, 135, 175, 155)] = 255

    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    h, theta, d = hough_line(image, theta=tested_angles)
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
    ax[2].set_axis_off()
    ax[2].set_title('Detected lines')

    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        ax[2].axline((x0, y0), slope=np.tan(angle + np.pi / 2))

    plt.tight_layout()
    plt.show()

    pass


def draw_vertex_lines(img, x0, y0, a, b):
    img = img.copy()

    cv.drawMarker(img, (int(x0), int(y0)), (0, 255, 0), cv.MARKER_CROSS, 20, 2)

    arrow_length = 50
    pt2 = (int(x0 + arrow_length * a), int(y0 + arrow_length * b))
    cv.arrowedLine(img, (int(x0), int(y0)), pt2, (0, 0, 255), 2, tipLength=0.3)

    plt.imshow(img)
    plt.show()


def main(argv):
    default_file = "../bottom_left.tif"
    filename = argv[0] if len(argv) > 0 else default_file

    prep = Preprocessing(filename)
    original_img = prep.original_image

    plt.imshow(original_img)
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

    hough_transform(processed_img, show=True)
    hough_transform_skimage_implementation(processed_img)  # funziona molto meglio

    return 0


if __name__ == "__main__":
    main(sys.argv[1:])