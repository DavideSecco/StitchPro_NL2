import sys
import math
import cv2 as cv
import numpy as np
import PIL
import matplotlib.pyplot as plt


def load_and_preprocess(img_path, show=False, scale_factor=1):
    # Carica l'immagine in formato RGBA
    src_rgba = cv.imread(img_path, cv.IMREAD_UNCHANGED)

    # Controlla se l'immagine è stata caricata correttamente
    if src_rgba is None:
        raise ValueError(f"Error opening image at {img_path}")

    width = int(src_rgba.shape[1] / scale_factor)
    height = int(src_rgba.shape[0] / scale_factor)
    dim = (width, height)
    src_resized = cv.resize(src_rgba, dim, interpolation=cv.INTER_AREA)

    # Verifica il numero di canali e converte in scala di grigi
    if len(src_resized.shape) == 2:  # L'immagine è già in scala di grigi
        src_gray = src_resized
    elif len(src_resized.shape) == 3:
        if src_resized.shape[2] == 4:  # L'immagine è in RGBA
            src_gray = cv.cvtColor(src_resized, cv.COLOR_RGBA2GRAY)
        elif src_resized.shape[2] == 3:  # L'immagine è in RGB o BGR
            src_gray = cv.cvtColor(src_resized, cv.COLOR_BGR2GRAY)  # OpenCV legge in BGR di default
    else:
        raise ValueError("Unsupported image format")

    if show:
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.title('Original RGBA Image')
        plt.imshow(cv.cvtColor(src_rgba, cv.COLOR_RGBA2RGB))

        plt.subplot(1, 2, 2)
        plt.title('Grayscale Image')
        plt.imshow(src_gray, cmap='gray')

        plt.show()

    return src_gray


def get_hull(src_gray, show=False):
    # Trova i contorni nell'immagine in scala di grigi
    contours, _ = cv.findContours(src_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=cv.contourArea)

    # Crea un'immagine vuota per disegnare l'hull convesso
    hull_image = np.zeros_like(src_gray).astype(np.uint8)

    # Calcola e disegna l'hull convesso per ogni contorno trovato
    for contour in contours:
        hull = cv.convexHull(contour)
        cv.drawContours(hull_image, [hull], -1, (0, 255, 0), thickness=cv.FILLED)

    if show:
        plt.imshow(hull_image, cmap='gray')
        plt.show()

    return hull_image


def main(argv):
    default_file = r"C:\Users\dicia\NL2_project\datasets\test-data\ur.tif"
    default_file = "../bottom_left.tif"
    filename = argv[0] if len(argv) > 0 else default_file
    src = load_and_preprocess(filename, scale_factor=3, show=True)


    # Check if image is loaded fine
    if src is None:
        print('Error opening image!')
        print('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1

    dst = cv.Canny(src, 50, 200, None, 3)
    cv.imshow("canny img", dst)
    # dst = get_hull(src, show=True)  # non ancora pronto

    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    Mx, My = dst.shape
    M = min(Mx, My)
    thresh = int(1*M/4)

    lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)


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

    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            # Calcola la pendenza della linea per filtrare le linee diagonali
            dx = l[2] - l[0]
            dy = l[3] - l[1]
            if dx == 0 or dy == 0 or abs(dy / dx) < 0.1:  # linee verticali, orizzontali o quasi orizzontali
                cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

    cv.imshow("Source", src)
    cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

    cv.waitKey()
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])