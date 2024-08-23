import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from PIL import Image







def load_image_as_array(image_path):
    # Apri l'immagine usando PIL
    img = Image.open(image_path)

    # Converti l'immagine in un array NumPy
    img_array = np.array(img)

    return img_array, img



def circle_arc_loss_cv(par, mask, pad=1500, save=False, name='best_ellipse_solution.png'):
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


path_to_mask = "C:/Users/dicia/NL2_project/debugging_series/debug5/upper_right/upper_right_debugging_x_contours.png"
x, pil_img = load_image_as_array(path_to_mask)

image_gray = pil_img.convert('L')
x = np.array(image_gray)
x = np.where(x > 200, 1, 0)

Mx, My = x.shape
M = min(Mx, My)

# Parametri per le perturbazioni
delta = 20  # Raggio di variazione per cx e cy
theta_delta = np.pi / 16  # Raggio di variazione per gli angoli

# Inizializzazioni variabili

i = 0

initialization_centre = [
    (0, My),
    (0, 0),
    (Mx, 0),
    (Mx, My)
]

initializations = []
# Numero di inizializzazioni da generare
num_initializations = 10

curr_theta = 0
segment_angle = 0

for _ in range(num_initializations):
    # Genera cx e cy intorno ai valori iniziali
    perturbed_cx = np.clip(initialization_centre[i][0] + np.random.uniform(-delta, delta), 0, M)
    perturbed_cy = np.clip(initialization_centre[i][1] + np.random.uniform(-delta, delta), 0, M)

    # Genera angoli theta intorno ai valori iniziali
    perturbed_theta_1 = curr_theta + np.random.uniform(-theta_delta, theta_delta)
    perturbed_segment_angle = segment_angle + np.random.uniform(-theta_delta, theta_delta)

    # Aggiungi l'inizializzazione perturbata alla lista
    initializations.append((perturbed_cx, perturbed_cy, perturbed_theta_1, perturbed_segment_angle))

# Per ogni inizializzazione, esegui l'ottimizzazione
solutions = []
for init in initializations:
    print("Trying initialization for segment {}: {}".format(i, init))

    # Definisci bounds per ogni inizializzazione
    bounds = [
        (max(0, init[0] - delta), min(M, init[0] + delta)),  # Limita cx vicino a init[0]
        (max(0, init[1] - delta), min(M, init[1] + delta)),  # Limita cy vicino a init[1]
        (M / 4, M / 2),  # Semiasse r1 (può essere più ristretto se necessario)
        (M / 4, M / 2),  # Semiasse r2 (può essere più ristretto se necessario)
        (init[2] - theta_delta, init[2] + theta_delta),  # Angolo iniziale limitato intorno a init[2]
        (init[3] * 0.8, init[3] * 1.2)  # Ampiezza dell'arco limitata
    ]

    # Valore iniziale per l'ottimizzazione
    x0 = [init[0], init[1], M / 2, M / 2, init[2], init[3]]

    pad = 1000
    # Esegui l'ottimizzazione
    solution = optimize.differential_evolution(
        circle_arc_loss_cv, bounds=bounds, args=[x, pad], x0=x0,
        popsize=50, maxiter=500, workers=1, updating='immediate'
    )

    solutions.append(solution)


