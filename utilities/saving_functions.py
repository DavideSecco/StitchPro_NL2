import csv
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import json


def salva_in_csv(dizionario, path, nome_file_csv):
    """
    Saves dictionary data into a CSV file, appending it to existing rows if the file exists.

    This function takes a dictionary containing information about a dataset and saves it into a CSV file at the
    specified path. If the file does not exist, it is created with a header. If the file exists, new data is appended
    to the existing rows.

    Args:
        dizionario (dict): A dictionary containing the data to be saved. It must include the keys
            'dataset' (description of the dataset), 'success' (operation result),
            'fun' (function name), and 'work_time' (execution time).
        path (str): The path to the directory where the CSV file will be saved.
        nome_file_csv (str): The name of the CSV file.

    Returns:
        None: The function writes the data to the CSV file without returning any value.
    """
    # Crea il percorso completo del file
    percorso_completo = os.path.join(path, nome_file_csv)

    # Prepara i dati da scrivere nella riga
    riga = [dizionario['dataset'], dizionario['success'], dizionario['fun'], dizionario['average_euclidean_distance_mm'], dizionario['work_time']]

    # Leggi le righe esistenti se il file esiste
    righe = []
    if os.path.isfile(percorso_completo):
        with open(percorso_completo, 'r', newline='') as file_letto:
            righe = list(csv.reader(file_letto))

    # Apri il file in modalità scrittura
    with open(percorso_completo, 'w', newline='') as file_csv:
        writer = csv.writer(file_csv)

        # Se il file è vuoto o non esiste, scrivi l'intestazione
        if not righe:
            writer.writerow(['Descrizione', 'Success', 'Fun', 'average euclidean distance mm',  'Work Time'])

        # Scrivi le righe precedenti
        writer.writerows(righe)

        # Aggiungi la nuova riga
        writer.writerow(riga)


def save_images_data_dict(data_dict, i, save_dir_image_i):
    """
    Saves and visualizes various images and annotations from a data dictionary.

    This function creates a figure with multiple subplots showing different components of the
    data, including the original image, tissue masks, closed masks, and annotated points and lines.
    The final image is saved to the specified directory.

    Args:
        data_dict (dict): A dictionary containing image data and annotations. The dictionary
            must include keys such as 'image', 'tissue_mask', 'tissue_mask_closed', 'ant_points',
            'pos_points', 'ant_line', and 'pos_line'.
        i (int): The index of the image in the data dictionary to visualize and save.
        save_dir_image_i (str): The directory path where the final image visualization will be saved.

    Returns:
        None: The function saves the image to the specified directory and closes all plots.
    """
    # Crea una figura con 3 sottotrame
    fig, axs = plt.subplots(2, 3, figsize=(15, 5))  # 1 riga, 3 colonne
    ax = axs.ravel()
    # Visualizza ciascuna immagine in una sottotrama
    ax[0].imshow(data_dict[i]["image"])
    ax[0].axis('off')  # Nasconde gli assi opzionale

    ax[1].imshow(data_dict[i]["tissue_mask"], cmap="gray")
    ax[1].axis('off')

    ax[2].imshow(data_dict[i]["tissue_mask_closed"], cmap="gray")
    ax[2].axis('off')

    # Mostra la visualizzazione con le tre immagini
    # plt.show()

    # Stampo i punti
    # aux_mask = cv2.cvtColor(aux_mask, cv2.COLOR_GRAY2RGB)
    aux_mask = data_dict[i]["tissue_mask_closed"]
    # Convert aux_mask from CV_32S to CV_8U
    aux_mask = aux_mask.astype(np.uint8)
    aux_mask = cv2.cvtColor(aux_mask, cv2.COLOR_GRAY2RGB)
    # disegna i punti per sola visualizzazione
    for point in data_dict[i]['ant_points']:
        cv2.drawMarker(aux_mask, tuple(point), color=(255, 255, 0), markerType=cv2.MARKER_SQUARE, markerSize=3,
                       thickness=2)

    for point in data_dict[i]['pos_points']:
        cv2.drawMarker(aux_mask, tuple(point), color=(255, 0, 0), markerType=cv2.MARKER_SQUARE, markerSize=3,
                       thickness=2)

    ax[3].imshow(aux_mask)
    ax[3].set_title("Contorni")

    aux_mask2 = data_dict[i]["tissue_mask_closed"]
    ax[4].imshow(aux_mask2)
    # Plot the line over the image
    ax[4].axline(data_dict[i]["ant_line"][0], xy2=data_dict[i]["ant_line"][1], color='yellow', linewidth=2,
                 marker='o')
    ax[4].axline(data_dict[i]["pos_line"][0], xy2=data_dict[i]["pos_line"][1], color='red', linewidth=2,
                 marker='o')

    plt.tight_layout()

    plt.savefig(os.path.join(save_dir_image_i, 'data_dict'), dpi=300)

    plt.close("all")


def save_data_dict(data_dict, i, save_dir_image_i):
    """
    Converts NumPy arrays to lists in a data dictionary and saves it as a JSON file.

    This function converts NumPy arrays in the data dictionary to lists for compatibility
    with JSON serialization. After saving the data as JSON, the lists are converted back to
    NumPy arrays to restore the original structure.

    Args:
        data_dict (dict): A dictionary containing image data and annotations. The dictionary
            must include keys such as 'image', 'tissue_mask', 'tissue_mask_closed', 'ant_line',
            'pos_line', 'ant_points', and 'pos_points'.
        i (int): The index of the image in the data dictionary to process and save.
        save_dir_image_i (str): The directory path where the JSON file will be saved.

    Returns:
        None: The function saves the JSON file to the specified directory and restores
        the original NumPy arrays in the data dictionary.
    """
    # Conversion needed since you can't save a nparray in a json file
    data_dict[i]['image'] = data_dict[i]['image'].tolist()
    data_dict[i]['tissue_mask'] = data_dict[i]['tissue_mask'].tolist()
    data_dict[i]['tissue_mask_closed'] = data_dict[i]['tissue_mask_closed'].tolist()
    data_dict[i]['ant_line'] = data_dict[i]['ant_line'].tolist()
    data_dict[i]['pos_line'] = data_dict[i]['pos_line'].tolist()
    data_dict[i]['ant_points'] = data_dict[i]['ant_points'].tolist()
    data_dict[i]['pos_points'] = data_dict[i]['pos_points'].tolist()

    # Save as json
    with open(os.path.join(save_dir_image_i, "data_dict.json"), "w") as file:
        # json.dump(data_dict[i], file)
        json.dump(dict((k, data_dict[i][k]) for k in ['ant_line', 'pos_line', 'ant_points', 'pos_points']),
                  file, separators=(',', ': '))

    data_dict[i]['image'] = np.array(data_dict[i]['image'])
    data_dict[i]['tissue_mask'] = np.array(data_dict[i]['tissue_mask'])
    data_dict[i]['tissue_mask_closed'] = np.array(data_dict[i]['tissue_mask_closed'])
    data_dict[i]['ant_line'] = np.array(data_dict[i]['ant_line'])
    data_dict[i]['pos_line'] = np.array(data_dict[i]['pos_line'])
    data_dict[i]['ant_points'] = np.array(data_dict[i]['ant_points'])
    data_dict[i]['pos_points'] = np.array(data_dict[i]['pos_points'])


if __name__ == "__main__":
    result_data = {
        "dataset": "dataset_n",
        "success": True,  # Esito dell'ottimizzazione
        "fun": 99,  # Valore della funzione obiettivo
        "work_time": 140.5  # Tempo di esecuzione
    }

    salva_in_csv(result_data, "results.csv")