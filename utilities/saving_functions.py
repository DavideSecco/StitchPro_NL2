import csv
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import json


def salva_in_csv(dizionario, path, nome_file_csv):
    # Crea il percorso completo del file
    percorso_completo = os.path.join(path, nome_file_csv)

    # Prepara i dati da scrivere nella riga
    riga = [dizionario['dataset'], dizionario['success'], dizionario['fun'], dizionario['work_time']]

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
            writer.writerow(['Descrizione', 'Success', 'Fun', 'Work Time'])

        # Scrivi le righe precedenti
        writer.writerows(righe)

        # Aggiungi la nuova riga
        writer.writerow(riga)


def save_images_data_dict(data_dict, i, save_dir_image_i):
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