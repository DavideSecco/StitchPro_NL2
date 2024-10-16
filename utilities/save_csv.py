import csv
import os


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

if __name__ == "__main__":
    result_data = {
        "dataset": "dataset_n",
        "success": True,  # Esito dell'ottimizzazione
        "fun": 99,  # Valore della funzione obiettivo
        "work_time": 140.5  # Tempo di esecuzione
    }

    salva_in_csv(result_data, "results.csv")