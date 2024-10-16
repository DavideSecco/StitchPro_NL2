import csv
import os


def salva_in_csv(dizionario, nome_file_csv):
    # Verifica se il file esiste
    file_esiste = os.path.isfile(nome_file_csv)

    # Apri il file in modalità append o scrittura (crea il file se non esiste)
    with open(nome_file_csv, mode='a' if file_esiste else 'w', newline='') as file_csv:
        writer = csv.writer(file_csv)

        # Se il file è appena creato, scrivi l'intestazione
        if not file_esiste:
            writer.writerow(['Descrizione', 'Success', 'Fun', 'Work Time'])

        # Prepara i dati da scrivere nella riga
        riga = ["Dataset_interopassato", dizionario['success'], dizionario['fun'], dizionario['work_time']]

        # Leggi le righe esistenti se il file esiste
        if file_esiste:
            with open(nome_file_csv, 'r', newline='') as file_letto:
                righe = list(csv.reader(file_letto))

            # Scrivi il contenuto aggiornato nel file
            with open(nome_file_csv, 'w', newline='') as file_scrittura:
                writer = csv.writer(file_scrittura)
                writer.writerows(righe)
        else:
            # Se il file non esiste, scrivi semplicemente la nuova riga
            writer.writerow(riga)

if __name__ == "__main__":
    result_data = {
        "dataset": "dataset_n",
        "success": True,  # Esito dell'ottimizzazione
        "fun": 99,  # Valore della funzione obiettivo
        "work_time": 140.5  # Tempo di esecuzione
    }

    salva_in_csv(result_data, "results.csv")