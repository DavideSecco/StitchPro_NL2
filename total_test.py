import os
import subprocess

if os.path.exists("/mnt/Volume/Mega/LaureaMagistrale/CorsiSemestre/A2S1/MultudisciplinaryProject/data/"):
    datasets_directory = "/mnt/Volume/Mega/LaureaMagistrale/CorsiSemestre/A2S1/MultudisciplinaryProject/data/"
else:
    print("Non ho trovato la cartella con tutti i dataset")

for folder in sorted(os.listdir(datasets_directory)):
    dataset_dir = os.path.join(datasets_directory, folder)
    print("Dataset dir: ", dataset_dir)

    # Comando da eseguire
    command = [
        'python', 'StichPro_our_implemetation_no_streamlit.py',
        '--input_path', dataset_dir
    ]

    # Esecuzione del comando
    subprocess.run(command, check=True)