import os
import subprocess
import argparse
import sys

# Possibilità 1: Passo il percorso della cartella tramite command line
parser = argparse.ArgumentParser(description="Script per caricare e ruotare immagini")
parser.add_argument(
    '--datasets_directory',
    type=str,
    default=None,  # Se non viene fornito dall'utente, rimane None
    help="Percorso opzionale per le immagini"
)

parser.add_argument(
    '--save_dir',
    type=str,
    default=None,  # Se non viene fornito dall'utente, rimane None
    help="Percorso dove salvare i risultati",
    required=True
)

parser.add_argument(
    '--script',
    type=str,
    choices=['vanilla', 'custom'],  # Limita i valori ammissibili a 'vanilla' e 'custom'
    default=None,                   # Se non viene fornito dall'utente, rimane None
    help="Script da testare (vanilla o custom)",
    required=True
)

# Parsing degli argomenti
args = parser.parse_args()

dataset_folder_command_line = args.datasets_directory
datasets_directory_Davide = "/mnt/Volume/Mega/LaureaMagistrale/CorsiSemestre/A2S1/MultudisciplinaryProject/data/"

save_dir = args.save_dir

print(dataset_folder_command_line)

if os.path.exists(dataset_folder_command_line):
    datasets_directory = dataset_folder_command_line
elif os.path.exists(datasets_directory_Davide):
    datasets_directory = datasets_directory_Davide
else:
    print("Percoso inserito dei datasets non esistente")
    sys.exit(1)


for folder in sorted(os.listdir(datasets_directory)):
    dataset_dir = os.path.join(datasets_directory, folder)
    print("Dataset dir: ", dataset_dir)

    if args.script == "vanilla":
        # Comando da eseguire
        command = [
            'python', 'StitchPro_vanilla_implementation_no_streamlit.py',
            '--input_path', dataset_dir,
            '--save_dir', save_dir
        ]
    elif args.script == "custom":
        # Comando da eseguire
        command = [
            'python', 'StichPro_our_implemetation_no_streamlit.py',
            '--input_path', dataset_dir,
            '--save_dir', save_dir
        ]

    # Esecuzione del comando
    subprocess.run(command, check=True)