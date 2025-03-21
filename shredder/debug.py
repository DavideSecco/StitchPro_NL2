import matplotlib.pyplot as plt
import numpy as np

def debug_plot_lines(self, save_path=None):
    """
    Quick debug visualization of the shredding lines 
    on top of the low-resolution image.
    If save_path is given, save the figure to disk as well.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    
    # Show the lowres image (assuming 3 channels or RGBA)
    if self.lowres_image.shape[2] == 3:
        ax.imshow(self.lowres_image)
    else:
        # If there's an alpha channel or more, just take the first 3 channels
        ax.imshow(self.lowres_image[:, :, :3])  

    # Vertical line in red
    ax.plot(self.v_line_interp[:, 0],
            self.v_line_interp[:, 1],
            linestyle='-', marker='', label='Vertical line', color='red')

    # Horizontal line in blue
    ax.plot(self.h_line_interp[:, 0],
            self.h_line_interp[:, 1],
            linestyle='-', marker='', label='Horizontal line', color='blue')

    # Mark the intersection in green
    ax.scatter(self.intersection[0],
               self.intersection[1],
               color='green', marker='x', s=80, label='Intersection')

    ax.set_title("Zig-Zag Lines Debug")
    ax.legend()

    # Se esiste un percorso in cui salvare, salviamo la figura
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[DEBUG] Debug plot saved to {save_path}")

    # Mostriamo a schermo
    plt.show()


import matplotlib.pyplot as plt

def vips2numpy(vips_img, force_grayscale=False):
    """
    Converte un'immagine pyvips in un array NumPy (dtype=uint8).
    Se force_grayscale=True, forza la visualizzazione in scala di grigi.
    """
    # Ricordati che vips_img.bands = numero di canali (RGB=3, RGBA=4, ecc.)
    # Per un canale solo, shape = (H, W, 1)
    np_arr = np.ndarray(
        buffer=vips_img.write_to_memory(),
        dtype=np.uint8,
        shape=(vips_img.height, vips_img.width, vips_img.bands)
    )
    if force_grayscale and np_arr.shape[2] == 1:
        # In caso servisse forzare la scala di grigi
        np_arr = np_arr[:, :, 0]
    return np_arr

def debug_show_image(vips_img, title="Debug Image", force_grayscale=False):
    """
    Mostra l'immagine pyvips (vips_img) con matplotlib in una singola figura.
    """
    arr = vips2numpy(vips_img, force_grayscale=force_grayscale)

    plt.figure()
    plt.title(title)
    # Se ha solo 1 canale, usiamo cmap='gray', altrimenti mostriamo l'RGB
    if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
        plt.imshow(arr, cmap='gray')
    else:
        plt.imshow(arr)
    plt.axis('on')
    plt.show()


# Funzione per visualizzare i frammenti come immagini
def display_fragments(fragments):
    # Numero di frammenti
    num_fragments = len(fragments)
    
    # Crea una figura per visualizzare le immagini
    fig, axes = plt.subplots(1, num_fragments, figsize=(15, 5))
    
    # Se c'è solo un frammento, axes non è una lista, quindi lo faccio diventare una lista
    if num_fragments == 1:
        axes = [axes]
    
    # Visualizza ogni frammento
    for i, fragment in enumerate(fragments):
        axes[i].imshow(fragment, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f"Fragment {i+1}")
    
    plt.show()

