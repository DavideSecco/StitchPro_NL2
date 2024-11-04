import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
from PIL import Image
import os


def visualize_image(image, name=None):
    # Display the image
    plt.imshow(image, cmap='gray')  # You can change 'gray' to other colormaps as needed
    plt.title(name)
    plt.axis('on')  # Hide the axis
    plt.show()

def visualize_images(images, names=None):
    rows = 2
    cols = 2

    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

    # Incompresible line to pass from a 1-D array to a 2-D array
    images = [images[i * cols:(i + 1) * cols] for i in range(rows)]
    names = [names[i * cols:(i + 1) * cols] for i in range(rows)]

    for row in range(0, rows):
        for col in range(0, cols):
            axes[row, col].imshow(images[row][col], cmap='gray')
            axes[row, col].set_title(names[row][col])
            axes[row, col].axis('on')

    plt.tight_layout()
    plt.show()

def find_centroid(image):
    # Convert the image to grayscale (optional, depending on the use case)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image to create a binary mask (you can adjust the threshold value)
    _, binary_mask = cv2.threshold(gray_image, 230, 255, cv2.THRESH_BINARY)

    # Find the indices of the non-zero pixels in the binary mask
    indices = np.argwhere(binary_mask == 0)

    # Calculate the centroid (mean of the indices)

    cX, cY = map(int,np.mean(indices, axis=0))
    # print("Centroid : cX", cX, ", cY:", cY)

    ###### Visualizzo centroid e true image (DA COMMENTARE IN PRODUZIONE) ######

    cv2.circle(image, (cY, cX), 70, (255, 0, 0), thickness=-1)  # Draw red circle
    plt.figure(figsize=(6, 6))
    plt.title("True Image")
    plt.imshow(image, cmap='gray') # cmap='gray'
    plt.axis('on')  # Hide axes for a cleaner look
    plt.show()

    ###### Visualizzo centroid e binary_mask image ######

    cv2.circle(binary_mask, (cY, cX), 70, (255, 0, 0), thickness=-1)  # Draw red circle
    plt.figure(figsize=(6, 6))
    plt.title("Binary Mask")
    plt.imshow(binary_mask, cmap='gray')  # cmap='gray'
    plt.axis('on')  # Hide axes for a cleaner look
    plt.show()

    # Print the centroid coordinates
    # print("Centroid (y, x):", centroid)
    return cX, cY

def divide_4_pieces(image, cX=None, cY=None):
    """
    :param image:
    :param cX: X coordinate of the centroid
    :param cY: Y coordinate of the centroid
    :return:

    The function divide an image in 4 part, if the point of the centroid are provided
    the image is divided using that point, otherwise in cutted in 4 equal part
    """
    if cX == None or cY == None:
        # Get the dimensions of the image
        height, width, levels = image.shape

        # Calculate the coordinates for the 4 pieces
        cX = width // 2
        cY = height // 2

    # Slice the image into four parts
    upper_left = image[:cX, :cY]
    upper_right = image[:cX, cY:]
    bottom_left = image[cX:, :cY]
    bottom_right = image[cX:, cY:]

    images = [upper_left, upper_right, bottom_left, bottom_right]
    names = ['upper_left.tif', 'upper_right.tif', 'bottom_left.tif', 'bottom_right.tif']

    return images, names

def add_noise_to_line(y, max_noise):
    noise = np.random.randint(-max_noise, max_noise, y.shape)
    return y + noise

def split_image_in_4_with_noise(iamge, cX=None, cY=None, max_noise=200):
    # Load the image
    # image = cv2.imread(image_path)

    # Get the dimensions of the image
    height, width, levels = image.shape

    if cX == None or cY == None:
        # Calculate the coordinates for the 4 pieces
        cX = width // 2
        cY = height // 2

    # Add noise to the horizontal and vertical lines
    noise_y = add_noise_to_line(np.full(height, cY), max_noise)
    print("Type:", type(noise_y))
    print(len(noise_y))
    print(noise_y)
    noise_x = add_noise_to_line(np.full(width, cX), max_noise)

    # Split the image into four parts
    upper_left = image[:min(noise_x), :min(noise_y)]
    upper_right = image[:min(noise_x), min(noise_y):]
    bottom_left = image[min(noise_x):, :min(noise_y)]
    bottom_right = image[min(noise_x):, min(noise_y):]

    # Save or display the pieces
    # cv2.imwrite('top_left.png', upper_left)
    # cv2.imwrite('top_right.png', upper_right)
    # cv2.imwrite('bottom_left.png', upper_right)
    # cv2.imwrite('bottom_right.png', bottom_right)

    # Display the pieces using PIL for better visualization
    # images = [upper_left, upper_right, bottom_left, bottom_right]
    # for idx, img in enumerate(images):
    #     Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).show(title=f'Piece {idx + 1}')

    images = [upper_left, upper_right, bottom_left, bottom_right]
    names = ['upper_left.tif', 'upper_right.tif', 'bottom_left.tif', 'bottom_right.tif']

    return images, names

def generate_wavy_line(length, amplitude, frequency):
    x = np.arange(length)
    # Generate a sinusoidal wave with added random noise
    y = amplitude * np.sin(2 * np.pi * frequency * x / length) + np.random.randint(-amplitude, amplitude, size=length)
    return y.astype(np.int32)

def split_image_with_wavy_cut(iamge, amplitude=30, frequency=5):
    # Load the image
    # image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Generate wavy lines for horizontal and vertical splits
    horizontal_wave = generate_wavy_line(width, amplitude, frequency) + height // 2
    vertical_wave = generate_wavy_line(height, amplitude, frequency) + width // 2

    # Ensure the waves stay within the image boundaries
    horizontal_wave = np.clip(horizontal_wave, 0, height - 1)
    vertical_wave = np.clip(vertical_wave, 0, width - 1)

    # Create masks for each piece
    mask_top_left = np.zeros((height, width), dtype=np.uint8)
    mask_top_right = np.zeros((height, width), dtype=np.uint8)
    mask_bottom_left = np.zeros((height, width), dtype=np.uint8)
    mask_bottom_right = np.zeros((height, width), dtype=np.uint8)

    for x in range(width):
        mask_top_left[:horizontal_wave[x], x] = 1
        mask_bottom_left[horizontal_wave[x]:, x] = 1

    for y in range(height):
        mask_top_right[y, :vertical_wave[y]] = 1
        mask_bottom_right[y, vertical_wave[y]:] = 1

    # Combine masks to get each piece
    upper_left = cv2.bitwise_and(image, image, mask=mask_top_left)
    upper_right = cv2.bitwise_and(image, image, mask=mask_top_right)
    bottom_left = cv2.bitwise_and(image, image, mask=mask_bottom_left)
    bottom_right = cv2.bitwise_and(image, image, mask=mask_bottom_right)

    # Save or display the pieces
    cv2.imwrite('top_left.png', upper_left)
    cv2.imwrite('top_right.png', upper_right)
    cv2.imwrite('bottom_left.png', bottom_left)
    cv2.imwrite('bottom_right.png', bottom_right)

    images = [upper_right, bottom_right, bottom_left, upper_left]
    names = ["upper_right.tif", "bottom_right.tif", "bottom_left.tif", "upper_left.tif"]

    return images, names

def convert_rgba_to_rgb(rgba_image, background=(255, 255, 255)):
    """Convert an RGBA image to RGB by blending it with a background color."""
    # Split the RGBA image into its components
    r, g, b, a = np.split(rgba_image, 4, axis=-1)

    # Normalize the alpha channel to be in the range [0, 1]
    alpha = a / 255.0

    # Blend the RGB channels with the background color using the alpha channel
    background = np.array(background).reshape(1, 1, 3)
    rgb_image = (1 - alpha) * background + alpha * np.concatenate([r, g, b], axis=-1)

    # Convert back to uint8 data type
    rgb_image = rgb_image.astype(np.uint8)

    return rgb_image


def is_rgb(image):
    """Check if the image is in RGB format."""
    return image.shape[-1] == 3 and image.dtype == np.uint8

def cut_image(image, save_dir=None, save=False):
    ###### Trovo centroid #######
    cX, cY = find_centroid(np.copy(image))
    print("Cordinate centroid: cX: ", cX, "cY: ", cY)

    ###### Divisione dell'immagine ######
    images, names = divide_4_pieces(image, cX, cY)
    # images, names = split_image_in_4_with_noise(image, cX, cY)
    # images, names = split_image_with_wavy_cut(image)

    ###### Aggiunta padding alle immagini #####
    padded_images = [None] * 4

    # Per tutti i nuovi frammenti dell'immagine
    for index in range(0, len(images)):
        padded_images[index] = np.pad(images[index], ((100, 100), (100, 100), (0, 0)), mode='constant',
                                      constant_values=np.iinfo(image.dtype).max)
        print("Dimensione frammento: ", index, ": ", padded_images[index].shape)
        # visualize_image(padded_images[index])

        # Save images with padding
        if save:
            tiff.imwrite(os.path.join(save_dir, names[index]), padded_images[index])

    print(type(padded_images[0]))

    visualize_images(padded_images, names)

    return padded_images

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Parse the argument passed
    parser = argparse.ArgumentParser(description="Script to process a file path")
    parser.add_argument('path', type=str, help='The path to the file')
    args = parser.parse_args()

    ####### Visualizzazione e eventuale conversione dell'immagine #####
    image = tiff.imread(args.path)
    print(image.shape)
    visualize_image(image)

    # Estrai la cartella in cui si trova l'immagine
    image_folder = os.path.dirname(args.path)
    print(f"La cartella contenente l'immagine è: {image_folder}")

    if not is_rgb(image):
        print("Not an RGB image: conversion from RGBA to RGB")
        image = convert_rgba_to_rgb(image)
        visualize_image(image)

    print("Dimensione dell'immagine RGB:", image.shape)

    cutted_images = cut_image(image, save_dir=image_folder, save=True)





