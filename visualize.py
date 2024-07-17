import os
import numpy as np
import matplotlib.pyplot as plt

def read_binary_image(file_path):
    with open(file_path, 'rb') as f:
        image = np.frombuffer(f.read(), dtype_=np.uint8).reshape(28, 28)
    return image

def read_binary_label(file_path):
    with open(file_path, 'rb') as f:
        label = np.frombuffer(f.read(), dtype_=np.uint8)[0]
    return label

def visualize_images_and_labels(image_dir, label_dir, num_samples=10):
    plt.figure(figsize=(10, 2))
    for i in range(num_samples):
        image_path = os.path.join(image_dir, f'{i + 1}.bin')
        label_path = os.path.join(label_dir, f'{i + 1}.bin')

        image = read_binary_image(image_path)
        label = read_binary_label(label_path)

        plt.subplot(1, num_samples, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f'Label: {label}')
        plt.axis('off')
    plt.show()

# Visualize the first 10 images and labels
visualize_images_and_labels('mnist_data/images', 'mnist_data/labels', num_samples=10)
