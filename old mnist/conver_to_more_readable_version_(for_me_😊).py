import os

# Define the directory structure
base_dir = 'mnist_data'
image_dir = os.path.join(base_dir, 'images')
label_dir = os.path.join(base_dir, 'labels')

# Create directories if they don't exist
os.makedirs(image_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

import numpy as np
import struct

def load_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, "Invalid magic number in image file"
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
    return images

def load_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        assert magic == 2049, "Invalid magic number in label file"
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def save_binary_files(images, labels, image_dir, label_dir):
    num_images = images.shape[0]
    for i in range(num_images):
        image_file_path = os.path.join(image_dir, f'{i + 1}.bin')
        label_file_path = os.path.join(label_dir, f'{i + 1}.bin')

        # Save image
        with open(image_file_path, 'wb') as img_f:
            img_f.write(images[i].tobytes())

        # Save label
        with open(label_file_path, 'wb') as lbl_f:
            lbl_f.write(labels[i].tobytes())

# Load the MNIST dataset
train_images = load_mnist_images('archive/train-images.idx3-ubyte')
train_labels = load_mnist_labels('archive/train-labels.idx1-ubyte')

# Save images and labels to binary files
save_binary_files(train_images, train_labels, image_dir, label_dir)
