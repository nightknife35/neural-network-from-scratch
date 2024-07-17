import numpy as np
import matplotlib.pyplot as plt
import os
import json
import jax.numpy as jnp
from jax import grad

class Layer:
    def __init__(self, number_of_nodes, activation, Type, number_of_prev_nodes):
        self.Type = Type
        self.number_of_nodes = number_of_nodes
        if self.Type == "Input":
            pass
        else:
            self.weights = np.random.randn(number_of_nodes, number_of_prev_nodes)
            self.biases = np.random.randn(number_of_nodes)
            self.activation = activation
            self.activation_prime = grad(self.activation)

    def calculate(self, inputs):
        self.z = np.dot(self.weights, inputs) + self.biases
        self.a = self.activation(self.z)
        return self.a

   
        
class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.default_activation = self.default_activation_function
        self.loss_function = self.default_loss_function
        self.loss_function_prime = grad(self.loss_function)

    def default_activation_function(self, x):
        return x 
    
    def default_loss_function(self, y_hat, y):
        return np.sum((y_hat - y) ** 2)
    
    def add_layer(self, number_of_nodes, activation=None , Type="Hidden"):
        if activation is None:
            activation = self.default_activation
        if Type == "Input":
            self.number_of_prev_nodes = 0
        layer = Layer(number_of_nodes, activation, Type, self.number_of_prev_nodes)
        self.number_of_prev_nodes = number_of_nodes
        self.layers.append(layer)
        if Type == "Output":
            self.all_weights = [layer.weights for layer in self.layers[1:]]
            self.all_biases = [layer.biases for layer in self.layers[1:]]

    def add_loss_function(self, loss_function):
        self.loss_function = loss_function
        self.loss_function_prime = grad(loss_function)

    def forward(self, inputs):
        x = inputs
        for layer in self.layers[1:]:  # Skip input layer
            x = layer.calculate(x)
        return x

    def predict(self, inputs):
        a = self.forward(inputs)
        return np.argmax(a)

    def backpropagation(self, x, y):

        output_layer = self.layers[-1]
        y_ = self.forward(x)
        # feel like its here only for metrics
        cost = self.loss_function(y_, y)
        
        delta_output = self.loss_function_prime(y_, y) * output_layer.activation_prime(output_layer.a)
        dL_dW = np.outer(delta_output, self.layers[-2].a)
        dL_db = delta_output
        self.weight_adjustment[-1] += dL_dW
        self.bias_adjustment[-1 ] += dL_db
        
        delta_ip1 = delta_output  # delta i+1

        for i in range(self.number_of_middle_layers, 1, -1):
            
            prev_layer = self.layers[i-1]
            layer = self.layers[i]
            next_layer = self.layers[i+1]

            delta_i = np.dot(next_layer.weights.T, delta_ip1) * layer.activation_prime(layer.a)
            dL_dW = np.outer(delta_i, prev_layer.a)
            dL_db = delta_i
            self.weight_adjustment[i] += dL_dW
            self.bias_adjustment[i] += dL_db

            delta_ip1 = delta_i

        layer_1 = self.layers[1]
        layer_2 = self.layers[2]

        delta_1 = np.dot(layer_2.weights.T, delta_ip1) * layer_1.activation_prime(layer_1.a)
        dL_dW = np.outer(delta_1, x)
        dL_db = delta_1
        self.weight_adjustment[1] += dL_dW
        self.bias_adjustment[1] += dL_db

    def update_weights_n_biases(self, lr):
        self.all_weights -= lr * self.weight_adjustment
        self.all_biases -= lr * self.bias_adjustment
        
    def train(self, path_to_images, path_to_labels, epochs, lr):

        image_files = os.listdir(path_to_images)[:1500] # all of the images 60.000    dataset now = 1500
        label_files = os.listdir(path_to_labels)[:1500]
        for epoch in range(epochs):
            i = 0
            for image_file, label_file in zip(image_files, label_files):

                image = get_image(path_to_images, image_file)
                label = get_label(path_to_labels, label_file)
                one_hot = np.zeros(10)
                one_hot[label] = 1      

                self.backpropagation(image, one_hot)



                if (i % 32) or (i==0): # 0-31,33-63,65-...
                    pass               # gather weights
                else:                  # 32,64,...
                    self.update_weights_n_biases(lr)
                    self.weight_adjustment = [np.zeros_like(matrix) for matrix in self.all_weights]
                    self.bias_adjustment = [np.zeros_like(matrix) for matrix in self.all_biases]







def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def softmax(x):
    exps = jnp.exp(x - jnp.max(x))
    return exps / jnp.sum(exps)

def mean_log_maybe(y, y_):
    return -jnp.mean(y * jnp.log(y_ + 1e-8))



# model creation
model = NeuralNetwork()
model.add_layer(784, Type="Input")
model.add_layer(100, activation=sigmoid)
model.add_layer(100, activation=sigmoid)
model.add_layer(10, activation=softmax, Type="Output")

model.add_loss_function(mean_log_maybe)



def get_image(path_to_images, image_file):
    x  = os.path.join(path_to_images, image_file)
    with open(x, 'rb') as file:
        byte_data = file.read()
        npimage = np.frombuffer(byte_data, dtype=np.uint8) / 255.0  # Normalize input
    return npimage

def get_label(path_to_labels, label_file):
    x  = os.path.join(path_to_images, label_file)
    with open(x, 'rb') as file:
        byte_data = file.read()
        nplabel = np.frombuffer(byte_data, dtype=np.uint8)[0]
    return nplabel


path_to_images = os.path.join("mnist_data", "images")
path_to_labels = os.path.join("mnist_data", "labels")


model.train(path_to_images, path_to_labels, 5,  0.01)
