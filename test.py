
import numpy as np
import matplotlib.pyplot as plt
import os
import json

def sig(x):
    return 1 / (1 + np.exp(-x))

def sig_(x):
    return x * (1 - x)

def loss(y_, y):
    return np.sum((y_ - y) ** 2)

def loss_(y_, y):
    return 2*(y_ - y)


def get_image(path_to_images, i):
    file_path = os.path.join(path_to_images, f'{i+1}.bin')
    with open(file_path, 'rb') as file:
        byte_data = file.read()
        npimage = np.frombuffer(byte_data, dtype=np.uint8) / 255.0  # Normalize input
    return npimage

def get_label(path_to_labels, i):
    file_path = os.path.join(path_to_labels, f'{i+1}.bin')
    with open(file_path, 'rb') as file:
        byte_data = file.read()
        nplabel = np.frombuffer(byte_data, dtype=np.uint8)[0]
    return nplabel





class NeuronLayer:
    def __init__(self, layer_id, type_, num_of_neurons, num_of_prev_neurons):
        self.layer_id = layer_id
        self.type = type_
        self.number_of_nodes = num_of_neurons
        self.activation = "sigmoid"
        if self.type != "input":
            self.weights = np.random.randn(num_of_neurons, num_of_prev_neurons) / num_of_prev_neurons
            self.bias = np.random.randn(num_of_neurons) / num_of_prev_neurons

    def calculate(self, inputs):
        # inputs = [1,2,3,4] (all the previous nodes outputs)
        # self.inputs = inputs
        self.z = np.dot(self.weights, inputs) + self.bias
        self.a = sig(self.z)
        return self.a

class NeuralNetwork:
    def __init__(self, input_nodes, output_nodes):
        self.number_of_middle_layers = 0
        self.layers = []
        self.layers.append(NeuronLayer(0, "input", input_nodes, None))
        self.layers.append(NeuronLayer(1, "output", output_nodes, input_nodes))

    def add_layer(self, number_of_neurons):
        self.number_of_middle_layers += 1
        num_of_prev_neurons = self.layers[self.number_of_middle_layers - 1].number_of_nodes
        new_layer = NeuronLayer(self.number_of_middle_layers, "normal", number_of_neurons, num_of_prev_neurons)
        self.layers.insert(self.number_of_middle_layers, new_layer)
        # Update ID and weights for output layer
        output_layer = self.layers[-1]
        output_layer.layer_id += 1
        output_layer.weights = np.random.randn(output_layer.number_of_nodes, number_of_neurons)

    def forward(self, inputs):
        x = inputs
        for layer in self.layers[1:]:  # Skip input layer
            x = layer.calculate(x)
        return x

    def predict(self, inputs):
        a = self.forward(inputs)
        return np.argmax(a)

    def backwards(self, x, y, lr):# lr = lr

        weight_adjustment = []
        bias_adjustment = []
        self.all_weights = []
        self.all_biases = []

        output_layer = self.layers[-1]
        y_ = self.forward(x)
        cost = loss(y_, y)
        
        delta_output = loss_(y_, y) * sig_(output_layer.a)
        dL_dW = np.outer(delta_output, self.layers[-2].a)
        dL_db = delta_output
        weight_adjustment.append(dL_dW)
        bias_adjustment.append(dL_db)
        self.all_weights.append(output_layer.weights)
        self.all_biases.append(output_layer.bias)
        #output_layer.weights -= lr * dL_dW
        #output_layer.bias -= lr * dL_db
        
        delta_ip1 = delta_output  # delta i+1

        for i in range(self.number_of_middle_layers, 1, -1):
            
            prev_layer = self.layers[i-1]
            layer = self.layers[i]
            next_layer = self.layers[i+1]

            delta_i = np.dot(next_layer.weights.T, delta_ip1) * sig_(layer.a)
            dL_dW = np.outer(delta_i, prev_layer.a)
            dL_db = delta_i
            weight_adjustment.append(dL_dW)
            bias_adjustment.append(dL_db)
            self.all_weights.append(layer.weights)
            self.all_biases.append(layer.bias)
            #layer.weights -= lr * dL_dW
            #layer.bias -= lr * dL_db

            delta_ip1 = delta_i

        layer_1 = self.layers[1]
        layer_2 = self.layers[2]

        delta_1 = np.dot(layer_2.weights.T, delta_ip1) * sig_(layer_1.a)
        dL_dW = np.outer(delta_1, x)
        dL_db = delta_1
        weight_adjustment.append(dL_dW)
        bias_adjustment.append(dL_db)
        self.all_weights.append(layer_1.weights)
        self.all_biases.append(layer_1.bias)
        #layer_1.weights -= lr * dL_dW
        #layer_1.bias -= lr * dL_db
        return weight_adjustment, bias_adjustment

    def update_weights(self,new_weight_adjustment, new_bias_adjustment, lr):
        for i in range(len(self.all_weights)):
            weight_adjustment = [lr * matrix for matrix in new_weight_adjustment]
            bias_adjustment = [lr * matrix for matrix in new_bias_adjustment]
            self.all_weights[i] -= weight_adjustment[i]
            self.all_biases[i] -= bias_adjustment[i]

    def update_weights_every_x_examples(self, i, image, label, lr, update_weights_every):
        if i == 0:
            return
        if (i - 1) % update_weights_every == 0: # (32+1) or 0                                           1,33,65
            self.prev_weight_adjustment, self.prev_bias_adjustment = self.backwards(image, label, lr)
        else:
            weight_adjustment, bias_adjustment = self.backwards(image, label, lr)
            new_weight_adjustment = [m1 + m2 for m1, m2 in zip(weight_adjustment, self.prev_weight_adjustment)]
            new_bias_adjustment= [m1 + m2 for m1, m2 in zip(bias_adjustment, self.prev_bias_adjustment)]
            self.prev_weight_adjustment = new_weight_adjustment 
            self.prev_bias_adjustment = new_bias_adjustment

        if (not(i % update_weights_every)) and (i != 0): # dont run at 0 and runs at every 32            32,64,96
            self.update_weights(new_weight_adjustment, new_bias_adjustment, lr)

    def train(self, path_to_images, path_to_labels, lr, epochs, batch):

        for epoch in range(epochs):
            correct = 0
            for i in range(int(len(os.listdir(path_to_images))/10)): # all of the images 60.000
                
                image = get_image(path_to_images, i)
                label = get_label(path_to_labels ,i)
                self.update_weights_every_x_examples(i, image, label, lr, update_weights_every=batch)
                
                if self.predict(image) == label:
                    correct+=1
            correct/=(len(os.listdir(path_to_images))/100)
            print(f'epoch: {epoch+1}, accuracy: {correct}')

        del self.prev_weight_adjustment
        del self.prev_bias_adjustment



# model creation
model = NeuralNetwork(784,10)
model.add_layer(100)
model.add_layer(100)

# "dataset creation"
path_to_images = os.path.join('mnist_data', 'images')
path_to_labels = os.path.join('mnist_data', 'labels')

# training
#model.train(path_to_images, path_to_labels, lr= 0.01, epochs = 5, batch = 32)

with open('mine.json', 'r') as json_file:
    weights_list = json.load(json_file)

# Convert the lists back to numpy arrays
weights = [np.array(w) for w in weights_list]


model.layers[1].weights = weights[0]
model.layers[1].biases = weights[1]
model.layers[2].weights = weights[2]
model.layers[2].biases = weights[3]
model.layers[3].weights = weights[4]
model.layers[3].biases = weights[5]






import numpy as np
import matplotlib.pyplot as plt

x = []
y = []
for i in range(20):
    image = get_image(path_to_images, i)
    label = get_label(path_to_labels, i)
    x.append(image)
    y.append(label)
x = np.array(x)  # Convert list to numpy array after appending all elements
y = np.array(y)  # Convert labels to numpy array if needed

y_hat = []
for i in range(20):
    y_hat.append(model.predict(x[i]))

correct_count = 0
results = []
for i in range(20):
    if y_hat[i] == y[i]:
        results.append("right")
        correct_count += 1
    else:
        results.append("wrong")

print(correct_count)

num_images = 20
num_cols = 5
num_rows = num_images // num_cols

fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

for i in range(num_images):
    ax = axes[i // num_cols, i % num_cols]
    ax.imshow(x[i].reshape(28, 28), cmap='gray')
    color = 'green' if results[i] == "right" else 'red'
    ax.set_title(f'{y[i]} vs {y_hat[i]}', color=color)
    ax.axis('off')

plt.show()


