import numpy as np
from random import choices


np.random.seed(7)

# Indexes
weight_i = 0
bias_i = 1
delta_i = 2


def sigmoid(x, derivative=False):
    sig = 1 / (1 + np.exp(-x))
    if derivative:
        return sig * (1 - sig)
    return sig


class Network:
    def __init__(self, network=None):
        network = [
            [self.layer(2, 20), sigmoid],
            [self.layer(20, 10), sigmoid],
            [self.layer(10, 2), sigmoid]
        ]
        self.network = network

    # Function to create weights and biases for a layer from input and output size (randomly initialised)
    def layer(self, input_size, output_size):
        w = np.subtract(np.multiply(np.random.rand(output_size, input_size), 2), 1)
        b = np.subtract(np.multiply(np.random.rand(output_size), 2), 1)
        return [w, b]

    # Training calls back propagation for images
    def train(self, images, labels, epochs=1, learning=0.1):
        for epoch in range(epochs):
            for i in choices(range(len(images))):
                image, label = images[i], labels[i]
                dw, db = self.backprop(image, label, learning)
                for j in range(len(self.network)):
                    self.network[j][0][bias_i] = np.add(self.network[j][0][bias_i], db[j])
                    self.network[j][0][weight_i] = np.add(self.network[j][0][weight_i], dw[j])
        return network

    # Single back propagation
    def backprop(self, image, label, learning):
        network = self.network
        layers = len(network)
        delta = np.array([np.zeros_like(network[i][0][bias_i]) for i in range(layers)])
        dw = np.array([np.zeros_like(network[i][0][weight_i]) for i in range(layers)])
        db = np.array([np.zeros_like(network[i][0][bias_i]) for i in range(layers)])
        _outs, _houts = self.forward_propagate(image, network)
        for i in reversed(range(len(network))):
            if i == len(network) - 1:
                delta[i] = np.multiply(label - _houts[i], sigmoid(_outs[i], derivative=True))
                delta[i] = np.multiply(label - _houts[i], sigmoid(_outs[i], derivative=True))
            else:
                delta[i] = np.multiply(np.dot(np.transpose(network[i + 1][0][weight_i]), delta[i + 1]),
                                       sigmoid(_outs[i], derivative=True))

            if i == 0:
                dw[i] = np.dot(np.outer(delta[i], np.transpose(image)), learning)
            else:

                dw[i] = np.dot(np.outer(delta[i], np.transpose(_houts[i - 1])), learning)
            db[i] = np.multiply(delta[i], learning)

        return dw, db

    # Forward propagate a layer, layer is just (w,b) tuple
    def forward_layer(self, _input, layer, activation_function):
        out = np.dot(layer[0], _input) + layer[1]
        hout = activation_function(out)
        return out, hout

    def forward_propagate(self, _input, network):
        outs, houts = [], []
        for _layer in network:
            # _input here is just the output before activation
            _out, _hout = self.forward_layer(_input, _layer[0], _layer[1])
            _input = _hout
            outs.append(_out)
            houts.append(_hout)
        return outs, houts

    # One hot encoding of labels
    def vectorise_label(self, _label, size=10):
        _label = int(_label)
        ar = [0] * size
        ar[_label] = 1
        return np.array(ar)

    def preprocess(self, mnist, index):
        _image = mnist['data'][index] / 255
        _label = self.vectorise_label(mnist['target'][index])
        return _image, _label


# Testing neural net with XOR
img1 = np.array([0, 1])
img2 = np.array([1, 0])
lab1 = np.array([1, 0])
lab2 = np.array([0, 1])
images = [img1, img2]
labels = [lab1, lab2]

network = Network()


print('-------------')
for i in range(100000):
    network.train(images=images, labels=labels, epochs=1, learning=0.1)
    if i % 1000 == 0:
        _, houts = network.forward_propagate(img1, network.network)
        print('Error 1 is {}'.format(sum(abs(lab1 - houts[-1])**2)))
        _, houts = network.forward_propagate(img2, network.network)
        print('Error 2 is {}'.format(sum(abs(lab2 - houts[-1])**2)))
        print('-------------')
