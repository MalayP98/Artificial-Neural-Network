import numpy as np
import math


class ANN:
    def __init__(self, define):
        self.define = define
        self.layer_list = []
        for i in range(len(self.define)):
            if i == 0:
                layer = Layer(self.define[i],1)
                self.layer_list.append(layer)
            else:
                layer = Layer(self.define[i], self.define[i-1])
                self.layer_list.append(layer)

    def forward_propogation(self):
        pass

    def backpropogation(self):
        pass

    def update_weight(self):
        pass

    def train(self):
        pass


class Layer:
    def __init__(self, neurons, previous_layer_neurons):
        self.neurons = neurons
        self.inputs = [0] * previous_layer_neurons
        self.outputs = [0] * neurons
        self.weights = np.empty((neurons, previous_layer_neurons))


obj = ANN([3,1,1])
for i in range(len(obj.define)):
    print(obj.layer_list[i].weights, "\n")