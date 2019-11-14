import numpy as np
import math

def sigmoid(z):
    return 1/(1+np.exp(-z))

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

    def forward_propogation(self, input):
        for i in range(len(self.layer_list)):
            if i == 0:
                self.layer_list[i].inputs = input
                self.layer_list[i].outputs = input
            else:
                self.layer_list[i].inputs = self.layer_list[i-1].outputs.dot(self.layer_list[i].weights.T)
                self.layer_list[i].outputs = sigmoid(self.layer_list[i].inputs)

    def backpropogation(self):
        pass

    def update_weight(self):
        pass

    def train(self):
        pass
    """
    checking forward propogation
        def print_layer(self):
        for i in self.layer_list:
            print("______layer______\n")
            print("layer input is {}, shape {}\n".format(i.inputs, i.inputs.shape))
            print("layer output is {}, shape {}\n".format(i.outputs, i.outputs.shape))
            print("layer weight is {}, shape {}\n".format(i.weights, i.weights.shape))
    """

class Layer:
    def __init__(self, neurons, previous_layer_neurons):
        self.neurons = neurons
        self.inputs = np.zeros((1, previous_layer_neurons))
        self.outputs = np.zeros((1, neurons))
        self.error = np.zeros((1, neurons))
        self.weights = np.empty((neurons, previous_layer_neurons))


obj = ANN([3,3,3,2])
obj.forward_propogation(np.array([[1,2,1]]))


