import numpy as np
import math


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def diff_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))


class ANN:
    def __init__(self, define, learning_rate):
        self.learning_rate = learning_rate
        self.define = define
        self.layer_list = []
        for i in range(len(self.define)):
            if i == 0:
                layer = Layer(self.define[i], 1)
                self.layer_list.append(layer)
            else:
                layer = Layer(self.define[i], self.define[i - 1])
                self.layer_list.append(layer)

    def forward_propogation(self, input):
        for i in range(len(self.layer_list)):
            if i == 0:
                self.layer_list[i].inputs = input
                self.layer_list[i].outputs = input
            else:
                self.layer_list[i].inputs = self.layer_list[i-1].outputs.dot(self.layer_list[i].weights.T)
                self.layer_list[i].outputs = sigmoid(self.layer_list[i].inputs)
                #print(self.layer_list[i].inputs,self.layer_list[i].outputs)

    def backpropogation(self, target):
        for i in range(len(self.layer_list) - 1, 0, -1):
            if i == len(self.layer_list) - 1:
                self.layer_list[i].error = self.layer_list[i].outputs - target
                for j in range(self.define[-1]):
                    self.layer_list[i].error[0, j] = self.layer_list[i].error[0, j] * diff_sigmoid(
                        self.layer_list[i].inputs[0, j])
                    #print(self.layer_list[i].error[0, j], self.layer_list[i].error[0, j].shape)
            else:
                error_prop = self.layer_list[i + 1].error.dot(self.layer_list[i + 1].weights)
                for j in range(error_prop.shape[1]):
                    self.layer_list[i].error[0, j] = error_prop[0, j] * diff_sigmoid(self.layer_list[i].inputs[0, j])
                    #print(self.layer_list[i].error[0, j], self.layer_list[i].error[0, j].shape)

    def update_weight(self):
        c = 0;
        for i in range(len(self.layer_list) - 1, 0, -1):
            for j in range(self.layer_list[i].neurons):
                c += 1
                weight_error = self.layer_list[i - 1].outputs * self.layer_list[i].error[0, j]
                """
                testing update_weight function
                print(self.layer_list[i - 1].outputs, "*", self.layer_list[i].error[0, j], "\n")
                print("shape is -- \n", weight_error.shape)
                """
                self.layer_list[i].weights[j] = self.layer_list[i].weights[j] - self.learning_rate * weight_error
                #print(self.layer_list[i].weights[j],"\n")

    def train(self, inputs, target, iterations):
        self.inputs = inputs
        self.target = target
        self.iteration = iterations
        for k in range(self.iteration):
            for l in range(self.inputs.shape[0]):
                #print(self.inputs[l].reshape(1, self.inputs.shape[1]),self.target[l].reshape(1, self.target.shape[1]))
                self.forward_propogation(self.inputs[l].reshape(1, self.inputs.shape[1]))
                self.backpropogation(self.target[l].reshape(1, self.target.shape[1]))
                self.update_weight()

            print(self.layer_list[len(self.define)-1].error)



    def predict(self, test):
        self.test = test
        self.forward_propogation(self.test.reshape(1, self.inputs.shape[1]))
        print(self.layer_list[len(self.define)-1].outputs)



    def print_layer(self):
        for i in self.layer_list:
            print("______layer______\n")
            """print("layer input is {}, shape {}\n".format(i.inputs, i.inputs.shape))
            print("layer output is {}, shape {}\n".format(i.outputs, i.outputs.shape))
            print("layer weight is {}, shape {}\n".format(i.weights, i.weights.shape))"""
            print("error is -- \n", i.error, i.error.shape)

    def print_layer2(self):
        for i in self.layer_list:
            print("______layer______\n")
            print("layer input is {}, shape {}\n".format(i.inputs, i.inputs.shape))
            print("layer output is {}, shape {}\n".format(i.outputs, i.outputs.shape))
            print("layer weight is {}, shape {}\n".format(i.weights, i.weights.shape))


class Layer:
    def __init__(self, neurons, previous_layer_neurons):
        self.neurons = neurons
        self.inputs = np.zeros((1, previous_layer_neurons))
        self.outputs = np.zeros((1, neurons))
        self.error = np.zeros((1, neurons))
        self.weights = np.zeros((neurons, previous_layer_neurons))


obj = ANN([3, 3, 3, 2], 0.1)
obj.forward_propogation(np.array([[1, 2, 1]]))
obj.print_layer2()
obj.backpropogation(np.array([[-1, -1]]))
obj.print_layer()
obj.update_weight()

inputs = np.array([[1,1,1],[1,0,1],[0,1,1],[0,0,1]])
target = np.array([[0],[1],[1],[0]])
obj2 = ANN([3,2,1], 0.9)
obj2.forward_propogation(np.array([[1,1]]))
obj2.backpropogation(np.array([[1]]))
obj2.update_weight()
obj2.print_layer2()
obj2.print_layer()
obj2.train(inputs, target, 20000)
obj2.predict(np.array([1,1,1]))
obj2.predict(np.array([1,0,1]))
obj2.predict(np.array([0,1,1]))
obj2.predict(np.array([0,0,1]))

