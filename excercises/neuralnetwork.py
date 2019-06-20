from neuron import *
import random
import numpy as np

# This class automatically constructs a NN given the number of desired nodes.
# The class assumes all nodes of each layer have a dependency on all nodes of the previous layer.
# Bias nodes don't have to be added manually
class NeuralNetwork:
    # This constructor initializes a NN with a number of input nodes corresponding with the length of initialValues and initializes them with those values
    # listOfNodes should be a list of ints, where each int represents the number of nodes in a layer, so the last int is the amount of endnodes
    # The learnRate (or eta) is used in the backPropagation function
    # The weights are assigned randomly between -1 and 1. Biases are added automatically
    def __init__(self, initialValues, listOfNodes, learnRate):
        self.neurons = [[StartNeuron(-1)] + [StartNeuron(init) for init in initialValues]]
        for layer in listOfNodes:
            self.neurons.append([StartNeuron(-1)] + [Neuron(self.neurons[-1], [random.uniform(-1,1) for _ in self.neurons[-1]]) for _ in range(layer)])
        self.neurons[-1] = self.neurons[-1][1:] # Throw away the bias that is present in the last layer

        self.learnRate = learnRate

    # Prints all the nodes
    def __str__(self):
        return "NeuralNetwork object with the following neurons:\n" + str([str(layer) + "\n" for layer in self.neurons])

    # This function evaluates the endNeurons of the NN with unchanged inputNeurons
    def evaluate(self):
        return [endNeuron.evaluate() for endNeuron in self.neurons[-1]]

    # This function evaluates the endNeurons with the values stored in inits
    def evaluateWith(self, inits):
        assert len(inits) + 1 == len(self.neurons[0]), "Not the correct number of input values"
        self.reset()
        self.changeInits(inits)
        return self.evaluate()

    # This function resets all the evaluated values in all the neurons
    def reset(self):
        for endNeuron in self.neurons[-1]:
            endNeuron.reset()

    # This function changes the inputs of the input layer to the values in the list initialValues
    def changeInits(self, initialValues):
        assert len(initialValues) + 1 == len(self.neurons[0]), "Not the correct number of input values"
        for i in range(len(initialValues)):
            self.neurons[0][i+1].changeInit(initialValues[i])

    def cost(self, trainingValues): # Assuming trainingValues is a list of tuples ([input], [expected output])
        result = 0
        for training in trainingValues:
            result += distance(self.evaluateWith(training[0]), training[1])
        return result/(2*len(trainingValues))

    # def singleLayerDelta(self, eta, trainingValues): # Same assumptions for traingingValues as for cost
    #     for training in trainingValues:
    #         desiredEndNeurons = training[1]
    #         realEvaluation = self.evaluateWith(training[0])
    #         for k in range(len(self.neurons[-1])): # Need to iterate over two lists at the same time
    #             desiredEndNeuronOutput = desiredEndNeurons[k]
    #             realEndNeuron = realEvaluation[k]
    #             newWeights = []
    #             for i in range(len(self.neurons[-1][k].weights)): # Again 2 different lists to iterate
    #                 newWeights.append(self.neurons[-1][k].weights[i] + eta * relu(self.neurons[-1][k].parents[i].evaluation) * 
    #                                   reluP(sum(zipWith([p.evaluation for p in self.neurons[-1][k].parents], self.neurons[-1][k].weights, operator.mul))) *
    #                                  (relu(realEndNeuron) - relu(desiredEndNeuronOutput)))
    #             self.neurons[-1][k].weights = newWeights

    # Assuming the errors for the whole layer are calculated
    # This function is used to calculate the delta rule for hidden layers
    # layerIndex is the index for the layer in which the neuron is residing you want the sumError of
    # neuronIndex is the index for the neuron in a layer you want the sumError of
    def nodeSumError(self, layerIndex, neuronIndex):
        result = 0
        for neuron in self.neurons[layerIndex]:
            if type(neuron) is Neuron:
                result += neuron.weights[neuronIndex] * neuron.error
        return result

    # This function applies backPropagation to a list of trainingValues, which should be tuples in the form (input, oracle)
    def backPropagation(self, trainingValues):
        for training in trainingValues:
            desiredOutput = training[1]
            self.evaluateWith(training[0])

            for index in range(len(self.neurons) - 1):
                i = len(self.neurons) - index - 1 # To get the layer, starting from the back, and we don't need to deltarule the input layer
                if index == 0:
                    for endNeuronIndex in range(len(self.neurons[i])):
                        self.neurons[i][endNeuronIndex].deltaRule(None, desiredOutput[endNeuronIndex])
                else:
                    for neuronIndex in range(len(self.neurons[i])):
                        hiddenError = self.nodeSumError(i + 1, neuronIndex)
                        self.neurons[i][neuronIndex].deltaRule(hiddenError, None)

            for layer in self.neurons:
                for neuron in layer:
                    neuron.updateWeights(self.learnRate)

# Helper function
def distance(x, y): # between two lists
    help = 0
    assert len(x) == len(y), "Hypo only possible for same dimensions"
    for i in range(len(x)):
        help += (y[i]-x[i])**2
    return math.sqrt(help)


# Used for testing
def main():
    # print("Testing of NeuralNetwork with NOR from excercise 4.1 combined with an AND")
    # testNN = NeuralNetwork([0, 0, 0], [2], 0.01)
    # for a in range(2):
    #     for b in range(2):
    #         for c in range(2):
    #             print(f"With start values {a}, {b} and {c}, the results are {testNN.evaluateWith([a,b,c])}")
    # print("\nTesting of singleLayerDelta:")
    # testDelta = NeuralNetwork([0,0,0],[2], 0.1)
    # print("We want to get two output neurons, a NOR and an AND gate, using the single layer delta rule. \nThe NN beforehand: " + str(testDelta))
    # trainingData = []
    # for a in range(2):
    #     for b in range(2):
    #         for c in range(2):
    #             trainingData.append(([a,b,c], [int(not (a or b or c)), int(a and b and c)]))
    # for _ in range(50):
    #     testDelta.singleLayerDelta(0.1, trainingData)
    # print("End result: " + str(testDelta))

    # print("\nTesting of backpropagation with XOR:\n")
    # trainingData = []
    # for a in range(2):
    #     for b in range(2):
    #         trainingData.append(([a, b], [int((a or b) and not (a == b))]))
    # testxor = NeuralNetwork([0,0],[2, 1], 0.01)
    # print("The NN beforehand: " + str(testxor))
    # for _ in range(1000):
    #     testxor.backPropagation(trainingData)
    # print("The NN afterwards: " + str(testxor))
    # for a in range(2):
    #     for b in range(2):
    #         print(f"With start values {a} and {b}, the test results are {testxor.evaluateWith([a,b])}")
    
    # Used for conversion of plants
    plantDic = {"Iris-setosa": [1, 0, 0],
                "Iris-versicolor": [0, 1, 0],
                "Iris-virginica": [0, 0, 1]}

    # Loading trainingdata
    data = np.genfromtxt("/home/philippe/Documents/school/aai/resources/data.txt", delimiter=',', usecols=[0, 1, 2, 3]).tolist()
    flowerNames = np.genfromtxt("/home/philippe/Documents/school/aai/resources//data.txt", dtype=str, delimiter=',', usecols=[4])

    flowerTraining = []
    for flowerType in flowerNames:
        flowerTraining.append(plantDic[flowerType])
    
    for i in range(len(data)):
        data[i] = (data[i], flowerTraining[i])

    # Loading testdata
    testData = np.genfromtxt("/home/philippe/Documents/school/aai/resources//testData.txt", delimiter=',', usecols=[0, 1, 2, 3]).tolist()
    testFlowerNames = np.genfromtxt("/home/philippe/Documents/school/aai/resources//testData.txt", dtype=str, delimiter=',', usecols=[4])

    flowerTests = []
    for flowerName in testFlowerNames:
        flowerTests.append(plantDic[flowerName])
    
    # I have chosen a network of 1 layer of 4 hidden nodes and 3 output nodes, corresponding with each possible iris
    flowerNN = NeuralNetwork([0,0,0,0], [4,4,3], 0.01)
    print("Iris NN before backpropagation: ", flowerNN)

    epochs = 20
    for _ in range(epochs):
        flowerNN.backPropagation(data)
    print("Iris NN after backpropagation: ", flowerNN)

    for testIndex in range(len(testData)):
        print(flowerNN.evaluateWith(testData[testIndex]), flowerTests[testIndex])

# Only execute main when testing the module, not when it is loaded by something else
if __name__ == '__main__':
    main()