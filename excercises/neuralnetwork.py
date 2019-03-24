from neuron import *
import random

# This class automatically constructs a NN given the number of desired nodes.
# The class assumes all nodes of each layer have a dependency on all nodes of the previous layer.
# Bias nodes don't have to be added manually, the weights have to be included in the list though
class NeuralNetwork:
    def __init__(self, initialValues, listsOfWeights):
        self.neurons = [[StartNeuron(-1)] + [StartNeuron(init) for init in initialValues]]
        for layer in listsOfWeights:
            self.neurons.append([StartNeuron(-1)] + [Neuron(self.neurons[-1], weights) for weights in layer])
        self.neurons[-1] = self.neurons[-1][1:] # Throw away the bias that is present in the last layer

    def __str__(self):
        return "NeuralNetwork object with the following neurons:\n" + str([str(layer) + "\n" for layer in self.neurons])

    def evaluate(self):
        return [endNeuron.evaluate() for endNeuron in self.neurons[-1]]

    def evaluateWith(self, inits):
        assert len(inits) + 1 == len(self.neurons[0]), "Not the correct number of input values"
        self.reset()
        self.changeInits(inits)
        return self.evaluate()

    def reset(self):
        for endNeuron in self.neurons[-1]:
            endNeuron.reset()

    def changeInits(self, initialValues):
        assert len(initialValues) + 1 == len(self.neurons[0]), "Not the correct number of input values"
        for i in range(len(initialValues)):
            self.neurons[0][i+1].changeInit(initialValues[i])

    def cost(self, trainingValues): # Assuming trainingValues is a list of tuples ([input], [expected output])
        result = 0
        for training in trainingValues:
            result += distance(self.evaluateWith(training[0]), training[1])
        return result/(2*len(trainingValues))

    def singleLayerDelta(self, eta, trainingValues): # Same assumptions for traingingValues as for cost
        for training in trainingValues:
            desiredEndNeurons = training[1]
            realEvaluation = self.evaluateWith(training[0])
            for k in range(len(self.neurons[-1])): # Need to iterate over two lists at the same time
                desiredEndNeuronOutput = desiredEndNeurons[k]
                realEndNeuron = realEvaluation[k]
                newWeights = []
                for i in range(len(self.neurons[-1][k].weights)): # Again 2 different lists to iterate
                    newWeights.append(self.neurons[-1][k].weights[i] + eta * sigmoid(self.neurons[-1][k].parents[i].evaluation) * 
                                      sigmoidP(sum(zipWith([p.evaluation for p in self.neurons[-1][k].parents], self.neurons[-1][k].weights, operator.mul))) *
                                     (sigmoid(realEndNeuron) - sigmoid(desiredEndNeuronOutput)))
                self.neurons[-1][k].weights = newWeights

    # def backpropagation(self, eta, trainingValues):
    #     for training in trainingValues:
    #         desiredEndNeurons = training[1]
    #         realEvaluation = self.evaluateWith(training[0])
    #         errorDelta = []
    #         for index in range(len(self.neurons) - 1): # For every layer except the starting layer, but gonna walk through in reverse order
    #             i = len(self.neurons) - index - 1
    #             neuronErrors = []
    #             if index == 0:
    #                 for j in range(len(self.neurons[i])): # So for every endneuron
    #                     weightErrors = []
    #                     desiredEndNeuronOutput = desiredEndNeurons[j]
    #                     for k in range(len(self.neurons[i][j].weights)): # k is the index of the weight in the neuron j of layer i
    #                         weightErrors.append(eta * sigmoid(self.neurons[i][j].parents[i].evaluation) * 
    #                                             sigmoidP(sum(zipWith([p.evaluation for p in self.neurons[i][j].parents], self.neurons[i][j].weights, operator.mul))) *
    #                                             (sigmoid(realEvaluation[j]) - sigmoid(desiredEndNeuronOutput)))
    #                     neuronErrors.append(weightErrors)
    #             else:
    #                 for j in range(len(self.neurons[i]) - 1): # j is the index of the current neuron in layer i, but we want only non-bias neurons, so j = j+1
    #                     weightErrors = []
    #                     childErrorsList = list(map(list, zip(*errorDelta[0])))# Tricky way to transpose matrix, and then we get the j + 1 element, which should be all parents
    #                     childErrors = childErrorsList[j+1]
    #                     for k in range(len(self.neurons[i][j+1].weights)): # k is the index of the weight in neuron j of layer i
    #                         weightErrors.append(eta * sigmoid(self.neurons[i][j+1].parents[k].evaluation) *
    #                                             sigmoidP(sum(zipWith([p.evaluation for p in self.neurons[i][j+1].parents], self.neurons[i][j+1].weights, operator.mul))) *
    #                                             sum(zipWith(childErrors[k], [child.weights[k] for child in self.neurons[i+1]], operator.mul)))
    #                     neuronErrors.append(weightErrors)
    #             errorDelta.insert(0, neuronErrors)

# Helper functions
def distance(x, y): # between two lists
    help = 0
    assert len(x) == len(y), "Hypo only possible for same dimensions"
    for i in range(len(x)):
        help += (y[i]-x[i])**2
    return math.sqrt(help)


# Used for testing
def main():
    print("Testing of NeuralNetwork with NOR from excercise 4.1 combined with an AND")
    testNN = NeuralNetwork([0, 0, 0], [[[-1, -2, -2, -2], [2.5, 1, 1, 1]]])
    for a in range(2):
        for b in range(2):
            for c in range(2):
                print(f"With start values {a}, {b} and {c}, the results are {testNN.evaluateWith([a,b,c])}")
    print("\nTesting of singleLayerDelta:")
    testDelta = NeuralNetwork([0,0,0],[[[random.uniform(-1,1) for i in range(4)],[random.uniform(-1,1) for i in range(4)]]])
    print("We want to get two output neurons, a NOR and an AND gate, using the single layer delta rule. \nThe NN beforehand: " + str(testDelta))
    trainingData = []
    for a in range(2):
        for b in range(2):
            for c in range(2):
                trainingData.append(([a,b,c], [int(not (a or b or c)), int(a and b and c)]))
    for n in range(50):
        testDelta.singleLayerDelta(0.1, trainingData)
    print("End result: " + str(testDelta))

    # print("\nTesting of backpropagation with XOR:\n")
    # trainingData = []
    # for a in range(2):
    #     for b in range(2):
    #         trainingData.append(([a, b], [int((a or b) and not (a == b))]))
    # testxor = NeuralNetwork([0,0],[[[random.uniform(-1,1) for i in range(2)],[random.uniform(-1,1) for i in range(2)]], [[random.uniform(-1, 1) for i in range(2)]]])
    # print("The NN beforehand: " + str(testxor))
    # for n in range(100):
    #     testxor.backpropagation(0.01, trainingData)

# Only execute main when testing the module, not when it is loaded by something else
if __name__ == '__main__':
    main()