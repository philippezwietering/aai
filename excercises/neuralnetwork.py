from neuron import *

# This class automatically constructs a NN given the number of desired nodes.
# The class assumes all nodes of each layer have a dependency of all nodes of the previous layer.
# Bias nodes don't have to be added manually, the weights have to be included in the list though
class NeuralNetwork:
    def __init__(self, initialValues, listsOfWeights):
        self.startNeurons = [StartNeuron(-1)] + [StartNeuron(init) for init in initialValues]
        self.neurons = []
        firstLayer = True
        for layer in listsOfWeights:
            if not firstLayer:
                self.neurons.append([StartNeuron(-1)] + [Neuron(self.neurons[-1], weights) for weights in layer])
            else:
                firstLayer = False
                self.neurons.append([StartNeuron(-1)] + [Neuron(self.startNeurons, weights) for weights in layer])
        self.neurons[-1] = self.neurons[-1][1:] # Throw away the bias that is present in the last layer

    def evaluate(self):
        self.reset()
        return [endNeuron.evaluate() for endNeuron in self.neurons[-1]]

    def evaluateWith(self, inits):
        if not len(inits) + 1 == len(self.startNeurons):
            return
        else:
            self.reset()
            self.changeInits(inits)
            return self.evaluate()

    def reset(self):
        for endNeuron in self.neurons[-1]:
            endNeuron.reset()

    def changeInits(self, initialValues):
        if not len(initialValues) + 1 == len(self.startNeurons):
            return
        else:
            for i in range(len(initialValues)):
                self.startNeurons[i+1].changeInit(initialValues[i])

    def cost(self, trainingValues): # Assuming trainingValues is a list of tuples (input, expected output)
        result = 0
        for training in trainingValues:
            result += distance(self.evaluateWith(training[0]), training[1])
        return result/(2*len(trainingValues))


# Helper functions
def distance(x, y): # between two lists
    help = 0
    if len(x) != len(y):
        return -1
    for i in range(len(x)):
        help += (y[i]-x[i])**2
    return math.sqrt(help)

def main():
    print("Testing of NeuralNetwork with NOR from excercise 4.1 combined with an AND")
    testNN = NeuralNetwork([0, 0, 0], [[[1, 2, 2, 2], [2.5, 1, 1, 1]]])
    for a in range(2):
        for b in range(2):
            for c in range(2):
                print(f"With start values {a}, {b} and {c}, the results are {testNN.evaluateWith([a,b,c])}")

# Only execute main when testing the module, not when it is loaded by something else
if __name__ == '__main__':
    main()