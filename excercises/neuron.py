import operator
import math

# Neuron class
# NB. End neurons are just normal neurons with a different meaning, but they have the exact same functionality
# The weight i,j is saved in node j, so it is way easier to evaluate stuff and it saves on functionality necessary for startNodes
class Neuron:
    def __init__(self, parents, weights):
        self.parents = parents
        self.weights = weights
        self.evaluated = False
        self.evaluation = 0
        self.error = 0

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Neuron with weights of " + str(self.weights)

    # Evaluation works kind of lazy, so it saves a lot of CPU in larger networks with a lot of interconnected neurons
    def evaluate(self):
        if not self.evaluated:
            parentEvaluations = [p.evaluate() for p in self.parents]
            self.evaluation = sigmoid(sum(zipWith(parentEvaluations, self.weights, operator.mul)))
            self.evaluated = True
        return self.evaluation

    # Again, lazy evaluated, should save a little bit of time
    def reset(self):
        if self.evaluated:
            self.evaluated = False
            for parent in self.parents:
                parent.reset()

    # If expectation is None then this is for hidden nodes, if weightsErrors is None then this is for output nodes (but one should be filled)
    def deltaRule(self, weightsErrors = None, oracle = None): 
        if weightsErrors is not None and oracle is not None or weightsErrors is None and oracle is None:
            return

        parentEvaluations = [p.evaluate() for p in self.parents]
        incoming = sum(zipWith(parentEvaluations, self.weights, operator.mul))

        if oracle is None:
            self.error = sigmoidP(incoming) * weightsErrors
        if weightsErrors is None:
            self.error = sigmoidP(incoming) * (oracle - self.evaluation)
        return self.error

    def updateWeights(self, learnRate):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + learnRate * self.parents[i].evaluation * self.error


# The little brother of neuron, the startneuron doesn't have a lot of functionality
class StartNeuron:
    def __init__(self, initialValue):
        self.evaluation = initialValue
        self.evaluated = True

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "StartNeuron with value " + str(self.evaluation)

    def evaluate(self):
        return self.evaluation

    def reset(self): # So we can recursively reset the evaluations without having to think about the startneurons, which are just less special
        return

    def changeInit(self, value):
        self.evaluation = value

    def getInit(self):
        return self.evaluation

class Bias:
    def __init__(self):
        self.evaluation = -1
        self.evaluated = True

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Bias node"

    def evaluate(self):
        return self.evaluation

    def reset(self):
        return

# Helper functions
def zipWith(l1, l2, f):
    return [f(a, b) for (a, b) in zip(l1, l2)]

def sigmoid(x):
    return 1/(1+math.exp(-x))

def sigmoidP(x):
    return x*(1-x)

def stepFunction(x):
    return 0 if x < 0 else 1

# For testing neuron functionality, testing with the NOR-gate from excercise 4.1
# and a half-adder from excercise 4.2:
def main():
    print("Testing Neuron with NOR-gate\n---------------------------------------")
    bias = Bias()
    xNeuron = StartNeuron(0)
    yNeuron = StartNeuron(0)
    zNeuron = StartNeuron(0)
    norNeuron = Neuron([bias, xNeuron, yNeuron, zNeuron], [-1, -2, -2, -2])
    for x in range(2):
        xNeuron.changeInit(x)
        for y in range(2):
            yNeuron.changeInit(y)
            for z in range(2):
                zNeuron.changeInit(z)
                norNeuron.reset()
                print(f"With start neurons x: {xNeuron.getInit()}, y: {yNeuron.getInit()} and z: {zNeuron.getInit()}")
                print(f"the nor neuron evaluates to {norNeuron.evaluate()}\n")

    print("\nTesting Neuron with half-adder network\n-------------------------------------")
    aNeuron = StartNeuron(0)
    bNeuron = StartNeuron(0)
    x1Neuron = Neuron([bias, aNeuron, bNeuron], [1, 2, 2])
    x2Neuron = Neuron([bias, aNeuron, bNeuron], [-3, -2, -2])
    sumNeuron = Neuron([bias, x1Neuron, x2Neuron], [3, 2, 2])
    carryNeuron = Neuron([bias, aNeuron, bNeuron], [3, 2, 2])

    for a in range(2):
        aNeuron.changeInit(a)
        for b in range(2):
            bNeuron.changeInit(b)
            sumNeuron.reset()
            carryNeuron.reset()
            print(f"With start neurons a: {aNeuron.getInit()} and b: {bNeuron.getInit()}")
            print(f"the sum neuron evaluates to {sumNeuron.evaluate()} and the carry neuron evaluates to {carryNeuron.evaluate()}")


# Standard weird main handling for Python
if __name__ == "__main__":
    main()