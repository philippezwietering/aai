"""
For this excercise the genotype is an array of booleans. The booleans represent if the (index + 1)th number is in the sum or multiplication pile,
with False being in the sum pile and True being in the multiplication pile. For this excercise every genotype will have a length of 10 of course.
"""

import random

def createPopulation(popSize, length):
    """
    Create a population
    :param popSize: population size
    :param length: needed for create_individual
    :return: returns a list of individuals; a population
    """
    return [createIndividual(length) for _ in range(popSize)]


def createIndividual(length):
    """
    Create an individual
    :param length: length of the created genotype
    :return: returns a list with values representing an individual
    """
    return [random.choice([True, False]) for _ in range(length)]

def fitness(genotype, desiredSum, desiredMultiplication):
    """
    Calculates fitness of an individual based on the desired sum and the desired multiplication
    :param genotype: list representing the genotype
    :param desiredSum: integer representing the desired sum
    :param desiredMultiplication: integer representing the desired multiplication
    :return: fitness as a number, closer to zero is better
    """
    sum = 0
    multiplication = 0
    for index in range(len(genotype)):
        if not genotype[index]:
            sum += index + 1
        else:
            if multiplication == 0:
                multiplication = index + 1
            else:
                multiplication *= index + 1
    resultSum = abs(sum - desiredSum)
    resultMultiply = abs(multiplication - desiredMultiplication)
    return resultSum+resultMultiply

def sortGeneration(pop, desiredSum, desiredMultiply):
    """
    Get a population and return it sorted based on the performance of the fitness function, so the best genotypes are easily selected from the population.
    This selection procedure usually isn't complete enough, because you can get stuck in a local maximum, but it appears it suffices for this excercise, probably
    because of the high mutation factor I have chosen to implement and in the textbook it is clearly stated that with a high enough mutation rate (not too high either),
    no recombination is needed
    :param pop: population to sort
    :param desiredSum: needed for fitness function
    :param desiredMultiply: needed for fitness function
    :return: returns a sorted population
    """
    fitnessList = []
    for genotype in pop:
        fitnessList.append(fitness(genotype, desiredSum, desiredMultiply))
    return [genotype for fit,genotype in sorted(zip(fitnessList, pop))]

def createNewGeneration(population, numberToKeep):
    """
    Create a new generation by inverting one gene in the genotype as mutation
    :param population: expects a list of a generation ordered by fitness, the best are first, worst are last
    :param numberToKeep: number of 'best' that you want to keep, the rest gets mutated
    :return: returns the new generation
    """
    for genotype in population[numberToKeep:]:
        index = random.randint(0, len(genotype)-1)
        genotype[index] = not genotype[index]
    return population # individuals are now mutated so just return the new generation

population = createPopulation(100, 10)
desiredSum = 36
desiredMultiplication = 360
numberOfGenerations = 10

for _ in range(numberOfGenerations):
    population = sortGeneration(population, desiredSum, desiredMultiplication)
    population = createNewGeneration(population, 10)

population = sortGeneration(population, desiredSum, desiredMultiplication) # sort generation so we can take the best 10
print("The best genotype is: {} with a fitness of: {}".format(population[0], fitness(population[0], desiredSum, desiredMultiplication))) # index 0 because of the sorted population

"""
The algorithm performs amazing, because it oftens finds the perfect combination of cards in the sum and multiplication pile. If you run it for example a 100 times, 
you can see that the fitness will rarely go over 1 or 2. 
"""