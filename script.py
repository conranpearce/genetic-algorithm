import random
import math
import plotly.express as px
import pandas as pd
from numpy import random as rand

# Global Variables
N = 20 # Chromosome length
P = 50 # Population
GENERATIONS = 300 # Increased to run more generations, decreasing the fitness
LOWER_BOUND = -32.0
UPPER_BOUND = 32.0
ITERATIONS = 10
SWAP_AMOUNT = int(P / 5)  # How many individuals are replaced with better solutions from the offspring
MUTATION_RATE = round(((1/P) + (1/N)) / 2, 2) # Mutation rate is between 1/population and 1/chromosome length
FITNESS_FUNCTION = "minimisationFitnessFunction2"

population, offspring = [], []
meanList, generationArr, bestArr, meanArr, minArr, minArrAverage, minAverageCalculated = [], [], [], [], [], [], []

# Setting the min average array values to 0 so that summation is able to take place
for i in range(0, GENERATIONS):
    minArrAverage.append(0)

class individual:
    gene = []
    fitness = 0

# Outputting a graph to display the average minimum value across the iterations of the GA
def setGraphMinimumAverage():
    d = {'Min Fitness Average': minAverageCalculated, 'Generations': generationArr }
    df = pd.DataFrame(data=d)
    fig = px.line(df, x='Generations', y='Min Fitness Average', title="GA")
    fig.show()

# Outputting a graph to visibly see the change in the fitness of the population after selection, crossover and mutation
def setGraph():
    d = {'Min Fitness': minArr, 'Mean Fitness': meanArr, 'Generations': generationArr } # This graph displays mean and minimum function
    df = pd.DataFrame(data=d)
    fig = px.line(df, x='Generations', y=['Min Fitness', 'Mean Fitness'], title="GA")
    fig.show()

# Setting and outputting the end fitness after tournament selection, crossover and mutation
def calcFitnesses(currentGeneration):
    endFitness, meanFitness = 0, 0
    minFitness = 10000
    # Calculate the minimum fitness by looping through the individuals in the population
    for i in population:
        endFitness = round(endFitness + i.fitness, 1)
        # Work out best fitness of the population (lowest fitness value)
        if (i.fitness < minFitness):
            minFitness = round(i.fitness,1)

    meanFitness = round(endFitness / P, 1) # Rounded mean fitness to nearest integer
    meanList.append(meanFitness)
    generationArr.append(currentGeneration)
    minArr.append(minFitness)
    meanArr.append(meanFitness)

    # Output the minimum fitness value and mean fitness value
    print("Min fitness is: ", minFitness)
    print("Mean fitness is: ", meanFitness)

# Setting individual genes to random values between the lower and upper bound
def setPopulation():
    for x in range (0, P):
        tempgene=[]
        for x in range (0, N):
            tempgene.append(round(random.uniform(LOWER_BOUND, UPPER_BOUND), 1))

        newIndividual = individual()
        newIndividual.gene = tempgene[:]
        population.append(newIndividual)

    # Setting fitness value in population
    for i in population:
        i.fitness = fitnessFunction(i.gene)

# Depending on which fitness function the user has chosen carry out that fitness calculation
def fitnessFunction(ind):
    if FITNESS_FUNCTION == "minimisationFitnessFunction":
        fitness = minimisationFitnessFunction(ind)
        return fitness
    elif FITNESS_FUNCTION == "minimisationFitnessFunction2":
        fitness = minimisationFitnessFunction2(ind)
        return fitness
    elif FITNESS_FUNCTION == "schwefelFitnessFunction":
        fitness = schwefelFitnessFunction(ind)
        return fitness

# Setting the fitness of an individual with a minimisation function. Lower and upper bounds are -5.12 and 5.12 respectively
def minimisationFitnessFunction(ind):
    fitness = N*len(ind)
    for i in range(0, len(ind)):
        fitness = fitness + (ind[i] * ind[i] - 10*math.cos(2*math.pi*ind[i]))
    return fitness

# Setting the fitness of an individual with a minimisation function. Lower and upper bounds are -32.0 and 32.0 respectively
def minimisationFitnessFunction2(ind): 
    fitnessSumFirst = 0
    fitnessSumSecond = 0
    for i in range(0, len(ind)):
        fitnessSumFirst += (ind[i] ** 2)
        fitnessSumSecond +=  math.cos(2*math.pi*ind[i])
    fitness = -20 * math.exp(-0.2 * math.sqrt((1/N) * fitnessSumFirst)) - math.exp((1/N) * fitnessSumSecond)

    return fitness

# Fitness function calculation for Schwefel's Function. Lower and upper bounds are -500.0 and 500.0 respectively
def schwefelFitnessFunction(ind):
    fitnessSum = 0
    for i in range(0, len(ind)):
        fitnessSum += ind[i] * math.sin(math.sqrt(math.fabs(ind[i])))
    fitness = (418.9829 * N) - fitnessSum

    return fitness

# Check if the fitness of the worst index in the original population is less than the fitness of the best index in the new population and swap the gene
def replaceChromosomes(lowestIndex, largestIndex, tempOffspring):
    if population[largestIndex].fitness > fitnessFunction(tempOffspring[lowestIndex]):
        population[largestIndex].gene = tempOffspring[lowestIndex]
        population[largestIndex].fitness = fitnessFunction(tempOffspring[lowestIndex])

# Find the worst solution in the population (largest fitness value) and return the index of this solution
def largestFitnessSolution():
    largest = -1000
    largestIndex = 0
    for i in range(0, P):
        tempFitness = fitnessFunction(population[i].gene)
        if tempFitness > largest:
            largest = tempFitness
            largestIndex = i

    return largestIndex

# Find the best solution in the new offspring population (lowest fitness value) and return the index of this solution
def lowestFitnessSolution(tempOffspring):
    lowest = 1000
    lowestIndex = 0
    for i in range(0, P):
        if (fitnessFunction(tempOffspring[i]) < lowest):
            lowest = fitnessFunction(tempOffspring[i])
            lowestIndex = i

    return lowestIndex

# Using tournament selection to set the offspring population
def tournamentSelection(population):
    for i in range(0, P):
        parent1 = random.randint(0, P-1)
        off1 = population[parent1]
        parent2 = random.randint(0, P-1)
        off2 = population[parent2]
        if off1.fitness < off2.fitness:
            offspring.append(off1)
        else:
            offspring.append(off2)

# Using roulette-Wheel selection to set the offspring population
def rouletteWheelSelection(population):
    # Calculating the total fitness of the population
    totalFitnessPopulation = 0
    for i in population:
        totalFitnessPopulation += i.fitness

    for i in range(0, P):
        # Select a random point from 0 to the total fitness value of the original population
        selectionPoint = random.randint(math.floor(totalFitnessPopulation), 0) 
        runningTotal = math.floor(totalFitnessPopulation)
        j = 0
        # While the running total is not less than the selection point append the fitness of value of an individual in the population to the running total
        while (runningTotal >= selectionPoint) and (j < P):
            runningTotal -= population[j].fitness
            j = j + 1
        # When the running total is less than the selection point, append the last individual from the population which fitness what added to the running total
        offspring.append(population[j - 1])

# Using rank selection to set the offspring population
def rankSelection(population):
    # Sort the individuals in the population in accessinding order based on the fitness value of the individuals
    for i in range(0, P):
            for j in range (0, P - i - 1):
                # Swap the individuals in the population position based on if the fitness is greater than another 
                if (population[j].fitness > population[j+1].fitness):
                    temp = population[j]
                    population[j] = population[j+1]
                    population[j+1] = temp

    # Give a ranking from 0 to the size of the population to the individuals
    rankSum = 0
    for i in range(0, P):
        # Setting the rank
        population[i].rank = P - i
        # Append to the rank sum value 
        rankSum += population[i].rank
 
    for i in range(0, P):
        # Setting the selection point based on a random integer between 0 and the sum of the ranked population
        selectionPoint = random.randint(0, rankSum)
        runningTotal = 0
        j = 0
        # While the running total is not greater than the selection point append the ranking of value of an individual in the population to the running total
        while runningTotal <= selectionPoint and (j < P):
            runningTotal += population[j].rank
            j = j + 1
        # When the running total is greater than the selection point, append the last individual from the population which fitness what added to the running total
        offspring.append(population[j - 1])

# Single point crossover
def singlePointCrossover(tempOffspring):
    # Iterate in 2 for pairs
    for i in range(0, P, 2):
        # Carry out crossover from a random point from the second position in the chromosome (array index 1)
        crossoverPoint = random.randint(1, N-1)
        # Setting the children equal to the original gene in the array before the crossover plus the alternative crossover
        tempA = offspring[i].gene[:crossoverPoint] + offspring[i+1].gene[crossoverPoint:]
        tempB = offspring[i+1].gene[:crossoverPoint] + offspring[i].gene[crossoverPoint:]
        # Append the new solutions to the new array
        tempOffspring.append(tempA)
        tempOffspring.append(tempB)

# Multi Point Crossover
def multiPointCrossover(tempOffspring):
    # Finding the two crossover points
    crossoverPoint1 = 0
    crossoverPoint2 = 0
    # If N mod 3 returns 0 then split the chromosome into three equal parts, using two crossover points
    if N % 3 == 0:
        crossoverPoint1 = (N / 3)
        crossoverPoint2 = (N / 3) * 2
    # If the chromosome does not split into three equal parts then work out where to put the crossover points
    else:
        crossoverPoint1 = round(N / 3)
        crossoverPoint2 = round(N / 3) * 2

    # Iterate in 2 for pairs
    for i in range(0, P, 2):
        # Carry out crossover for two crossover points (multi-point crossover)
        tempA = offspring[i].gene[:crossoverPoint1] + offspring[i+1].gene[crossoverPoint1:crossoverPoint2] + offspring[i].gene[crossoverPoint2:]
        tempB = offspring[i+1].gene[:crossoverPoint1] + offspring[i].gene[crossoverPoint1:crossoverPoint2] + offspring[i+1].gene[crossoverPoint2:]
        # Append the new solutions to the new array
        tempOffspring.append(tempA)
        tempOffspring.append(tempB)

# Uniform Crossover
def uniformCrossover(tempOffspring):
    # Iterate in 2 for pairs
    for i in range(0, P, 2):
        tempA = []
        tempB = []
        # Flip a coin (random integer of 0 or 1) to decide if each chromosome will be included in the off-spring (crossed over)
        # for j in  range(0, len(offspring[i].gene)):
        for j in  range(0, N):
            # Coin flip - random integer of 0 or 1 is produced
            if random.randint(0, 1) == 0:
                tempA.append(offspring[i+1].gene[j])
                tempB.append(offspring[i].gene[j])
            else:
                tempA.append(offspring[i].gene[j])
                tempB.append(offspring[i+1].gene[j])
        tempOffspring.append(tempA)
        tempOffspring.append(tempB)

# Random mutation within a range of bounds
def randomMutation(tempOffspring):
    for i in range(0, P):
        for j in range(0, N):
            mutationProbability = random.randint(0,100) # Randomly generate a number between 0 and 100 
            # If the number generated is less than the mutation rate * 100 then flip the gene in the chromosome
            if mutationProbability < (100 * MUTATION_RATE):
                # Carry out mutation of randomly adding or minusing a number in range from 0.0 to the mutation step
                addOrMinus = random.randint(0,1) # Set variable to randomly select minus or plus
                # Create a random integer between 0 and the upper bound for mutation step, then alter the genes value by a random integer between 0.0 and the mutation step
                mutationStep = round(random.uniform(0.0, UPPER_BOUND),1)
                alter = round(random.uniform(0.0, mutationStep),1)

                # If variable equals 0 then minus a random integer in range 0.0 to the mutation step
                if (addOrMinus == 0):
                    if ((tempOffspring[i][j] - alter) >= LOWER_BOUND):
                        tempOffspring[i][j] = round((tempOffspring[i][j] - alter), 1)
                    # If the value goes below the lower bound after the minus then set to the lower bound as the minimum value it can be
                    else:
                        tempOffspring[i][j] = LOWER_BOUND
                # If variable does not equal 0 then plus a random integer in range 0.0 to the mutation step
                else:
                    if ((tempOffspring[i][j] + alter) <= UPPER_BOUND):
                        tempOffspring[i][j] = round((tempOffspring[i][j] + alter), 1)
                    # If the value goes above 1.0 after the addition then set to the upper bound as the maximum value it can be
                    else:
                        tempOffspring[i][j] = UPPER_BOUND

# Gaussian mutation, mutation within a range of a normal distribution
def gaussianMutation(tempOffspring):
    # Carry out mutation on every individual in population 
    for i in range(0, P):
        for j in range(0, N):
            mutationProbability = random.randint(0,100) 
            if mutationProbability < (100 * MUTATION_RATE):
                # Loc indicates the center of the distribution and Scale indicates the spread of the distribution
                alter = round(float(rand.normal(loc=0, scale=5, size=(1))),1) 
                if ((tempOffspring[i][j] + alter) >= LOWER_BOUND) and ((tempOffspring[i][j] + alter) <= UPPER_BOUND):
                    tempOffspring[i][j] = tempOffspring[i][j] + alter
                elif ((tempOffspring[i][j] + alter) < LOWER_BOUND):
                    tempOffspring[i][j] = LOWER_BOUND
                elif ((tempOffspring[i][j] + alter) > UPPER_BOUND):
                    tempOffspring[i][j] = UPPER_BOUND

# Creating a random end point for scrambled mutation
def calculateEndPoint(startPoint):
    endPoint = random.randint(startPoint, N)
    # Checking that not the whole gene is scrambled
    if endPoint == N:
        calculateEndPoint(startPoint)
    return endPoint

# Scrambled Mutation
def scrambleMutation(tempOffspring):
    # Carry out mutation on every individual in population 
    for i in range(0, P):
        mutationProbability = random.randint(0,100) 
        if mutationProbability < (100 * MUTATION_RATE):
            # Create a starting and end point of where the scrambled mutation on the individuals should take place
            startingPoint = random.randint(0, N-1) # Making sure that more than one gene is mutated
            endPoint = calculateEndPoint(startingPoint)
            # Shuffle the genes in the chromosome between the start and end point
            shuffledArray = []
            for j in range(startingPoint, endPoint):
                shuffledArray.append(tempOffspring[i][j])
            rand.shuffle(shuffledArray)
            # Put the scrambled array back into the individual 
            for x in range(startingPoint, endPoint):
                for y in range(0, len(shuffledArray)):
                    tempOffspring[i][x] = shuffledArray[y]

# Clear the array values so that the GA is able to iterate again
def clearArrays():
    meanList.clear()
    generationArr.clear()
    bestArr.clear()
    meanArr.clear()
    minArr.clear()
    population.clear()
    offspring.clear()

# Main function to start code from
def main():
    # Carry out iterations of the GA so that we can plot the average of the results
    for x in range(0,ITERATIONS):
        setPopulation() # Setting population of individuals
        # Carry out crossover and mutation for as many generations set, this is the termination condition of the algorithm 
        for i in range(1, GENERATIONS + 1):
            print("\nGeneration ", i)
            tempOffspring = []

            # Selection 
            tournamentSelection(population)  
            # Crossover 
            singlePointCrossover(tempOffspring)
            # Mutation 
            randomMutation(tempOffspring)
    
            # This range determines how many solutions are selected and swapped per run
            for j in range(0, SWAP_AMOUNT):
                largestIndex = largestFitnessSolution() # Finding worst solution in the population (individual with the largest fitness value)
                lowestIndex = lowestFitnessSolution(tempOffspring) # Finding best solution in the temporary offspring (lowest fitness value)
                replaceChromosomes(lowestIndex, largestIndex, tempOffspring) # Set the worst case of the original population to equal the best case of the temp offspring population

            # Calculate and print the fitness of the population after selection, mutation and crossover
            calcFitnesses(i)

        # setGraph() # Setting individual iteration graph using plotly

        for i in range(0, GENERATIONS):
            minArrAverage[i] += minArr[i]

        clearArrays() # Re set the values of the arrays so that other iterations of the GA can occur

    # Appending the average minimum fitness results from the GA run 
    for i in range(0, len(minArrAverage)):
        minAverageCalculated.append(round(float(minArrAverage[i] / ITERATIONS),1))
        generationArr.append(i+1) # Set the generation array equal to how many generations occur per run

    setGraphMinimumAverage() # Output the average results on the graph

if __name__ == "__main__":
    main()
