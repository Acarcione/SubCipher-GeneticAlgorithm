import sys
import math
import random
import getopt
import numpy as np
import matplotlib.pyplot as plot
import matplotlib.patches as mpatches
import time
from scipy import stats

def preprocess(train):
    line = train.readline()
    newTrain = ""
    while line:
        line = line.strip(" ")
        line = line.replace("\n", " ")
        if(line == " "):
            line = train.readline()
            continue
        i = 0
        newLine = ""
        for char in line:
            if(char == ' '):
                newLine = newLine + char
            if((ord(char) > 64 and ord(char) < 91)):
                newLine = newLine + char
            if((ord(char) > 96 and ord(char) < 123)):
               newLine = newLine + char
               
        line = newLine
                
        newTrain = newTrain + line
        line = train.readline()
        
    return newTrain
            

def biGrams(biDict, train):
    i = 0
    j = 3
    while(j < len(train)):
        digram = train[i:j]
        if(" " in digram):
            i += 1
            j += 1
            continue
        else:
            key = digram[0].upper() + digram[1].upper() + digram[2].upper()
            biDict[key] = biDict[key] + 1
        
            i += 1
            j += 1
            
    return biDict

def frequencyTable(freqDict):
    for key in freqDict.keys():
        val = freqDict[key]
        if val != 0:
            freqDict[key] = math.log(val, 2.0)
        
    return freqDict

def fitness(biDict, freqDict):
    biSum = 0
    for key in biDict.keys():
        if(biDict[key] != 0):
            biSum += biDict[key] * freqDict[key]
            
    return biSum

def encrypt(alphabet, mixedAlph, text):
    encryptedText = ""
    for item in text:
        if(item.upper() != " "):
            idx = alphabet.index(item.upper())
            encryptedText += mixedAlph[idx]
        else:
            encryptedText += " "
            
    return encryptedText

def decrypt(alphabet, mixedAlph, text):
    decryptedText = ""
    for item in text:
         if(item.upper() != " "):
            idx = mixedAlph.index(item.upper())
            decryptedText += alphabet[idx]
         else:
             decryptedText += " "
            
    return decryptedText

def generatePop(pop, popSize, alphabet):
    for i in range(popSize):
        temp = alphabet.copy()
        random.shuffle(temp)
        pop.append(temp)
    
    return pop


def insMut(child):  #Insert Mutation, preserves most order and adjacency information
    #prob = random.randint(0,100)/100
    #if (prob <= 1):  #Mutate
    ind = sorted(random.sample(range(0,len(child)), 2))

    val1 = child[ind[0]]
    val2 = child[ind[1]]
    indAcc = val2
    while (child[ind[0]+1] != val2): #While the number after the first index isnt the desired value
        temp = child[ind[1]-1]
        child[ind[1]-1] = child[ind[1]]
        child[ind[1]] = temp
        ind[1] -= 1

    return child

def invMut(child):  #Inverse Mutation, Preserves Adjacency and disrupts order
    #prob = random.randint(0,100)/100
    acc = 0
    newChild = []
    #if (prob <= pMut):  #Mutate
    ind = sorted(random.sample(range(0,len(child)), 2))
    s1 = child[0:ind[0]]
    seg = child[ind[0]:ind[1]]
    s2 = child[ind[1]:]

    
    seg.reverse()
    newChild = s1 + seg + s2

    return newChild


def Tournament(population, popFits, p, size, numElite):
    parents = []
    subParents = []
    for i in range(len(population) - numElite): #tournament process
        tournamentPop = []
        tournamentFit = []
        if(len(population) < size):
            tournamentPop = random.sample(range(0, len(population)), len(population))
        else:
            tournamentPop = random.sample(range(0,len(population)), size)
        for item in tournamentPop:
            tournamentFit.append(popFits[item])
        winner = 0
        pLimit = p
        place = 0
        probability = float("inf")
        maxScore = 0
        winnerLocation = 0
        probability = random.random()
        while(probability >= pLimit and place < len(tournamentPop)):
            place += 1
            if(probability > pLimit):
                pLimit += pLimit * (1-p)
        if(place == 1):
            maxScore = max(tournamentFit)
            winnerLocation = tournamentFit.index(maxScore)
            winner = tournamentPop[winnerLocation]

        else:
            for i in range(0, place-1):
                maxScore = max(tournamentFit)
                winnerLocation = tournamentFit.index(maxScore)
                tournamentFit.pop(winnerLocation)
                tournamentPop.pop(winnerLocation)
            maxScore = max(tournamentFit)
            winnerLocation = tournamentFit.index(maxScore) 
            winner = tournamentPop[winnerLocation]

        subParents.append(population[winner])
        if(len(subParents) == 2):
            parents.append(subParents)
            subParents = []

        popFits.pop(winner)
        population.pop(winner)
    return parents
            
def FPS(pop, fit, numElite):
    popFit = []
    for i in range(len(pop)):
        tup = (pop[i], fit[i])
        popFit.append(tup)

    popFit = sorted(popFit, key = lambda x : x[1])  #Sort Population by Fitness
    accNorm = [] #Will contain accumulated normalized values (w/ final value = 1)
    totalFit = 0

    #Normalize Values
    for i in range(len(popFit)):
        totalFit += popFit[i][1]

    for i in range(len(fit)):
        accNorm.append(popFit[i][1]/totalFit)
    
    normSum = 0
    for i in range(len(popFit)):
        accNorm[i] = accNorm[i] + normSum
        normSum = accNorm[i]
    
    #accNorm contains Slices for selecting parents
    parents = []
    indexArr = []
    for i in range(int((len(popFit)-numElite)/2)): 
        parentss = []
        for j in range(2):
            randNum = random.randint(0,100)/100
            for j in range(len(accNorm)):
                if (j == 0):
                    if (randNum < accNorm[j]):
                        parentss.append(popFit[j][0])
                else:
                    if (randNum < accNorm[j] and randNum > accNorm[j-1]):
                        parentss.append(popFit[j][0])
        parents.append(parentss)


    for i in range(len(parents)-1):
        if (len(parents[i]) == 0):
            parents.pop(i)
            
    for i in range(len(parents)-1):
        if (len(parents[i]) != 2):
            parents[i].append(parents[i][0])

    return parents

def Crossover(parents, pCross, numElite):
    newPop = []
    for i in range(int(len(parents))):
        prob = random.randint(0,100)/100
        if (prob > pCross): #Dont Crossover
            if (len(parents[i]) == 1):
                parents[i].append(parents[i][0])
            newPop.append(parents[i][0])
            newPop.append(parents[i][1])
            pass
        elif (prob <= pCross):  #Perform Crossover
            portionArr = []
            child1 = []
            child2 = []
            childArr = [child1, child2]
            cut = []
            cut2 = []
            cutArr = [cut, cut2]

            acc = 0

            if (len(parents[i]) == 1):
                parents[i].append(parents[i][0])

            parentsArr = [parents[i][0],parents[i][1]]   #Parents
            for j in range(2):  #One Iteration per parent
                portionInd = random.sample(range(0, len(parents[j][0])+1), 2)  #Indexes used for isolation crossover portion
                portionInd = sorted(portionInd) 
                portion = parentsArr[j][portionInd[0]:portionInd[1]]
                portionArr.append(portion)

                for k in range(len(parentsArr[j])):     #Check if every value from pX is in the portion, if it is:ignore, if not:append to cutArr
                    if (j == 0):   
                        if (parentsArr[1][k] not in portion):
                            cutArr[0].append(parentsArr[1][k])
                    if (j == 1):
                        if (parentsArr[0][k] not in portion):
                            cutArr[1].append(parentsArr[0][k])

                tempInd = 0
                tempInd2 = 0

                for l in range(len(parentsArr[j])):
                    if (l < portionInd[0]):
                        childArr[j].append(cutArr[j][l])
                        tempInd +=1
                    if (l >= portionInd[0] and l < portionInd[1]):
                        childArr[j].append(portionArr[j][tempInd2])
                        tempInd2 +=1
                    if (l >= portionInd[1]):
                        childArr[j].append(cutArr[j][tempInd])
                        tempInd +=1
            newPop.append(childArr[0])
            newPop.append(childArr[1])

    return newPop

def Mutation(pop, pMut, numElite):
    for i in range(int(len(pop)-numElite)):
        prob = random.randint(0,100)/100
        if (prob <= pMut):  #Mutate
            chrom = pop[i]
            ind = sorted(random.sample(range(0,len(pop[i])), 2))

            val1 = pop[i][ind[0]]
            val2 = pop[i][ind[1]]
            indAcc = val2
            while (pop[i][ind[0]+1] != val2): #While the number after the first index isnt the desired value
                temp = pop[i][ind[1]-1]
                pop[i][ind[1]-1] = pop[i][ind[1]]
                pop[i][ind[1]] = temp
                ind[1] -= 1
    return pop

def swapMut(pop, pMut, numElite): #Takes a child after crossover
    for i in range(int(len(pop)-numElite)):
        prob = random.randint(0,100)/100
        if(prob <= pMut):  #Mutate
            chrom = pop[i]
            ind = sorted(random.sample(range(0,len(pop[i])), 2))
            val1 = pop[i][ind[0]]
            val2 = pop[i][ind[1]]
            temp = val1
            pop[i][ind[0]] = val2
            pop[i][ind[1]] = temp
    return pop

def main():
    popSize = 500
    size = 20
    pTourn = .75
    pCross = .65
    pMut = .2
    percentElite = 0.15
    crossPoints = 5
    maxGenerations = 1000
    converge = 20
    trainName = ""
    encryptName = ""
    decryptName = ""
    graph = False
    useFPS = False
    err = "The valid arguments are:\n-t or --trainText for the name of the training text\n-i or --encryptText for the name of the encrpyted text\n-o or --decryptText for the name of the output decrypted text\n-h for enabling a graph of the max, min and average fitnesses over the span of the generations\n-f for enbling FPS selection\n--popSize for the population size\n--tSize for the tournament size\n--pTournament for the tournament probability\n--pCrossover for the crossover probability\n--pMutation for the mutation probability\n--percentElite for the percentage of elitism\n--crossPoints for the number of crossover points\n--converge for the number of generations with the same max fitness"
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], 't:i:o:hm:gf', ['trainText=', 'encryptText=', 'decryptText=', 'popSize=', 'tSize=', 'pTournament=', 'pCrossover=', 'pMutation=', 'percentElite=', 'crossPoints=', 'maxGenerations=', 'converge='])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)
        
    for opt, arg in opts:
        if opt == '-h':
            print(err)
            sys.exit(2)
        elif opt in ('-t', '--trainText'):
            trainName = arg
            train = open(trainName, "r")
        elif opt in ('-i', '--encryptText'):
            encryptName = arg
            encryptText = open(encryptName, "r")
        elif opt in ("-o", '--decryptText'):
            decryptName = arg
            decryptText = open(decryptName, "w")
        elif opt == '--popSize':
            popSize == int(arg)
        elif opt == '--tSize':
            size = int(arg)
        elif opt == '--pTournament':
            pT = float(arg)
        elif opt == '--pCrossover':
            pC = float(arg)
        elif opt == '--pMutation':
            pM = float(arg)
        elif opt == '--percentElite':
            percentElite = int(arg)
        elif opt == '--crossPoints':
            crossPoints = int(arg)
        elif opt in ('-m', '--maxGenerations'):
            maxGenerations = arg
            maxGenerations = int(maxGenerations)
        elif opt == '--converge':
            converge = arg
        elif opt == '-g':
            graph = True
        elif opt == '-f':
            useFPS = True
        else:
            print(err)
            sys.exit(2)

    if(trainName == ""):
        print("---------------------------------------------------------------\nYou must input the name of a file containing the training text. Try using -h for a list of commands\n--------------------------------------------------------------")
        sys.exit(2)
    elif(encryptName == ""):
        print("---------------------------------------------------------------\nYou must input the name of a file containing the text to encrypt. Try using -h for a list of commands\n---------------------------------------------------------------")
        sys.exit(2)
    elif(decryptName == ""):
        print("--------------------------------------------------------------\nYou must input the name of a file to write the decrypted message to. Try using -h for a list of commands\n--------------------------------------------------------------")
        sys.exit(2)
        
    alph = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    freqDict = {}
    biGramDict = {}
    population = []
    
    randEncryptAlph = alph.copy()
    random.shuffle(randEncryptAlph)

    for i in range(26):
        for j in range(26):
            for k in range(26):
                freqDict[alph[i] + alph[j] + alph[k]] = 0
                biGramDict[alph[i] + alph[j] + alph[k]] = 0
            
    encryptedText = open("empty.txt", "w")

    newTrain = preprocess(train)
    eText = preprocess(encryptText)

    encryptedMessage = encrypt(alph, randEncryptAlph, eText)
    encryptedText.write(encryptedMessage)
    
    freqTable = biGrams(freqDict, newTrain)
    freqTable = frequencyTable(freqTable)

    population = generatePop(population, popSize, alph)

    fitScores = []
    generation = 0
    tournamentPop = [] # participants in the tournament
    tournamentFit = [] # fitnesses of participants
    maxFitness = 0
    convergeCount = 0
    numberElite = int(len(population)*percentElite)
    bestKey = []

    allFit = []
    avgFit = []
    minFit = []
    xaxis = []
    #xaxis.append(0)

    #elite = []
    startTime = time.clock()
    ################# START LOOP ########################################
    for i in range(maxGenerations):
        elite = []
        print("Generation:", generation+1)
        fitScores = []
        for key in population:
            fitnessScore = 0
            decryptedMessage = ""
            biTable = biGramDict.copy()
            decryptedMessage = decrypt(alph, key, encryptedMessage)
            biTable = biGrams(biTable, decryptedMessage)
            fitnessScore = fitness(biTable, freqTable)
            fitScores.append(fitnessScore)
        
        if(max(fitScores) > maxFitness): # checking for max fitness score
            maxFitness = max(fitScores)
            idxBest = fitScores.index(maxFitness)
            bestKey = population[idxBest]
            convergeCount = 0 #reset convergeCount
        elif(maxFitness == max(fitScores)):
            convergeCount += 1

        if (convergeCount == converge):
            generation += 1
            break


        ####### Used for Testing and Plotting ######
        allFit.append(max(fitScores))
        minFit.append(min(fitScores))
        a = fitScores
        a = np.asarray(a)
        avgFit.append(np.average(a))
        ############################################

        popFitArr = []
        for i in range(len(fitScores)):     # Link the fitness scores to the index of their population
            tup = (population[i],fitScores[i])
            popFitArr.append(tup)


        tempElite = sorted(popFitArr, key = lambda x : x[1])  #Get top X% of fitScores and add them to Elite
        tempElite.reverse()
        for i in range(int(numberElite)):
            elite.append(tempElite[i][0])

        ''' Selection '''
        if(generation != maxGenerations - 1):
            if(useFPS == True):
                parents = FPS(population, fitScores, numberElite)   #Fitness Proportional Selection
            else:
                parents = Tournament(population, fitScores, pTourn, size, numberElite)   #Tournament Selection
        else:
           if(useFPS == True):
               parents = FPS(population, fitScores, numberElite)   #Fitness Proportional Selection
           else:
               parents = Tournament(population, fitScores, pTourn, size, numberElite)   #Tournament Selection

        ''' Crossover '''
        population = Crossover(parents, pCross, numberElite) #Crossover without duplicates

        ''' Mutation '''
        #population = Mutation(population, pMut, numberElite)    #Insert Mutation (on whole population)
        population = swapMut(population, pMut, numberElite)
        '''
        for j in range(len(population)):        #Mutation of a single Child (inv or ins)
            prob = random.randint(0,100)/100
            if (prob <= pMut):  #Mutate
                population[i] = invMut(population[i])   #Inverse Mutation
        '''


        population += elite

        generation += 1
        xaxis.append(generation)
    ################## END LOOP ######################################

    endTime = time.clock()
    runTime = endTime - startTime

    #maxFitness = max(fitScores)
    #idxBest = fitScores.index(maxFitness)
    #bestKey = population[idxBest]

    count = 0
    for i in range(26):
        if(bestKey[i] == randEncryptAlph[i]):
            count += 1
    
    decryptedMessage = decrypt(alph, bestKey, encryptedMessage)
    decryptText.write(decryptedMessage)
    
    pop = np.array(population)
    mode = stats.mode(pop)
    mode = mode[0]
    mode = np.array(mode).tolist()[0]
    simCount = population.count(mode)
    
    print("\nConverged At Generation:", str(generation-20))
    print("number of generations: " + str(generation))
    print("maxFitness Score: " + str(maxFitness))
    print("percent of key correct: " + str(count / 26) + " %")
    print("runTime: " + str(runTime))
    print(str(float(simCount/len(population))) + " % similarity")
    
    
    
    #PLOT
    if graph:
        allFit = np.asarray(allFit)
        avgFit = np.asarray(avgFit)
        minFit = np.asarray(minFit)
        plot.plot(xaxis, allFit, "b") #plot max fitness of each generation
        plot.plot(xaxis, avgFit, "g") #plot average fitness of each generation
        plot.plot(xaxis, minFit, "r") #plot min fitness of each generation
        plot.xlabel("Generation")
        plot.ylabel("Fitness")

        blue_patch = mpatches.Patch(color = 'blue', label = "Max Fitness")
        green_patch = mpatches.Patch(color = 'green', label = "Average Fitness")
        red_patch = mpatches.Patch(color = 'red', label = "Min Fitness")

        plot.legend(handles = [blue_patch, red_patch, green_patch])

        plot.axis([0,generation-1,0, max(allFit) + 100])
        plot.show()
main()
