import ea                  
import fnn                
import agent
import random       
import numpy as np
import matplotlib.pyplot as plt

# ANN Params
layers = [5,25,25,4]

# Task Params
duration = 20
stepsize = .02   
noisestd = 0.01 

maxFoodTrials = 5
numberOfFoodTrials =8*maxFoodTrials
#trainingFoodPositions = [(10,10),(5,5),(-5,5),(-10,10),(5,-5),(10,-10),(-5,-5),(-10,-10)]
trainingFoodPositions = []


for i in range(1,maxFoodTrials,1):
    trainingFoodPositions.append((i,i))
    trainingFoodPositions.append((-i,i))
    trainingFoodPositions.append((i,-i))
    trainingFoodPositions.append((-i,-i)) 
    trainingFoodPositions.append((0,i)) 
    trainingFoodPositions.append((0,-i)) 
    trainingFoodPositions.append((-i,0)) 
    trainingFoodPositions.append((i,0)) 

evalutateFoodPositions = [(10,10),(-10,10),(10,-10),(-10,-10)]
# Time
time = np.arange(0.0,duration,stepsize)

popsize = 10
genesize = np.sum(np.multiply(layers[1:],layers[:-1])) + np.sum(layers[1:]) 
recombProb = 0.1
mutatProb = 0.01
tournaments = 50*popsize


def fitnessFunction(genotype):
    nn = fnn.FNN(layers)
    nn.setParams(genotype)
    body = agent.Agent()
    fit = 0.0
    for foodPos in trainingFoodPositions:
        body.xPos = 0
        body.yPos = 0
        body.food = 0
        body.foodPosX = foodPos[0]
        body.foodPosY = foodPos[1]
        f = stepsize
        t = 0
        while t<duration:
            inp = body.state()
            out = nn.forward(inp)*2 - 1 + np.random.normal(0.0,noisestd)
            #print(out[0])
            f = body.step(out[0])
            fit += f
            t += stepsize
        #print("Time: "+str(body.time))
    return -fit/(duration*numberOfFoodTrials)

# Evolve and visualize fitness over generations
ga = ea.MGA(fitnessFunction, genesize, popsize, recombProb, mutatProb, tournaments)
ga.run()
ga.showFitness()

bestind_num = int(ga.bestind[-1])
print(bestind_num)
bestind_genotype = ga.pop[bestind_num]

def evaluate(genotype): # repeat of fitness function but saving theta
    nn = fnn.FNN(layers)
    nn.setParams(genotype)
    body = agent.Agent()
    out_hist = np.zeros((len(time),5))
    f_hist=np.zeros(len(time))
    body.xPos = 0
    body.yPos = 0 
    body.foodPosX = 3
    body.foodPosY = 3
    k=0
    for t in time:
        inp = body.state()
        out = nn.forward(inp)*2-1 + np.random.normal(0.0,noisestd)
        f = body.step(out[0])
        out_hist[k] = inp
        f_hist[k] = f
        k += 1
        if body.done == True:
            break
    #print("Walk length: "+str(body.walkLength))
    body.graph()
    print("Food: "+str(body.food))
    return out_hist, f_hist


out_hist1, f_hist1 = evaluate(bestind_genotype)
#out_hist2, f_hist2 = evaluate(bestind_genotype)

plt.plot(out_hist1)
#plt.plot(out_hist2)
plt.plot(f_hist1*50,'k',label="Output")
#plt.plot(f_hist2*50,'k',label="Output")
plt.show()
