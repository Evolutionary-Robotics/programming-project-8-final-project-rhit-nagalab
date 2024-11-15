import random
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import numpy as np


class Agent:
    def __init__(self):
        self.xPos = 0
        self.yPos = 0
        self.time = 0
        self.age = 0
        self.walkLength = 1
        self.food = 0
        self.foodPosX = 0
        self.foodPosY = 0
        self.done = False
        self.XPositions = []
        self.YPositions = []
        
    def getIndexOfMax(self,array):
        maxNum = max(array)
        for index,value in enumerate(array):
            if value==maxNum:
                return index
        return 0
    
    def setFoodPosition(self,xPos,yPos):
        self.foodPosX = xPos
        self.foodPosY = yPos

    def step(self, action):
        #print("Step")
        self.time += 1
        #print(action)
        #print(stepsize)
        index = self.getIndexOfMax(action)

        if index == 0:
            self.yPos += self.walkLength
        elif index == 1:
            self.yPos -= self.walkLength
        elif index == 2:
            self.xPos += self.walkLength
        elif index == 3:
            self.xPos -= self.walkLength
        
        if self.xPos==self.foodPosX and self.yPos==self.foodPosY:
            #print("Food")
            self.food += 1
            self.done = True

        self.XPositions.append(self.xPos)
        self.YPositions.append(self.yPos)

        return self.getDistanceFromFood()

    def getDistanceFromOrigin(self):
        return math.sqrt(math.exp2(self.xPos)+math.exp2(self.yPos))
    
    def getDistanceFromClosestFood(self):
        min = sys.maxsize
        for food in self.foodPositions:
            #print(food)
            dist = math.sqrt(math.exp2(self.xPos-food[0])+math.exp2(self.yPos-food[1]))
            if dist<min:
                min = dist
        return min
    
    def getDistanceFromFood(self):
        return math.sqrt(math.exp2(self.xPos-self.foodPosX)+math.exp2(self.yPos-self.foodPosY))

    def state(self):
        return np.array([self.xPos, self.yPos, self.foodPosX, self.foodPosY,self.getDistanceFromFood()])
    
    def graph(self):
        plt.plot(self.foodPosX,self.foodPosY,'o')
        plt.plot(self.XPositions,self.YPositions)
        plt.show()

