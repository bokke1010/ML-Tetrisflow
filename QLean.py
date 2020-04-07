import random

class Model ():
    def __init__(self, data: dict = {}, learningRate: float = 0.75, discountFactor: float = 0.9,
                randomize: float = 0.05, actionList: list = []):
        self.data = data
        self.learningRate = learningRate
        self.discountFactor = discountFactor
        self.randomize = randomize
        self.actions = actionList

    def bestAction(self, state):
        if random.random() < self.randomize:
            return random.choice(self.actions)
        q = self.getQValues(state)
        maxValue = q[0]
        bestAction = 0
        untestedActions = []
        for index in range(len(q)):
            if q[index] == 0:
                untestedActions.append(index)
            if q[index] > maxValue:
                maxValue = q[index]
                bestAction = index
        if maxValue == 0:
            return self.actions[random.choice(untestedActions)]
        return self.actions[bestAction]


    def getQValues(self, state):
        if not state in self.data:
            q = [0]*len(self.actions)
            self.data[state] = q
        return self.data[state]

    def updateModel(self, oldState, newState, reward, action):
        q0, q1 = self.getQValues(oldState), self.getQValues(newState)
        newValue = reward + self.discountFactor * max(q1)
        actIndex = self.actions.index(action)
        q0[actIndex] *= 1 - self.learningRate
        q0[actIndex] += self.learningRate * newValue
