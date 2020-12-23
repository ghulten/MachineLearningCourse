import math

class QLearning(object):
    """Simple table based q learning"""
    def __init__(self, stateSpaceShape, numActions, discountRate=.5):
        self.numActions = numActions
        self.stateSpaceShape = stateSpaceShape
        self.discountRate = discountRate

    def GetAction(self, state, learningMode=True, randomActionRate=0, actionProbabilityBase=0):
        print("Stub GetAction in ", __file__)
        return 0

    def ObserveAction(self, oldState, action, newState, reward, learningRateScale=0):
        print("Stub ObserveAction in ", __file__)
        return 0
