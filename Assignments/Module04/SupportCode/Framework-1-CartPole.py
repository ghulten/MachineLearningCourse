
import gym

env = gym.make('CartPole-v0')

import random
import MachineLearningCourse.MLUtilities.Reinforcement.QLearning as QLearning
import MachineLearningCourse.Assignments.Module04.SupportCode.GymSupport as GymSupport

## Hyperparameters to tune:
discountRate = 0.999          # Controls the discount rate for future rewards -- this is gamma from 13.10
actionProbabilityBase = 1.9  # This is k from the P(a_i|s) expression from section 13.3.5 and influences how random exploration is
randomActionRate = 0.001      # Percent of time the next action selected by GetAction is totally random
learningRateScale = 0.001     # Should be multiplied by visits_n from 13.11.
binsPerDimension = 5
trainingIterations = 20000

continuousToDiscrete = GymSupport.ContinuousToDiscrete(binsPerDimension, [ -4.8000002e+00, -4, -4.1887903e-01, -4 ], [ 4.8000002e+00, 4, 4.1887903e-01, 4 ])

qlearner = QLearning.QLearning(stateSpaceShape=continuousToDiscrete.StateSpaceShape(), numActions=env.action_space.n, discountRate=discountRate)

# Learn the policy
for trialNumber in range(trainingIterations):
    observation = env.reset()
    reward = 0
    for i in range(300):
        #env.render()

        currentState = continuousToDiscrete.Convert(observation)
        action = qlearner.GetAction(currentState, learningMode=True, randomActionRate=randomActionRate, actionProbabilityBase=actionProbabilityBase)

        oldState = continuousToDiscrete.Convert(observation)
        observation, reward, isDone, info = env.step(action)
        newState = continuousToDiscrete.Convert(observation)

        qlearner.ObserveAction(oldState, action, newState, reward, learningRateScale=learningRateScale)

        if isDone:
            if(trialNumber%1000) == 0:
                print(trialNumber, i, reward)
            break

# Evaluate the policy
n = 20
totalRewards = []
for runNumber in range(n):
    observation = env.reset()
    totalReward = 0
    reward = 0
    for i in range(300):
        renderDone = env.render()

        currentState = continuousToDiscrete.Convert(observation)
        observation, reward, isDone, info = env.step(qlearner.GetAction(currentState, learningMode=False))

        totalReward += reward

        if isDone:
            renderDone = env.render()
            print(i, totalReward)
            totalRewards.append(totalReward)
            break

