import random

def TrainValidateTestSplit(x, y, percentValidate = .1, percentTest = .1):
    if(len(x) != len(y)):
        raise UserWarning("Attempting to split into training and testing set.\n\tArrays do not have the same size. Check your work and try again.")

    numTest = round(len(x) * percentTest)
    numValidate = round(len(x) * percentValidate)

    if(numValidate == 0 or numValidate > len(y)):
        raise UserWarning("Attempting to split into training, validation and testing set.\n\tSome problem with the percentValidate or data set size. Check your work and try again.")

    if(numTest == 0 or numTest > len(y)):
        raise UserWarning("Attempting to split into training, validation and testing set.\n\tSome problem with the percentTest or data set size. Check your work and try again.")

    xTest = x[:numTest]
    xValidate = x[numTest:numTest + numValidate]
    xTrain = x[numTest + numValidate:]
    yTest = y[:numTest]
    yValidate = y[numTest: numTest+numValidate]
    yTrain = y[numTest+numValidate:]

    return (xTrain, yTrain, xValidate, yValidate, xTest, yTest)

def BootstrapSample(x, y, limit=None):
    if len(x) != len(y):
        raise UserWarning("x list and y list are different lengths.")
    
    if limit == None:
        limit = len(x)
        
    sampleIndexes = [ random.randint(0, len(x) - 1) for i in range(limit) ]
    sampledX = [ x[i] for i in sampleIndexes ]
    sampledY = [ y[i] for i in sampleIndexes ]
    
    return (sampledX, sampledY)

def Shuffle(x, y):
    # use to randomly reorder samples & labels while keeping the correct label with the correct sample
    tmp = list(zip(x, y))
    random.shuffle(tmp)
    xShuffled, yShuffled = zip(*tmp)
    return (xShuffled, yShuffled)