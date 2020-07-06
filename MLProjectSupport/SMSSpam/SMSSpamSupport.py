# some environments have trouble with text encodings, try:
#   f = open(path, 'r', encoding='unicode_escape')

import collections
import os

def LoadRawData(path):
    print("Loading data from: %s" % os.path.abspath(path))
    f = open(path, 'r')
    
    lines = f.readlines()

    kNumberExamplesExpected = 5474

    if(len(lines) != kNumberExamplesExpected):
        message = "Attempting to load %s:\n" % (path)
        message += "   Expected %d lines, got %d.\n" % (kNumberExamplesExpected, len(lines))
        message += "    Check the path to training data and try again."
        raise UserWarning(message)

    x = []
    y = []

    for l in lines:
        if(l.startswith('ham')):
            y.append(0)
            x.append(l[4:].strip())
        elif(l.startswith('spam')):
            y.append(1)
            x.append(l[5:].strip())
        else:
            message = "Attempting to process %s\n" % (l)
            message += "   Did not match expected format."
            message += "    Check the path to training data and try again."
            raise UserWarning(message)

    return (x, y)

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




