def CrossValidation(x, y, numberOfFolds, foldIDForEvaluation):
    if(len(x) != len(y)):
        raise UserWarning("Attempting to split into training and testing set.\n\tx and y arrays do not have the same size. Check your work and try again.")

    if(numberOfFolds <= 1 or numberOfFolds > len(y)):
        raise UserWarning("Attempting to split into %d numberOfFolds, must be between 2 and the number of samples.\n." % numberOfFolds)

    if(foldIDForEvaluation < 0 or foldIDForEvaluation >= numberOfFolds):
        raise UserWarning("Attempting to select fold %d for evaluation, must be 0 - %d." % (foldIDForEvaluation, numberOfFolds - 1))

    numberPerFold = round(len(y) / numberOfFolds)

    xTrain = []
    yTrain = []
    xEvaluate = []
    yEvaluate = []

    for i in range(numberOfFolds):
        if i == foldIDForEvaluation:
            xThis = xEvaluate
            yThis = yEvaluate
        else:
            xThis = xTrain
            yThis = yTrain

        firstIndex = numberPerFold * i
        lastIndex = numberPerFold * (i+1)

        if i == (numberOfFolds -1):
            # last, pick up any stragglers
            lastIndex = len(y)

        xThis += x[firstIndex:lastIndex]
        yThis += y[firstIndex:lastIndex]

    return(xTrain, yTrain, xEvaluate, yEvaluate)