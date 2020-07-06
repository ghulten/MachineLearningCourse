import math

def __CheckEvaluationInput(y, yPredicted):
    # Check sizes
    if(len(y) != len(yPredicted)):
        raise UserWarning("Attempting to evaluate between the true labels and predictions.\n   Arrays contained different numbers of samples. Check your work and try again.")

    # Check values
    valueError = False
    for value in y:
        if value < 0 or value > 1:
            valueError = True
    for value in yPredicted:
        if value < 0 or value > 1:
            valueError = True

    if valueError:
        raise UserWarning("Attempting to evaluate between the true labels and predictions.\n   Arrays contained unexpected values. Must be between 0 and 1.")

def MeanSquaredErrorLoss(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    print("Stub MeanSquaredErrorLoss in ", __file__)

def LogLoss(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    print("Stub LogLoss in ", __file__)

