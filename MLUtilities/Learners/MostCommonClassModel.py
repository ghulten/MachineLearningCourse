import collections

class MostCommonClassModel(object):
    """A classification model that predicts the most common label from the training data."""

    def __init__(self):
        pass

    # Every model should have a fit function that takes:
    #   x - an array of the training data
    #   y - the correct lables for x (in a parallel array)
    #   <hyper parameters> - as needed. These are extra information that controls how the model is fit. We'll see many of these over time...
    def fit(self, x, y, optionalHyperParameters=None):
        # This model ignores the data in x. It finds the most common value of y and remembers it in variable 'self.mostCommonClass'
        count = collections.Counter()

        for label in y:
            count[label] += 1

        self.mostCommonClass = count.most_common(1)[0][0]

    # Every model should have a predict function that takes:
    #   x - an aray of the data that you want to make predictions for
    def predict(self, x):
        # This model predicts 'self.mostCommonClass' for every x, no matter what is in the feature vector, x (which probably won't be very accurate)
        return [ self.mostCommonClass for example in x ]

    # It's a good idea to have some methods that let you inspect the parameters your model learned (in this case self.mostCommonClass is the only parameter)
    def visualize(self):
        try:
            print("This model always predicts: %d" % self.mostCommonClass)
        except:
            print("This model has not been fit yet.")