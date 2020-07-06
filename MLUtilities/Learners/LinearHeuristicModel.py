class LinearHeuristicModel(object):
    """A heuristic (hand crafted) linear model. You will need to pass in all weights in the call to fit()"""

    def __init__(self):
        # weight_0 is the 'bias' weight. It is added to every example's score at prediction time.
        self.weight0 = None
        
        # weights array contains one weight per feature
        self.weights = None
        
    def fit(self, x, y, biasWeight, weights):
        # this model is hand-tuned so fit doesn't look at the data at all.
        
        self.weight0 = biasWeight
        self.weights = weights

    def predict(self, x, threshold = 0):
        assert len(x[0]) == len(self.weights)

        predictions = []

        for example in x:
            # do a dot product between the feature values and the weights
            score = sum( 
                        [ self.weight0 * 1.0 ] +                                        # the bias weight is used as is (multipled by 1.0).
                        [ example[i] * self.weights[i] for i in range(len(example)) ]   # each feature is multiplied by the associated weight
                        )

            predictions.append( 1 if score > threshold else 0 )
        
        return predictions
    
    def visualize(self):
        print("w0 (bias): %f " % (self.weight0), end='')

        for i in range(len(self.weights)):
            print("w%d: %f " % (i+1, self.weights[i]), end='')

        print("\n")