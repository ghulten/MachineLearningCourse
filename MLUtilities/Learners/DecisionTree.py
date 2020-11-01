import collections
import math
import time

def Entropy(y):
    print("Stub Entropy in ", __file__)
    return 0.0

def FindBestSplitOnFeature(x, y, featureIndex):
    if len(y) < 2:
        # there aren't enough samples so there is no split to make
        return None

    ### Do more tests to make sure you haven't hit a terminal case...
        
    # HINT here is how to get an array that has indexes into the data (x, y) arrays in sorded 
    # order based on the value of the feature at featureIndex
    indexesInSortedOrder = sorted(range(len(x)), key = lambda i : x[i][featureIndex])
    
    # so x[indexesInSortedOrder[0]] will be the training sample with the smalles value of 'featureIndex'
    # and y[indexesInSortedOrder[0]] will be the associated label

    print("Stub FindBestSplitOnFeature in ", __file__)

    # HINT: might like to return the partitioned data and the
    #  entropy after partitioning based on the threshold
    return (bestThreshold, splitData, entropyAfterSplit)


class TreeNode(object):
    def __init__(self, depth = 0):
        self.depth = depth
        self.labelDistribution = collections.Counter()
        self.splitIndex = None
        self.threshold = None
        self.children = []
        self.x = []
        self.y = []

    def isLeaf(self):
        return self.splitIndex == None

    def addData(self, x, y):
        self.x += x
        self.y += y

        for label in y:
            self.labelDistribution[label] += 1

    def growTree(self, maxDepth):
        if self.depth == maxDepth:
            return

        print("Stub growTree in ", __file__)
        

    def predictProbability(self,x):
        # Remember to find the correct leaf then use an m-estimate to smooth the probability:
        #  (#_with_label_1 + 1) / (#_at_leaf + 2)
        
        print("Stub predictProbability in ", __file__)

    
    def visualize(self, depth=1):
        ## Here is a helper function to visualize the tree (if you choose to use the framework class)
        if self.isLeaf():
            print(self.labelDistribution)

        else:
            print("Split on: %d" % (self.splitIndex))

            # less than
            for i in range(depth):
                print(' ', end='', flush=True)
            print("< %f -- " % self.threshold, end='', flush=True)
            self.children[0].visualize(depth+1)

            # greater than or equal
            for i in range(depth):
                print(' ', end='', flush=True)
            print(">= %f -- " % self.threshold, end='', flush=True)
            self.children[1].visualize(depth+1)

    def countNodes(self):
        if self.isLeaf():
            return 1

        else:
            return 1 + self.children[0].countNodes() + self.children[1].countNodes()

class DecisionTree(object):
    """Wrapper class for decision tree learning."""

    def __init__(self):
        pass

    def fit(self, x, y, maxDepth = 10000, verbose=True):
        self.maxDepth = maxDepth
        
        startTime = time.time()

        self.treeNode = TreeNode(depth=0)

        self.treeNode.addData(x,y)
        self.treeNode.growTree(maxDepth)
        
        endTime = time.time()
        runtime = endTime - startTime
        
        if verbose:
            print("Decision Tree completed with %d nodes (%.2f seconds) -- %d features. Hyperparameters: maxDepth=%d." % (self.countNodes(), runtime, len(x[0]), maxDepth))

    def predictProbabilities(self, x):
        y = []

        for example in x:
            y.append(self.treeNode.predictProbability(example))        
            
        return y

    def predict(self, x, classificationThreshold=0.5):
        return [ 1 if probability >= classificationThreshold else 0 for probability in self.predictProbabilities(x) ]

    def visualize(self):
        self.treeNode.visualize()

    def countNodes(self):
        return self.treeNode.countNodes()
