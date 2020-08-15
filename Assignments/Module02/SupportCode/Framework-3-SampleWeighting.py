import MachineLearningCourse.MLUtilities.Learners.DecisionTreeWeighted as DecisionTreeWeighted

# some sample tests. You may not have implemented the same way, so you might have to adapt.

WeightedEntropyUnitTest = True
if WeightedEntropyUnitTest:
    y = [1, 1, 0, 1, 1, 0, 0, 0, 1, 0]
    
    print("Unweighted: ")
    print(DecisionTreeWeighted.Entropy(y, [ 1.0 for label in y ]))

    print("All 1s get 0 weight: ")
    print(DecisionTreeWeighted.Entropy(y, [ 0.0 if label == 1 else 1.0 for label in y ]))

    print("All 1s get .1 weight: ")
    print(DecisionTreeWeighted.Entropy(y, [ 0.1 if label == 1 else 1.0 for label in y ]))


WeightedSplitUnitTest = False
if WeightedSplitUnitTest:
    x = [[.1], [.2], [.3], [.4], [.5], [.6], [.7], [.8], [.9], [1.0]]
    y = [1, 1, 0, 1, 1, 0, 0, 0, 1, 0]
    
    print("Unweighted: ")
    print(DecisionTreeWeighted.FindBestSplitOnFeature(x, y, [ 1.0 for label in y ], 0))

    print("All 1s get 0 weight: ")
    print(DecisionTreeWeighted.FindBestSplitOnFeature(x, y, [ 0.0 if label == 1 else 1.0 for label in y ], 0))

    print("All 1s get .1 weight: ")
    print(DecisionTreeWeighted.FindBestSplitOnFeature(x, y, [ 0.1 if label == 1 else 1.0 for label in y ], 0))


WeightTreeUnitTest = False
if WeightTreeUnitTest:
    xTrain = [[.1], [.2], [.3], [.4], [.5], [.6], [.7], [.8], [.9], [1.0]]
    yTrain = [1, 1, 0, 1, 1, 0, 0, 0, 1, 0]

    print("Unweighted:")
    model = DecisionTreeWeighted.DecisionTreeWeighted()
    model.fit(xTrain, yTrain, maxDepth = 1)

    model.visualize()

    print("Weighted 1s:")
    model = DecisionTreeWeighted.DecisionTreeWeighted()
    model.fit(xTrain, yTrain, weights=[ 10 if y == 1 else 0.1 for y in yTrain ], maxDepth = 1)

    model.visualize()

    print("Weighted 0s:")
    model = DecisionTreeWeighted.DecisionTreeWeighted()
    model.fit(xTrain, yTrain, weights=[ 1 if y == 0 else 0.1 for y in yTrain ], maxDepth = 1)

    model.visualize()


