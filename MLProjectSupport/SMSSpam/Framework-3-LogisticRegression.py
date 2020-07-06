
runUnitTest = False
runSMSSpam  = True
    
if runUnitTest:
    # Little synthetic dataset to help with implementation. 2 features, 8 samples.
    xTrain = [[.1, .1], [.2, .2], [.2, .1], [.1, .2], [.95, .95], [.9, .8], [.8, .9], [.7, .6]]
    yTrain = [0, 0, 0, 0, 1, 1, 1, 1]

    # create a linear model with weights initialized to 0
    import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
    model = LogisticRegression.LogisticRegression(featureCount = len(xTrain[0]))

    # To use this function you need to install the PIL imaging library. Instructions are in the lecture notes.
    import MachineLearningCourse.MLUtilities.Visualizations.Visualize2D as Visualize2D

    while not model.converged:
        # do 10 iterations of training
        model.fit(xTrain, yTrain, iterations=10, stepSize=.1, convergence=0.001, verbose=True)
        
        # then look at the models weights
        model.visualize()

        # and visualize the model's decision boundary
        visualization = Visualize2D.Visualize2D("C:\\temp\\visualize", "{0:04}.test".format(model.iterations))
        visualization.PlotDataAndBinaryConcept(xTrain, yTrain, model)
        visualization.Save()
        


if runSMSSpam:
    import MachineLearningCourse.MLProjectSupport.SMSSpam.SMSSpamSupport as SMSSpamSupport

    kDataPath = "MachineLearningCourse\\MLProjectSupport\\SMSSpam\\dataset\\SMSSpamCollection"

    (xRaw, yRaw) = SMSSpamSupport.LoadRawData(kDataPath)
    (xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = SMSSpamSupport.TrainValidateTestSplit(xRaw, yRaw, percentValidate=.1, percentTest=.1)

    import MachineLearningCourse.MLProjectSupport.SMSSpam.SMSSpamFeaturize as SMSSpamFeaturize
    featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=True)
    featurizer.CreateVocabulary(xTrainRaw, yTrain, supplementalVocabularyWords=['call','to','your'])

    xTrain      = featurizer.Featurize(xTrainRaw)
    xValidate   = featurizer.Featurize(xValidateRaw)
    xTest       = featurizer.Featurize(xTestRaw)

    #############################
    # Learn the logistic regression model
    
    print("Learning the logistic regression model:")
    print("<iterations>, <train set loss>")
    import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
    logisticRegressionModel = LogisticRegression.LogisticRegression()
    
    logisticRegressionModel.fit(xTrain, yTrain, iterations=5000, stepSize=0.1, convergence=.0001, verbose=True)
    
    #############################
    # Comparing the models
    
    import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification

    print ("\nHeuristic model:")
    import MachineLearningCourse.MLUtilities.Learners.LinearHeuristicModel as LinearHeuristicModel
    heuristicModel = LinearHeuristicModel.LinearHeuristicModel()

    heuristicModel.fit(xTrain, yTrain, -1.0, [ .75, .75, .75, .25, .25 ])
    
    heuristicModel.visualize()
    EvaluateBinaryClassification.ExecuteAll(yValidate, heuristicModel.predict(xValidate))
    
    print ("\nLogistic regression model:")
    logisticRegressionModel.visualize()
    EvaluateBinaryClassification.ExecuteAll(yValidate, logisticRegressionModel.predict(xValidate, classificationThreshold=0.5))
