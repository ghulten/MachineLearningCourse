import MachineLearningCourse.MLProjectSupport.SMSSpam.SMSSpamDataset as SMSSpamDataset

(xRaw, yRaw) = SMSSpamDataset.LoadRawData()

import MachineLearningCourse.MLUtilities.Data.Sample as Sample
import MachineLearningCourse.Assignments.Module01.SupportCode.SMSSpamFeaturize as SMSSpamFeaturize
import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
import MachineLearningCourse.MLUtilities.Learners.MostCommonClassModel as MostCommonClassModel
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification

(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw, percentValidate=.1, percentTest=.1)

doModelEvaluation = True
if doModelEvaluation:
    ######
    ### Build a model and evaluate on validation data
    stepSize = 1.0
    convergence = 0.001

    featurizer = SMSSpamFeaturize.SMSSpamFeaturize()
    featurizer.CreateVocabulary(xTrainRaw, yTrain, numMutualInformationWords = 25)

    xTrain      = featurizer.Featurize(xTrainRaw)
    xValidate   = featurizer.Featurize(xValidateRaw)
    xTest       = featurizer.Featurize(xTestRaw)

    frequentModel = LogisticRegression.LogisticRegression()
    frequentModel.fit(xTrain, yTrain, convergence=convergence, stepSize=stepSize, verbose=True)

    ######
    ### Use equation 5.1 from Mitchell to bound the validation set error and the true error
    import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds

    print("Logistic regression with 25 features by mutual information:")
    validationSetAccuracy = EvaluateBinaryClassification.Accuracy(yValidate, frequentModel.predict(xValidate))
    print("Validation set accuracy: %.4f." % (validationSetAccuracy))
    for confidence in [.5, .8, .9, .95, .99]:
        (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(validationSetAccuracy, len(xValidate), confidence)    
        print(" %.2f%% accuracy bound: %.4f - %.4f" % (confidence, lowerBound, upperBound))

    ### Compare to most common class model here...

# Set this to true when you've completed the previous steps and are ready to move on...
doCrossValidation = False
if doCrossValidation:
    import MachineLearningCourse.MLUtilities.Data.CrossValidation as CrossValidation    
    numberOfFolds = 5

    import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds
    
    # Good luck!