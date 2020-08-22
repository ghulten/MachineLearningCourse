kOutputDirectory = "C:\\temp\\visualize"

import MachineLearningCourse.MLUtilities.Data.Generators.SampleUniform2D as SampleUniform2D
import MachineLearningCourse.MLUtilities.Data.Generators.ConceptCircle2D as ConceptCircle2D
import MachineLearningCourse.MLUtilities.Data.Generators.ConceptSquare2D as ConceptSquare2D
import MachineLearningCourse.MLUtilities.Data.Generators.ConceptLinear2D as ConceptLinear2D
import MachineLearningCourse.MLUtilities.Data.Generators.ConceptCompound2D as ConceptCompound2D

## remember this helper function
# Charting.PlotSeriesWithErrorBars([yValues], [errorBars], [series names], xValues, chartTitle=", xAxisTitle="", yAxisTitle="", yBotLimit=0.5, outputDirectory=kOutputDirectory, fileName="")

## generat some synthetic data do help debug your learning code

generator = SampleUniform2D.SampleUniform2D(seed=100)
#conceptSquare = ConceptSquare2D.ConceptSquare2D(width=.2)
conceptLinear = ConceptLinear2D.ConceptLinear2D(bias=0.05, weights=[0.3, -0.3])
conceptCircle = ConceptCircle2D.ConceptCircle2D(radius=.3)

concept = ConceptCompound2D.ConceptCompound2D(concepts = [ conceptLinear, conceptCircle ])

xTest = generator.generate(1000)
yTest = concept.predict(xTest)

xTrain = generator.generate(1000)
yTrain = concept.predict(xTrain)


import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting
import MachineLearningCourse.MLUtilities.Visualizations.Visualize2D as Visualize2D

## this code outputs the true concept.
visualize = Visualize2D.Visualize2D(kOutputDirectory, "Generated Concept")
visualize.Plot2DDataAndBinaryConcept(xTest,yTest,concept)
visualize.Save()

## you can use this to visualize what your model is learning.
# visualize = Visualize2D.Visualize2D(kOutputDirectory, "Your Boosted Tree...", size=150)
# visualize.PlotBinaryConcept(model)

# Or you can use it to visualize individual models that you learened, e.g.:
# visualize.PlotBinaryConcept(model->modelLearnedInRound[2])
    
## you might like to see the training or test data too, so you might prefer this to simply calling 'PlotBinaryConcept'
#visualize.Plot2DDataAndBinaryConcept(xTrain,yTrain,model)

# And remember to save
# visualize.Save()
