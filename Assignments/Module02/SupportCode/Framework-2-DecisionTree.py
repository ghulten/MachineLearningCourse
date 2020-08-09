import MachineLearningCourse.MLUtilities.Learners.DecisionTree as DecisionTree

### Some simple test cases to get you started. You'll have to work out the correct answers yourself.

print("test simple split")
x=[[1],[1],[1],[1],[1],[0],[0],[0],[0],[0]]
y = [1,1,1,1,1,1,0,0,0,1]
model = DecisionTree.DecisionTree()
model.fit(x, y)

model.visualize()

print("test no split")
x=[[1],[1],[1],[1],[1],[0],[0],[0],[0],[0]]
y = [1,1,0,1,1,1,1,1,0,1]
model = DecisionTree.DecisionTree()
model.fit(x, y)

model.visualize()

print("test numeric feature sort")
x = [[1,3], [2,2], [19,7], [4,1]]
y = [1,1,0,0]

model = DecisionTree.DecisionTree()
model.fit(x, y)

model.visualize()

print("Bigger tree")
x = [[10, 7], [9,8], [101, 71], [44, 44], [19, 111], [1, 2], [1,3], [2,2], [19,7], [4,1]]
y = [1,1,0,0,1,1,0,0,1,1]

model = DecisionTree.DecisionTree()
model.fit(x, y)

model.visualize()

## These might help you debug...
doVisualize = True
if doVisualize:
    kOutputDirectory = "C:\\Users\\ghult\\Desktop\\visualize"
    
    import MachineLearningCourse.MLUtilities.Data.Generators.SampleUniform2D as SampleUniform2D
    import MachineLearningCourse.MLUtilities.Data.Generators.ConceptCircle2D as ConceptCircle2D
    import MachineLearningCourse.MLUtilities.Data.Generators.ConceptSquare2D as ConceptSquare2D
    import MachineLearningCourse.MLUtilities.Data.Generators.ConceptLinear2D as ConceptLinear2D

    generator = SampleUniform2D.SampleUniform2D(seed=100)
    #concept = ConceptSquare2D.ConceptSquare2D(width=.2)
    concept = ConceptLinear2D.ConceptLinear2D(bias=0.05, weights=[0.2, -0.2])
    #concept = ConceptCircle2D.ConceptCircle2D(radius=.3)

    x = generator.generate(100)
    y = concept.predict(x)

    import MachineLearningCourse.MLUtilities.Visualizations.Visualize2D as Visualize2D

    visualize = Visualize2D.Visualize2D(kOutputDirectory, "Generated Concept")

    visualize.Plot2DDataAndBinaryConcept(x,y,concept)
    visualize.Save()

    model = DecisionTree.DecisionTree()
    model.fit(x, y, maxDepth = 2)
    model.visualize()

    visualize = Visualize2D.Visualize2D(kOutputDirectory, "DecisionTree on Generated Concept")

    visualize.Plot2DDataAndBinaryConcept(x,y,model)
    visualize.Save()
