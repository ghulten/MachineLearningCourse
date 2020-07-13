import matplotlib
import matplotlib.pyplot as plt

def PlotTrainValidateTestSeries(trainValues, validationValues, testValues=None, xAxisPoints=None, chartTitle=None, xAxisTitle=None, yAxisTitle=None, yTopLimit=None, yBotLimit=None, outputDirectory=None, fileName=None):
   if chartTitle == None or xAxisTitle == None or yAxisTitle == None:
      raise UserWarning("Label your chart -- title and axes!")
      
   plt.clf()
   fig, ax = plt.subplots()
   
   if xAxisPoints == None:
      xAxisPoints = [i for i in range(len(train))]
   
   trainLine = ax.plot(xAxisPoints, trainValues, color='0.0', marker='x', linestyle='dashed', label="Train")
   validationLine = ax.plot(xAxisPoints, validationValues, color='0.7', marker='o', label="Validation")
   
   if testValues != None:
      testLine = ax.plot(xAxisPoints, testValues, color='0.5', maker='+', label="Test")
   
   ax.set(xlabel=xAxisTitle, ylabel=yAxisTitle)
   
   if chartTitle != None:
      ax.set(title=chartTitle)
      
   ax.grid()
   ax.legend()

   if yBotLimit != None:
      ax.set_ylim(bottom=yBotLimit)
   else:
      ax.set_ylim(bottom=0)
   
   if yTopLimit != None:
      ax.set_ylim(top=yTopLimit)
   
   if outputDirectory != None:
      filePath = "%s\\%s" % (outputDirectory, fileName)
      fig.savefig(filePath)

   else:
      plt.show()

