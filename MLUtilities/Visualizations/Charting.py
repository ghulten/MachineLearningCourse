import matplotlib
import matplotlib.pyplot as plt

# Note: matplotlib keeps internal state, so it's hard to wrap it cleanly. Keeping each charting call self contained for API simplicity.

def __SetUpChart(chartTitle=None, xAxisTitle=None, yAxisTitle=None, yTopLimit=None, yBotLimit=None):
   if chartTitle == None or xAxisTitle == None or yAxisTitle == None:
      raise UserWarning("Label your chart -- title and axes!")
   
   plt.clf()
   fig, ax = plt.subplots()
 
   ax.grid()

   if yBotLimit != None:
      ax.set_ylim(bottom=yBotLimit)
   else:
      ax.set_ylim(bottom=0)
   
   if yTopLimit != None:
      ax.set_ylim(top=yTopLimit)
      
   ax.set(title=chartTitle)      
   ax.set(xlabel=xAxisTitle, ylabel=yAxisTitle)
   
   return fig, ax

      
def __CompleteChart(fig, ax, outputDirectory=None, fileName=None):
   ax.legend()
      
   if outputDirectory != None:
      filePath = "%s\\%s" % (outputDirectory, fileName)
      fig.savefig(filePath)

   else:
      plt.show()

def __GetLineStyle(index, useLines):
   if useLines == False:
      return 'None'
   
   styles = ['-', ':', '-.', '--']
   
   return styles[index % len(styles)]

def __GetLineColor(index):
   colors = ['0.0', '0.25', '0.5']
   
   return colors[index % len(colors)]

def __GetMarker(index):
   markers = ['x', 'o', '+', '*', 's']
   
   return markers[index % len(markers)]

def PlotSeries(seriesData, seriesLabels, xAxisPoints, useLines=True, chartTitle=None, xAxisTitle=None, yAxisTitle=None, yTopLimit=None, yBotLimit=None, outputDirectory=None, fileName=None):
   if len(seriesData) != len(seriesLabels):
      raise UserWarning("Mismatched number of seriesData and seriesLabels")
   
   for i in range(len(seriesData)):
      if len(seriesData[i]) != len(xAxisPoints):
         raise UserWarning("Number of points in series %d does not match the number of xAxisPoints" % (i))

   fig, ax =__SetUpChart(chartTitle, xAxisTitle, yAxisTitle, yTopLimit, yBotLimit)
   
   for i in range(len(seriesData)):
      ax.plot(xAxisPoints, seriesData[i], marker= __GetMarker(i), color = __GetLineColor(i), linestyle=__GetLineStyle(i, useLines), label = seriesLabels[i])
              
   __CompleteChart(fig, ax, outputDirectory, fileName)

def PlotROCs(seriesFalsePositiveRates, seriesFalseNegativeRates, seriesLabels, useLines=True, chartTitle=None, xAxisTitle=None, yAxisTitle=None, yTopLimit=None, yBotLimit=None, outputDirectory=None, fileName=None):
   if len(seriesFalsePositiveRates) != len(seriesFalseNegativeRates):
      raise UserWarning("Mismatched number of seriesFalsePositiveRates and seriesFalseNegativeRates")
   
   if len(seriesFalsePositiveRates) != len(seriesLabels):
      raise UserWarning("Mismatched number of seriesFalsePositiveRates and seriesLabels")
   
   for i in range(len(seriesFalsePositiveRates)):
      if len(seriesFalsePositiveRates[i]) != len(seriesFalseNegativeRates[i]):
         raise UserWarning("Number of Y points in series %d does not match the number of X points" % (i))
      
   fig, ax =__SetUpChart(chartTitle, xAxisTitle, yAxisTitle, yTopLimit, yBotLimit)

   ax.set_xlim(left=0.0)
   ax.set_xlim(right=1.0)
   ax.invert_yaxis()
   
   for i in range(len(seriesFalsePositiveRates)):
      ax.plot(seriesFalseNegativeRates[i], seriesFalsePositiveRates[i], label = seriesLabels[i], marker ='', color = __GetLineColor(i), linestyle=__GetLineStyle(i, useLines))
              
   __CompleteChart(fig, ax, outputDirectory, fileName)

def PlotTrainValidateTestSeries(trainValues, validationValues, testValues=None, xAxisPoints=None, chartTitle=None, xAxisTitle=None, yAxisTitle=None, yTopLimit=None, yBotLimit=None, outputDirectory=None, fileName=None):
   fig, ax = __SetUpChart(chartTitle, xAxisTitle, yAxisTitle, yTopLimit, yBotLimit)
   
   if xAxisPoints == None:
      xAxisPoints = [i for i in range(len(train))]
   
   trainLine = ax.plot(xAxisPoints, trainValues, color='0.0', marker='x', linestyle='dashed', label="Train")
   validationLine = ax.plot(xAxisPoints, validationValues, color='0.7', marker='o', label="Validation")
   
   if testValues != None:
      testLine = ax.plot(xAxisPoints, testValues, color='0.5', maker='+', label="Test")
         
   __CompleteChart(fig, ax, outputDirectory, fileName)

