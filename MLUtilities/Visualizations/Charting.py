import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Note: matplotlib keeps internal state, so it's hard to wrap it cleanly. Keeping each charting call self contained for API simplicity.

def __SetUpChart(chartTitle=None, xAxisTitle=None, yAxisTitle=None):
   if chartTitle == None or xAxisTitle == None or yAxisTitle == None:
      raise UserWarning("Label your chart -- title and axes!")
   
   plt.clf()
   fig, ax = plt.subplots()
 
   ax.grid()
      
   ax.set(title=chartTitle)      
   ax.set(xlabel=xAxisTitle, ylabel=yAxisTitle)
   
   return fig, ax

      
def __CompleteChart(fig, ax, outputDirectory=None, fileName=None, yTopLimit=None, yBotLimit=None, invertYAxis=False):
   if yBotLimit != None:
      ax.set_ylim(bottom=yBotLimit)
   else:
      ax.set_ylim(bottom=0)
   
   if yTopLimit != None:
      ax.set_ylim(top=yTopLimit)

   ax.legend()
      
   if outputDirectory != None:
      filePath = "%s\\%s" % (outputDirectory, fileName)
      fig.savefig(filePath)

   else:
      plt.show()
      
   if invertYAxis:
      ax.invert_yaxis()
    
   matplotlib.pyplot.close(fig)

def __GetLineStyle(index, useLines):
   if useLines == False:
      return 'None'
   
   styles = ['-', ':', '-.', '--']
   
   return styles[index % len(styles)]

def __GetLineColor(index):
   colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
   
   return colors[index % len(colors)]

def __GetMarker(index, useMarkers=False):
   markers = ['x', 'o', '+', '*', 's']
   
   if useMarkers:
      return markers[index % len(markers)]
   else:
      return None

def PlotSeries(seriesData, seriesLabels, xAxisPoints, useLines=True, useMarkers=True, chartTitle=None, xAxisTitle=None, yAxisTitle=None, yTopLimit=None, yBotLimit=None, outputDirectory=None, fileName=None):
   if len(seriesData) != len(seriesLabels):
      raise UserWarning("Mismatched number of seriesData and seriesLabels")
   
   for i in range(len(seriesData)):
      if len(seriesData[i]) != len(xAxisPoints):
         raise UserWarning("Number of points in series %d does not match the number of xAxisPoints" % (i))

   fig, ax =__SetUpChart(chartTitle, xAxisTitle, yAxisTitle)
   
   for i in range(len(seriesData)):
      ax.plot(xAxisPoints, seriesData[i], marker= __GetMarker(i, useMarkers), color = __GetLineColor(i), linestyle=__GetLineStyle(i, useLines), label = seriesLabels[i])
              
   __CompleteChart(fig, ax, outputDirectory, fileName, yTopLimit, yBotLimit)

def PlotSeriesWithErrorBars(seriesData, seriesErrorBars, seriesLabels, xAxisPoints, useLines=True, useMarkers=True, chartTitle=None, xAxisTitle=None, yAxisTitle=None, yTopLimit=None, yBotLimit=None, outputDirectory=None, fileName=None):
   if len(seriesData) != len(seriesLabels):
      raise UserWarning("Mismatched number of seriesData and seriesLabels")
   
   if len(seriesErrorBars) != len(seriesLabels):
      raise UserWarning("Mismatched number of error bars")
   
   for i in range(len(seriesData)):
      if len(seriesData[i]) != len(xAxisPoints):
         raise UserWarning("Number of points in series %d does not match the number of xAxisPoints" % (i))

      if len(seriesData[i]) != len(seriesErrorBars[i]):
         raise UserWarning("Number of points in series %d does not match the number of error bars" % (i))


   fig, ax =__SetUpChart(chartTitle, xAxisTitle, yAxisTitle)
   
   for i in range(len(seriesData)):
      ax.errorbar(xAxisPoints, seriesData[i], seriesErrorBars[i], marker= __GetMarker(i, useMarkers), color = __GetLineColor(i), linestyle=__GetLineStyle(i, useLines), label = seriesLabels[i])
              
   __CompleteChart(fig, ax, outputDirectory, fileName, yTopLimit, yBotLimit)

def PlotROCs(seriesFalsePositiveRates, seriesFalseNegativeRates, seriesLabels, useLines=True, chartTitle=None, xAxisTitle=None, yAxisTitle=None, outputDirectory=None, fileName=None):
   if len(seriesFalsePositiveRates) != len(seriesFalseNegativeRates):
      raise UserWarning("Mismatched number of seriesFalsePositiveRates and seriesFalseNegativeRates")
   
   if len(seriesFalsePositiveRates) != len(seriesLabels):
      raise UserWarning("Mismatched number of seriesFalsePositiveRates and seriesLabels")
   
   for i in range(len(seriesFalsePositiveRates)):
      if len(seriesFalsePositiveRates[i]) != len(seriesFalseNegativeRates[i]):
         raise UserWarning("Number of Y points in series %d does not match the number of X points" % (i))
      
   fig, ax =__SetUpChart(chartTitle, xAxisTitle, yAxisTitle)

   for i in range(len(seriesFalsePositiveRates)):
      ax.plot(seriesFalseNegativeRates[i], seriesFalsePositiveRates[i], label = seriesLabels[i], marker ='', color = __GetLineColor(i), linestyle=__GetLineStyle(i, useLines))

   ax.set_xlim(left=0.0)
   ax.set_xlim(right=1.0)
              
   __CompleteChart(fig, ax, outputDirectory, fileName, 0.0, 1.0, invertYAxis=True)


def PlotTrainValidateTestSeries(trainValues, validationValues, testValues=None, xAxisPoints=None, chartTitle=None, xAxisTitle=None, yAxisTitle=None, yTopLimit=None, yBotLimit=None, outputDirectory=None, fileName=None):
   fig, ax = __SetUpChart(chartTitle, xAxisTitle, yAxisTitle)
   
   if xAxisPoints == None:
      xAxisPoints = [i for i in range(len(trainValues))]
   
   trainLine = ax.plot(xAxisPoints, trainValues, color='0.0', marker='x', linestyle='dashed', label="Train")
   validationLine = ax.plot(xAxisPoints, validationValues, color='0.7', marker='o', label="Validation")
   
   if testValues != None:
      testLine = ax.plot(xAxisPoints, testValues, color='0.5', maker='+', label="Test")
         
   __CompleteChart(fig, ax, outputDirectory, fileName, yTopLimit, yBotLimit)

