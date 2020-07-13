import PIL
from PIL import Image

kDarkRed = (.65,.55,.55)
kDarkGreen = (.55,.65,.55)
kBrightRed = (.99,0,0)
kBrightGreen = (0,.99,0)

class Visualize2D(object):
    def __init__(self, directory, name, size=400):
        self.outputPath = "%s\\%s.png" % (directory,name)
        self.image = Image.new("RGB", (size,size), "White")
        self.pixels = self.image.load()
        self.size = size

    def Plot2DPoints(self, x, y, RGBFloat, pointSize=5, weights=None, labelMask=None):
        if weights == None:
            weights = [1 for i in range(len(x))]
        
        for i in range(len(x)):
            if labelMask == None or y[i] in labelMask:
                color = (int(255 * RGBFloat[0] * weights[i]), int(255 * RGBFloat[1] * weights[i]), int(255 * RGBFloat[2] * weights[i]))

                xCoordinate = x[i][0] * self.size
                yCoordinate = x[i][1] * self.size

                if xCoordinate + pointSize >= self.size:
                    xCoordinate = xCoordinate - pointSize

                if yCoordinate + pointSize >= self.size:
                    yCoordinate = yCoordinate - pointSize
                    
                for xOffset in range(pointSize):
                    for yOffset in range(pointSize):
                        self.pixels[xCoordinate + xOffset, yCoordinate + yOffset] = color

    def PlotBinaryConcept(self, model, colorTrue=kDarkGreen, colorFalse=kDarkRed):
        points = []
        samples = int(self.size + self.size * 0.1)
        for x in range(samples):
            for y in range(samples):
                points.append([x / samples, y / samples])

        predicted = model.predict(points)

        self.Plot2DPoints(points, predicted, colorFalse, pointSize=1, weights = [1 for i in predicted], labelMask = [0])
        self.Plot2DPoints(points, predicted, colorTrue, pointSize=1, weights =  [1 for i in predicted], labelMask = [1])

    def Plot2DDataAndBinaryConcept(self, x, y, model):
        self.PlotBinaryConcept(model)
        self.Plot2DPoints(x, y, kBrightRed, labelMask=[0])
        self.Plot2DPoints(x, y, kBrightGreen, labelMask=[1])

    def Save(self):
        self.image.save(self.outputPath)