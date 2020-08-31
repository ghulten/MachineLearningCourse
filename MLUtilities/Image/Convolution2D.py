Identity       = [[0, 0, 0],  [0, 1, 0], [0, 0, 0]]
SobelX         = [[1, 0, -1], [2, 0,-2], [1, 0, -1]]
SobelY         = [[1, 2, 1],  [0, 0, 0], [-1, -2, -1]]

def Convolution3x3(image, filter):
    # check that the filter is formated correctly
    if not (len(filter) == 3 and len(filter[0]) == 3 and len(filter[1]) == 3 and len(filter[2]) == 3):
        raise UserWarning("Filter is not formatted correctly, should be [[x,x,x], [x,x,x], [x,x,x]]")

    xSize = image.size[0]
    ySize = image.size[1]
    pixels = image.load()

    # initialize the answer array-nest
    answer = []
    for x in range(xSize):
        answer.append([ 0 for y in range(ySize) ])

    # skip the edges
    for x in range(1, xSize - 1):
        for y in range(1, ySize - 1):

            value = 0
            for i in range(3):
                for j in range(3):
                    value += filter[2-j][2-i] * pixels[x + (i-1), y + (j-1)]
            answer[x][y] = value

    return answer