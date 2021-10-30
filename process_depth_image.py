import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from numpy import asarray
from itertools import product
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# this will iterate through the pixels and show every single one :)
# photo received will be 640x480


def read_data(filename):
    path="../data/images/depth/"
    photos = []
    file = open(filename, 'r')
    for line in file:
        elements = line.split(' ')
        elements[0] = path + elements[0] + '_z16.jpg'
        photos.append(elements)
    return photos


def crop_roi(img, roi):
    y, x = img.shape
    startx = int(x * (1 - roi))
    endx = int(x * roi)
    starty = int(y * (1 - roi))
    endy = int(y * roi)
    return img[starty:endy, startx:endx]


def getGrayPhoto(photoName):
    # this function gets the name of the file
    return Image.open(photoName).convert('L')


def getArrayPhoto(photo):
    # this function gets an image object
    return asarray(photo)


def getBaseline(photo, height, width):
    # this function gets the photo as a np.array
    list = []
    for x in range(height):
        for y in range(width):
            list.append(photo.item((x, y)))
    list.sort()
    return list[int(len(list) / 2)]


def getDimensions(photo):
    return photo.shape


def getCorrespondingPhoto(photo, height, width, percentage=1):
    photoSeen = []
    medianLine = getBaseline(photo, height, width)
    for x in range(height):
        line = []
        for y in range(width):
            pos = (x, y)
            if photo.item(pos) < medianLine * (1 - percentage):
                green = (0, 255, 0)
                line.append(green)

            elif photo.item(pos) > medianLine * (1 + percentage):
                red = (255, 0, 0)
                line.append(red)
            else:
                white = (255, 255, 255)
                line.append(white)
        photoSeen.append(line)

    return photoSeen


def showImageFromArray(photoArray):
    array = np.array(photoArray, dtype=np.uint8)
    newPhoto = Image.fromarray(array)
    newPhoto.show()


def showImage(image):
    image.show()


def showPlane(array):
    # ph = np.array(array, dtype=np.uint8)
    # photo = Image.fromarray(ph)
    xt = []
    yt = []
    zt = []
    for x in range(len(array) - 1):
        for y in range(len(array[0]) - 1):
            if y % 50 == 0:
                xt.append(x)
                yt.append(y)
                zt.append(array.item((x, y)))

    X = np.array(xt)
    Y = np.array(yt)
    Z = np.array(zt)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X, Y, Z)

    plt.show()


def getPercentagesCovered(photo, height, width, percentage=1):
    medianLine = getBaseline(photo, height, width)
    area = height * width
    areaUnderLevel = 0
    areaOverLevel = 0
    areaInLevel = 0
    for x in range(height):
        for y in range(width):
            pos = (x, y)
            if photo.item(pos) < medianLine * (1 - percentage):
                areaOverLevel += 1

            elif photo.item(pos) > medianLine * (1 + percentage):
                areaUnderLevel += 1
            else:
                areaInLevel += 1

    percentageCoveredInLevel = areaInLevel / area * 100
    percentageCoveredOverLevel = areaOverLevel / area * 100
    percentageCovererUnderLevel = areaUnderLevel / area * 100
    return percentageCoveredInLevel, percentageCovererUnderLevel, percentageCoveredOverLevel


def getSmallestPixel(photo):
    smallestPixel = 255
    for x in range(height):
        for y in range(width):
            pos = (x, y)
            if photo.item(pos) < smallestPixel:
                smallestPixel = photo.item(pos)

    return smallestPixel


def getBiggestPixel(photo):
    biggestPixel = 255
    for x in range(height):
        for y in range(width):
            pos = (x, y)
            if photo.item(pos) > biggestPixel:
                biggestPixel = photo.item(pos)
    return biggestPixel


def getDistanceFromPixel(pixel, minDepth, maxDepth):
    '''
    min pixel -> the closest to black
    max pixel -> the closest to white
    This function will get the value from a pixel, the minimum depth of the photo, the maximum one, the minimum pixel and the maximum pixel
     and return the distance between the camera and that pixel
    Solution: minDepth will be corresponding to the minPixel, such that every centimeter between the depths will increase/decrease the pixel with
              a specified coefficient ( coefficient = maxDepth-minDepth/ maxPixel-minPixel )
    '''

    NUMBER_OF_PIXELS = 256  # WE USE GRAY PHOTOS
    coefficient = (maxDepth - minDepth) / NUMBER_OF_PIXELS
    distance = maxDepth  # at the beginning
    print(coefficient)
    for __ in range(255, pixel, -1):
        distance = distance - coefficient

    return distance


def showDistances(photo, height, width, minDepth, maxDepth):
    for x in range(height):
        for y in range(width):
            pos = (x, y)
            pixel = photo.item(pos)
            print(x, y, pixel)
            distance = getDistanceFromPixel(pixel, minDepth, maxDepth)
            print(str(x) + ' & ' + str(y) + " => distance: " + str(distance))


def showPercentages(array, height, width, percentage=1):
    p1, p2, p3 = getPercentagesCovered(array, height, width, percentage=percentage)
    print("__________________________________________________")
    print("This photo has ", p1, " % covered fine!", )
    print("This photo has ", p2, " % covered under the level!", )
    print("This photo has ", p3, " % covered over the level!", )
    print("__________________________________________________")


if __name__ == '__main__':
    photos = read_data("..\\data\\annotations\\min_max_depth.txt")
    for photo in photos:
        photoName = photo[0]
        minDepth = float(photo[1])
        maxDepth = float(photo[2])
        grayPhoto = getGrayPhoto(photoName)
        arrayC = getArrayPhoto(grayPhoto)
        array = crop_roi(arrayC, 0.9)
        height, width = getDimensions(array)
        baseline = getBaseline(array, height, width)
        detectedPhoto = getCorrespondingPhoto(array, height, width, percentage=0.3)
        showPercentages(array, height, width, percentage=0.3)
       # showDistances(array, height, width, minDepth, maxDepth)
        showImageFromArray(detectedPhoto)
        showImage(grayPhoto)
       # showPlane(array)
