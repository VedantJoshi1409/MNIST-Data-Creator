import numpy as np
import pandas as pd
from PIL import Image
import random


def translate(arr, x, y):
    newArr = np.zeros_like(arr)
    for row in range(arr.shape[0]):
        newRow = row + y
        if 0 <= newRow < arr.shape[0]:
            for col in range(arr.shape[1]):
                newCol = col + x
                if 0 <= newCol < arr.shape[1]:
                    newArr[newRow, newCol] = arr[row, col]
    return newArr


def rotate(arr, angle):
    angle = np.radians(angle)
    centerRow = arr.shape[0] // 2
    centerCol = arr.shape[1] // 2
    newArr = np.zeros_like(arr)

    for row in range(arr.shape[0]):
        for col in range(arr.shape[1]):
            x = col - centerCol
            y = row - centerRow

            newX = int(x * np.cos(angle) - y * np.sin(angle))
            newY = int(x * np.sin(angle) + y * np.cos(angle))

            newCol = newX + centerCol
            newRow = newY + centerRow

            if 0 <= newRow < arr.shape[0] and 0 <= newCol < arr.shape[1]:
                newArr[row, col] = arr[newRow, newCol]
    return newArr


def resize(arr, xScale, yScale):
    newArr = np.zeros_like(arr)
    centerRow = arr.shape[0] // 2
    centerCol = arr.shape[1] // 2

    for row in range(arr.shape[0]):
        for col in range(arr.shape[1]):
            newRow = centerRow + int((row - centerRow) * xScale)
            newCol = centerCol + int((col - centerCol) * yScale)

            if 0 <= newRow < arr.shape[0] and 0 <= newCol < arr.shape[1]:
                newArr[newRow, newCol] = arr[row, col]
    return newArr


def noise(arr, amount):
    section = int(arr.shape[0] / amount) + 1
    newArr = np.copy(arr)
    xStart = 0
    yStart = 0
    for i in range(amount):
        for j in range(amount):
            xEnd = xStart + section
            yEnd = yStart + section
            if xEnd >= arr.shape[0]:
                xEnd = arr.shape[0] - 1
            if yEnd >= arr.shape[1]:
                yEnd = arr.shape[1] - 1
            row = random.randint(xStart, xEnd)
            col = random.randint(yStart, yEnd)
            pixel = random.randint(0, 255)
            newArr[row, col] = pixel
            xStart += section
            if xStart >= arr.shape[0]:
                break

        yStart += section
        xStart = 0
        if yStart >= arr.shape[1]:
            break
    return newArr


def createData(data, minXTranslate, maxXTranslate, minYTranslate, maxYTranslate,
               minRotation, maxRotation, minXScale,
               maxXScale, minYScale, maxYScale, minNoise, maxNoise, transformationChance, loops):
    data = data.T
    labels = data[0]
    pixels = data[1:].T
    firstLoop = True
    totalData = []

    for j in range(loops):
        newPixels = []
        for i in range(len(labels)):
            if i % 100 == 0:
                print("Processing Image {}".format(j * data.shape[1] + i))
            newImage = np.reshape(pixels[i], (28, 28))
            if random.randint(1, transformationChance) == 1:
                newImage = resize(newImage, random.uniform(minXScale, maxXScale), random.uniform(minYScale, maxYScale))
            if random.randint(1, transformationChance) == 1:
                newImage = rotate(newImage, random.randint(minRotation, maxRotation))
            if random.randint(1, transformationChance) == 1:
                newImage = translate(newImage, random.randint(minXTranslate, maxXTranslate),
                                     random.randint(minYTranslate, maxYTranslate))
            if random.randint(1, transformationChance) == 1:
                newImage = noise(newImage, random.randint(minNoise, maxNoise))

            newImage = newImage.flatten()
            newPixels.append(newImage)
        if firstLoop:
            firstLoop = False
            totalData = np.vstack([labels, np.array(newPixels).T]).T
        else:
            newData = np.vstack([labels, np.array(newPixels).T]).T
            totalData = np.vstack([totalData, newData])

    return totalData


def getData(data, label):
    data = data.T
    labels = data[0]
    pixels = data[1:].T
    newLabels = []
    specificPixels = []
    for i in range(labels.size):
        if labels[i] == label:
            newLabels.append(label)
            specificPixels.append(pixels[i])
    return np.vstack([np.array(newLabels), np.array(specificPixels).T]).T


def displayImage(data, index, name):
    pixels = data.T[1:].T
    image = Image.fromarray(np.reshape(pixels[index], (28, 28)).astype('uint8'))
    image.save('ImageBox/outputImage{}.png'.format(name))


def interweave_arrays(arr1, arr2):
    ratio = int(arr1.shape[0] / arr2.shape[0] + 1)
    print(ratio)
    counter = 0
    shape = arr1.shape
    print(ratio * shape[1])
    for i in range(arr2.shape[0]):
        if i % 10 == 0:
            print("Processing data {} of {}".format(i, arr2.shape[0]))
        arr1 = np.insert(arr1, counter, arr2[i])
        counter += ratio * shape[1]
    return np.reshape(arr1, (shape[0] + arr2.shape[0], shape[1]))


originalData = np.array(pd.read_csv('digit-recognizer/train.csv'))
# originalData4 = np.array(pd.read_csv('customData/Custom4.csv'))
# originalData = np.loadtxt('customData/customTrain.csv', delimiter=',')

data = createData(originalData, -4, 4, -4, 4, -30, 30, 0.8, 1.2, 0.8, 1.2, 1, 2, 3, 10)
# data = getData(originalData, 4)
# data = interweave_arrays(originalData, originalData4)
np.savetxt("customData/CustomTrain.csv", data, delimiter=",", fmt='%d')
