import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
import utils
import os
import numpy as np


class DataProcessor():
    numberOfClasses = 0

    def __init__(self, classLabelValue):
        self.dataset = []
        self.training = []
        self.validation = []
        self.testing = []

    def splitData(data):
        np.random.shuffle(data)
        seventyPercent = int(np.floor(0.75*len(data[:])))
        return [np.array(data[:seventyPercent]), np.array(data[seventyPercent:])]

    def featureExtraction(image3d):
        averageMatrixRgb = []

        # loop every channel
        for rgbChannelIndex in range(0,image3d.shape[-1]):
            averageMatrixOFChannel = DataProcessor.getChannelMatrix(image3d[:,:, rgbChannelIndex], stepSize=10)
            averageMatrixRgb.append(averageMatrixOFChannel)
        
        averageMatrixRgb = np.array(averageMatrixRgb, dtype=int)
        averageMatrixRgb = averageMatrixRgb.T

        # flatten the features
        flattened = averageMatrixRgb.reshape((averageMatrixRgb.shape[0]*averageMatrixRgb.shape[1]*averageMatrixRgb.shape[2]))    

        return flattened

    # returns a 5x5 matrix for a channel 
    def getChannelMatrix(matrix, stepSize):
        averageChannelMatrix = np.empty((int(matrix.shape[0]/stepSize), int(matrix.shape[1]/stepSize)))
        for i in range(0, averageChannelMatrix.shape[0]):
            for ii in range(0, averageChannelMatrix.shape[1]):
                average = DataProcessor.averageValue(matrix[i:(i+1)*stepSize, ii:(ii+1)*stepSize])
                averageChannelMatrix[i, ii] = average.astype(int)
        return averageChannelMatrix

    def averageValue(matrix):
        try:
            return np.average(matrix)
        except:
            print("error")
      
    def createDataset(dataDirectoryPath):
        directories = os.listdir(dataDirectoryPath)
        np.savetxt("classesLength.csv", [len(directories)])
    
        imagesFeatures = []
        # len(directories)
        for i in range(0, len(directories)):
            classDirectory = directories[i]

            print("directory: {}".format(classDirectory))
            classDirectoryPath = "{}/{}".format(dataDirectoryPath,classDirectory)

            if os.path.isdir(classDirectoryPath):
                imageFileNames = os.listdir(classDirectoryPath)

                # len(imageFileNames)
                for ii in range(0, len(imageFileNames)):
                    imageFile = imageFileNames[ii]
                    imageFilePath = "{}/{}/{}".format(dataDirectoryPath,classDirectory,imageFile)

                    if os.path.isfile(imageFilePath):
                        image3dArray = utils.get_image_3d(imageFilePath)
                        features = DataProcessor.featureExtraction(image3dArray)
                        features = np.append(features,[i])
                        imagesFeatures.append(features)

        imagesFeatures = np.array(imagesFeatures)
        [training, testing] = DataProcessor.splitData(imagesFeatures)
        np.savetxt("training.csv", training, delimiter=",")
        np.savetxt("testing.csv", testing, delimiter=",")
        return imagesFeatures

