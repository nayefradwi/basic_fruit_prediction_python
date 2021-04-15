import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
import utils
import os
import numpy as np


class DataProcessor():
    numberOfClasses = 0
    classLabelsList =[]
    globalTrainingSet = []
    globalTestingSet = []
    featureLength = 0

    def __init__(self, classLabelValue):
        data = DataProcessor.getTrainingSetForClass(classLabelValue)
        self.training = data[0]
        self.validation = data[1]


    def getTrainingSetForClass(classLabelValue):
        # get only classlabel==classLabelValue from data
        classData = DataProcessor.globalTrainingSet[np.where(DataProcessor.globalTrainingSet[:, -1] == classLabelValue)]

        # get classlabel!=classLabelValue from data
        nonClassData = DataProcessor.globalTrainingSet[np.where(DataProcessor.globalTrainingSet[:, -1] != classLabelValue)]
        
        # undersampling
        nonClassData = nonClassData[:len(classData)]
        
        # returning training and validation set
        dataSetForOneClass = np.append(classData, nonClassData, axis=0)
        return DataProcessor.splitData(np.copy(dataSetForOneClass))
        

    def initializeDataProcessorClass():
        DataProcessor.classLabelsList = open("classLabels.csv", "r").read().split('\n')[:-1]
        DataProcessor.numberOfClasses = len(DataProcessor.classLabelsList)
        DataProcessor.globalTrainingSet = np.genfromtxt('training.csv', delimiter=',')
        DataProcessor.globalTestingSet = np.genfromtxt('testing.csv', delimiter=',')
        DataProcessor.featureLength = DataProcessor.globalTrainingSet.shape[-1]

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
        return np.average(matrix)
      
    def createDataset(dataDirectoryPath):
        directories = os.listdir(dataDirectoryPath)
        
        classes = []
        imagesFeatures = []
        for i in range(0, len(directories)):
            classDirectory = directories[i]
            
            print("directory: {}".format(classDirectory))
            classes.append(classDirectory)
            
            classDirectoryPath = "{}/{}".format(dataDirectoryPath,classDirectory)

            if os.path.isdir(classDirectoryPath):
                imageFileNames = os.listdir(classDirectoryPath)

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
        np.savetxt("classLabels.csv", np.array(classes), delimiter=',',fmt="%s")
        return imagesFeatures

# DataProcessor.createDataset("./data")