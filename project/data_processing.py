import matplotlib.pyplot as plt
import utils
import os
import numpy as np


class DataProcessor():
    classesList = []
    numberOfClasses = 0

    def __init__(self, classLabelValue):
        self.dataset = DataProcessor.classesList[classLabelValue]
        self.training = []
        self.validation = []
        self.testing = []

    def addClass(classImages):
        DataProcessor.classesList.append(np.array(classImages))
        DataProcessor.numberOfClasses = DataProcessor.numberOfClasses+1

    def featureExtraction(image3d):
        pass

    def absolute_scale(feature):
        return feature/np.max(feature)
      
def createDataset(dataDirectoryPath):
    directories = os.listdir(dataDirectoryPath)
    images = []

    for i in range(0,len(directories)):
        classDirectory = directories[i]

        print("directory: {}".format(classDirectory))
        classDirectoryPath = "{}/{}".format(dataDirectoryPath,classDirectory)

        if os.path.isdir(classDirectoryPath):
            imageFileNames = os.listdir(classDirectoryPath)

            for ii in range(0, len(imageFileNames)):
                imageFile = imageFileNames[ii]
                imageFilePath = "{}/{}/{}".format(dataDirectoryPath,classDirectory,imageFile)

                if os.path.isfile(imageFilePath):
                    image1dArray = utils.get_image_1d(imageFilePath)
                    images.append(image1dArray)
    return images

