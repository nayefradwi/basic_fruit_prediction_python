from matplotlib import pyplot as plt
import utils
import os
import numpy as np
import file_processing as fp
import time

class DataProcessor():
    classesList = []

    def __init__(self, filePath):
        data = np.genfromtxt(filePath, delimiter=',')
        splitData = self.splitData(data)
        self.training = splitData[0]
        self.testing = splitData[1]

    def splitData(self, data):
        np.random.shuffle(data)
        seventyPercent = int(np.floor(0.75*len(data[:])))
        return [data[:seventyPercent], data[seventyPercent:]]
      

def getImageOriginalShape(newShape, oldShape):
    return newShape[0], newShape[1] // oldShape[2], oldShape[2]

def loadClassCsvFile(classPath):
    data = np.genfromtxt(classPath,delimiter=",", dtype=int)
    return data.reshape(getImageOriginalShape(data.shape, (data.shape[0], data.shape[-1], 3 )))
    

def createDataset(dataDirectoryPath):
    timeBeforeExecution = time.time()

    # create a dataset directory that will store .csv files
    # fp.createDatasetDirectory()

    # get list of directories in the dataset (classes)
    directories = os.listdir(dataDirectoryPath)
    
    # loop over the data directory 
    # len(directories)
    for i in range(0,1):
        classDirectory = directories[i]
        print("directory: {}".format(classDirectory))
        classDirectoryPath = "{}/{}".format(dataDirectoryPath,classDirectory)

        if os.path.isdir(classDirectoryPath):
            imageFileNames = os.listdir(classDirectoryPath)
            # imagesStack = np.empty((0,100, 3),dtype=int)

            images = []
            # loop over each image in a specific directory
            for ii in range(0, len(imageFileNames)):
                imageFile = imageFileNames[ii]
                imageFilePath = "{}/{}/{}".format(dataDirectoryPath,classDirectory,imageFile)

                if os.path.isfile(imageFilePath):
                    
                    image3dArray = utils.get_image_3d(imageFilePath)
                    images.append(image3dArray)
                    # imagesStack = np.vstack((i,image3dArray))
            # fp.save3dFileAs2dCsv("dataset/{}".format(classDirectory.lower()), imagesStack)
            DataProcessor.classesList.append(np.array(images))
            
    timeTaken = time.time()-timeBeforeExecution
    print("time took for createDataset: ")
    print(timeTaken)

def loadClasses():
    timeBeforeExecution = time.time()
    stackedImagesArray = []
    classes = os.listdir("{}/dataset".format(os.curdir))
    for classCsvFile in classes:
        print("file: {}".format(classCsvFile))
        stackedImagesArray.append(loadClassCsvFile("{}/dataset/{}".format(os.curdir, classCsvFile)))
    timeTaken = time.time()-timeBeforeExecution
    print("time took for loadclasses: ")
    print(timeTaken)
    return stackedImagesArray

# def calculateMacroRecall(actuals, predictions):
#     pass

# def calculateMacroPrecision(actuals, predictions):
#     pass

# def calculateMacroF1(actuals, predictions):
#     pass

createDataset("./data")
# loadClasses()