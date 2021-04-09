from datetime import date
import matplotlib.pyplot as plt
from numpy.lib.type_check import imag
import utils
import os
import numpy as np
from sklearn.decomposition import PCA

# todo: remove time and pca before submitting 
import time

class DataProcessor():
    classesList = []
    numberOfClasses = 0
    def __init__(self, classLabelValue):
        data = DataProcessor.classesList[classLabelValue]

    def addClass(classImages):
        DataProcessor.classesList.append(np.array(classImages))
        DataProcessor.numberOfClasses = DataProcessor.numberOfClasses+1
    
    def dimensionReduction(image):
        image = DataProcessor.convertImageToGrayScale(image)
        DataProcessor.pcaFunction(image)
        DataProcessor.pc(image)
        return image

    def pcaFunction(data):
        data = DataProcessor.center(data)
        cov = DataProcessor.getCovMatrix(data)
        eigVector, eigValue = np.linalg.eig(cov)
        idx = eigVector.argsort()[::-1] # Sort descending and get sorted indices
        eigVector = eigVector[idx] # Use indices on eigv vector
        eigValue = eigValue[:,idx] #
        # print(eigVector[:5]) 
        data = np.abs(data.dot(eigValue[:, :10]))
        print(data.shape)

    def pc(A):
        # print(A[0][:10])
        # A = DataProcessor.scaleData(A)
        # print(A[0][:10])
        pca = PCA(n_components=0.9)
        pca.fit(A)
        components = pca.transform(A)
        projected = pca.inverse_transform(components)
        print(components.shape)
        # print(components[0])
        # print(projected.shape)
        # utils.show_image(projected)
        # plt.show()
        
    def getCovMatrix(data):
        return np.cov(data.T) / data.shape[0]
        
    def center(data):
        return data - data.mean(axis=0)

    def convertImageToGrayScale(image):
        rgbWeights = [0.2989, 0.5870, 0.1140]
        return np.dot(image, rgbWeights)

      
def createDataset(dataDirectoryPath):
    # timeBeforeExecution = time.time()
    directories = os.listdir(dataDirectoryPath)
    
    # loop over the data directory 
    # len(directories)
    for i in range(0,1):
        classDirectory = directories[i]
        print("directory: {}".format(classDirectory))
        classDirectoryPath = "{}/{}".format(dataDirectoryPath,classDirectory)

        if os.path.isdir(classDirectoryPath):
            imageFileNames = os.listdir(classDirectoryPath)

            # list of images for a specific class
            images = []

            # loop over each image in a specific directory
            # len(imageFileNames)
            for ii in range(0, 1):
                imageFile = imageFileNames[ii]
                imageFilePath = "{}/{}/{}".format(dataDirectoryPath,classDirectory,imageFile)
                if os.path.isfile(imageFilePath):
                    image3dArray = utils.get_image_3d(imageFilePath)
                    image3dArray = DataProcessor.dimensionReduction(image3dArray)
                    images.append(image3dArray)

            # add class images to be stored in static list
            DataProcessor.addClass(images)

    # timeTaken = time.time()-timeBeforeExecution
    # print("time took for createDataset: ")
    # print(timeTaken)


createDataset("./data")