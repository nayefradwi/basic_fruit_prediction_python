import numpy as np
import os

def save3dFileAs2dCsv(filename, imageStack):
    images = imageStack.reshape(imageStack.shape[0], -1)
    filename = "{}_{}.csv".format(filename, imageStack.shape[0])
    np.savetxt(filename, images, delimiter=',')

def createDatasetDirectory():
    try:
        os.mkdir(os.curdir+"/dataset")
    except FileExistsError:
        pass
    except:
        print("unkown error occured when creating dataset")

# def getClassNameFromFileName(fileName):
#     return fileName.split("_")[0]

