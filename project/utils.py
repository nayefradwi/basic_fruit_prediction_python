# -*- coding: utf-8 -*-
"""
Utilites to read images and display/plot them. Also, a simple demonstration of how to list files
in a folder. To be used for the second term project of Machine Learning course.

********DO NOT MODIFY ANY OF THE CONTENTS IN THIS FILE.**********

@author: Abdulaziz Al-Ali (c)
"""

# ********DO NOT MODIFY ANY OF THE CONTENTS IN THIS FILE.**********


from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
from numpy.core.shape_base import vstack

def get_image_3d(path):
    """
    This function receives a path to an image and returns a 3D numpy array
    containing the pixel (red, green, blue) colors.
    
    An example path is "./data/Apple/apple_1.jpg" where we assume there is
    a directory called data in "this" folder and it has a directory called "Apple"
    
    On the returned numpy array, if you use pixels[1][4] it will return the
    three color values for the pixel in the position x=1 and y=4
    
    """
    im = Image.open(path, 'r')
    pixels = np.array(list(im.getdata())).reshape(100,100,3)
    
    return pixels


def get_image_1d(path):
    """
    This function receives a path to an image and returns a 1D numpy array
    containing all the colors (red, blue, green) of all pixels.
    So, since our image is 100x100 pixels, this will return a numpy array
    that has 30000 values (100x100x3).
    
    An example path is "./data/Apple/apple_1.jpg" where we assume there is
    a directory called data in "this" folder and it has a directory called "Apple"
    
    On the returned numpy array, if you use pixels[1] it will return the
    pixel value (red, green, or blue depends on order).
    
    """
    im = Image.open(path, 'r')
    pixels = np.array(list(im.getdata())).reshape(30000)
    
    return pixels


def show_image(pixels):
    """
    given a 3D numpy array of an image (such as that returned by get_image_3d above)
    it will display it in a new plot
    """
    
    plt.figure()
    plt.imshow(pixels)
    return None


def print_files(directory):
    """
    This function is just to demonstrate how you can access all the files
    inside a directory.
    """
    
    for file in os.listdir(directory):
        print(directory+file)
        
    return None

