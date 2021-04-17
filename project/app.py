import utils
import tkinter as tk
from perceptron import Perceptron
from data_processing import DataProcessor
from tkinter.constants import BOTTOM, CENTER, LEFT, RIGHT, TOP, TRUE
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from tkinter.filedialog import askopenfilename

image = utils.get_image_3d("./data/Apple/apple_0.jpg")
figure = utils.show_image(image)
isInitialized = False

# window related 
root = tk.Tk()
epochsVar = tk.StringVar(value="Epochs")
labelText = tk.StringVar(value="apple")
root.geometry("800x600")
root.resizable(False, False)

# state variables

# window functions
def on_closing():
    root.quit()

def create_dataset():
    filename = askopenfilename(filetypes=[("images", "*.jpg")])
    if os.isdir(filename):
        pass

def train_dataset():
    pass

def test_dataset():
    pass

def selectImage():
    global isInitialized
    if not isInitialized:
        status = Perceptron.initialize()
        if status == 0:
            print("perceptron initialized")
            isInitialized = True
        else:
            print("could not locate weights file")
            return
    filename = askopenfilename(filetypes=[("images", "*.jpg")])
    if filename:
        image3d = utils.get_image_3d(filename)
        imageFeature = DataProcessor.featureExtraction(image3d)
        predicition = Perceptron.predictModel(example=imageFeature)
        classificationLabel.configure(text=predicition[-1])


# buttons
bottomFrame = tk.Frame(root, bg="#B8D5CD")
bottomFrame.pack(side=tk.BOTTOM, expand=True, fill="both")
buttonListFrame = tk.Frame(bottomFrame)
buttonListFrame.pack(side=tk.RIGHT, fill='x',)
createDataSetBtn = tk.Button(buttonListFrame,text="create Data set")
createDataSetBtn.pack(side=tk.TOP,  fill="both")
runTrainingBtn = tk.Button(buttonListFrame,text="run training set")
runTrainingBtn.pack(side=tk.TOP,  fill="both")
epochEntry = tk.Entry(buttonListFrame, textvariable=epochsVar)
epochEntry.pack(side=tk.TOP,  fill="both")
runTestSetBtn = tk.Button(buttonListFrame,text="run test set")
runTestSetBtn.pack(side=tk.TOP,   fill="both")
selectImgBtn = tk.Button(buttonListFrame,text="select image from file explorer", command=selectImage)
selectImgBtn.pack(side=tk.TOP,  fill="both")

# title
tk.Label(root, text="The fruits problem", bg="#2E856E", font=("Courier", 44)).pack(side=TOP,  fill="x")

# image 
imageAndClassificationFrame = tk.Frame(bottomFrame, bg="#B8D5CD")
imageAndClassificationFrame.pack(side=RIGHT,  fill="x", expand=True)
chart_type = FigureCanvasTkAgg(figure, imageAndClassificationFrame)
chart_type.get_tk_widget().pack(side=TOP, padx=50)
classificationLabel = tk.Label(imageAndClassificationFrame, text="apple")
classificationLabel.pack(side=BOTTOM, fill="x", padx=50)


# main loop
root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()