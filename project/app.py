import utils
import tkinter as tk
from perceptron import Perceptron
from data_processing import DataProcessor
from tkinter.constants import BOTTOM, CENTER, DISABLED, INSERT, LEFT, RIGHT, TOP, TRUE
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
from threading import Thread

root = tk.Tk()
image = utils.get_image_3d("./data/Apple/apple_0.jpg")
figure = utils.show_image(image)
isInitialized = False
isDpInitialized = False

# window related 
root.geometry("800x800")
root.resizable(False, False)

# state variables
labelText = tk.StringVar(value="apple")
epochsVar = tk.StringVar(value="Epochs")
learningRateVar = tk.StringVar(value="learning rate")

def redirector(inputStr):
    global textbox
    textbox.config(state="normal",autoseparators=True)
    textbox.insert(INSERT, inputStr)
    textbox.config(state="disabled",autoseparators=True)
    textbox.see(tk.END)

# window functions
def on_closing():
    root.quit()

def onProgressIsDone(thread):
    thread.join()
    print("DONE")

def create_dataset():
    filename = askdirectory()
    if os.path.isdir(filename):
        thread = Thread(target = DataProcessor.createDataset, args=[filename])
        thread.start()
        print("creating dataset...")
        thread1 = Thread(target=onProgressIsDone, args=[thread])
        thread1.start()
     
def updateImage():
    global chart_type
    chart_type.draw_idle()
 
def train_dataset():
    global epochsVar, learningRateVar,isDpInitialized
    if not isDpInitialized:
        status = DataProcessor.initializeDataProcessorClass()
        if status == 0:
            print("data processor initialized")
            isDpInitialized = True
        else:
            print("could not locate dataset file")
            return
    try:
        epoch = int(epochsVar.get())
        learningrate = float(learningRateVar.get())
        thread = Thread(target =  Perceptron.trainModel, args=[epoch, learningrate])
        thread.start()
        print("started training...")
    except Exception as E:
        print("please use numbers for epochs(int) and learning rate(float)")

def test_dataset():
    global isDpInitialized
    if not isDpInitialized:
        status = DataProcessor.initializeDataProcessorClass()
        if status == 0:
            print("data processor initialized")
            isDpInitialized = True
        else:
            print("could not locate dataset file")
            return
    thread = Thread(target =  Perceptron.testModel)
    thread.start()
    print("running tests...")

def selectImage():
    global isInitialized, labelText, chart_type, figure
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
        labelText.set(predicition[-1])
        figure = utils.show_image(image3d)
        updateImage()


# buttons
bottomFrame = tk.Frame(root, bg="#B8D5CD")
bottomFrame.pack(side=tk.BOTTOM, expand=True, fill="both")
buttonListFrame = tk.Frame(bottomFrame)
buttonListFrame.pack(side=tk.RIGHT, fill='x',)
createDataSetBtn = tk.Button(buttonListFrame,text="create Data set", command=create_dataset)
createDataSetBtn.pack(side=tk.TOP,  fill="both")
epochEntry = tk.Entry(buttonListFrame, textvariable=epochsVar)
epochEntry.pack(side=tk.TOP,  fill="both")
learningRateEntry = tk.Entry(buttonListFrame, textvariable=learningRateVar)
learningRateEntry.pack(side=tk.TOP,  fill="both")
runTrainingBtn = tk.Button(buttonListFrame,text="run training set", command=train_dataset)
runTrainingBtn.pack(side=tk.TOP,  fill="both")
runTestSetBtn = tk.Button(buttonListFrame,text="run test set", command=test_dataset)
runTestSetBtn.pack(side=tk.TOP,   fill="both")
selectImgBtn = tk.Button(buttonListFrame,text="select image from file explorer", command=selectImage)
selectImgBtn.pack(side=tk.TOP,  fill="both")

# title
tk.Label(root, text="The fruits problem", bg="#2E856E", font=("Courier", 20)).pack(side=TOP,  fill="x")

# image 
imageAndClassificationFrame = tk.Frame(bottomFrame, bg="#B8D5CD")
imageAndClassificationFrame.pack(side=RIGHT,  fill="x", expand=True)
chart_type = FigureCanvasTkAgg(figure, imageAndClassificationFrame)
chart_type.get_tk_widget().pack(side=TOP, padx=50,)
classificationLabel = tk.Label(imageAndClassificationFrame, textvariable=labelText, fg="black",width=100)
classificationLabel.pack(side=TOP, fill="x", padx=50)
textbox=tk.Text(imageAndClassificationFrame,  height=10, state="disabled")
textbox.pack(side=BOTTOM, padx=50, pady=10)


# main loop
sys.stdout.write = redirector
root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()