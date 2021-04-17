import utils
import tkinter as tk
from perceptron import Perceptron
from tkinter.constants import BOTTOM, CENTER, LEFT, RIGHT, TOP, TRUE
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

image = utils.get_image_3d("./data/Apple/apple_0.jpg")
figure = utils.show_image(image)


# window related 
root = tk.Tk()
root.geometry("800x600")
root.resizable(False, False)

# functions


def on_closing():
    root.quit()


# buttons
bottomFrame = tk.Frame(root, bg="#B8D5CD")
bottomFrame.pack(side=tk.BOTTOM, expand=True, fill="both")
buttonListFrame = tk.Frame(bottomFrame)
buttonListFrame.pack(side=tk.RIGHT, fill='x',)
createDataSetBtn = tk.Button(buttonListFrame,text="create Data set")
createDataSetBtn.pack(side=tk.TOP,  fill="both")
runTrainingBtn = tk.Button(buttonListFrame,text="run training set")
runTrainingBtn.pack(side=tk.TOP,  fill="both")
runTestSetBtn = tk.Button(buttonListFrame,text="run test set")
runTestSetBtn.pack(side=tk.TOP,   fill="both")
selectImgBtn = tk.Button(buttonListFrame,text="select image from file explorer")
selectImgBtn.pack(side=tk.TOP,  fill="both")

# title
test = tk.Label(root, text="The fruits problem", bg="#2E856E", font=("Courier", 44))
test.pack(side=TOP,  fill="x")

# image 
imageAndClassificationFrame = tk.Frame(bottomFrame, bg="#B8D5CD")
imageAndClassificationFrame.pack(side=RIGHT,  fill="x", expand=True)
chart_type = FigureCanvasTkAgg(figure, imageAndClassificationFrame)
chart_type.get_tk_widget().pack(side=TOP, padx=50)
test = tk.Label(imageAndClassificationFrame, text="apple")
test.pack(side=BOTTOM, fill="x", padx=50)


# main loop
root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()