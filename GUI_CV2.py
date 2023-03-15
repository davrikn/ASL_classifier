from models.dropoutModel import DropoutModel
from models.GPTModel import CNNGPT
import tkinter
import cv2
import PIL.Image, PIL.ImageTk
from PIL import ImageOps
import time
#from PIL import Image
from tkinter import *
import random
import numpy as np
from matplotlib import cm
import copy
import torch
import numpy.ma as ma
from predictor import predict, load_model
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import keyboard
from image_datasets import imagepathloader
from torchvision.transforms import transforms


class App:
    
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.i=0
        self.total_imgs_num=0
        self.last_time = 0
        self.keyboard_on_off=False
        self.index_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'del', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'nothing', 16: 'O', 17: 'P', 18: 'Q', 19: 'R', 20: 'S', 21: 'space', 22: 'T', 23: 'U', 24: 'V', 25: 'W', 26: 'X', 27: 'Y', 28: 'Z'}  
        self.norm_transform = transforms.Normalize(
            (132.3501, 127.2977, 131.0638),
            (55.5031, 62.3274, 64.1869)
        )
        self.softmax = torch.nn.Softmax(dim=1)



        greeting = tkinter.Label(text="ASL Sign Language Detection!",font=("Arial", 25))
        greeting.pack()

        # Open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()

        #Display predictions
        self.my_output_text=""
        self.output_text = tkinter.Label(text=self.my_output_text)
        self.output_text.pack()

        # Buttons for settings and keyboard
        top = Frame(self.window)
        top.pack(side=TOP)
        self.btn_settings=tkinter.Button(window, text="Settings", width=10, height=2, command=self.openNewWindow)
        self.btn_keyboard=tkinter.Button(window, text="Keyboard", width=10, height=2, command=self.keyboard)
        self.btn_learn_mode=tkinter.Button(window, text="Learn ASL!", width=10, height=2)#Add command
        #self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=10, height=2, command=self.snapshot)
        self.btn_settings.pack(anchor=tkinter.SW, expand=True, in_=top, side=LEFT)
        self.btn_keyboard.pack(anchor=tkinter.SW, expand=True, in_=top, side=LEFT)
        self.btn_learn_mode.pack(anchor=tkinter.SW, expand=True, in_=top, side=LEFT)
        top.pack(side=TOP)

        #Initializing our model
        self.model = DropoutModel()
        load_model(self.model, model_path="./models/saved/model_1.pth")


        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 50
        #window.update_idletasks()


        self.fig, self.ax = plt.subplots()
        self.ax.plot(0,0)

        self.canvas2 = FigureCanvasTkAgg(self.fig, self.window)
        self.canvas2.draw()
        self.canvas2.get_tk_widget().pack(fill=tkinter.BOTH, expand=True)

        self.update()
        self.window.mainloop()
    

    def openNewWindow(self):
        # Toplevel object which will be treated as a new window
        newWindow = Toplevel(self.window)
    
        # sets the title of the Toplevel widget
        newWindow.title("New Window")
    
        # sets the geometry of toplevel
        a=str(self.vid.width)+"w"+str(self.vid.height)
        newWindow.geometry("600x600")

        top = Frame(self.window)
        b = Button(self.window, text="Enter", width=10, height=2)
        c = Button(self.window, text="Clear", width=10, height=2)
        b.pack(in_=top, side=LEFT)
        c.pack(in_=top, side=RIGHT)
    
        # A Label widget to show in toplevel
        Label(newWindow,text ="Made for the course TMA4851. \n Thanks to our teacher.").pack()

    def keyboard(self):

        """
        Turns the input from webcam into keyboard outputs.
        """
        if not(self.keyboard_on_off):
            self.keyboard_on_off=True
            print("Activated keyboard")
        else:
            self.keyboard_on_off=False
            print("Deactivated keyboard")

    def snapshot(self):
        """
        Take a screenshot
        """
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        time0 = time.time()
        frame=cv2.flip(frame,1)              
        #Converting to PIL
        #frame=PIL.Image.open("C:/Users/Øyvind/Desktop/ASL_sign_language/ASL_classifier/data/asl_alphabet_train/A/A2.jpg")       #testing on image A2 works.
        im = PIL.Image.fromarray(np.array(frame).astype("uint8"))
        
        #Cropping to 192x192
        width,height=im.size
        im = im.crop(((width-height)/2, 0, width-((width-height)/2), height))
        im = im.resize((192, 192))
        
        #im.save("Real_image"+str(self.total_imgs_num)+".jpg")              #if we want to create more training data by using images from webcam
        
        #Transposing and transforming into a tensor
        im=np.array(im)
        im=im.transpose(2,0,1)
        im=self.norm_transform(torch.tensor(im).float())

        #Make prediction
        prediction=predict(self.model, im)[:-4]#abcd,del,...,nothing,...,space

        #Display output from prediction
        best=torch.argmax(prediction)
        softmax_obj=torch.nn.Softmax(dim=1)
        softmax=softmax_obj(prediction/25)

        predicted_prob = softmax[0][best]
        predicted_letter = self.index_map[int(best)]


        self.output_text.config(text=f"{predicted_letter} with probability {np.round(predicted_prob.item(), 3)}. fps: {np.round(1/(time0 - self.last_time))}")

        self.ax.cla()
        print(softmax)
        self.ax.bar(self.index_map.values(), softmax[0])
        #If we want to draw a graph...
        self.canvas2.draw()
        self.canvas2.get_tk_widget().pack(fill=tkinter.BOTH, expand=True)

        if ret:
            #frame=cv2.flip(frame,1)
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
        
        if self.keyboard_on_off:
            keyboard.write(predicted_letter)
            
        self.last_time = time0
        self.total_imgs_num+=1
        self.window.after(self.delay, self.update)

        


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Create a window and pass it to the Application object
App(tkinter.Tk(), "Tkinter and OpenCV")
