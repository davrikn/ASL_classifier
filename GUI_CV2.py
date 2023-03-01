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
import dropoutModel
import torch
import numpy.ma as ma
from predictor import predict


class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.i=0

        greeting = tkinter.Label(text="ASL Sign Language Detection!",font=("Arial", 25))
        greeting.pack()

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()

        #Display predictions
        self.my_output_text=""
        self.output_text = tkinter.Label(text=self.my_output_text)
        self.output_text.pack()

        # Button for settings
        self.btn_settings=tkinter.Button(window, text="Settings", width=10, command=self.openNewWindow)
        self.btn_settings.pack(anchor=tkinter.SW, expand=True)

        #Initializing our model
        self.model=dropoutModel.DropoutModel()

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        #window.update_idletasks()
        self.update()
        self.window.mainloop()
    
    def openNewWindow(self):
        # Toplevel object which will
        # be treated as a new window
        newWindow = Toplevel(self.window)
    
        # sets the title of the
        # Toplevel widget
        newWindow.title("New Window")
    
        # sets the geometry of toplevel
        a=str(self.vid.width)+"w"+str(self.vid.height)
        newWindow.geometry("600x600")
    
        # A Label widget to show in toplevel
        Label(newWindow,text ="Made for the course TMA4851. \n Thanks to our teacher.").pack()

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        #Converting to PIL
        #frame=PIL.Image.open("C:/Users/Øyvind/Desktop/ASL_sign_language/A1.jpg")       #testing on image A1 works.
        im = PIL.Image.fromarray(np.array(frame).astype("uint8"))

        #Cropping to 200x200
        width,height=im.size
        im = im.crop(((width-height)/2, 0, width-((width-height)/2), height))
        im = im.resize((200, 200))
        #Transposing and transforming into a tensor
        im=np.array(im)
        copied=copy.copy(im)
        copied2=copied.transpose(2,0,1)
        copied2=torch.tensor(copied2)

                                            #Ignore these lines.
        #Previously had and issue with data types, so I looped through and converted to float. Works without.
        #for i in range(len(frame3)):
        #    for j in range(len(frame3[i])):
        #        for k in range(len(frame3[i][j])):
        #            #print("k:",k)
        #            frame3[i][j][k]=float(frame3[i][j][k])
        #frameq=np.array(frame3,dtype=int)
        #frameq=np.array(frame3,dtype=float)


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #make prediction
        prediction=predict(self.model,"C:/Users/Øyvind/Desktop/ASL_sign_language/ASL_classifier/model_dropout_v3.pth",copied2,device=device)
        #display output from prediction
        best=torch.argmax(prediction)
        softmax_obj=torch.nn.Softmax(dim=1)
        softmax=softmax_obj(prediction)
        print("type of softmax is:",(softmax))
        tall=chr(best+65)

        self.output_text.config(text = str(softmax))
            
        if ret:
            frame=cv2.flip(frame,1)
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)

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
