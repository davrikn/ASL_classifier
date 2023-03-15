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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
root = tk.Tk()
fig, ax = plt.subplots()
 
canvas = FigureCanvasTkAgg(fig, root)
canvas.draw()
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
 
root.mainloop()
"""

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.i=0
        self.aaaaaa=0
        self.last_times=[]

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
        #self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=10, height=2, command=self.snapshot)
        self.btn_settings.pack(anchor=tkinter.SW, expand=True, in_=top, side=LEFT)
        self.btn_keyboard.pack(anchor=tkinter.SW, expand=True, in_=top, side=LEFT)
        top.pack(side=TOP)

        #Initializing our model
        self.model=dropoutModel.DropoutModel()
        #print("conv1",self.model.conv1)
        #print("size",self.model.size())
        #top = Frame(self.window)
        #top.pack(side=TOP)
        #bottom = Frame(self.window)
        #b = Button(self.window, text="Enter", width=10, height=2)
        #c = Button(self.window, text="Clear", width=10, height=2)
        #self.btn_settings.pack(in_=top, side=LEFT)
        #self.btn_keyboard.pack(in_=top, side=LEFT)
        #bottom = Frame(self.window)
        #top.pack(side=TOP)
        #bottom.pack(side=BOTTOM, fill=BOTH, expand=True)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
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
        # create the main sections of the layout, 


        # and lay them out
        #top = Frame(self.window)
        #bottom = Frame(self.window)
        #top.pack(side=TOP)
        #bottom.pack(side=BOTTOM, fill=BOTH, expand=True)

        # create the widgets for the top part of the GUI,
        # and lay them out
        top = Frame(self.window)
        b = Button(self.window, text="Enter", width=10, height=2)
        c = Button(self.window, text="Clear", width=10, height=2)
        b.pack(in_=top, side=LEFT)
        c.pack(in_=top, side=RIGHT)
    
        # A Label widget to show in toplevel
        Label(newWindow,text ="Made for the course TMA4851. \n Thanks to our teacher.").pack()

    def keyboard(self):
        self.keyboard=True
        print("Activated keyboard")

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def update(self):
        time_consumption=[]
        if self.aaaaaa==1:
            self.time_list=time_consumption
            print(self.time_list)
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        now = time.time()
        time_consumption.append(time.time())
        if len(self.last_times) == 20:
            self.last_times.append(round(now-self.last_time,3))
            self.last_times.pop(0)
        else:
            try:
                self.last_times.append(round(now-self.last_time,3))
            except:
                useless_variable=1
        time_consumption.append(time.time())
        #Converting to PIL
        #frame=PIL.Image.open("C:/Users/Øyvind/Desktop/ASL_sign_language/ASL_classifier/data/asl_alphabet_train/A/A2.jpg")       #testing on image A2 works.
        im = PIL.Image.fromarray(np.array(frame).astype("uint8"))
        time_consumption.append(time.time())
        #Cropping to 200x200
        width,height=im.size
        im = im.crop(((width-height)/2, 0, width-((width-height)/2), height))
        time_consumption.append(time.time())
        im = im.resize((200, 200))
        time_consumption.append(time.time())
        #im.save("aaaaaaaaaa.jpg")
        #Transposing and transforming into a tensor
        im=np.array(im)
        time_consumption.append(time.time())
        copied=copy.copy(im)
        time_consumption.append(time.time())
        copied2=copied.transpose(2,0,1)
        time_consumption.append(time.time())
        copied2=torch.tensor(copied2)
        time_consumption.append(time.time())

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
        #Make prediction
        #prediction=predict(self.model,"C:/Users/Øyvind/Desktop/ASL_sign_language/ASL_classifier/model_dropout_v3.pth",copied2,device=device)
        prediction=predict(self.model,copied2)
        time_consumption.append(time.time())
        #Display output from prediction
        best=torch.argmax(prediction)
        time_consumption.append(time.time())
        softmax_obj=torch.nn.Softmax(dim=1)
        time_consumption.append(time.time())
        softmax=softmax_obj(prediction)
        time_consumption.append(time.time())
        #print("type of softmax is:",(softmax))
        tall=chr(best+65)
        ting=softmax[0][best]
        #print("ting",ting)

        #self.output_text.config(text = str(softmax))
        try:
            #print(len(self.last_times))
            self.output_text.config(text = str(tall)+" Med sannsynlighet: "+str(ting.item())+"\n Time since last frame:"+str(now-self.last_time)+"\n Last 10 frame-times:"+str(self.last_times))
        except:
            useless_variable=1
        time_consumption.append(time.time())
        self.ax.cla()
        self.ax.plot(self.last_times)
 
        self.canvas2.draw()
        self.canvas2.get_tk_widget().pack(fill=tkinter.BOTH, expand=True)
        time_consumption.append(time.time())
        if ret:
            frame=cv2.flip(frame,1)
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
        time_consumption.append(time.time())
        self.aaaaaa+=1

        #Prints time consumption of various steps in code
        output_thing=[]
        for i in range(len(time_consumption)):
            output_thing.append(time_consumption[i]-time_consumption[0])
        print("time_consumption",output_thing)
        #try:
        #    self.last_times.append(round(now-self.last_time,3))
        #except:
        #    useless_variable=1
        self.last_time=now
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
