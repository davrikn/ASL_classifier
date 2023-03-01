import tkinter
import cv2
import PIL.Image, PIL.ImageTk
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


def change_format(frame):#Redundant function to change format of image array. Use transpose instead...
    """Changes format from webcam: [[[r,g,b],[r,g,b]][[r,g,b]]]
    to correct format: [[[r,200,r],[r,200,r]], [[b,200,b],[b,200,b]], [[g,200,c],...,[c,200,c]]]"""
    r_verdier=[]
    g_verdier=[]
    b_verdier=[]
    r_verdier_temp=[]
    g_verdier_temp=[]
    b_verdier_temp=[]
    counter=0
    g_counter=0
    b_counter=0
    for i in range(len(frame)):
        for j in range(len(frame[i])):

            r_verdier_temp.append(frame[i][j][0])
            g_verdier_temp.append(frame[i][j][1])
            b_verdier_temp.append(frame[i][j][2])
            counter+=1
            if counter==200:#200
                r_verdier.append(r_verdier_temp)
                g_verdier.append(g_verdier_temp)
                b_verdier.append(b_verdier_temp)
                counter=0
                r_verdier_temp=[]
                g_verdier_temp=[]
                b_verdier_temp=[]

    final=[[],[],[]]
    for i in range(len(r_verdier)):#number of lists?
        final[0].append(r_verdier[i])
        final[1].append(g_verdier[i])
        final[2].append(b_verdier[i])

    return final

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.i=0

        #greeting = tk.Label(text="Hello, Tkinter")
        greeting = tkinter.Label(text="ASL Sign Language Detection!",font=("Arial", 25))
        greeting.pack()

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()

        # Button that lets the user take a snapshot
        #self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
        #self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)

        self.my_output_text="aa"
        self.output_text = tkinter.Label(text=self.my_output_text)
        self.output_text.pack()

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        #window.update_idletasks()
        #print("updated")
        self.model=dropoutModel.DropoutModel()

        self.update()
        self.window.mainloop()

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        #print("frame_type:",type(frame))
        if self.i<10:
            self.snapshot()
            #print("my frame is",frame)
            frame2=copy.copy(frame)
            frame2=frame2
            im = PIL.Image.fromarray(np.array(frame2).astype("uint8"))

            #a=PIL.Image.open('frame-08-02-2023-15-38-51.jpg')
            a=im
            print("TYPE:",type(a))
            width,height=a.size
            im1 = a.crop(((width-height)/2, 0, width-((width-height)/2), height))
            im1.save('image_400_lol.jpg')
            new_image = im1.resize((200, 200))
            new_image.save('image_400.jpg')
            self.i+=1

        frameA=np.array(frame)
        frameA=frameA.transpose(2,0,1)
        frame3=frameA.tolist()
        """
        for i in range(len(frame3)):
            for j in range(len(frame3[i])):
                for k in range(len(frame3[i][j])):
                    #print("k:",k)
                    frame3[i][j][k]=float(frame3[i][j][k])"""
        frameq=np.array(frame3,dtype=int)
        #frameq=np.array(frame3,dtype=float)
        frame_changed=torch.tensor(frameq)
        prediction=self.model.predict(frame_changed)               #remove #

        self.output_text.config(text = str(prediction))               #remove #
            
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

"""a=PIL.Image.open('frame-08-02-2023-14-07-18.jpg')
width,height=a.size
print("width",width)
print("height",height)
#im1 = im.crop((left, top, right, bottom))
#80,480,560,0
print("a",(width-height)/2)
print("b",width-((width-height)/2))
im1 = a.crop(((width-height)/2, 0, width-((width-height)/2), height))
#im1 = a.crop((1, 1, 200,300))
im1.save('image_400_lol.jpg')
new_image = im1.resize((200, 200))
new_image.save('image_400.jpg')"""
