import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
#from PIL import Image
from tkinter import *

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.i=0

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()

        #greeting = tk.Label(text="Hello, Tkinter")
        greeting = tkinter.Label(text="Hello, Tkinter")
        greeting.pack()

        # Button that lets the user take a snapshot
        self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15

        self.update()
        self.window.mainloop()

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        #print("frame",frame)
        #print(" is :::",len(frame))
        #print("[] is:::",len(frame[0]))
        #print("ret",ret)

        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if self.i<30:
            self.snapshot()
            a=PIL.Image.open('frame-08-02-2023-15-38-51.jpg')          #remove #
            width,height=a.size                #remove #
            #print("width",width)
            #print("height",height)
            #im1 = im.crop((left, top, right, bottom))
            #80,480,560,0
            im1 = a.crop(((width-height)/2, 0, width-((width-height)/2), height))
            #im1 = a.crop((1, 1, 200,300))
            im1.save('image_400_lol.jpg')
            new_image = im1.resize((200, 200))
            new_image.save('image_400.jpg')
            self.i+=1
            

        if ret:
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
