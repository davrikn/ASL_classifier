from models.dropoutModel3 import DropoutModel
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


class ALSPredictorApplication:
    
    def __init__(self, window, window_title, video_source=0) -> None:

        """ PyTorch """
        self.norm_transform = transforms.Normalize(
            (132.3501, 127.2977, 131.0638),
            (55.5031, 62.3274, 64.1869)
        )
        self.softmax = torch.nn.Softmax(dim=1)
        # Initializing our model
        self.model = DropoutModel()
        load_model(self.model, model_path="./models/saved/model_v3_1.pth")

        """ Other """
        self.__last_time = 0
        self.index_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'del', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'nothing', 16: 'O', 17: 'P', 18: 'Q', 19: 'R', 20: 'S', 21: 'space', 22: 'T', 23: 'U', 24: 'V', 25: 'W', 26: 'X', 27: 'Y', 28: 'Z'}  
        self.frame = 0
        self.fig, self.ax = plt.subplots()
        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 50
        self.last_five_images=[]

        """ OpenCV """
        # Initializing a window
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.keyboard_on_off = False
        self.video_on=True
        # Display predictions
        self.my_output_text = ""
        self.output_text = tkinter.Label(text=self.my_output_text)
        self.output_text.pack()
        # Open video source (by default this will try to open the computer webcam)
        self.vid = VideoCapture(self.video_source)
        greeting = tkinter.Label(text="ASL Sign Language Detection", font=("Arial", 25))
        greeting.pack()
        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()
        # Buttons for settings and keyboard
        top = Frame(self.window)
        top.pack(side=TOP)
        self.btn_settings=tkinter.Button(window, text="Credits", width=10, height=2, command=self.__openNewWindow)
        self.btn_keyboard=tkinter.Button(window, text="Keyboard", width=10, height=2, command=self.__keyboard)
        self.btn_learn_mode=tkinter.Button(window, text="Learn ASL!", width=10, height=2)#Add command
        self.btn_learn_mode=tkinter.Button(window, text="Toggle show video", width=10, height=2, command=self.__toggleVideo)
        #self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=10, height=2, command=self.snapshot)
        self.btn_settings.pack(anchor=tkinter.SW, expand=True, in_=top, side=LEFT)
        self.btn_keyboard.pack(anchor=tkinter.SW, expand=True, in_=top, side=LEFT)
        self.btn_learn_mode.pack(anchor=tkinter.SW, expand=True, in_=top, side=LEFT)
        top.pack(side=TOP)
        # Second canvas for plotting
        self.plt_canvas = FigureCanvasTkAgg(self.fig, self.window)
        self.plt_canvas.draw()
        self.plt_canvas.get_tk_widget().pack(fill=tkinter.BOTH, expand=True)

        # Initializing main loop
        self.update()
        self.window.mainloop()
    

    def __openNewWindow(self):
        """
        Opens the settings window
        """
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
        Label(newWindow,text ="Made for the course TMA4851. \n Thanks to our teacher. \n\n\n\n\n\n Made by:David\nMikkel\nOle\nRimba\nØyvind",font=("Arial", 25)).pack()


    def __keyboard(self):
        """
        Turns the input from webcam into keyboard outputs.
        """
        if not(self.keyboard_on_off):
            self.keyboard_on_off=True
            print("Activated keyboard")
        else:
            self.keyboard_on_off=False
            print("Deactivated keyboard")

    def __toggleVideo(self):
        self.video_on=not(self.video_on)

    def snapshot(self):
        """
        Take a screenshot
        """
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


    def __process_image(self) -> tuple:
        """
        Gets an image from OpenCV feed and returns it as a 3D
        numpy array.
        
        Args:
            None
        Returns:
            has_image: boolean of whether an image was returned
            frame:     Raw output from OpenCV
            image:     3D numpy array of image
        """
        # Retrieving image
        has_image, frame = self.vid.get_frame()
        frame = cv2.flip(frame, 1)              
        image = PIL.Image.fromarray(np.array(frame).astype("uint8"))
        #image.save("webcam_images/A/A"+str(self.frame)+".jpg")
        
        # Cropping and resizing
        width, height = image.size
        image = image.crop(((width-height)/2, 0, width-((width-height)/2), height))
        image = image.resize((192, 192))
                
        # Transposing and transforming into a tensor
        image = np.array(image)
        image = image.transpose(2,0,1)

        return has_image, frame, image


    def __predict(self, image: np.ndarray, reducer: float = 5.0, reduce_nothing: bool = False) -> tuple:
        """
        Given an image in the form of a numpy array, uses the class model
        to make a prediction with PyTorch.

        Args:
            image   (NumPy array): Source image
            reducer       (float): Scaler for raw output, making distribution flatter
            reduce_nothing (bool): Whether to make the prediction "nothing" less likely
        Returns:
            best_ind:        The index of the most likely prediction
            distr:           Estimated probability distribution
            predicted_letter
        """
        
        image = self.norm_transform(torch.tensor(image).float())
        prediction = self.model(image) / reducer
        if reduce_nothing: prediction[0, 15] /= 2


        distr = self.softmax(prediction)
        return distr
    

    def __set_output_text(self, best_ind: bool, predicted_letter: str, distr: np.ndarray, time0: float) -> None:
        """
        Sets the output text in the window based on this frames predictions.

        Args:
            best_ind            (int): The index of the most likely prediction
            distr       (NumPy array): Estimated probability distribution
            predicted_letter (string): String of actual predicted letter
            time0             (float): Time at start of frame
        Returns:
            None
        """
        predicted_prob = distr[0][best_ind]
        self.output_text.config(
            text="{pred} with probability {prob}. fps: {fps}".format(
                pred = predicted_letter,
                prob = np.round(predicted_prob.item(), 3),
                fps  = np.round(1/(time0 - self.__last_time))
            )
        )

    
    def __plot_distr(self, distr) -> None:
        """
        Plots barplot of the distribution every fifth frame.

        Args:
            distr (NumPy array): Distribution to plot
        Returns:
            None
        """
        # Plotting barplot every fifth frame:
        if self.frame % 5 == 0:
            self.ax.cla()
            with torch.no_grad():
                self.ax.bar(self.index_map.values(), distr[0])

        # Drawing barplot on window
        self.plt_canvas.draw()
        self.plt_canvas.get_tk_widget().pack(fill=tkinter.BOTH, expand=True)
    

    def __draw_image_on_update(self, has_image: bool, frame) -> None:
        """
        Draws the image at the end of the update function.

        Args:
            has_image          (bool): Whether an image was returned
            frame                 (?): Raw output from OpenCV to be drawn
        Returns:
            None
        """
        if has_image and self.video_on:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
                    
        self.window.after(self.delay, self.update)
        self.frame += 1
        

    def update(self) -> None:
        """
        This function is called every time openCV updates the frame.
        It is responsible for retrieving an image and printing a
        valid prediction.
        """
        time0 = time.time()      

        # Get Image from webcam and do image processing:
        has_image, frame, image = self.__process_image()

        if(len(self.last_five_images)==5):
            self.last_five_images.pop(0)
            self.last_five_images.append(image)
        elif (len(self.last_five_images)<5):
            self.last_five_images.append(image)
        else:
            print("ERROR!")

        # PyTorch prediction:
        if(self.frame % 5 == 0):
            #Make prediction on batch
            predictions=[]
            for i in self.last_five_images:
                distr = self.__predict(image)
                predictions.append(distr)
            batch_prediction=0
            for i in predictions:
                batch_prediction+=i
            batch_prediction/=5
            
            best_ind = torch.argmax(batch_prediction)
            predicted_letter = self.index_map[int(best_ind)]

            if self.keyboard_on_off:
                keyboard.write(predicted_letter)


        # Setting output text on window:
        try:
            self.__set_output_text(best_ind, predicted_letter, distr, time0)
        except:
            pass
        # Plotting estimated distribution:
        
        try:
            self.__plot_distr(distr)
        except:
            pass

        # Drawing image:
        self.__draw_image_on_update(has_image, frame)

        self.__last_time = time0


class VideoCapture:
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


def main() -> None:
    ALSPredictorApplication(tkinter.Tk(), "Tkinter and OpenCV")


if __name__ == "__main__":
    main()