from models.dropoutModel4 import DropoutModel
import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
from tkinter import *
import numpy as np
import torch
import numpy.ma as ma
from predictor import load_model
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import keyboard
from torchvision.transforms import transforms
from HMM.uniform_predict import uniform_predict, semi_uniform_predict
from utility.torch_queue import TorchQueue
import copy


class ASLPredictorApplication:
    
    def __init__(self, window, window_title, save_loc = None, cache_size=10, video_source=0, HMM=True) -> None:

        """ PyTorch """
        self.norm_transform = transforms.Normalize(
            (132.3501, 127.2977, 131.0638),
            (55.5031, 62.3274, 64.1869)
        )
        self.softmax = torch.nn.Softmax(dim=1)
        # Initializing our model
        self.model = DropoutModel()
        load_model(self.model, model_path="./models/saved/model_v4_2.pth")
        self.cache_size = cache_size
        self.distr_cache = TorchQueue(torch.ones((cache_size, 29))/29)
        self.last_distr_pred = torch.ones(29)/29

        """ Other """
        self.__last_time = 0
        self.index_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'del', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'nothing', 16: 'O', 17: 'P', 18: 'Q', 19: 'R', 20: 'S', 21: 'space', 22: 'T', 23: 'U', 24: 'V', 25: 'W', 26: 'X', 27: 'Y', 28: 'Z'}  
        self.frame = 0
        self.fig, self.ax = plt.subplots()
        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 50
        self.save_loc = save_loc
        self.HMM = HMM

        """ OpenCV """
        # Initializing a window
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.keyboard_on_off = False
        self.video_on=True
        self.graph_on=True
        self.last_predicted_letter=""
        self.second_last_predicted_letter=""
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
        self.btn_learn_mode=tkinter.Button(window, text="Toggle video-display", width=20, height=2, command=self.__toggleVideo)
        self.btn_toggle_graph=tkinter.Button(window, text="Toggle display graph", width=20, height=2, command=self.__toggleGraph)
        #self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=10, height=2, command=self.snapshot)
        self.btn_settings.pack(anchor=tkinter.SW, expand=True, in_=top, side=LEFT)
        self.btn_keyboard.pack(anchor=tkinter.SW, expand=True, in_=top, side=LEFT)
        self.btn_learn_mode.pack(anchor=tkinter.SW, expand=True, in_=top, side=LEFT)
        self.btn_toggle_graph.pack(anchor=tkinter.SW, expand=True, in_=top, side=LEFT)
        top.pack(side=TOP)
        # Display predictions
        self.my_output_text = ""
        self.output_text = tkinter.Label(text=self.my_output_text,font=("Arial", 20))
        self.output_text.pack()
        # Second canvas for plotting
        self.plt_canvas = FigureCanvasTkAgg(self.fig, self.window)
        self.plt_canvas.draw()
        self.plt_canvas.get_tk_widget().pack(fill=tkinter.BOTH, expand=True)

        # Initializing main loop
        self.update()
        self.window.mainloop()
    

    def __openNewWindow(self):
        """
        Initializes a new window
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
        Label(newWindow,text ="Made for the course TMA4851. \n Thanks to our teacher. \n\n\n\n Made by:\nDavid\nMikkel\nOle\nRimba\nØyvind",font=("Arial", 25)).pack()


    def __keyboard(self):
        """
        Turns the input from webcam into keyboard outputs. Waits 2 seconds so that he user may click away before the keyboard starts writing.
        """
        if not(self.keyboard_on_off):
            self.keyboard_on_off=True
        else:
            self.keyboard_on_off=False
        time.sleep(2)

    def __toggleVideo(self):
        self.video_on=not(self.video_on)

    def __toggleGraph(self):
        self.graph_on=not(self.graph_on)

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


    def __predict(self, image: np.ndarray, reducer: float = 5.0, reduce_nothing: bool = False) -> None:
        """
        Given an image in the form of a numpy array, uses the class model
        to make a prediction with PyTorch and adds it to the cache.

        Args:
            image   (NumPy array): Source image
            reducer       (float): Scaler for raw output, making distribution flatter
            reduce_nothing (bool): Whether to make the prediction "nothing" less likely
        Returns:
            None
        """
        image = self.norm_transform(torch.tensor(image).float())
        prediction = self.model(image) / reducer
        if reduce_nothing: prediction[0, 15] /= 2

        self.distr_cache.insert(self.softmax(prediction)[0])
    

    def __predict_from_cache(self, func = None) -> torch.Tensor:
        """
        Makes a prediction based on the values in the cache.
        Args:
            func: Function to use for prediction.
                  If None, returns the last value
        """
        if func is None:
            return self.distr_cache[0]
        else:
            return func(self.distr_cache)
    

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
        predicted_prob = distr[best_ind]
        self.output_text.config(
            text="{pred} with probability {prob}. Fps: {fps}".format(
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
        if self.graph_on:
            # Plotting barplot every fifth frame:
            if self.frame % self.cache_size == self.cache_size-1:
                self.ax.cla()
                with torch.no_grad():
                    self.ax.bar(self.index_map.values(), distr)

            # Drawing barplot on window
            self.plt_canvas.draw()
            self.plt_canvas.get_tk_widget().pack(fill=tkinter.BOTH, expand=True)
        else:
            self.canvas.delete(self.plt_canvas)
    

    def __draw_image_on_update(self, has_image: bool, frame) -> None:
        """
        Draws the image at the end of the update function.
        Also handles keyboard functionality

        Args:
            has_image          (bool): Whether an image was returned
            frame                 (?): Raw output from OpenCV to be drawn
            predicted_letter (string)
        Returns:
            None
        """
        if has_image and self.video_on:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.displayed_image=self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
        if not(self.video_on):
            self.canvas.create_image(0, 0, image = None, anchor = tkinter.NW)
            self.canvas.delete(self.displayed_image)
                    
        self.window.after(self.delay, self.update)
        self.frame += 1
        

    def update(self) -> None:
        """
        This function is called every time openCV updates the frame.
        It is responsible for retrieving an image and printing a
        valid prediction.
        """
        time0 = time.time()
        if not 'distr' in locals(): distr = self.last_distr_pred

        # Image processing:
        has_image, frame, image = self.__process_image()

        # PyTorch/HMM prediction:
        with torch.no_grad():

            self.__predict(image)

            if self.HMM:
                distr = torch.tensor(semi_uniform_predict(
                    self.distr_cache[None].numpy(),
                    self.last_distr_pred.numpy()
                ))

                self.last_distr_pred = distr
            else:
                distr = self.distr_cache[self.cache_size-1]

            best_ind = torch.argmax(distr)
            predicted_letter = self.index_map[int(best_ind)]

            if self.keyboard_on_off and float(self.last_distr_pred[best_ind])>0.4:#Only writes a number if its probability is larger than 0.4
                if not(self.last_predicted_letter==predicted_letter): #Current letter is different than the previous written letter. If we want double letters, we can invent a repeat sign.
                    if (predicted_letter!="nothing") and (predicted_letter !="del") and(predicted_letter!="space"):
                        keyboard.write(predicted_letter)
                    if predicted_letter=="space":
                        keyboard.write(" ")
                if predicted_letter=="del" and float(self.last_distr_pred[best_ind])>0.5:#Want to be really sure that we want to delete.
                        keyboard.write('\b')
                self.second_last_predicted_letter=copy.copy(self.last_predicted_letter)
                self.last_predicted_letter=copy.copy(predicted_letter)


        # Setting output text on window:
        try:
            self.__set_output_text(best_ind, predicted_letter, distr, time0)
        except:
            pass

        # Plotting estimated distribution:
        self.__plot_distr(distr)

        # Drawing image:
        self.__draw_image_on_update(has_image, frame)

        # Saving image:
        if self.save_loc is not None and has_image:
            cv2.imwrite(self.save_loc + f"{lab}%d.jpg" % self.frame, frame)

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


lab = "E"
def main() -> None: 
    ASLPredictorApplication(tkinter.Tk(), "Tkinter and OpenCV", cache_size=5, HMM=True)


if __name__ == "__main__":
    main()