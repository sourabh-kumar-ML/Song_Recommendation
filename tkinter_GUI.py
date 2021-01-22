from PIL import Image, ImageTk
import tkinter as tk
import argparse
import datetime
import cv2
import os
import pickle
import pandas as pd
import numpy as np
import webbrowser
import warnings
warnings.filterwarnings("ignore")

class EmotionDetection:
    def __init__(self, ):
        self.emotion_labels = ['sad','neutral', 'happy', 'angry']
        with open('cnn_model.pkl','rb') as f:
            self.model = pickle.load(f)
            
        self.vs = cv2.VideoCapture(0) 
        self.current_image = None  
        self.faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        self.root = tk.Tk()  
        self.root.title("Music Recommendation System")  
        
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)

        self.panel = tk.Label(self.root)  
        self.panel.pack(padx=10, pady=10)

        
        btn = tk.Button(self.root, text="Predict Music!", command=self.classify)
        btn.pack(fill="both", expand=True, padx=10, pady=10)
        
        
        self.video_loop()

    def video_loop(self):
        
        img, frame = self.vs.read()
        image = frame.copy()
        if img:  
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY,1)
            faces = self.faceCascade.detectMultiScale(img_gray,
                                                 scaleFactor=1.1,
                                                 minNeighbors=5,
                                                 minSize=(30,30),
                                                 flags = cv2.CASCADE_SCALE_IMAGE)

            emotions = []
            for (x,y,w,h) in faces:
                face_image_gray = img_gray[y:y+h,x:x+w]
                cv2.rectangle(frame,(x, y), (x+w, y+h), (0, 255, 0), 2)
                
                sad,neutral, happy, angry = self.predict_emotion(face_image_gray/255)
                self.emt = self.emotion_labels[np.argmax([sad,neutral, happy, angry])]

                font = cv2.FONT_HERSHEY_SIMPLEX 
                org = (50, 50) 
                fontScale = 1
                color = (255, 0, 0) 
                thickness = 2
                cv2.putText(frame,self.emt,org, font,  
                   fontScale, color, thickness, cv2.LINE_AA)

            
            self.current_image = Image.fromarray(frame)  
            imgtk = ImageTk.PhotoImage(image=self.current_image)  
            self.panel.imgtk = imgtk  
            self.panel.config(image=imgtk)  
        self.root.after(30, self.video_loop)  


    def predict_emotion(self,face_image_gray): # a single cropped face
        resized_img = cv2.resize(face_image_gray, (48,48), interpolation = cv2.INTER_AREA)
        # cv2.imwrite(str(index)+'.png', resized_img)
        image = resized_img.reshape(1, 48, 48, 1)
        list_of_list = self.model.predict(image, batch_size=1)
        sad,neutral, happy, angry = [prob for lst in list_of_list for prob in lst]
        return [sad,neutral, happy, angry]

    def clear(self):
        self.rd1.destroy()
        self.rd2.destroy()
        self.rd3.destroy()
        self.rd4.destroy()
        self.rd5.destroy()
        self.button.destroy()
    def play(self):
        
        webbrowser.open("https://www.youtube.com/results?search_query={}".format(self.v.get().replace(" ","+")))
            
    def classify(self):
                #self.current_image.save(".\{}".format("abc.jpg"))
        
        if self.emt == "sad":
            file = pd.read_csv("Sad.csv")
        elif self.emt == "neutral":
            file = pd.read_csv("Calm.csv")
        elif self.emt == "happy":
            file = pd.read_csv("Happy.csv")
        elif self.emt == "angry":
            file = pd.read_csv("Energetic.csv")
        music = file.sample(5)
        music.reset_index(inplace=True)
        
        self.v = tk.StringVar(self.root,"1")
        self.rd1 = tk.Radiobutton(self.root,text=music.music_name[0],variable = self.v,value =music.music_name[0],command= self.play)
        self.rd1.pack(fill="both", expand=True, padx=10, pady=1)
        self.rd2 = tk.Radiobutton(self.root,text=music.music_name[1],variable = self.v,value = music.music_name[1],command= self.play)
        self.rd2.pack(fill="both", expand=True, padx=10, pady=1)
        self.rd3 = tk.Radiobutton(self.root,text=music.music_name[2],variable = self.v,value = music.music_name[2],command= self.play)
        self.rd3.pack(fill="both", expand=True, padx=10, pady=1)
        self.rd4 = tk.Radiobutton(self.root,text=music.music_name[3],variable = self.v,value = music.music_name[3],command= self.play)
        self.rd4.pack(fill="both", expand=True, padx=10, pady=1)
        self.rd5 = tk.Radiobutton(self.root,text=music.music_name[4],variable = self.v,value = music.music_name[4],command= self.play)
        self.rd5.pack(fill="both", expand=True, padx=10, pady=1)

        self.button = tk.Button(self.root,text="Clear!",command=self.clear)
        self.button.pack(fill="both",expand=True, padx=10, pady=1)

    def destructor(self):
        """ Destroy the root object and release all resources """
        print("[INFO] closing...")
        self.root.destroy()
        self.vs.release()  # release web camera
        cv2.destroyAllWindows()  # it is not mandatory in this application


# start the app
print("[INFO] starting...")
pba = EmotionDetection()
pba.root.mainloop()
