import cv2
import numpy as np
import opencv_jupyter_ui as jcv2
from feat import Detector
from IPython.display import Image
from feat.utils import FEAT_EMOTION_COLUMNS
import time as t
from time import sleep
from furhat_remote_api import FurhatRemoteAPI
from numpy.random import randint

import time
import sys
import trace
from pathlib import Path
import multiprocessing
import threading
import os
from torch import nn
import torch



# DETECTOR SETUP -------------------------------------------------------------

detector = Detector(device="cuda")

class MLP(nn.Module):
    def __init__(self, features_in=2, features_out=3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(features_in, 10),
            nn.ReLU(),
            nn.Linear(10, features_out),
        )

    def forward(self, input):
        return self.net(input)
    
path = Path(os.getcwd()).parent
DIR_PATH = str(Path(__file__).parent.parent.absolute()) + r"\\"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load(Path(DIR_PATH + 'models\\best_model_12.pt') ).to(device)

expression = {"anger": 6, "disgust": 5 , "fear": 4, "happiness": 1, "neutral": 0, "sadness": 2, "surprise": 3}
def return_emo(loc):
    for key,_ in expression.items():
        if expression[key] == loc:
            return key
        
EMO_LIST= []

def capture_and_return_emos():
    cap = cv2.VideoCapture(1)
    print("Capture on")
    flag=True
    while flag: #or not event.is_set()
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to capture the frame.")
            break

        detected_faces = detector.detect_faces(frame)
        detected_landmarks = detector.detect_landmarks(frame, detected_faces)
        detected_aus = detector.detect_aus(frame, detected_landmarks)

        for faces,au_units in zip(detected_faces,detected_aus): #access only one frame
            for i in range(len(faces)): #access all faces detected in the frame
                au_arr=model(torch.tensor(au_units[i]).to(device)).cpu()
                max_loc=np.argmax(au_arr.softmax(dim=0).numpy())
                emotion=return_emo(max_loc)
                EMO_LIST.append(emotion)
                x, y, w, h, p = faces[i]
                # Drawing a rectangle around the detected face
                cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), (0,0 , 255), 2)

                # Displaying the emotion label on top of the rectangle
                cv2.putText(frame, emotion, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Displaying the frame with detected faces and emotions
                jcv2.imshow("Emotion Detection", frame)

                # Press Esc to exit the program
                key = jcv2.waitKey(1) or 0xFF

                # if event.is_set():
                #     flag=False
                #     break
                if key == 27:
                    flag=False
                    break
    cap.release()
    jcv2.destroyAllWindows()
    print("Capture off")

def main():
    process1 = multiprocessing.Process(target=capture_and_return_emos())
    process1.start()
    time.sleep(5)
    process1.terminate()
    process1.kill()

if __name__=="__main__":
    main()