import cv2
import numpy as np
from feat import Detector
from feat.utils import FEAT_EMOTION_COLUMNS
import time
from numpy.random import randint
import time
import torch
import multiprocessing

from furhat_remote_api import FurhatRemoteAPI
import time
import pygame
from furhat.excercises import visualizationExercise
from furhat.furhatConfig import furhatConfig
from furhat.furhatConfig import LOOK_DOWN, LOOK_BACK
from utils import best_emo_model, return_emo, visualizationResponse
# Create an instance of the FurhatRemoteAPI class, providing the address of the robot or the SDK running the virtual robot
furhat = FurhatRemoteAPI("localhost")
furhat.set_led(red=100, green=50, blue=50)

# ----- Initializing robot's characteristics -----
furhatConfig(furhat)

# ----- Initializing detector -----

detector = Detector(device="cuda")

# ----- Initializing model -----
EMO_LIST = []
device = "cuda" if torch.cuda.is_available() else "cpu"
model = best_emo_model('models\\best_model_12.pt').to(device)

def return_detected_emo():
    max=0
    for i in np.unique(EMO_LIST):
        print(i, np.char.count(EMO_LIST, i).sum())
        if np.char.count(EMO_LIST, i).sum() > max:
            max = np.char.count(EMO_LIST, i).sum()
            emo = i
    {"anger": 6, "disgust": 5 , "fear": 4, "happiness": 1, "neutral": 0, "sadness": 2, "surprise": 3}
    if emo == 'anger' or emo == 'disgust':
        return 'angry'
    elif emo== 'happiness' or emo == 'surprise':
        return 'happy'
    elif emo == 'fear' or emo == 'sadness':
        return 'sad'
    elif emo == 'neutral':
        return emo
    else:
        return 'calm'

def capture_and_return_emos(detector, model, device, t):
    fl = False
    start= time.time()
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
                stop = time.time()

                if stop - start >= t:
                    print(fl)
                    fl = True
                    flag=False
                    break                                   

    cap.release()
    print("Capture off")

# ----- Introduction dialogues & excercises  -----
def introduction():

    # ----- Set initial color LED -----
    furhat.set_led(red=200, green=200, blue=200)
    
    # ----- Start playing the background music -----
    pygame.mixer.init()
    pygame.mixer.music.load("furhat/backgroundWaves.mp3")
    pygame.mixer.music.set_volume(0.3)
    pygame.mixer.music.play()

    # ----- Starting Gesture  -----
    furhat.gesture(name="CloseEyes")
    furhat.gesture(body=LOOK_DOWN(speed=1), blocking=True)
    time.sleep(0.5)
    furhat.gesture(body=LOOK_BACK(speed=1), blocking=True)
    furhat.gesture(name="OpenEyes")

    # ----- Starting Dialogue  -----
    furhat.say(text="Welcome to today's meditation session.")
    time.sleep(2)
    furhat.say(text="My name is Mary, and I will be your guide today. Now let us begin the 1 minute session")
    time.sleep(2)

    # ----- Starting Visualization Exercise  -----

    # Furhat conducts the 'exercise' and does tries to detect the user's emotion simultaneously  
    process1 = multiprocessing.Process(target=capture_and_return_emos(50))
    process2 = multiprocessing.Process(target=visualizationExercise(furhat))

    process1.start()
    process2.start()

    process1.join()
    process2.join()

    response = visualizationResponse(return_detected_emo())

    if response=='maintain':
        furhat.say(text = f"I understand you're feeling {return_detected_emo()}. This is ideal, embrace visualization, and let it guide you towards a maintaining this calm state.")
    else:
        furhat.say(text = f"I understand you're feeling {return_detected_emo()}. Embrace visualization, and let it guide you towards a {response} state.")

# ----- Conclusion Exercise  -----
def conclusion():
    furhat.say(text="Now, slowly bring your awareness back to the present moment.")
    time.sleep(3)
    furhat.say(text="When you are ready, open your eyes.")
    time.sleep(3)

    # Custom response depending on the total emotions in the class

    furhat.say(text="Thank you for joining me today.")
    time.sleep(3)
    furhat.say(text="I hope you feel a sense of calm and peace, and have a wonderful day.")
    time.sleep(3)
    
if __name__=="__main__":
    introduction()

# Add pauses to allow time for the user to follow the instructions
# furhat.pause(duration=10)  # You may need to adjust the duration based on your preference

# Play an audio file (with lipsync automatically added) 
# furhat.say(url="https://drive.google.com/uc?export=open&id=1c-Re2aUo1mQHaJkxLJd6kTUA0CM-i3A_", lipsync=False)
