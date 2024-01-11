import cv2
import numpy as np
from feat import Detector
from feat.utils import FEAT_EMOTION_COLUMNS
import time
from numpy.random import randint
from time import sleep
import torch
#import tensorflow as tf
import multiprocessing
from pathlib import Path
import os
import sys

parent_dir = os.path.abspath('../IIS/IIS_Project/')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from furhat_remote_api import FurhatRemoteAPI
import time
import pygame
from furhat.furhatConfig import furhatConfig, LOOK_DOWN, LOOK_BACK
from utils import best_emo_model, return_emo, MLP
# Create an instance of the FurhatRemoteAPI class, providing the address of the robot or the SDK running the virtual robot
furhat = FurhatRemoteAPI("localhost")
furhat.set_led(red=100, green=50, blue=50)

# ----- Initializing robot's characteristics -----
furhatConfig(furhat)

# ----- Initializing detector -----

detector = Detector(device="cuda")

# ----- Initializing model -----
EMO_LIST = []
DIR_PATH = str(Path(__file__).parent.parent.absolute()) + r"\\"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = best_emo_model(DIR_PATH + 'models\\Emo\\best_model_12.pt').to(device)
# model_path = Path(str(path) + '/models/model_larger_architecture.h5')
# model = tf.keras.models.load_model(model_path)

def return_detected_emo(emo_list):
    max=0
    for i in np.unique(emo_list):
        if np.char.count(emo_list, i).sum() > max:
            max = np.char.count(emo_list, i).sum()
            emo = i

    if emo == 'anger' or emo == 'disgust':
        return 'disturbed'
    
    elif emo== 'happiness' or emo == 'surprise':
        return 'energetic and positive'
    
    elif emo == 'fear' or emo == 'sadness':
        return 'depressed'
    
    elif emo == 'neutral':
        return 'stoic'
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
    return True

# ------- Furhat ----------

def say_with_delay(furhat, tex_t, t):
    furhat.say(text=tex_t, blocking=True)
    sleep(t)

def visualizationExercise(furhat):

    # Visualization Exercise (1 minute)
    say_with_delay(furhat, "Now, let's enter a moment of visualization. Please close your eyes!", 1)
    
    flag = capture_and_return_emos(detector, model, device,10)
    
    say_with_delay(furhat, "Picture a serene place in your mind. It could be a beach, a forest, or any place that brings you peace.", 1)
    
    flag = capture_and_return_emos(detector, model, device,10)

    say_with_delay(furhat, "Imagine the sights, sounds, and smells of this tranquil setting.", 1)
    
    flag = capture_and_return_emos(detector, model, device,10)
    
    init_emo = EMO_LIST.copy()
    EMO_LIST.clear()

    say_with_delay(furhat, "With each breath, feel yourself absorbing the calm and positive energy from this place.", 1)
    
    flag = capture_and_return_emos(detector, model, device,10)
    
    say_with_delay(furhat, "Let the peaceful energy wash over you, soothing your body and mind.", 1)
    
    flag = capture_and_return_emos(detector, model, device,10)
    # Call the function that returns the user's current emotion
    if flag:
        furhat.say(text = "How are you feeling right now?", blocking=True)
        result = furhat.listen()
        texts = result.__dict__['_message'].split(' ')
        if any(el in texts for el in ['great','good','super', 'happy']):
            furhat.say(text = "I am glad to hear that! ", blocking=True)
        else:
            furhat.say(text = "I see. I believe attending more of these stress relaxing exercises may bring calmness to your mind.", blocking = True)
        furhat.say(text = "Now, during the exercise, ")
    return init_emo
    # sleep(3)


def visualizationResponse(furhat, emotion):
    if emotion == "angry":
        furhat.say(text="I saw that you're angry. While in this visualization, acknowledge your anger and visualize letting it go. Picture your serene place absorbing any negative energy.")
        sleep(3)
        askForBreak(furhat)
        return "happy and eventually a calm"
    
    elif emotion == "happy":
        furhat.say(text="I sensed that you're feeling happy! In your visualization, amplify this happiness. Picture your serene place filled with joyful energy.")
        sleep(3)
        askForBreak(furhat)
        return "calm"
    
    elif emotion == "sad":
        furhat.say(text="I notice that you're feeling sad. During this visualization, allow the peaceful setting to envelop you with comfort. Visualize releasing any sadness with each breath.")
        sleep(3)
        askForBreak(furhat)
        return "calm"

    elif emotion == "calm":
        furhat.say(text="I sensed that you're feeling calm. Use this calmness to deepen your visualization experience. Picture your serene place with heightened clarity and tranquility.")
        sleep(3)
        askForBreak(furhat)
        return "maintain"
    
    else:
        furhat.say(text="I couldn't find an apt word to describe your emotion. Anyway, allow gratitude to be a source of comfort and positivity.")
        sleep(3)
        askForBreak(furhat)
        return "calm"

def askForBreak(furhat):
    while True:
        furhat.say(text="Would you like to take a break?", blocking=True)
        result = furhat.listen()
        texts = result.__dict__['_message'].split(' ')
        if "yes" in texts:
            furhat.say(text="Great! let's take a 10 second break")
            furhat.gesture(name="CloseEyes")
            furhat.gesture(body=LOOK_DOWN(speed=1))
            sleep(10)
            furhat.gesture(body=LOOK_BACK(speed=1))
            furhat.gesture(name="OpenEyes")
            furhat.say(text="Let's continue.")
            break  # Exit the loop if the user wants a break
        
        elif "no" in texts:
            furhat.say(text="Great! Let's continue.", blocking=True)
            break  # Exit the loop if the user doesn't want a break
       
        else:
            furhat.say(text="I'm sorry, I didn't get that, can you repeat it?")

# ----- Introduction dialogues & excercises  -----
def introduction():

    # ----- Set initial color LED -----
    furhat.set_led(red=200, green=200, blue=200)
    
    # ----- Start playing the background music -----
    pygame.mixer.init()
    pygame.mixer.music.load(DIR_PATH + "furhat/backgroundWaves.mp3")
    pygame.mixer.music.set_volume(0.3)
    pygame.mixer.music.play()

    # ----- Starting Gesture  -----
    furhat.gesture(name="CloseEyes")
    furhat.gesture(body=LOOK_DOWN(speed=1), blocking=True)
    time.sleep(0.5)
    furhat.gesture(body=LOOK_BACK(speed=1), blocking=True)
    furhat.gesture(name="OpenEyes")

    # ----- Starting Dialogue  -----
    furhat.say(text="Welcome to today's meditation session.",blocking=True)
    furhat.say(text="My name is Lucia, and I will be your guide today. Now let us begin the 1 minute session")
    sleep(2)
    # ----- Starting Visualization Exercise  -----

    # Furhat conducts the 'exercise' and does tries to detect the user's emotion simultaneously  
    initial_emo = return_detected_emo(visualizationExercise(furhat))
    print(EMO_LIST)
    emo_after_exercise = return_detected_emo(EMO_LIST)
    
    response = visualizationResponse(furhat, emo_after_exercise)

    if response=='maintain':
        furhat.say(text = f"At the beginning of the exercise, I could see that you were feeling {initial_emo}. As the exercise progressed, I sensed that you're feeling {return_detected_emo(EMO_LIST)}. This is ideal, embrace visualization, and let it guide you towards a maintaining this calm state.",blocking=True)
    else:
        furhat.say(text = f"At the beginning of the exercise, I could see that you were feeling {initial_emo}. As the exercise progressed, I sensed that you're feeling {return_detected_emo(EMO_LIST)}. Embrace visualization, and let it guide you towards a {response} state.",blocking=True)
    sleep(2)
# ----- Conclusion  -----
def conclusion():
    # furhat.say(text="Now, slowly bring your awareness back to the present moment.")
    # time.sleep(3)
    # furhat.say(text="When you are ready, open your eyes.")
    # time.sleep(3)
    furhat.say(text="Thank you for joining me today.")
    furhat.say(text="I hope you feel a sense of calm and peace, and have a wonderful day!")
    
if __name__=="__main__":
    introduction()
    conclusion()

