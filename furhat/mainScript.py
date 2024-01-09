from furhat_remote_api import FurhatRemoteAPI
import time
# import pygame
from .excercises import visualizationExercise
from .furhatConfig import furhatConfig
from .furhatConfig import LOOK_DOWN
from .furhatConfig import LOOK_BACK

# Create an instance of the FurhatRemoteAPI class, providing the address of the robot or the SDK running the virtual
# robot
furhat = FurhatRemoteAPI("localhost")
furhat.set_led(red=100, green=50, blue=50)

# ----- Initializing robot's characteristics -----
furhatConfig(furhat)


# ----- Introduction dialogues & exercises  -----
def introduction(emotion):

    # ----- Set initial color LED -----
    furhat.set_led(red=200, green=200, blue=200)
    
    # ----- Start playing the background music -----
    # pygame.mixer.init()
    # pygame.mixer.music.load(r"C:\Users\NickosKal\Desktop\University\Semester 3\Period 4\Intelligent Interactive Systems\IIS_Project\furhat\backgroundWaves.mp3")
    # pygame.mixer.music.set_volume(0.3)
    # pygame.mixer.music.play()

    # ----- Starting Gesture  -----
    furhat.gesture(name="CloseEyes")
    furhat.gesture(body=LOOK_DOWN(speed=1), blocking=True)
    time.sleep(0.5)
    furhat.gesture(body=LOOK_BACK(speed=1), blocking=True)
    furhat.gesture(name="OpenEyes")

    # ----- Starting Dialogue  -----
    furhat.say(text="Welcome to today's meditation session.")
    time.sleep(4)
    furhat.say(text="My name is Amany, and I will be your guide today.")
    time.sleep(4)
    furhat.say(text="Close your eyes and take a few deep breaths, inhaling through your nose and exhaling through "
                    "your mouth.")
    time.sleep(12)
    furhat.say(text="Begin by bringing your awareness to the present moment.")
    time.sleep(12)
    furhat.say(text="Feel the sensation of your body sitting or lying down.")
    time.sleep(12)

    furhat.say(text="Our first exercise will be a visualization exercise.")

    # ----- Starting Visualization Exercise  -----
    visualizationExercise(furhat, emotion=emotion)


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
    
        
introduction(None)

# Add pauses to allow time for the user to follow the instructions
furhat.pause(duration=10)  # You may need to adjust the duration based on your preference

# Play an audio file (with lipsync automatically added) 
# furhat.say(url="https://drive.google.com/uc?export=open&id=1c-Re2aUo1mQHaJkxLJd6kTUA0CM-i3A_", lipsync=False)
