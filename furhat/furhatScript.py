
from furhat_remote_api import FurhatRemoteAPI
from time import sleep
import asyncio
import pygame

# Create an instance of the FurhatRemoteAPI class, providing the address of the robot or the SDK running the virtual robot
furhat = FurhatRemoteAPI("localhost")
furhat.set_led(red=100, green=50, blue=50)

# ----- Initializing robot's characteristics -----

# Initializing the face of the robot
FACES = {
    'Loo'    : 'Patricia',
    'Amany'  : 'Nazar'
}
furhat.set_face(character=FACES['Amany'], mask="Adult")

# Intializing the voice of the robot
VOICES_EN = {
    'FemaleOne' : 'Kimberly-Neural',
    'FemaleTwo' : 'Salli-Neural',
    'Male' : 'Kevin-Neural'
}

furhat.set_voice(name=VOICES_EN['FemaleTwo'])

def LOOK_BACK(speed):
    return {
    "frames": [
        {
            "time": [
                0.33 / speed
            ],
            "persist": True,
            "params": {
                'LOOK_DOWN' : 0,
                'LOOK_UP' : 0,
                'NECK_TILT' : 0
            }
        }, {
            "time": [
                1 / speed
            ],
            "params": {
                "NECK_PAN": 0,
                'LOOK_DOWN' : 0,
                'LOOK_UP' : 0,
                'NECK_TILT' : 0
            }
        }
    ],
    "class": "furhatos.gestures.Gesture"
    }

def LOOK_DOWN(speed=1):
    return {
    "frames": [
        {
            "time": [
                0.33 / speed
            ],
            "persist": True,
            "params": {
                'LOOK_DOWN' : 1.0
            }
        }, {
            "time": [
                1 / speed
            ],
            "persist": True,
            "params": {
                "NECK_TILT": 20
            }
        }
    ],
    "class": "furhatos.gestures.Gesture"
    }

# async def generate_color_gradient(furhat, duration=20, steps=200):
#     for step in range(steps):
#         hue = step / float(steps)
#         # Adjust hue to transition only between blue, green, and white
#         hue = hue * 0.5  # Scale the hue range to cover blue, green, and white
#         rgb_color = tuple(int(val * 255) for val in colorsys.hsv_to_rgb(hue, 1.0, 1.0))
        
#         # Set LED to only display green, white, and blue
#         furhat.set_led(red=0, green=rgb_color[1], blue=rgb_color[2])
        
#         await asyncio.sleep(duration / steps)


# Example usage:
# furhat.set_led(red=200, green=50, blue=50)  # Set initial color
# generate_color_gradient(duration=10, steps=100)  # Generate a color gradient over 10 seconds with 100 steps

async def introduction():

    # Set initial color to white
    furhat.set_led(red=200, green=200, blue=200)
    # asyncio.create_task(generate_color_gradient(furhat, duration=20, steps=200))
    
    # Play an audio file (with lipsync automatically added) 
    pygame.mixer.init()
    pygame.mixer.music.load("furhat/backgroundWaves.mp3")
    pygame.mixer.music.play()

    furhat.gesture(name="CloseEyes")
    furhat.gesture(body=LOOK_DOWN(speed=1), blocking=True)
    sleep(0.5)
    furhat.gesture(body=LOOK_BACK(speed=1), blocking=True)
    furhat.gesture(name="OpenEyes")
    furhat.say(text="Welcome to today's meditation session.")
    sleep(3)
    furhat.say(text="My name is Amany, and I will be your guide today.")

asyncio.run(introduction())

# Play an audio file (with lipsync automatically added) 
# furhat.say(url="https://drive.google.com/uc?export=open&id=1c-Re2aUo1mQHaJkxLJd6kTUA0CM-i3A_", lipsync=False)
