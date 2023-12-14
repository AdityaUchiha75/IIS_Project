
# ----- File with Furhat gestures, voice initialization, and face initialization -----

# Initializing the face of the robot
FACES = {
    'Loo'    : 'Patricia',
    'Amany'  : 'Nazar'
}

# Intializing the voice of the robot
VOICES_EN = {
    'FemaleOne' : 'Kimberly-Neural',
    'FemaleTwo' : 'Salli-Neural',
    'Male' : 'Kevin-Neural'
}

def furhatConfig(furhat):
    furhat.set_face(character=FACES['Amany'], mask="Adult")
    # Setting the voice of the robot
    furhat.set_voice(name=VOICES_EN['FemaleTwo'])


# ----- Gestures functions -----

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
