import torch
from torch import nn
from pathlib import Path
from time import sleep
from furhat.furhatConfig import LOOK_DOWN, LOOK_BACK

DIR_PATH = str(Path(__file__).parent.parent.absolute()) + r"\\"

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
    
def best_emo_model(local_path):
    return torch.load(Path(DIR_PATH + local_path) )

def return_emo(loc):
    expression = {"anger": 6, "disgust": 5 , "fear": 4, "happiness": 1, "neutral": 0, "sadness": 2, "surprise": 3}
    for key,_ in expression.items():
        if expression[key] == loc:
            return key


# ------- Furhat ----------

def visualizationExercise(furhat):

    # Visualization (1 minute)
    furhat.say(text="Now, let's enter a moment of visualization.")
    sleep(12)
    furhat.say(text="Picture a serene place in your mind. It could be a beach, a forest, or any place that brings you peace.")
    sleep(12)
    furhat.say(text="Imagine the sights, sounds, and smells of this tranquil setting.")

    # Add pauses to allow time for the user to visualize
    sleep(12) 

    furhat.say(text="With each breath, feel yourself absorbing the calm and positive energy from this place.")
    sleep(8)
    furhat.say(text="Let the peaceful energy wash over you, soothing your body and mind.")

    # Add pauses to allow time for the user to experience the visualization
    # sleep(12)

    # Call the function that returns the user's current emotion
    # furhat.say("How are you feeling right now?") # angry, sad, calm, happy
    # emotion = furhat.listen().lower()
    # sleep(3)


def visualizationResponse(furhat, emotion):
    if emotion == "angry":
        furhat.say("I see that you're angry. While in this visualization, acknowledge your anger and visualize letting it go. Picture your serene place absorbing any negative energy.")
        sleep(3)
        askForBreak()
        return "happy"
    
    elif emotion == "happy":
        furhat.say("Great to see you're feeling happy! In your visualization, amplify this happiness. Picture your serene place filled with joyful energy.")
        sleep(3)
        askForBreak()
        return "calm/relaxed"
    
    elif emotion == "sad":
        furhat.say("I notice that you're feeling sad. During this visualization, allow the peaceful setting to envelop you with comfort. Visualize releasing any sadness with each breath.")
        sleep(3)
        askForBreak()
        return "calm/relaxed"

    elif emotion == "calm":
        furhat.say("I sense that you're feeling calm. Use this calmness to deepen your visualization experience. Picture your serene place with heightened clarity and tranquility.")
        sleep(3)
        askForBreak()
        return "maintain"
    
    else:
        furhat.say("I understand that emotions can be complex. Whatever you're feeling, allow gratitude to be a source of comfort and positivity.")
        sleep(3)
        askForBreak()
        return "calm/relaxed"

def askForBreak(furhat):
    while True:
        furhat.say("Would you like to take a break?")
        result = furhat.listen()
        
        if result.lower() == "yes":
            furhat.say("Great! let's take a 10 second break")
            furhat.gesture(name="CloseEyes")
            furhat.gesture(body=LOOK_DOWN(speed=1), blocking=True)
            sleep(10)
            furhat.gesture(body=LOOK_BACK(speed=1), blocking=True)
            furhat.gesture(name="OpenEyes")
            furhat.say("Let's continue.")
            break  # Exit the loop if the user wants a break
        
        elif result.lower() == "no":
            furhat.say("Great! Let's continue.")
            break  # Exit the loop if the user doesn't want a break
       
        else:
            furhat.say("I'm sorry, I didn't get that")


# if __name__ == '__main__':
#     print(DIR_PATH)
#     t = best_emo_model('models\\best_model_12.pt')