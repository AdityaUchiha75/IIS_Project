
from time import sleep
from furhatConfig import LOOK_DOWN, LOOK_BACK

# ----- Visualization Excercise  -----
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
    sleep(12)

    # Call the function that returns the user's current emotion
    furhat.say("How are you feeling right now?") # angry, sad, calm, happy
    emotion = furhat.listen().lower()
    sleep(3)

    # Call the function that returns the response to the user's current emotion
    response = visualizationResponse(emotion)
    furhat.say(f"I understand you're feeling {emotion}. Embrace visualization, and let it guide you towards a {response} state.")

# ----- Gratitude Excercise  -----
def gratitudeExercise(furhat):
    furhat.say(text="Shift your focus to gratitude.")
    sleep(12)
    furhat.say(text="Bring to mind three things you are thankful for today.")
    sleep(3)
    furhat.say(text="They can be simple or profound.")
    sleep(12)
    furhat.say(text="As you reflect on each one, feel a sense of gratitude and warmth.")

    # Add pauses to allow time for the user to experience the gratitude
    sleep(12) 

    # Prompt the user to share their current emotion
    furhat.say("How are you feeling right now?") # (angry, sad, calm, happy, something)
    emotion = furhat.listen().lower()
    sleep(3)

     # Call the custom emotion response function
    response = gratitudeEmotionResponse(emotion)
    furhat.say(f"I understand you're feeling {emotion}. Embrace gratitude, and let it guide you towards a {response} state.")

# ----- Visualization : Responses to each of the emotions : angry, happy, sad, calm/relaxed  -----
def visualizationResponse(furhat, emotion):
    if emotion == "angry":
        furhat.say("I see that you're angry. While in this visualization, acknowledge your anger and visualize letting it go. Picture your serene place absorbing any negative energy.")
        sleep(3)
        askForBreak()

    elif emotion == "happy":
        furhat.say("Great to see you're feeling happy! In your visualization, amplify this happiness. Picture your serene place filled with joyful energy.")
        sleep(3)
        askForBreak()

    elif emotion == "sad":
        furhat.say("I notice that you're feeling sad. During this visualization, allow the peaceful setting to envelop you with comfort. Visualize releasing any sadness with each breath.")
        sleep(3)
        askForBreak()

    elif emotion == "calm":
        furhat.say("I sense that you're feeling calm. Use this calmness to deepen your visualization experience. Picture your serene place with heightened clarity and tranquility.")
        sleep(3)
        askForBreak()
    
    else:
        furhat.say("I understand that emotions can be complex. Whatever you're feeling, allow gratitude to be a source of comfort and positivity.")
        sleep(3)
        askForBreak()

# ----- Gratitude : Responses to each of the emotions : angry, happy, sad, calm/relaxed  -----
def gratitudeEmotionResponse(furhat, emotion):
    if emotion == "angry":
        furhat.say("I see that you're Angry. I understand that anger can be challenging. Acknowledge it and allow gratitude to bring a sense of peace and positivity.")
        sleep(3)
        return "happy"
    
    elif emotion == "sad":
        furhat.say("I see that you're Sad. Feeling sad is a natural part of life. Embrace gratitude to lift your spirits and find comfort in the positive aspects of your day.")
        sleep(3)
        return "calm/relaxed"
    
    elif emotion == "calm":
        furhat.say("I see that you're calm. It's wonderful that you're feeling calm. Use this calmness to deepen your sense of gratitude and appreciation for the present moment.")
        sleep(3)
        return "happy"
    
    elif emotion == "happy":
        furhat.say("I see that you're happy. That's Fantastic! Embrace the joy you're feeling. Let gratitude amplify this happiness and create a positive ripple effect in your day.")
        sleep(3)
        return "calm/relaxed"
    
    else:
        furhat.say("I understand that emotions can be complex. Whatever you're feeling, allow gratitude to be a source of comfort and positivity.")
        sleep(3)
        return "something"

# ----- Break : Function that triggers when asking for a break  -----
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

# ----- Conclusion Exercise  -----
def conclusionExercise(furhat):
    furhat.say(text="Now, slowly bring your awareness back to the present moment.")
    sleep(3)
    furhat.say(text="When you are ready, open your eyes.")
    sleep(3)

    # Prompt the user to share their current emotion
    emotion = input("How are you feeling right now? (angry, sad, calm, happy): ").lower()

    # Call the custom emotion response function
    # conclusionEmotionResponse(emotion)

    furhat.say(text="Thank you for joining me today.")
    sleep(3)
    furhat.say(text="I hope you feel a sense of calm and peace, and have a wonderful day.")
    sleep(3)
