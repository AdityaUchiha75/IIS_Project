import cv2
import numpy as np
import os
from pathlib import Path
import tensorflow as tf

path = Path(os.getcwd()).parent.parent
model_path = Path(str(path) + '/model_larger_architecture.h5')


def emotion_detection():
    # Load the emotion recognition model
    model = tf.keras.models.load_model(model_path)

    # Define the emotion labels
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    # Load the Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Open a connection to the webcam (0 is usually the default webcam)
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Convert the frame to grayscale (assuming your model was trained on grayscale images)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert the frame to grayscale (assuming your model was trained on grayscale images)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform face detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Draw rectangles around the detected faces and make predictions
        for (x, y, w, h) in faces:
            # Crop the face from the frame
            face_roi = gray[y:y + h, x:x + w]

            # Resize the face to match the input size expected by your model
            resized_face = cv2.resize(face_roi, (48, 48))

            # Preprocess the face for prediction
            img_array = tf.keras.preprocessing.image.img_to_array(resized_face)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            # Make predictions
            predictions = model.predict(img_array)

            # Get the predicted emotion label
            predicted_emotion = emotion_labels[np.argmax(predictions)]

            # Draw rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Display the predicted emotion label
            cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

            yield predicted_emotion

        # Display the frame with rectangles and emotion labels
        cv2.imshow('Emotion Recognition', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

