import os
import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.model_from_json(open(r"/IIS_Project/models/model.json", "r").read())
model.load_weights(r'C:\Users\NickosKal\Desktop\University\Semester 3\Period 4\Intelligent Interactive '
                   r'Systems\IIS_Project\models\fer2013_model.h5')
# model = load_model('static\Fer2013.h5')
face_haar_cascade = cv2.CascadeClassifier(r'/IIS_Project/frontal_face_features.xml')
cap = cv2.VideoCapture(0)

while cap.isOpened():
    res, frame = cap.read()

    height, width, channel = frame.shape
    sub_img = frame[0:int(height / 6), 0:int(width)]

    black_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0
    res = cv2.addWeighted(sub_img, 0.77, black_rect, 0.23, 0)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.8
    FONT_THICKNESS = 2
    label_color = (10, 10, 255)
    label = "Emotion Detection made by Abhishek"
    label_dimension = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)[0]
    textX = int((res.shape[1] - label_dimension[0]) / 2)
    textY = int((res.shape[0] + label_dimension[1]) / 2)
    cv2.putText(res, label, (textX, textY), FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS)
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_haar_cascade.detectMultiScale(gray_image)
    try:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = gray_image[y - 5:y + h + 5, x - 5:x + w + 5]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            image_pixels = tf.keras.preprocessing.image.img_to_array(roi_gray)
            image_pixels = np.expand_dims(image_pixels, axis=0)
            image_pixels /= 255
            predictions = model.predict(image_pixels)
            max_index = np.argmax(predictions[0])
            emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            emotion_prediction = emotion_detection[max_index]
            cv2.putText(res, "Sentiment: {}".format(emotion_prediction), (0, textY + 22 + 5), FONT, 0.7, label_color, 2)
            label_violation = 'Confidence: {}'.format(str(np.round(np.max(predictions[0]) * 100, 1)) + "%")
            violation_text_dimension = cv2.getTextSize(label_violation, FONT, FONT_SCALE, FONT_THICKNESS)[0]
            violation_x_axis = int(res.shape[1] - violation_text_dimension[0])
            cv2.putText(res, label_violation, (violation_x_axis, textY + 22 + 5), FONT, 0.7, label_color, 2)
    except:
        pass
    frame[0:int(height / 6), 0:int(width)] = res
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()