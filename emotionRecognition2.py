import argparse

from keras.models import load_model
import numpy as np
import cv2
import argparse

# Setup argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-vw", "--isVideoWriter", type=bool, default=False)
args = vars(ap.parse_args())

# Emotion label and color dictionary
emotions = {
    0: {"emotion": "Angry", "color": (193, 69, 42)},
    1: {"emotion": "Disgust", "color": (164, 175, 49)},
    2: {"emotion": "Fear", "color": (40, 52, 155)},
    3: {"emotion": "Happy", "color": (23, 164, 28)},
    4: {"emotion": "Sad", "color": (164, 93, 23)},
    5: {"emotion": "Suprise", "color": (218, 229, 97)},
    6: {"emotion": "Neutral", "color": (108, 72, 200)}
}

# Load the pre-trained emotion classification model
emotionModelPath = 'models/emotionModel.hdf5'
emotionClassifier = load_model(emotionModelPath, compile=False)
emotionTargetSize = emotionClassifier.input_shape[1:3]

# Initialize video capture
cap = cv2.VideoCapture(0)

# Video writer setup
videoWrite = None
if args["isVideoWriter"]:
    fourrcc = cv2.VideoWriter_fourcc(*"MJPG")
    capWidth, capHeight = int(cap.get(3)), int(cap.get(4))
    videoWrite = cv2.VideoWriter("output.avi", fourrcc, 22, (capWidth, capHeight))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for better performance
    frame = cv2.resize(frame, (720, 480))
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using OpenCV face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(grayFrame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Prepare the face for the model
        grayFace = grayFrame[y:y + h, x:x + w]
        try:
            grayFace = cv2.resize(grayFace, emotionTargetSize)
        except:
            continue

        grayFace = grayFace.astype('float32') / 255.0
        grayFace = (grayFace - 0.5) * 2.0
        grayFace = np.expand_dims(np.expand_dims(grayFace, 0), -1)

        # Predict emotion
        emotion_prediction = emotionClassifier.predict(grayFace)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)

        # Display emotion label if probability is high
        if emotion_probability > 0.36:
            color = emotions[emotion_label_arg]['color']
            emotion_text = emotions[emotion_label_arg]['emotion']
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Write to video file if needed
    if args["isVideoWriter"]:
        videoWrite.write(frame)

    # Display the result
    cv2.imshow("Emotion Recognition", frame)

    # Break the loop when 'ESC' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
if videoWrite is not None:
    videoWrite.release()
cv2.destroyAllWindows()
