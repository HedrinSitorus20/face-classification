from keras.models import load_model
import numpy as np
import cv2

# Load gender model
gender_model_path = 'models\genderModel_VGG16.hdf5'
gender_classifier = load_model(gender_model_path, compile=False)
gender_target_size = gender_classifier.input_shape[1:3]

# Define genders dictionary
genders = {
    0: {"label": "Female", "color": (245, 215, 130)},
    1: {"label": "Male", "color": (148, 181, 192)},
}

# Load face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open video capture
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using OpenCV face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        resized = frame[y-20: y+h+30, x-10:x+w+10]
        cv2.imshow("resized: ", resized)
        try:
            frame_resize = cv2.resize(resized, gender_target_size)
        except:
            continue

        frame_resize = frame_resize.astype('float32')
        frame_scaled = frame_resize/255.0
        frame_reshape = np.reshape(frame_scaled, (1, *gender_target_size, 3))
        frame_vstack = np.vstack([frame_reshape])
        gender_prediction = gender_classifier.predict(frame_vstack)
        gender_probability = np.max(gender_prediction)
        color = (255, 255, 255)

        if gender_probability > 0.6:
            gender_label = np.argmax(gender_prediction)
            gender_result = genders[gender_label]["label"]
            color = genders[gender_label]["color"]
            cv2.putText(frame, gender_result, (x+5, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow("Gender Classification", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
