# Importing required packages
import cv2
import numpy as np
from keras.models import load_model

def rect_points(rect):
    x, y, w, h = rect
    return (x, y, w, h)

gender_model_path = 'models/genderModel_VGG16.hdf5'
gender_classifier = load_model(gender_model_path, compile=False)
gender_target_size = gender_classifier.input_shape[1:3]

genders = {
    0: {"label": "Female", "color": (245, 215, 130)},
    1: {"label": "Male", "color": (148, 181, 192)},
}

# pre-trained model
model_file = "faceDetection/models/dnn/res10_300x300_ssd_iter_140000.caffemodel"
# prototxt has the information of where the training data is located.
config_file = "faceDetection/models/dnn/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(config_file, model_file)

def detect_faces_with_dnn(frame):
    # A neural network that really supports the input value
    size = (300, 300)

    # After executing the average reduction, the image needs to be scaled
    scale_factor = 1.0

    # These are our mean subtraction values. They can be a 3-tuple of the RGB means or
    # they can be a single value in which case the supplied value is subtracted from every
    # channel of the image.
    swap_RB = (104.0, 117.0, 123.0)

    height, width = frame.shape[:2]
    resized_frame = cv2.resize(frame, size)
    blob = cv2.dnn.blobFromImage(resized_frame, scale_factor, size, swap_RB)
    net.setInput(blob)
    dnn_faces = net.forward()
    for i in range(dnn_faces.shape[2]):
        confidence = dnn_faces[0, 0, i, 2]
        if confidence > 0.5:
            box = dnn_faces[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x1, y1) = box.astype("int")
            resized = frame[y: y1, x:x1]
            try:
                frame_resize = cv2.resize(resized, gender_target_size)
            except:
                continue

            frame_resize = frame_resize.astype("float32")
            frame_scaled = frame_resize / 255.0
            frame_reshape = np.reshape(frame_scaled, (1, *gender_target_size, 3))
            frame_vstack = np.vstack([frame_reshape])
            gender_prediction = gender_classifier.predict(frame_vstack)
            gender_probability = np.max(gender_prediction)
            color = (255, 255, 255)
            if gender_probability > 0.4:
                gender_label = np.argmax(gender_prediction)
                gender_result = genders[gender_label]["label"]
                color = genders[gender_label]["color"]
                cv2.putText(frame, gender_result, (x + 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x1, y1), color, 2)
            else:
                cv2.rectangle(frame, (x, y), (x1, y1), color, 2)
    return frame

photo = cv2.imread("images/tiktok2.jpg")
photo = cv2.resize(photo, (1080, 720))
frame = detect_faces_with_dnn(photo)
cv2.imshow("Gender Classification", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
