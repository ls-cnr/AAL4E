import cv2

import signal
import sys

from deepface import DeepFace
from deepface.basemodels import VGGFace
from deepface.extendedmodels import Emotion
from deepface.commons import distance as dst
from deepface.commons import functions

import numpy as np


class CameraWrapper:

    def __init__(self, model_path, display_frame, ux):
        self.ux = ux
        self.path = model_path
        ux.message("init")
        self.display_the_frame = display_frame
        self.static_back = None

        try:
            ux.message("loading the face detection model")
            self.face_recognition_model = VGGFace.loadModel()
            ux.message("loading the emotion analysis model")
            self.emotion_model = Emotion.loadModel()
            ux.message("loading classifier")
            self.face_cascade = cv2.CascadeClassifier(model_path + 'haarcascade_frontalface_default.xml')
            ux.message("opening the stream")
            self.streaming = cv2.VideoCapture(0)


        except Exception as e:
            ux.message("camera error: " + e)

        signal.signal(signal.SIGINT, self.exit_handler)

    def exit_handler(self, sig, frame):
        self.ux.message('You pressed Ctrl+C!')
        self.release()
        sys.exit(0)

    def release(self):
        self.ux.message("releasing the cam")
        self.streaming.release()
        cv2.destroyAllWindows()

    def capture_image(self, greyscale=False, blurred=False):
        # Gets the frame of the video source
        flag, image = self.streaming.read()

        if self.display_the_frame:
             cv2.imshow('Frame', image)
             cv2.waitKey(1)

        # Switching the color space to Grayscale to ease of use
        if greyscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Using a Gaussian Blur to even more improve ease of use
        if blurred:
            image = cv2.GaussianBlur(image, (21, 21), 0)

        return image

    def calculate_representation(self, image_or_imagefile):
        image_repr = DeepFace.represent(image_or_imagefile, model=self.face_recognition_model)
        return image_repr

    def face_detection(self):
        blurred_gray = self.capture_image(greyscale=True,blurred=True)
        faces = self.face_cascade.detectMultiScale(blurred_gray, 1.1, 4)
        return len(faces) > 0

    def face_recognition(self, known_faces):
        try:
            image = self.capture_image()
            cam_repr = self.calculate_representation(image)

            threshold = 0.4
            selected = None
            for key in known_faces:
                known_face_representation = known_faces[key]

                distance = dst.findCosineDistance(cam_repr, known_face_representation)
                distance = np.float64(distance)
                if distance < threshold:
                    selected = key
                    threshold = distance

            return selected

        except Exception as e:
            print(e)

        return None

    def emotion_analysis(self):
        try:
            image = self.capture_image()

            emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            img, region = functions.preprocess_face(img=image, target_size=(48, 48), grayscale=True,
                                                    enforce_detection=True,
                                                    detector_backend="opencv", return_region=True)
            emotion_predictions = self.emotion_model.predict(img)[0, :]

            resp_obj = {}
            sum_of_predictions = emotion_predictions.sum()
            for i in range(0, len(emotion_labels)):
                emotion_label = emotion_labels[i]
                emotion_prediction = round( 100 * emotion_predictions[i] / sum_of_predictions ,3 )
                resp_obj[emotion_label] = emotion_prediction

            resp_obj["dominant_emotion"] = emotion_labels[np.argmax(emotion_predictions)]
            print(resp_obj)

        except Exception as e:
            print(e)


def detect_movement(self):
    # print("capturing the image")
    blurred_gray = self.capture_grey_blurred_image()
    if self.static_back is None:
        self.static_back = blurred_gray
        return False

    else:
        # Gets the absolute difference between the static background and the current analysed frame
        frame_difference = cv2.absdiff(self.static_back, blurred_gray)
        self.static_back = blurred_gray

        # Thresholds the frame in order to recognise foreground and background, then appies morphological operations
        # in order to show the foreground as bigger
        threshold_frame = cv2.threshold(frame_difference, 30, 255, cv2.THRESH_BINARY)[1]
        # threshold_frame = cv2.dilate(cv2.threshold(frame_difference, 30, 255, cv2.THRESH_BINARY)[1], None, iterations=2)

        # Finds the countours of the foreground
        contours, _ = cv2.findContours(threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        relevant = False
        # For each countour
        for contour in contours:
            th = cv2.contourArea(contour)
            if th > 1000:  # If an area with relevant area is found a motion is detected
                relevant = True

        return relevant
