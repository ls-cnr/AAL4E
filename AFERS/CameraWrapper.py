import cv2
import numpy

import signal
import sys

from deepface import DeepFace
from deepface.basemodels import VGGFace
from deepface.commons import distance as dst
import numpy as np


class CameraWrapper:

    def __init__(self, model_path, display_frame,ux):
        self.ux = ux
        self.path = model_path
        ux.message("init")
        self.display_the_frame = display_frame

        try:
            ux.message("loading the model")
            self.face_recognition_model = VGGFace.loadModel()
            ux.message("loading classifier")
            self.face_cascade = cv2.CascadeClassifier(model_path + 'haarcascade_frontalface_default.xml')
            ux.message("opening the stream")
            self.streaming = cv2.VideoCapture(0)
            ux.message("capturing the background")
            self.static_back = self.capture_grey_blurred_image()
        except Exception as e:
            ux.message("camera error: "+e)

        signal.signal(signal.SIGINT, self.exit_handler)

    def exit_handler(self, sig, frame):
        self.ux.message('You pressed Ctrl+C!')
        self.release()
        sys.exit(0)

    def release(self):
        self.ux.message("releasing the cam")
        self.streaming.release()
        cv2.destroyAllWindows()

    def capture_grey_blurred_image(self):
        # Gets the frame of the video source
        flag, image = self.streaming.read()

        # Switching the color space to Grayscale to ease of use
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Using a Gaussian Blur to even more improve ease of use
        blurred_gray = cv2.GaussianBlur(gray_image, (21, 21), 0)

        if self.display_the_frame:
            cv2.imshow('Frame', blurred_gray)
            cv2.waitKey(1)

        return blurred_gray


    def calculate_representation(self, path):
        image_repr = DeepFace.represent(path, model=self.face_recognition_model)
        return image_repr

    def face_detection(self):
        blurred_gray = self.capture_grey_blurred_image()
        faces = self.face_cascade.detectMultiScale(blurred_gray, 1.1, 4)
        return len(faces) > 0

    def face_recognition(self, known_faces):
        try:
            ret, image = self.streaming.read()
            cam_repr = DeepFace.represent(image, model=self.face_recognition_model)

            if self.display_the_frame:
                cv2.imshow('Last Capture', image)
                cv2.waitKey(1)

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


def detect_movement(self):
    # print("capturing the image")
    blurred_gray = self.capture_grey_blurred_image()

    # Gets the absolute difference between the static background and the current analysed frame
    frame_difference = cv2.absdiff(self.static_back, blurred_gray)

    # Thresholds the frame in order to recognise foreground and background, then appies morphological operations
    # in order to show the foreground as bigger
    threshold_frame = cv2.threshold(frame_difference, 30, 255, cv2.THRESH_BINARY)[1]
    # threshold_frame = cv2.dilate(cv2.threshold(frame_difference, 30, 255, cv2.THRESH_BINARY)[1], None, iterations=2)

    # Finds the countours of the foreground
    contours, _ = cv2.findContours(threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Number of Contours found = " + str(len(contours)))

    relevant = False
    # For each countour
    for contour in contours:
        th = cv2.contourArea(contour)
        if th > 1000:  # If an area with relevant area is found a motion is detected
            relevant = True

    return relevant
