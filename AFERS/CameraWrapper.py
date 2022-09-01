import cv2
import numpy

import signal
import sys

from deepface import DeepFace
from deepface.basemodels import VGGFace
from deepface.commons import distance as dst
import numpy as np


class CameraWrapper:

    def __init__(self, model_path, display_frame):

        self.path = model_path
        print("init")
        self.display_the_frame = display_frame

        try:
            print("loading the model")
            self.face_recognition_model = VGGFace.loadModel()
            print("loading classifier")
            self.face_cascade = cv2.CascadeClassifier(model_path + 'haarcascade_frontalface_default.xml')
            print("opening the stream")
            self.streaming = cv2.VideoCapture(0)
            print("capturing the background")
            self.static_back = self.capture_grey_blurred_image()
        except Exception as e:
            print("camera error: "+e)

        signal.signal(signal.SIGINT, self.exit_handler)

    def exit_handler(self, sig, frame):
        print('You pressed Ctrl+C!')
        self.release()
        sys.exit(0)

    def release(self):
        print("releasing the cam")
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

        # cv2.drawContours(image=blurred_gray, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,lineType=cv2.LINE_AA)
        # if self.display_the_frame:
        #     cv2.imshow('Frame2', blurred_gray)
        #     cv2.waitKey(1)
        # return False

        relevant = False
        # For each countour
        for contour in contours:
            th = cv2.contourArea(contour)
            # print("detected th: "+str(th))
            # The area is checked and if it is not so relevant (the area is less than 10000), the next area is checked
            if th > 1000:  # If an area with relevant area is found a motion is detected
                # print("is relevant!")
                relevant = True

        # print("exiting with "+str(relevant))
        return relevant

    def detect_face(self):
        blurred_gray = self.capture_grey_blurred_image()
        faces = self.face_cascade.detectMultiScale(blurred_gray, 1.1, 4)
        return len(faces) > 0

    def calculate_representation(self, path):
        image_repr = DeepFace.represent(path, model=self.face_recognition_model)
        print(image_repr)
        return image_repr

    def recognize_face_optimized(self, known_faces):
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

    def recognize_face(self, known_faces):
        try:
            ret, image = self.streaming.read()
            image_repr = DeepFace.represent(image, model=self.face_recognition_model)
            # print(embedding2)

            cv2.imshow('Last Capture', image)

            for key in known_faces:
                img_path = known_faces[key]
                print(img_path)

                img_path_repr = DeepFace.represent(img_path, model=self.face_recognition_model)
                distance = dst.findCosineDistance(image_repr, img_path_repr)
                distance = np.float64(distance)
                if distance < 0.4:
                    print("primo metodo => " + key + ": appatta")
                else:
                    print("primo metodo => " + key + ": non appatta")

                result = DeepFace.verify(img_path, image)
                print(result)
                if result["verified"]:
                    print("secondo metodo => " + key + ": appatta")
                else:
                    print("secondo metodo => " + key + ": non appatta")

        except Exception as e:
            print(e)

        return True
