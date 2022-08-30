import cv2
import numpy
from deepface import DeepFace
from deepface.basemodels import VGGFace


class CameraWrapper:

    def __init__(self,display_frame=True):
        print("init")
        self.display_the_frame = display_frame
        #print("loading the model")
        #self.face_recognition_model = VGGFace.loadModel()
        print("opening the stream")
        self.streaming = cv2.VideoCapture(0)
        print("capturing the background")
        self.static_back = self.capture_grey_blurred_image()

    def capture_grey_blurred_image(self):
        # Gets the frame of the video source
        flag, image = self.streaming.read()

        # Switching the color space to Grayscale to ease of use
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Using a Gaussian Blur to even more improve ease of use
        blurred_gray = cv2.GaussianBlur(gray_image, (21, 21), 0)

        if self.display_the_frame:
            cv2.imshow('Frame', blurred_gray)

        return blurred_gray

    def detect_movement(self):
        print("capturing the image")
        blurred_gray = self.capture_grey_blurred_image()

        # Gets the absolute difference between the static background and the current analysed frame
        frame_difference = cv2.absdiff(numpy.array(self.static_back), blurred_gray)

        # Thresholds the frame in order to recognise foreground and background, then appies morphological operations
        # in order to show the foreground as bigger
        threshold_frame = cv2.dilate(cv2.threshold(frame_difference, 30, 255, cv2.THRESH_BINARY)[1], None, iterations=2)

        # Finds the countours of the foreground
        contours, _ = cv2.findContours(threshold_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        relevant = False
        # For each countour
        for contour in contours:
            th = cv2.contourArea(contour)
            print("detected th: "+th)
            # The area is checked and if it is not so relevant (the area is less than 10000), the next area is checked
            if cv2.contourArea(contour) > 10000:  # If an area with relevant area is found a motion is detected
                relevant = True

        return relevant

    def check_face(self, known_faces):
        pass
