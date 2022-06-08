import cv2
import numpy

'''
    Script for the recognition of a state of idling or the presence of motion in a scene. The video used can be real live captured using "dev/video0" source 
'''

#This method recognise if there's a change in the scene and exits, otherwise it loops indefinitely
def motion_recognition(globals_pointer):

    #Definition of a static background
    static_back = None
    


    #Until the exit condition inside this block is verified, it loops indefinitely
    while True:
        
        #Gets the frame of the video source
        check, image = globals_pointer.streaming.read()


        #Switching the color space to Grayscale to ease of use 
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #Using a Gaussian Blur to even more improve ease of use
        blurred_gray = cv2.GaussianBlur( gray_image , (21, 21), 0)

        #This "if" statement has its condition verified only if we are looking at the first frame of the scene
        if static_back is None:
            #Sets the static background to the first frame of the scene, then exit this block
            static_back = blurred_gray 
            continue

        #Gets the absolute difference between the static background and the current analysed frame
        frame_difference = cv2.absdiff(numpy.array(static_back), blurred_gray)

        #Thresholds the frame in order to recognise foreground and background, then appies morphological operations in order to show the foreground as bigger
        threshold_frame = cv2.dilate(cv2.threshold(frame_difference, 30, 255, cv2.THRESH_BINARY)[1] , None, iterations = 2)

        #Finds the countours of the foreground
        contours,_ = cv2.findContours(threshold_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #For each countour
        for contour in contours:
            #The area is checked and if it is not so relevant (the area is less than 10000), the next area is checked
            if cv2.contourArea(contour) < 10000:
                continue
            return 1 #If an area with relevant area is found a motion is detected



#This method recognise if there's no change in the scene for a fixed number of frames and exits, otherwise it loops indefinitely
def idle_recognition(globals_pointer):

    #Definition of a static background and a variable to store the last analysed frame
    static_back = None
    prev_frame = None

    #Variable that counts how many consecutive idle frames are detected
    idle_frame_check = 0

    #Gets the video source and its resolution
    video_input = globals_pointer.streaming

    #Gets the frame area for later calculations
    frame_area = int(video_input.get(cv2.CAP_PROP_FRAME_WIDTH)) * int(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #Until tow seconds of idling  frames are found, it keeps looping
    while idle_frame_check < video_input.get(cv2.CAP_PROP_FPS):
        

        #Gets the frame of the video source
        check, image = video_input.read()

        #Switching the color space to Grayscale to ease of use
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #Using a Gaussian Blur to even more improve ease of use
        blurred_gray = cv2.GaussianBlur( gray_image , (21, 21), 0)

        #This "if" statement has its condition verified only if we are looking at the first frame of the scene
        if static_back is None:
            #Sets the static background to the first frame of the scene, then exit this block
            static_back = blurred_gray
            continue

        #Gets the absolute difference between the static background and the current analysed frame
        frame_difference = cv2.absdiff(numpy.array(static_back), blurred_gray)

        #Thresholds the frame in order to recognise foreground and background, then appies morphological operations in order to show the foreground as bigger
        threshold_frame = cv2.dilate(cv2.threshold(frame_difference, 30, 255, cv2.THRESH_BINARY)[1] , None, iterations = 2)


        if prev_frame is not None:

            #Calculates the norm distance between the two images
            norm =  cv2.norm(threshold_frame, prev_frame, cv2.NORM_L2)
            #Checks how similar the two images are
            similarity = 1 - norm/ frame_area

            #If the two images are similar...
            if similarity >= 0.95:
                    #... increment the number of idling frame analysed
                idle_frame_check = idle_frame_check + 1
            else:
                #If not, resets the counter to zero
                idle_frame_check = 0
            
        else:
            #If there's not a previous frame (hence we are in the first frame) it is considere as an idling frame, the counter is incremented
            idle_frame_check = idle_frame_check + 1

        #Stores the current analysed frame in the prev_frame variable 
        prev_frame = threshold_frame
    return 1