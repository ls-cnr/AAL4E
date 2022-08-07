import os
from warnings import catch_warnings
import cv2
import sys
import pandas
from gtts import gTTS
import speech_recognition
import numpy
from database_handling import DataBaseHandler
import time

from deepface import DeepFace
from deepface.commons import functions
from deepface.detectors import FaceDetector
from tensorflow.python.keras import Model as pred

#Definition of the class for the initialization of the system
class Globals:

    #Initialization of the object
    def __init__(self, database_path, API, model = 'VGG-Face'):

        #Important properties set to zero at the beginning
        self.backend_model = None
        self.emotion_model = None
        self.age_model = None
        self.gender_model = None
        self.input_shape  = (0,0)
        self.input_shape_x = 0
        self.input_shape_y = 0
        self.face_detector = None
        self.model = None
        self.model_name = model
        self.database_path = database_path
        self.mediaAPI = API

        self.recognizer_instance = speech_recognition.Recognizer()

        #Loading the stream
        self.load_webcam_stream()
        self.build_model(model)


    #Loading the Database
    def load_database_faces(self):
        #If the Database is loaded, continue; otherwise kill the program after printing the exception
        try:
            #The database will be stored here
            self.elderly = []

            #If the path exists, continue; otherwise kill the program
            if os.path.isdir(self.database_path) == True:

                #Gets all the relevant information about the directory
                for r, d, f in os.walk(self.database_path):

                    #Add all files with their exact path to the list of people recognised
                    for file in f:
                        if('.jpg' in file):
                            exact_path = r + "/" + file
                            self.elderly.append(exact_path)

                #If the list is empty, exit
                if self.elderly is None:
                    sys.exit(1)

            else:
                print(self.database_path)
                print("Specified path not working. Exiting...")
                sys.exit(1)

        except OSError as ose:
            print(ose)
            sys.exit(1)

        except Exception as e:
            print(e)
            sys.exit(1)


    #Loading gender model
    def load_gender_model(self):
        #Try tot load the model, if not possible kill the program after printing the exception
        try:
            if self.gender_model is None:

                #Call to deepface.DeepFace.build_model function
                self.gender_model = DeepFace.build_model('Gender')

                #If the call fails, kill the program
                if self.gender_model is None:
                    sys.exit(1)

        except Exception as e:
            print(e)
            sys.exit(1)
    

    #Loading gender model
    def load_age_model(self):
        #Try tot load the model, if not possible kill the program after printing the exception
        try:
            if self.age_model is None:

                #Call to deepface.DeepFace.build_model function
                self.age_model = DeepFace.build_model('Age')

                #If the call fails, kill the program
                if self.age_model is None:
                    sys.exit(1)

        except Exception as e:
            print(e)
            sys.exit(1)


    #Loading emotion model
    def load_emotion_model(self):
        #Try tot load the model, if not possible kill the program after printing the exception
        try:
            if self.emotion_model is None:
                
                #Call to deepface.DeepFace.build_model function
                self.emotion_model = DeepFace.build_model('Emotion')
                #If the call fails, kill the program
                if self.emotion_model is None:
                    sys.exit(1)
        except Exception as e:
            print(e)
            sys.exit(1)


    #Loading Webcam
    def load_webcam_stream(self):
        #Try tot load the stream, if not possible kill the program after printing the exception  
        try:
            self.streaming = cv2.VideoCapture(0)
            #Get the shape of the stream
            self.input_shape = tuple((int(self.streaming.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.streaming.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            

        except Exception as e:
            print(e)
            sys.exit(1)


    #Backend model building
    def build_backend_model(self, detector_backend):
        #Try tot load the model, if not possible kill the program after printing the exception
        try:
            #Call to deepface.detector.FaceDetector.build_model function
            if self.backend_model is None:
                self.backend_model = FaceDetector.build_model(detector_backend)

                #If the call fails, exit
                if self.backend_model is None:
                    sys.exit(1)

        except Exception as e:
            print(e)
            sys.exit(1)


    #Model building
    def build_model(self, model):
        #Try tot load the model, if not possible kill the program after printing the exception
        try:
            #Call to deepface.DeepFace.build_model function
            if self.model is None:
                self.model = DeepFace.build_model(model)

                #If the call fails, exit
                if self.model is None:
                    sys.exit(1)

        except Exception as e:
            print(e)
            sys.exit(1)
            

    #Preprocessing
    def preprocessing(self):
        #Try to Preprocess, if something goes wrong, exit after printing the error
        try:
            
            embeddings = []
            #Preprocess any image contained in the database
            for index in range(0, len(self.elderly)):
                elder = self.elderly[index]
                
                embedding = []
                #Actual Preprocessing
                img = functions.preprocess_face(img = elder, target_size =(224,224), enforce_detection = False, detector_backend = 'opencv')
                if self.model is None:
                    self.build_model(self.model_name)
                img_representation = self.model.predict(x=img)[0,:]
                embedding.append(elder)
                embedding.append(img_representation)
                embeddings.append(embedding)
                
            #Creating a dataframe for later to be returned
            self.embeddings_df = pandas.DataFrame(embeddings, columns= ['elder', 'embedding'])
            self.embeddings_df['distance_metric'] = 'cosine'
        except Exception as e:
            print(e)
            sys.exit(1)


    #Speech Analysis method
    def speech_analysis(self, tts='', lang='en'):

        with speech_recognition.Microphone() as source:
            #Reduce the Ambient Noise of the Microphone (taking a second of environmental noise takes a second or so)
            self.recognizer_instance.adjust_for_ambient_noise(source)

            #If a text is passed as a parameter, call the TTSInterface in order to say it
            if tts:
                self.TTSInterface(tts, lang=lang)
                time.sleep(1)

            #Listen to what the user has to see
            audio = self.recognizer_instance.listen(source)
            time.sleep(1.5)
        try:
            #Contact the Google Speech Recogniser to analyse the audio produced
            text = self.recognizer_instance.recognize_google(audio, language="it-IT")

            #Return the result as a lower case text
            return text.lower()
        
        #If the request to the Google Speech Recogniser happens to fail, throw an Exception
        except Exception as e:
            print(e)
    

    #Text To Speech Interface
    def TTSInterface(self, text='', lang='en'):
        #Reference to the class gTTS()
        tts = gTTS(text, lang=lang)
        
        #Save the audio as temporary MP3 file in order to call the playsound() function, then delete the file
        tts.save("dump.mp3")
        os.system('mpg321 dump.mp3 && exit')
        os.remove("dump.mp3")
    

    #Embedding of the Registration Module
    def registration(self):
        try:
            #Get the path of the folder for any single person as /<root_folder>/DB/<name>-<surname>/
            name = self.speech_analysis("State your name, wait a moment before speaking", lang='en')
            input(name)
            surname = self.speech_analysis("State your surname, wait a moment before speaking", lang='en')
            input(surname)

            #Initialize the connection to the database
            dh = DataBaseHandler(database_path=self.database_path)
            folder_name = self.database_path + name.replace(' ', '-').lower() + '_' + surname.replace(' ', '-').lower() + '/'
            input(folder_name)
            #If the person's entry does not exists in the database, create it

            
            dh.DBHElderlyCommit(name=name, surname=surname, picture=folder_name)

            name = name.replace(' ', '-').lower()
            surname = surname.replace(' ', '-').lower()
            #If the path does not exists, create it
            if not os.path.isdir(folder_name):
                os.makedirs(folder_name)

            self.TTSInterface("Taking a picture. Hold still")
            ret, frame = self.streaming.read()
            if ret == True:
                #Check if the folder is empty. If it is, then write the image
                if len(os.listdir(folder_name)) == 0:
                    cv2.imwrite(folder_name + name.lower() + '_' + surname.lower() + '_1'+ '.jpg', frame)
                    
                else:
                    #If the folder is not empty, ask if you want to add another picture
                    answer = self.speech_analysis(tts="The system recognises you as an user. Do you want to add another image?", lang='en')
                    if answer == "yes":
                        cv2.imwrite(folder_name + name.lower() + '_' + surname.lower() + '_' + str(len(os.listdir(folder_name))) + '.jpg', frame)
                    else:
                        self.TTSInterface("Image saving aborted", lang='en')
            else:
                self.TTSInterface("Image saving aborted 2", lang='en')
        except Exception as e:
            print(e)

    
    #This method recognise if there's a change in the scene and exits, otherwise it loops indefinitely
    def motion_recognition(self):

        #Definition of a static background
        static_back = None
        


        #Until the exit condition inside this block is verified, it loops indefinitely
        while True:
            
            #Gets the frame of the video source
            check, image = self.streaming.read()


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
    def idle_recognition(self):

        #Definition of a static background and a variable to store the last analysed frame
        static_back = None
        prev_frame = None

        #Variable that counts how many consecutive idle frames are detected
        idle_frame_check = 0
        
        #Gets the video source and its resolution
        video_input = self.streaming

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


    
    def emotion_check(self, res):
        emotions = res['Emotions']
        mood = emotions['Emotion'].iat[0]

        if mood == "Happy" or mood =="Surprise":
            return 'Positive'
        elif mood == 'Neutral':
            return 'neutral'
        else:
            return 'negative'
    
