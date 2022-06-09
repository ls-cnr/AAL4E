import os
import cv2
import sys
import pandas
from gtts import gTTS
from playsound import playsound
from database_handling import DataBaseHandler
import speech_recognition

from deepface import DeepFace
from deepface.commons import functions
from deepface.detectors import FaceDetector

#Definition of the class for the initialization of the system
class Globals:

    #Initialization of the object
    def __init__(self, database_path):

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
        self.database_path = database_path

        self.recognizer_instance = speech_recognition.Recognizer()

        #Loading the stream
        self.load_webcam_stream()


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
                img = functions.preprocess_face(img = elder, target_size = self.input_shape, enforce_detection = False, detector_backend = 'opencv')
                img_representation = self.model.predict(img)[0,:]

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

            #Listen to what the user has to see
            audio = self.recognizer_instance.listen(source)
        try:
            #Contact the Google Speech Recogniser to analyse the audio produced
            text = self.recognizer_instance.recognize_google(audio, language="it-IT")

            #Return the result as a lower case text
            return text.lower()
        
        #If the request to the Google Speech Recogniser happens to fail, throw an Exception
        except Exception as e:
            print(e)
    

    #Text To Speech Interface
    def TTSInterface(text, lang='en'):
        #Reference to the class gTTS()
        tts = gTTS(text, lang=lang)
        
        #Save the audio as temporary MP3 file in order to call the playsound() function, then delete the file
        tts.save("dump.mp3")
        playsound("dump.mp3")
        os.remove("dump.mp3")
    

    #Embedding of the Registration Module
    def registration(self, name, surname, frame):
        try:
            #Get the path of the folder for any single person as /<root_folder>/DB/<name>-<surname>/
            db_path = self.database_path
            folder_name = db_path + name.lower() + '-' + surname.lower() + '/'

            #Initialize the connection to the database
            d = DataBaseHandler()

            #If the person's entry does not exists in the database, create it
            if not d.DBHElderExists(name=name, surname=surname):
                d.DBHElderlyCommit(name=name, surname=surname, picture=folder_name)

            #If the path does not exists, create it
            if not os.path.isdir(folder_name):
                os.makedirs(folder_name)

            #Check if the folder is empty. If it is, then write the image
            if len(os.listdir(folder_name)) == 0:
                cv2.imwrite(name.lower() + '_' + surname.lower() + '_1'+ '.jpg', frame)
                
            else:
                #If the folder is not empty, ask if you want to add another picture
                answer = self.speech_analysis(tts="Do you want to add another image?", lang='en')
                if answer is "yes":
                    cv2.imwrite(name.lower() + '_' + surname.lower() + '_' + (len(os.listdir(folder_name))) + '.jpg', frame)
                else:
                    self.TTSInterface("Image saving aborted", lang='en')
            d.DBHClose()
        except Exception as e:
            print(e)