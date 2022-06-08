import os
import cv2
import sys
import pandas

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