import os
import cv2
import sys
import numpy
import pandas
import re

from deepface import DeepFace
from deepface.commons import functions, distance as Distance
from deepface.detectors import FaceDetector
from deepface.extendedmodels import Age

#Definition of the class for the initialization of the system
class Globals:

    def __init__(self):
        self.backend_model = None
        self.emotion_model = None
        self.age_model = None
        self.gender_model = None
        self.input_shape  = (0,0)
        self.input_shape_x = 0
        self.input_shape_y = 0
        self.face_detector = None
        self.model = None

        self.load_webcam_stream()

    #Loading Database
    def load_database_faces(self, db_path):
        try:
            #The database will be stored here
            self.elderly = []
            if os.path.isdir(db_path) == True:
                for r, d, f in os.walk(db_path):
                    for file in f:
                        if('.jpg' in file):
                            exact_path = r + "/" + file
                            self.elderly.append(exact_path)
                if self.elderly is None:
                    sys.exit(1)
                print("load_database_faces function has ended")
            else:
                print(os.getcwd() + db_path)
                print("Specified path not working. Exiting...")
                sys.exit(1)
        except OSError as ose:
            print("Error loading the database")
            print(ose)

        except Exception as e:
            print("Generic Error")
            print(e)
            sys.exit(1)


    #Loading gender model
    def load_gender_model(self):
        try:
            if self.gender_model is None:
                self.gender_model = DeepFace.build_model('Gender')
                if self.gender_model is None:
                    sys.exit(1)
        except Exception as e:
            print("Error loading the gender model")
            print(e)
        finally:
            print("load_gender_model function has ended")
    

    #Loading gender model
    def load_age_model(self):
        try:
            if self.age_model is None:
                self.age_model = DeepFace.build_model('Age')
                if self.age_model is None:
                    sys.exit(1)
        except Exception as e:
            print("Error loading the age model")
            print(e)
        finally:
            print("load_age_model function has ended")


    #Loading emotion model
    def load_emotion_model(self):
        try:
            if self.emotion_model is None:
                self.emotion_model = DeepFace.build_model('Emotion')
                if self.emotion_model is None:
                    sys.exit(1)
        except Exception as e:
            print("Error loading the emotion model")
            sys.exit(1)
        finally:
            print("load_emotion_model function has ended")

        
    #Loading Webcam
    def load_webcam_stream(self):
        try:
            self.streaming = cv2.VideoCapture(0)
            temporary_shape = (int(self.streaming.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.streaming.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self.input_shape = tuple(temporary_shape)

            print("load-webcam_stream function has ended")
        except Exception as e:
            print("Error loading video stream")
            print(e)
            sys.exit(1)


    #Backend model building
    def build_backend_model(self, detector_backend):
        try:
            if self.backend_model is None:
                self.backend_model = FaceDetector.build_model(detector_backend)
            if self.backend_model is None:
                print("Backend problem")
                sys.exit(1)
            print(self.backend_model)
        except Exception as e:
            print("Error building the backend model")
            print(e)
            sys.exit(1)
        except self.backend_model is not None:
            print("cazzone, non hai creato il modello")
            sys.exit(1)


    #Model building
    def build_model(self, model):
        try:
            if self.model is None:
                self.model = DeepFace.build_model(model)
            if self.model is None:
                sys.exit(1)
            print(self.backend_model)
            print("build_model function has ended")
        except Exception as e:
            print("Error building", model, "model")
            print(e)
            sys.exit(1)
            

    #Preprocessing
    def preprocessing(self):
        try:
            embeddings = []
            for index in range(0, len(self.elderly)):
                elder = self.elderly[index]

                embedding = []

                img = functions.preprocess_face(img = elder, target_size = self.input_shape, enforce_detection = False, detector_backend = 'opencv')
                img_representation = self.model.predict(img)[0,:]

                embedding.append(elder)
                embedding.append(img_representation)
                embeddings.append(embedding)
            
            embeddings_df = pandas.DataFrame(embeddings, columns= ['elder', 'embedding'])
            embeddings_df['distance_metric'] = 'cosine'

            print("preprocessing function has ended")
            if embeddings_df is None:
                sys.exit(1)
            else:
                return embeddings_df
        except Exception as e:
            print("Error while preprocessing")
            print(e)
            sys.exit(1)
    
class AFERS:

    def analysis(self, model_name = 'VGG-Face', detector_backend = 'opencv', distance_metric = 'cosine', used_models = [], db_path='DB', frame_threshold = 5):

        global_functions = Globals()
        result = pandas.DataFrame()

        global_functions.model = DeepFace.build_model(model_name)
        if global_functions.model is None:
            sys.exit(1)
        else:
            print(global_functions.model)
        global_functions.build_backend_model(detector_backend=detector_backend)

        global_functions.load_database_faces(db_path=db_path)

        #Works Perfectly
        if global_functions.elderly is not None:
            
            global_functions.input_shape = functions.find_input_shape(global_functions.model)
            global_functions.input_shape_x = global_functions.input_shape[0]
            global_functions.input_shape_y = global_functions.input_shape[1]

            threshold = Distance.findThreshold(model_name, distance_metric)

        #Works Perfectly
        if used_models is not None:
            if 'Emotion' in used_models:
                emotion_model = global_functions.load_emotion_model()
            if 'Age' in used_models:
                age_model = global_functions.load_age_model()
            if 'Gender' in used_models:
                gender_model = global_functions.load_gender_model()

        
        #Works Perfectly
        if global_functions.elderly is not None:
            embeddings = global_functions.preprocessing()

        freeze = False
        face_detected = False
        face_included_frames = 0

        dec_copy = global_functions.backend_model
        print(dec_copy)

        while(True):

            ret, image = global_functions.streaming.read()

            if image is None:
                break
            
            

            image_copy = image.copy()
            faces = []
            try:
                faces = FaceDetector.detect_faces(face_detector= dec_copy, detector_backend= detector_backend, img=image, align = False)
            except:
                faces = []
            
            if len(faces) == 0:
                face_included_frames = 0

            detected_faces = []
            face_index = 0

            for faces, (x, y, w, h) in faces:
                if w > 130:
                    print("face detected!")
                    face_detected = True
                    if face_index == 0:
                        face_included_frames = face_included_frames + 1
                    
                    detected_faces.append((x, y, w, h))
                    face_index = face_index + 1
            
            if face_detected == True and face_included_frames == frame_threshold:
                base_image = image_copy.copy()
                detected_faces_final = detected_faces.copy()


                for detected_face in detected_faces_final:
                    x = detected_face[0]; y = detected_face[1]
                    w = detected_face[2]; h = detected_face[3]

                    custom_face = base_image[y:y+h, x:x+w]

                    face_224 = functions.preprocess_face(img= custom_face, target_size=(224, 224), grayscale= False, enforce_detection=False, detector_backend=detector_backend)
                    
                    if used_models == True:
                        if 'Emotion' in used_models:

                            gray_img = functions.preprocess_face(img=custom_face, target_size=(48,48), grayscale=True, enforce_detection=False, detector_backend=detector_backend)
                            emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
                            emotion_predictions = emotion_model.predict(gray_img)[0,:]
                            sum_of_predictions = emotion_predictions.sum()

                            moods = []
                            for i in range(0, len(emotion_labels)):
                                mood = []
                                emotion_label = emotion_labels[i]
                                emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions

                                mood.append(emotion_label)
                                mood.append(emotion_prediction)
                                moods.append(mood)

                            emotion_df = pandas.DataFrame(moods, columns=['Emotion', 'Score'])
                            emotion_df = emotion_df.sort_values(by=['Score'], ascending=False).reset_index(drop=True)

                            result = result.append({'Emotions' : emotion_df}, ignore_index=True)
                            print(result)
                        if 'Age' in used_models:

                            age_predictions = age_model.predict(face_224)[0,:]
                            apparent_age = Age.findApparentAge(age_predictions)

                            result = result.append({'Age' : apparent_age}, ignore_index= True)
                            print(result)
                        if 'Gender' in used_models:
                            gender_prediction = gender_model.predict(face_224)[0,:]

                            if numpy.argmax(gender_prediction) == 0:
                                gender = 'W'
                            elif numpy.argmax(gender_prediction) == 1:
                                gender = 'M'

                            result = result.append({'Gender': gender}, ignore_index=True)
                            print(result)

                    custom_face = functions.preprocess_face(img= custom_face, target_size= (global_functions.input_shape_x, global_functions.input_shape_y), enforce_detection= False, detector_backend= 'opencv')


                    if custom_face.shape[1:3] == global_functions.input_shape:

                        if embeddings.shape[0] > 0:
                            img1_representation = global_functions.model.predict(custom_face)[0,:]

                            def findDistance(row):
                                embeddings = row['distance_metric']
                                img2_representation = row['embedding']

                                distance = 1000
                                if distance_metric == 'cosine':
                                    distance = Distance.findCosineDistance(img1_representation, img2_representation)
                                elif distance_metric == 'euclidean':
                                    distance = Distance.findEuclideanDistance(img1_representation, img2_representation)
                                elif distance_metric == 'euclidean_l2':
                                    distance = Distance.findEuclideanDistance(Distance.l2_normalize(img1_representation), Distance.l2_normalize(img2_representation))
                                
                                return distance
                            
                            embeddings['distance'] = embeddings.apply(findDistance, axis= 1)
                            embeddings = embeddings.sort_values(by= ['distance'])

                            candidate = embeddings.iloc[0]
                            elder_name = candidate['elder']
                            best_distance = candidate['distance']

                            if best_distance <= threshold:
                                label = elder_name.split("/")[-1].replace(".jpg", "")
                                label = re.sub('[0-9]', '', label)

                                result.append({'Person' : label}, ignore_index=True)


        input(result)          
        return result