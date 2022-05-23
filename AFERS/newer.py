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
        self.backend_model = []
        self.emotion_model = []
        self.age_model = []
        self.gender_model = []
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
        except Exception as e:
            print("Error building the backend model")
            print(e)
            sys.exit(1)
        finally:
            print("build_backend_model function has ended")


    #Model building
    def build_model(self, model):
        try:
            if self.model is None:
                self.model = DeepFace.build_model(model)
        except Exception as e:
            print("Error building ", model, " model")
            print(e)
        finally:
            print("build_model function has ended")


    #Preprocessing
    def preprocessing(self):
        try:
            embeddings = []
            for index in range(0, len(self.elderly)):
                elder = self.elderly[index]

                embedding = []

                img = functions.preprocess_face(img = elder, target_size = self.input_shape, enforce_detection = False, detector_backend = 'opencv')
                img_representation = self.models.predict(img)[0,:]

                embedding.append(elder)
                embedding.append(img_representation)
                embeddings.append(embedding)
            
            embeddings_df = pandas.DataFrame(embeddings, columns= ['elder', 'embedding'])
            embeddings_df['distance_metric'] = 'cosine'

            print("preprocessing function has ended")
            return embeddings_df
        except Exception as e:
            print("Error while preprocessing")
            print(e)
            sys.exit(1)
    
class AFERS:

    def analysis(self, model_name = 'VGG-Face', detector_backend = 'opencv', distance_metric = 'cosine', used_models = [], db_path='DB', frame_threshold = 5):

        global_functions = Globals()
        result = pandas.DataFrame()

        face_detector = global_functions.build_backend_model(detector_backend=detector_backend)

        elderly = global_functions.load_database_faces(db_path=db_path)

        if elderly is not None:

            model = global_functions.build_model(model=model_name)

            input_shape = functions.find_input_shape(model)
            input_shape_x = input_shape[0]
            input_shape_y = input_shape[1]

            threshold = Distance.findThreshold(model_name, distance_metric)

        if used_models is not None:

            input(used_models)
            if 'Emotion' in used_models:
                emotion_model = global_functions.load_emotion_model()
            if 'Age' in used_models:
                age_model = global_functions.load_age_model()
            if 'Gender' in used_models:
                gender_model = global_functions.load_gender_model()

        
        #it does not enter here
        if elderly is not None:
            embeddings = global_functions.preprocessing()

        freeze = False
        face_detected = False
        face_included_frames = 0
        freezed_frame = 0

        iteration = 0

        while(result.empty):
            print("Iteration number ", iteration)
            print(result)
            iteration = iteration + 1
            input()

            ret, image = global_functions.streaming.read()

            if image is None:
                break

            image_copy = image.copy()

            try:
                faces = FaceDetector.detect_faces(face_detector, detector_backend, image, align = False)
                print["face detected", faces]
            except:
                faces = []
                print("No face detected in this frame")
                face_included_frames = 0
            
            detected_faces = []
            face_index = 0

            for faces, (x, y, w, h) in faces:
                if w > 130:
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

                    face_224 = functions.preprocess_face(img= custom_face, target_size=( 224, 224), grayscale= False, detector_backend= 'opencv')
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
                        
                        if 'Age' in used_models:

                            age_predictions = age_model.predict(face_224)[0,:]
                            apparent_age = Age.findApparentAge(age_predictions)

                            result = result.append({'Age' : apparent_age}, ignore_index= True)

                        if 'Gender' in used_models:
                            gender_prediction = gender_model.predict(face_224)[0,:]

                            if numpy.argmax(gender_prediction) == 0:
                                gender = 'W'
                            elif numpy.argmax(gender_prediction) == 1:
                                gender = 'M'

                            result = result.append({'Gender': gender}, ignore_index=True)
                    

                    custom_face = functions.preprocess_face(img= custom_face, target_size= (input_shape_x, input_shape_y), enforce_detection= False, detector_backend= 'opencv')


                    if custom_face.shape[1:3] == input_shape:

                        if detected_face.shape[0] > 0:
                            img1_representation = model.predict(custom_face)[0,:]

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


        input("Does it arrrive to the end?")          
        return result