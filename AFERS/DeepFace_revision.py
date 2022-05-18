import os
from pyexpat import model
import time
import numpy as np
from tqdm import tqdm
import pandas as pd
import cv2
import re

from deepface import DeepFace
from deepface.commons import functions, distance as dst
from deepface.detectors import FaceDetector
from deepface.extendedmodels import Age





class AFERS:


    def __init__(self):
        self.db_path = "DB"
        self.load_database_faces(self.db_path)
        self.load_webcam_stream()

    def load_database_faces(self, db_path):
        self.elderly = []
        if os.path.isdir(db_path) == True:
            for r, d, f in os.walk(db_path):
                for file in f:
                    if ('.jpg' in file):
                        exact_path = r + "/" + file
                        self.elderly.append(exact_path)
        

    def load_gender_model(self):
        return DeepFace.build_model('Gender')
    
    def load_emotion_model(self):
        return DeepFace.build_model('Emotion')

    def load_age_model(self):
        return DeepFace.build_model('Age')

    def load_webcam_stream(self):
        self.streaming = cv2.VideoCapture(0)
        self.input_shape_x = int(self.streaming.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.input_shape_y = int(self.streaming.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.input_shape = (self.input_shape_x, self.input_shape_y)
        self.input_shape = tuple(self.input_shape)

    def preprocessing(self):
        pbar = tqdm(range(0, len(self.elderly)), desc='Finding embeddings')

        #preprocessing delle facce nel DB
        embeddings = []
        #for employee in employees:
        for index in pbar:
            elder = self.elderly[index]
            pbar.set_description("Finding embedding for %s" % (elder.split("/")[-1]))
            embedding = []

            #preprocess_face returns single face. this is expected for source images in db.
            img = functions.preprocess_face(img = elder, target_size = (self.input_shape_y, self.input_shape_x), enforce_detection = False, detector_backend = 'opencv')
            img_representation = self.models.predict(img)[0,:]

            embedding.append(elder)
            embedding.append(img_representation)
            embeddings.append(embedding)

        df = pd.DataFrame(embeddings, columns = ['elder', 'embedding'])
        df['distance_metric'] = 'cosine'
        return

    def analysis(self, model_name = 'VGG-Face', detector_backend = 'opencv', distance_metric = 'cosine', used_models = (), source = 0, time_threshold = 5, frame_threshold = 5):
        
        results = pd.DataFrame()

        #Creazione del modello backend
        face_detector = FaceDetector.build_model(detector_backend)

        elderly = self.load_database_faces(self.db_path)

        #Se ci sono vecchi, allora...
        if elderly is not None:

            #...crea il modello 
            models = DeepFace.build_model(model_name)
            
            input_shape = functions.find_input_shape(models)
            input_shape_x = input_shape[0]; input_shape_y = input_shape[1]      
            #tuned thresholds for model and metric pair
            threshold = dst.findThreshold(model_name, distance_metric)

        #------------------------
        #facial attribute analysis models
        #caricamento dei modelli
        if used_models is not None:

            if "Emotion" in used_models:
                emotion_model = self.load_emotion_model()

            elif "Age" in used_models:
                age_model = self.load_age_model()

            elif "Gender" in used_models:
                gender_model = self.load_gender_model()
        
        if elderly is not None:
            embbeddings = self.preprocessing()


        #------------------------------------------------------

        pivot_img_size = 112 #face recognition result image
        freeze = False
        face_detected = False
        face_included_frames = 0 #freeze screen if face detected sequantially 5 frames
        freezed_frame = 0
        tic = time.time()

        #Loop infinito per la lettura del video
        while(True):

            #Lettura video
            ret, img = self.streaming.read()

            #Se c'è qualche problema, esci
            if img is None:
                break
            
            #Copia dell'immagine e della risoluzione
            raw_img = img.copy()

            #Nel caso in cui non dobbiamo fare il display dell'output
            if freeze == False:

                #Prova a riconoscere le facce
                try:
                    #faces store list of detected_face and region pair
                    faces = FaceDetector.detect_faces(face_detector, detector_backend, img, align = False)
                except: #to avoid exception if no face detected
                    faces = []

                #se non ne riconosce, setta l'iteratore a 0
                if len(faces) == 0:
                    face_included_frames = 0
            #se viene fatto il display dell'output, svuota le facce
            else:
                faces = []

            #Facce riconosciute
            detected_faces = []
            face_index = 0
            #dettagli di nome, dimensione e posizione della faccia per lavorare all'output
            for faces, (x, y, w, h) in faces:
                if w > 130: #discard small detected faces

                    face_detected = True
                    if face_index == 0:
                        face_included_frames = face_included_frames + 1 #increase frame for a single face

                    detected_faces.append((x,y,w,h))
                    face_index = face_index + 1

                    #-------------------------------------

            #Se le facce sono riconosciute per 5 frame consecutivi
            if face_detected == True and face_included_frames == frame_threshold and freeze == False:
                freeze = True
                base_img = raw_img.copy()
                detected_faces_final = detected_faces.copy()

            #Se è l'ora di fare il display...
            if freeze == True:

                if freezed_frame == 0:

                    #si ottengono dimensioni e posizione della faccia riconosciuta al quinto frame
                    for detected_face in detected_faces_final:
                        x = detected_face[0]; y = detected_face[1]
                        w = detected_face[2]; h = detected_face[3]

                        #apply deep learning for custom_face

                        custom_face = base_img[y:y+h, x:x+w]

                        #-------------------------------
                        #Analisi e attributi facciali

                        #se c'è almento un modello utilizzato allora vai qui, altrimenti skippa
                        if used_models == True:

                            if "Emotion" in used_models:
                            #Emotion Recognition
                                gray_img = functions.preprocess_face(img = custom_face, target_size = (48, 48), grayscale = True, enforce_detection = False, detector_backend = 'opencv')
                                emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
                                emotion_predictions = emotion_model.predict(gray_img)[0,:]
                                sum_of_predictions = emotion_predictions.sum()

                                mood_items = []
                                for i in range(0, len(emotion_labels)):
                                    mood_item = []
                                    emotion_label = emotion_labels[i]
                                    emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
                                    mood_item.append(emotion_label)
                                    mood_item.append(emotion_prediction)
                                    mood_items.append(mood_item)

                                emotion_df = pd.DataFrame(mood_items, columns = ["emotion", "score"])
                                emotion_df = emotion_df.sort_values(by = ["score"], ascending=False).reset_index(drop=True)

                                results.append({"Emotions" : emotion_df}, ignore_index=True)

                            
                            elif "Age" in used_models:
                            #ANALISI DELLE ETÀ

                                face_224 = functions.preprocess_face(img = custom_face, target_size = (224, 224), grayscale = False, enforce_detection = False, detector_backend = 'opencv')

                                age_predictions = age_model.predict(face_224)[0,:]
                                apparent_age = Age.findApparentAge(age_predictions)

                                results.append({'Age' : apparent_age}, ignore_index=True)

                                #-------------------------------
                            elif "Gender" in used_models:
                                gender_prediction = gender_model.predict(face_224)[0,:]

                                if np.argmax(gender_prediction) == 0:
                                    gender = "W"
                                elif np.argmax(gender_prediction) == 1:
                                    gender = "M"

                                results.append({'Gender' : gender}, ignore_index=True)
                                
                            #FINE ANALISI DELLE ETÀ
                            #-------------------------------

                        #-------------------------------
                        #face recognition

                        custom_face = functions.preprocess_face(img = custom_face, target_size = (self.input_shape_y, self.input_shape_x), enforce_detection = False, detector_backend = 'opencv')

                        #check preprocess_face function handled
                        if custom_face.shape[1:3] == self.input_shape:
                            #Se ci sono delle immagini nel DB, allora analizza

                            if df.shape[0] > 0: #if there are images to verify, apply face recognition
                                img1_representation = models.predict(custom_face)[0,:]

                                def findDistance(row):
                                    distance_metric = row['distance_metric']
                                    img2_representation = row['embedding']

                                    distance = 1000 #initialize very large value
                                    if distance_metric == 'cosine':
                                        distance = dst.findCosineDistance(img1_representation, img2_representation)
                                    elif distance_metric == 'euclidean':
                                        distance = dst.findEuclideanDistance(img1_representation, img2_representation)
                                    elif distance_metric == 'euclidean_l2':
                                        distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation), dst.l2_normalize(img2_representation))

                                    return distance

                                df['distance'] = df.apply(findDistance, axis = 1)
                                df = df.sort_values(by = ["distance"])

                                candidate = df.iloc[0]
                                elder_name = candidate['elder']
                                best_distance = candidate['distance']


                                #if True:
                                if best_distance <= threshold:
                                    label = elder_name.split("/")[-1].replace(".jpg", "")
                                    label = re.sub('[0-9]', '', label)
                                    
                                    results.append({'Person' : label}, ignore_index=True)
                                
                                return results