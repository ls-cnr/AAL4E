import os
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

    def __init__(self) -> None:
        pass

    def analysis(db_path, model_name = 'VGG-Face', detector_backend = 'opencv', distance_metric = 'cosine', used_models = (), source = 0, time_threshold = 5, frame_threshold = 5):
        
        #Creazione del modello backend
        face_detector = FaceDetector.build_model(detector_backend)

        #------------------------

        #Quanti vecchi ci sono
        elderly = []
        #Se ci sono vecchi nel DB, aggiungili nella cartella
        if os.path.isdir(db_path) == True:
            for r, d, f in os.walk(db_path): # r=root, d=directories, f = files
                for file in f:
                    if ('.jpg' in file):
                        #exact_path = os.path.join(r, file)
                        exact_path = r + "/" + file
                        #print(exact_path)
                        elderly.append(exact_path)

        #Se ci sono vecchi, allora...
        if len(elderly) > 0:

            #...crea il modello 
            model = DeepFace.build_model(model_name)
            

            input_shape = functions.find_input_shape(model)
            input_shape_x = input_shape[0]; input_shape_y = input_shape[1]      
            #tuned thresholds for model and metric pair
            threshold = dst.findThreshold(model_name, distance_metric)

        #------------------------
        #facial attribute analysis models
        #caricamento dei modelli (da fare all'inizio della creazione del )
        if used_models is not None:

            if "emotion" in used_models:
                emotion_model = DeepFace.build_model('Emotion')

            elif "age" in used_models:
                age_model = DeepFace.build_model('Age')

            elif "gender" in used_models:
                gender_model = DeepFace.build_model('Gender')
        
        
        #una progress bar in command line
        pbar = tqdm(range(0, len(elderly)), desc='Finding embeddings')

        #TODO: why don't you store those embeddings in a pickle file similar to find function?

        #preprocessing delle facce nel DB
        embeddings = []
        #for employee in employees:
        for index in pbar:
            elder = elderly[index]
            pbar.set_description("Finding embedding for %s" % (elder.split("/")[-1]))
            embedding = []

            #preprocess_face returns single face. this is expected for source images in db.
            img = functions.preprocess_face(img = elder, target_size = (input_shape_y, input_shape_x), enforce_detection = False, detector_backend = detector_backend)
            img_representation = model.predict(img)[0,:]

            embedding.append(elder)
            embedding.append(img_representation)
            embeddings.append(embedding)

        df = pd.DataFrame(embeddings, columns = ['elder', 'embedding'])
        df['distance_metric'] = distance_metric
        #Fine Preprocessing


        #------------------------------------------------------

        pivot_img_size = 112 #face recognition result image
        freeze = False
        face_detected = False
        face_included_frames = 0 #freeze screen if face detected sequantially 5 frames
        freezed_frame = 0
        tic = time.time()


        #Apertura canale Video
        cap = cv2.VideoCapture(source) #webcam

        #Loop infinito per la lettura del video
        while(True):

            #Lettura video
            ret, img = cap.read()

            #Se c'è qualche problema, esci
            if img is None:
                break
            
            #Copia dell'immagine e della risoluzione
            raw_img = img.copy()
            resolution_x = img.shape[1]; resolution_y = img.shape[0]

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
            for face, (x, y, w, h) in faces:
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
                tic = time.time()

            #Se è l'ora di fare il display...
            if freeze == True:

                toc = time.time()
                #Se la differenza di tempo tra il riconoscimento e il display è minore del threshold che abbiamo attribuito, allora fai il display
                if (toc - tic) < time_threshold:
                    

                    if freezed_frame == 0:
                        freeze_img = base_img.copy()
                        #freeze_img = np.zeros(resolution, np.uint8) #here, np.uint8 handles showing white area issue

                        #si ottengono dimensioni e posizione della faccia riconosciuta al quinto frame
                        for detected_face in detected_faces_final:
                            x = detected_face[0]; y = detected_face[1]
                            w = detected_face[2]; h = detected_face[3]

                            #apply deep learning for custom_face (filtri e make-up) NON CI INTERESSA

                            custom_face = base_img[y:y+h, x:x+w]


                            #-------------------------------
                            #Analisi e attributi facciali

                            #se c'è almento un modello utilizzato allora vai qui, altrimenti skippa
                            if used_models == True:

                                #Emotion Recognition - LA PARTE CHE SERVE A NOI
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

                                #FINE PARTE CHE SERVE A NOI

                                #ANALISI DELLE ETÀ

                                face_224 = functions.preprocess_face(img = custom_face, target_size = (224, 224), grayscale = False, enforce_detection = False, detector_backend = 'opencv')

                                age_predictions = age_model.predict(face_224)[0,:]
                                apparent_age = Age.findApparentAge(age_predictions)

                                #-------------------------------

                                gender_prediction = gender_model.predict(face_224)[0,:]

                                if np.argmax(gender_prediction) == 0:
                                    gender = "W"
                                elif np.argmax(gender_prediction) == 1:
                                    gender = "M"

                                analysis_report = str(int(apparent_age))+" "+gender
                                
                                #FINE ANALISI DELLE ETÀ
                                #-------------------------------

                            #-------------------------------
                            #face recognition

                            custom_face = functions.preprocess_face(img = custom_face, target_size = (input_shape_y, input_shape_x), enforce_detection = False, detector_backend = 'opencv')

                            #check preprocess_face function handled
                            if custom_face.shape[1:3] == input_shape:
                                #Se ci sono delle immagini nel DB, allora analizza

                                if df.shape[0] > 0: #if there are images to verify, apply face recognition
                                    img1_representation = model.predict(custom_face)[0,:]

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
                                        display_img = cv2.imread(elder_name)

                                        display_img = cv2.resize(display_img, (pivot_img_size, pivot_img_size))

                                        label = elder_name.split("/")[-1].replace(".jpg", "")
                                        label = re.sub('[0-9]', '', label)


                            #Mostra quanto tempo rimane per la visualizzazione dellì'immagine ad output
                            tic = time.time() #in this way, freezed image can show 5 seconds

                            #-------------------------------

                    #Altro relativo al display del tempo rimanente
                    time_left = int(time_threshold - (toc - tic) + 1)

                    cv2.rectangle(freeze_img, (10, 10), (90, 50), (67,67,67), -10)
                    cv2.putText(freeze_img, str(time_left), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

                    cv2.imshow('img', freeze_img)

                    freezed_frame = freezed_frame + 1
                
                #Se la differenza di tempo tra il riconoscimento e il display è minore del threshold che abbiamo attribuito, allora fai il display ( qui non fare nulla, resetta le variabili e passa alla prossima iterazione)
                else:
                    face_detected = False
                    face_included_frames = 0
                    freeze = False
                    freezed_frame = 0

            #Se non si deve fare display, allora mostra l'immagine img
            else:
                cv2.imshow('img',img)

            #premere q per uscire
            if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
                break
    

