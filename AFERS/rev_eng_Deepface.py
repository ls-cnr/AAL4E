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
from deepface.extendedmodels import Age, Gender, Race, Emotion

class AFERS:

    def __init__(self) -> None:
        pass

    def stream(db_path = '', model_name ='VGG-Face', detector_backend = 'opencv', distance_metric = 'cosine', used_models = (), source = 0, time_threshold = 5, frame_threshold = 5):

        """
        This function applies real time face recognition and facial attribute analysis

        Parameters:
            db_path (string): facial database path. You should store some .jpg files in this folder.

            model_name (string): VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib or Ensemble

            detector_backend (string): opencv, ssd, mtcnn, dlib, retinaface

            distance_metric (string): cosine, euclidean, euclidean_l2

            enable_facial_analysis (boolean): Set this to False to just run face recognition

            source: Set this to 0 for access web cam. Otherwise, pass exact video path.

            time_threshold (int): how many second analyzed image will be displayed

            frame_threshold (int): how many frames required to focus on face

        """
        AFERS.analysis(db_path, model_name, detector_backend, distance_metric, used_models
                            , source = source, time_threshold = time_threshold, frame_threshold = frame_threshold)

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

        #-----------------------

        pivot_img_size = 112 #face recognition result image

        #-----------------------

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

                    #DISPLAY MOVING RECTANGLE OUTPUT
                    cv2.rectangle(img, (x,y), (x+w,y+h), (67,67,67), 1) #draw rectangle to main image

                    cv2.putText(img, str(frame_threshold - face_included_frames), (int(x+w/4),int(y+h/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2)

                    detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
                    #DISPLAY MOVING RECTANGLE OUTPUT END
                    #-------------------------------------

                    detected_faces.append((x,y,w,h))
                    face_index = face_index + 1

                    #-------------------------------------

            #Se le facce sono riconosciute per 5 frame consecutivi
            if face_detected == True and face_included_frames == frame_threshold and freeze == False:
                freeze = True
                #base_img = img.copy()
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

                            #si disegna un rettangolo sulla faccia
                            cv2.rectangle(freeze_img, (x,y), (x+w,y+h), (67,67,67), 1) #draw rectangle to main image

                            #-------------------------------

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
                                
                                #INIZIO DISPLAY DELLE EMOZIONI
                                #background of mood box

                                #transparency
                                overlay = freeze_img.copy()
                                opacity = 0.4

                                if x+w+pivot_img_size < resolution_x:
                                    #right
                                    cv2.rectangle(freeze_img
                                        #, (x+w,y+20)
                                        , (x+w,y)
                                        , (x+w+pivot_img_size, y+h)
                                        , (64,64,64),cv2.FILLED)

                                    cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

                                elif x-pivot_img_size > 0:
                                    #left
                                    cv2.rectangle(freeze_img
                                        #, (x-pivot_img_size,y+20)
                                        , (x-pivot_img_size,y)
                                        , (x, y+h)
                                        , (64,64,64),cv2.FILLED)

                                    cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

                                for index, instance in emotion_df.iterrows():
                                    emotion_label = "%s " % (instance['emotion'])
                                    emotion_score = instance['score']/100

                                    bar_x = 35 #this is the size if an emotion is 100%
                                    bar_x = int(bar_x * emotion_score)

                                    if x+w+pivot_img_size < resolution_x:

                                        text_location_y = y + 20 + (index+1) * 20
                                        text_location_x = x+w

                                        if text_location_y < y + h:
                                            cv2.putText(freeze_img, emotion_label, (text_location_x, text_location_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                                            cv2.rectangle(freeze_img
                                                , (x+w+70, y + 13 + (index+1) * 20)
                                                , (x+w+70+bar_x, y + 13 + (index+1) * 20 + 5)
                                                , (255,255,255), cv2.FILLED)

                                    elif x-pivot_img_size > 0:

                                        text_location_y = y + 20 + (index+1) * 20
                                        text_location_x = x-pivot_img_size

                                        if text_location_y <= y+h:
                                            cv2.putText(freeze_img, emotion_label, (text_location_x, text_location_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                                            cv2.rectangle(freeze_img
                                                , (x-pivot_img_size+70, y + 13 + (index+1) * 20)
                                                , (x-pivot_img_size+70+bar_x, y + 13 + (index+1) * 20 + 5)
                                                , (255,255,255), cv2.FILLED)

                                #FINE DISPLAY DELLE EMOZIONI
                                #-------------------------------


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


                                #INIZIO DISPLAY ANALISI DELLE ETÀ
                                info_box_color = (46,200,255)

                                #top
                                if y - pivot_img_size + int(pivot_img_size/5) > 0:

                                    triangle_coordinates = np.array( [
                                        (x+int(w/2), y)
                                        , (x+int(w/2)-int(w/10), y-int(pivot_img_size/3))
                                        , (x+int(w/2)+int(w/10), y-int(pivot_img_size/3))
                                    ] )

                                    cv2.drawContours(freeze_img, [triangle_coordinates], 0, info_box_color, -1)

                                    cv2.rectangle(freeze_img, (x+int(w/5), y-pivot_img_size+int(pivot_img_size/5)), (x+w-int(w/5), y-int(pivot_img_size/3)), info_box_color, cv2.FILLED)

                                    cv2.putText(freeze_img, analysis_report, (x+int(w/3.5), y - int(pivot_img_size/2.1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 111, 255), 2)

                                #bottom
                                elif y + h + pivot_img_size - int(pivot_img_size/5) < resolution_y:

                                    triangle_coordinates = np.array( [
                                        (x+int(w/2), y+h)
                                        , (x+int(w/2)-int(w/10), y+h+int(pivot_img_size/3))
                                        , (x+int(w/2)+int(w/10), y+h+int(pivot_img_size/3))
                                    ] )

                                    cv2.drawContours(freeze_img, [triangle_coordinates], 0, info_box_color, -1)

                                    cv2.rectangle(freeze_img, (x+int(w/5), y + h + int(pivot_img_size/3)), (x+w-int(w/5), y+h+pivot_img_size-int(pivot_img_size/5)), info_box_color, cv2.FILLED)

                                    cv2.putText(freeze_img, analysis_report, (x+int(w/3.5), y + h + int(pivot_img_size/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 111, 255), 2)


                                #FINE RICONOSCIMENTO DELLE ETÀ

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

                                        #Prova a stampare in output, altrimenti nulla
                                        try:
                                            #CASI RELATIVI ALLA POSIZIONE DELLA FACCIA
                                            if y - pivot_img_size > 0 and x + w + pivot_img_size < resolution_x:
                                                #top right
                                                freeze_img[y - pivot_img_size:y, x+w:x+w+pivot_img_size] = display_img

                                                overlay = freeze_img.copy(); opacity = 0.4
                                                cv2.rectangle(freeze_img,(x+w,y),(x+w+pivot_img_size, y+20),(46,200,255),cv2.FILLED)
                                                cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

                                                #cv2.putText(freeze_img, label, (x+w, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

                                                #connect face and text
                                                cv2.line(freeze_img,(x+int(w/2), y), (x+3*int(w/4), y-int(pivot_img_size/2)),(67,67,67),1)
                                                cv2.line(freeze_img, (x+3*int(w/4), y-int(pivot_img_size/2)), (x+w, y - int(pivot_img_size/2)), (67,67,67),1)

                                            elif y + h + pivot_img_size < resolution_y and x - pivot_img_size > 0:
                                                #bottom left
                                                freeze_img[y+h:y+h+pivot_img_size, x-pivot_img_size:x] = display_img

                                                overlay = freeze_img.copy(); opacity = 0.4
                                                cv2.rectangle(freeze_img,(x-pivot_img_size,y+h-20),(x, y+h),(46,200,255),cv2.FILLED)
                                                cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

                                                #cv2.putText(freeze_img, label, (x - pivot_img_size, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

                                                #connect face and text
                                                cv2.line(freeze_img,(x+int(w/2), y+h), (x+int(w/2)-int(w/4), y+h+int(pivot_img_size/2)),(67,67,67),1)
                                                cv2.line(freeze_img, (x+int(w/2)-int(w/4), y+h+int(pivot_img_size/2)), (x, y+h+int(pivot_img_size/2)), (67,67,67),1)

                                            elif y - pivot_img_size > 0 and x - pivot_img_size > 0:
                                                #top left
                                                freeze_img[y-pivot_img_size:y, x-pivot_img_size:x] = display_img

                                                overlay = freeze_img.copy(); opacity = 0.4
                                                cv2.rectangle(freeze_img,(x- pivot_img_size,y),(x, y+20),(46,200,255),cv2.FILLED)
                                                cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

                                                #cv2.putText(freeze_img, label, (x - pivot_img_size, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

                                                #connect face and text
                                                cv2.line(freeze_img,(x+int(w/2), y), (x+int(w/2)-int(w/4), y-int(pivot_img_size/2)),(67,67,67),1)
                                                cv2.line(freeze_img, (x+int(w/2)-int(w/4), y-int(pivot_img_size/2)), (x, y - int(pivot_img_size/2)), (67,67,67),1)

                                            elif x+w+pivot_img_size < resolution_x and y + h + pivot_img_size < resolution_y:
                                                #bottom righ
                                                freeze_img[y+h:y+h+pivot_img_size, x+w:x+w+pivot_img_size] = display_img

                                                overlay = freeze_img.copy(); opacity = 0.4
                                                cv2.rectangle(freeze_img,(x+w,y+h-20),(x+w+pivot_img_size, y+h),(46,200,255),cv2.FILLED)
                                                cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

                                                #cv2.putText(freeze_img, label, (x+w, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

                                                #connect face and text
                                                cv2.line(freeze_img,(x+int(w/2), y+h), (x+int(w/2)+int(w/4), y+h+int(pivot_img_size/2)),(67,67,67),1)
                                                cv2.line(freeze_img, (x+int(w/2)+int(w/4), y+h+int(pivot_img_size/2)), (x+w, y+h+int(pivot_img_size/2)), (67,67,67),1)
                                            #FINE CASI RELATIVI ALLA POSIZIONE DELLA FACCIA
                                        except Exception as err:
                                            print(str(err))


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