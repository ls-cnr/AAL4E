import sys
import numpy
import pandas
import re
import pandas

from deepface import DeepFace
from deepface.commons import functions, distance as Distance
from deepface.detectors import FaceDetector
from deepface.extendedmodels import Age


class AFERS:

    #Initialization of the AFERS objects. Calling important Functions in the Globals class
    def __init__(self, globals_pointer, model_name = 'VGG-Face',detector_backend = 'opencv', distance_metric= 'cosine', used_models = ['Emotion', 'Age', 'Gender']):
    #Object to call the common functions used in this program
        self.global_functions = globals_pointer
        
        #Build the model, if something goes wrong, exit
        self.global_functions.model = DeepFace.build_model(model_name)
        if self.global_functions.model is None:
            sys.exit(1)

        #Build the backend model
        self.global_functions.build_backend_model(detector_backend=detector_backend)


        #Load the database
        self.global_functions.load_database_faces()


        #If there are some faces in our database find the input shape and set a threshold for the defined model name and distance metric
        if self.global_functions.elderly is not None:
            
            self.global_functions.input_shape = functions.find_input_shape(self.global_functions.model)
            self.global_functions.input_shape_x = self.global_functions.input_shape[0]
            self.global_functions.input_shape_y = self.global_functions.input_shape[1]

            self.threshold = Distance.findThreshold(model_name, distance_metric)


        #Load the models for Emotion or Age or Gender if those has been specified in the function parameters
        if used_models is not None:
            if 'Emotion' in used_models:
                self.global_functions.load_emotion_model()

            if 'Age' in used_models:
                self.global_functions.load_age_model()

            if 'Gender' in used_models:
                self.global_functions.load_gender_model()
    

        
        #Start preprocessing the images in the database if it is not empty
        if self.global_functions.elderly is not None:
            self.embeddings = self.global_functions.preprocessing()
            print("faces were preprocessed")
        

    #Function to analyse the informations took from the stream
    def analysis(self, detector_backend = 'opencv', distance_metric = 'cosine', used_models = [], frame_threshold = 5):
        

        #Dictionary that will be returned later with all the information obtained
        result = {}

        #Variable used to exit the following while cycle
        face_detected = False
        face_included_frames = 0

        #While loop that exits only when a face is recognised for exactly <frame_threshold> frames
        while(face_included_frames != frame_threshold):
            #Get the frame and break if it is not possible to get it
            ret, image = self.global_functions.streaming.read()
            if image is None:
                break

            #Make a copy of the image
            image_copy = image.copy()
            
            #Make a list to store the faces detected for the frame
            faces = []

            #Try to pick faces, if not possible make the face list void
            try:
                faces = FaceDetector.detect_faces(face_detector = self.global_functions.backend_model, detector_backend = detector_backend, img = image, align = False)
            except:
                faces = []

            
            #If there are not faces present in this frame, reset the variable that allows us to exit the loop and proceed with the code 
            if len(faces) == 0:
                face_included_frames = 0
                face_detected = False

            #List to store faces and relative index
            detected_faces = []
            face_index = 0

            #If faces are present in the frame...
            for faces, (x, y, w, h) in faces:
                #... and if the width of the face is at least 130 pixels ...
                if w > 130:

                    #... then modify the face_detected flag and increment the frame variable (only if we are looking at the first face of the frame), then append the face on the aforementioned list 
                    face_detected = True
                    if face_index == 0:
                        face_included_frames = face_included_frames + 1
                    
                    detected_faces.append((x, y, w, h))
                    face_index = face_index + 1
            

        #If a face has been detected and it has been detected for <frame_threshold> frames
        if face_detected == True and face_included_frames == frame_threshold:
            
            #Copy the image and the faces detected on the last frame
            base_image = image_copy.copy()
            detected_faces_final = detected_faces.copy()

            #For any face in the frame
            for detected_face in detected_faces_final:
                
                #Get the coordinates and the shape of the face
                x = detected_face[0]; y = detected_face[1]
                w = detected_face[2]; h = detected_face[3]

                custom_face = base_image[y:y+h, x:x+w]

                #Preprocessing of the face
                face_224 = functions.preprocess_face(img= custom_face, target_size= (224, 224), grayscale= False, enforce_detection= False, detector_backend=detector_backend)
                
                #If ancillary modules were loaded, then let them operate on the face
                if used_models is not None:

                    #Emotion Model Scope
                    if 'Emotion' in used_models:
                        #Preprocessing the face
                        gray_img = functions.preprocess_face(img=custom_face, target_size=(48,48), grayscale=True, enforce_detection=False, detector_backend=detector_backend)
                        
                        #Setting the emotion labels
                        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

                        #Make the prediction
                        emotion_predictions = self.global_functions.emotion_model.predict(gray_img)[0,:]

                        sum_of_predictions = emotion_predictions.sum()

                        #List where to store the information about the emotions
                        moods = []

                        for i in range(0, len(emotion_labels)):
                            mood = []
                            emotion_label = emotion_labels[i]

                            #Get a percentage of the prediction
                            emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions

                            #Append the prediction to the list
                            mood.append(emotion_label)
                            mood.append(emotion_prediction)
                            moods.append(mood)

                        #Make a dataframe out of the list and sort it by descending score
                        emotion_df = pandas.DataFrame(moods, columns=['Emotion', 'Score'])
                        emotion_df = emotion_df.sort_values(by=['Score'], ascending=False).reset_index(drop=True)

                        #Add ther result of the model to the dictionary
                        result['Emotions'] = emotion_df

                    #Age Model Scope
                    if 'Age' in used_models:
                        
                        #Make the prediction and find the apparent age
                        age_predictions = self.global_functions.age_model.predict(face_224)[0,:]
                        apparent_age = Age.findApparentAge(age_predictions)

                        #Add ther result of the model to the dictionary
                        result['Age'] = apparent_age

                    #Gender Model Scope
                    if 'Gender' in used_models:
                        
                        #Make the prediction
                        gender_prediction = self.global_functions.gender_model.predict(face_224)[0,:]

                        #The prediction can have values 0 (female face) or 1 (male face)
                        if numpy.argmax(gender_prediction) == 0:
                            gender = 'W'
                        elif numpy.argmax(gender_prediction) == 1:
                            gender = 'M'

                        #Add ther result of the model to the dictionary
                        result['Gender'] = gender

                #Preprocessing the face
                custom_face = functions.preprocess_face(img= custom_face, target_size= (self.global_functions.input_shape_x, self.global_functions.input_shape_y), enforce_detection= False, detector_backend= 'opencv')

                #If the face has a certain face
                if custom_face.shape[1:3] == self.global_functions.input_shape:
                    
                    #If the shape of the embeddings is greater than 0
                    if self.global_functions.embeddings_df.shape[0] > 0:

                        #Predict the identity
                        img1_representation = self.global_functions.model.predict(custom_face)[0,:]

                        #Embedded Function
                        #Gets the distance, relative to the distance metric
                        def findDistance(row):
                            distance_metric = row['distance_metric']
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

                        #Sort the images of the database for whatever has smaller distance
                        embeddings = embeddings.sort_values(by= ['distance'])

                        #Gets the identity of the person
                        candidate = embeddings.iloc[0]
                        elder_name = candidate['elder']
                        best_distance = candidate['distance']


                        if best_distance <= self.threshold:
                            #Gets the name of the person
                            label = elder_name.split("/")[-1].replace(".jpg", "")
                            label = re.sub('[0-9]', '', label)

                            try:
                                #Add ther result of the model to the dictionary
                                result['Person'] =  label
                            except:
                                sys.exit(1)
        
        return result

