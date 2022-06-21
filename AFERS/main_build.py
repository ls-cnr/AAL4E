from globals import Globals
from afers import AFERS
from database_handling import DataBaseHandler
from PER import PER
import time

#Important Data to store elsewhere during 
API = '563492ad6f9170000100000192345faa49b644a28763f37812c770a0'
database_path = '/home/pi/AAL4E/AFERS/DB/'

#Preprocessing
preprocessing_time = time.process_time()
glob = Globals(database_path = database_path, API= API, model='VGG-Face')
AFERS = AFERS(globals_pointer=glob, used_models = ['Emotion'])
dh = DataBaseHandler(database_path=database_path)
#Program's main core
print("Preprocessing done in " + str(time.process_time() - preprocessing_time) + " seconds")
#Indefinitely cycling loop
while(True):

    #If a motion is recognised
    if glob.motion_recognition() == 1:
        time.sleep(1)
        #Call the function in order to catch a face
        person_recognition = AFERS.analysis(used_models=[])
        #If a person is not recognised...
        if person_recognition == {}:
            #...start the registration
            glob.registration()
            glob.preprocessing()

        #else, if a person is recognised...
        else:
            #... start by saluting the user
            glob.TTSInterface(text="Welcome Back." + person_recognition["Name"] + " Hold still in order to catch your emotions", lang='en')
            #catch their emotions
            emotion = AFERS.analysis(used_models=['Emotion'])
            mood = glob.emotion_check(emotion)
            #If the emotions are positive...
            if mood == 'positive':
                #...do something
                glob.TTSInterface(text="positive")
                dh.DBHDetectionCommit(name=emotion['Name'], surname=emotion['Surname'], mood= mood, angry=(emotion['Emotions'])['Angry'], disgust=(emotion['Emotions'])['Disgust'], fear=(emotion['Emotions'])['Fear'], happiness=(emotion['Emotions'])['Happiness'], sad=(emotion['Emotions'])['Sad'], surprise=(emotion['Emotions'])['Surprise'], neutral=(emotion['Emotions'])['Neutral'])
                pass
            #Else, if the emotions are neutral...
            elif mood == 'neutral':
                #...do something
                glob.TTSInterface(text="neutral")
                dh.DBHDetectionCommit(name=emotion['Name'], surname=emotion['Surname'], mood= mood, angry=(emotion['Emotions'])['Angry'], disgust=(emotion['Emotions'])['Disgust'], fear=(emotion['Emotions'])['Fear'], happiness=(emotion['Emotions'])['Happiness'], sad=(emotion['Emotions'])['Sad'], surprise=(emotion['Emotions'])['Surprise'], neutral=(emotion['Emotions'])['Neutral'])
                pass
            #Else, if the emotions are negative...
            elif mood == 'negative':
                #...start Proactive Emotion Recognition
                glob.TTSInterface("pronto per il PER", lang="it")
                #ADD INDENTATION HERE AFTER YOU REMOVE YOUR COMMENTED IF STATEMENTS
                PER(emotion['Name'], emotion['Surname'], globals_reference=glob , afers=AFERS)
            else:
                input("Something is wrong, I can feel it")

            #Wait until an Idle scene is recognised
            glob.idle_recognition()
            pass
