from globals import Globals
from afers import AFERS
from database_handling import DataBaseHandler
from PER import PER

#Important Data to store elsewhere during 
API = '563492ad6f9170000100000192345faa49b644a28763f37812c770a0'
database_path = '/home/pierfrancesco/AAL4E/AFERS/DB/'

#Preprocessing
glob = Globals(database_path = database_path, API= API, model='VGG-Face')
input(type(glob))
AFERS = AFERS(globals_pointer=glob, used_models = ['Emotion'])
dh = DataBaseHandler(database_path=database_path)
#Program's main core

#Indefinitely cycling loop
while(True):

    #If a motion is recognised
    if glob.motion_recognition() == 1:

        #Call the function in order to catch a face
        person_recognition = AFERS.analysis(used_models=['Emotion'])
        #If a person is not recognised...
        if person_recognition == {}:
            #...start the registration
            glob.registration()

        #else, if a person is recognised...
        else:
            #... start by saluting the user
            glob.TTSInterface(text="Welcome Back. Hold still in order to catch your emotions", lang='en')

            #catch their emotions
            emotion = AFERS.analysis(used_models=['Emotion'])
            mood = glob.emotion_check(emotion)
            
            #If the emotions are positive...
            if mood == 'positive':
                #...do something
                pass
            #Else, if the emotions are neutral...
            elif mood == 'neutral':
                #...do something
                pass
            #Else, if the emotions are negative...
            elif mood == 'negative':
                #...start Proactive Emotion Recognition
                PER(emotion[1]['Name'], emotion[1]['Surname'], glob)
            else:
                input("Something is wrong, I can feel it")

            #Wait until an Idle scene is recognised
            glob.idle_recognition()
            pass