from globals import Globals
from afers import AFERS
from database_handling import DataBaseHandler
from PER import PER
import time
import os

#Important Data to store elsewhere during
API = '563492ad6f9170000100000192345faa49b644a28763f37812c770a0'
database_path = os.getcwd() + '/AFERS/DB/'

#Preprocessing
state_variable = 0
person_recognition = None
emotion = None
glob = None
afers = None
dh = None

#Indefinitely cycling loop
while(state_variable != -1):
    if(state_variable == 0):
        #Intialization
        print('Init globals')
        glob = Globals(database_path = database_path, API= API, model='VGG-Face')
        print('Init afers')
        afers = AFERS(globals_pointer=glob, used_models = ['Emotion'])
        print('Init db')
        dh = DataBaseHandler(database_path=database_path)
        state_variable = 1

    elif(state_variable == 1):
        #Idle
        print('State: Idle')
        if(glob.motion_recognition()):
            time.sleep(1.5)
            state_variable = 2

    elif(state_variable == 2):
        #Person Recognition
        print('State: face entered the cam')
        person_recognition = afers.analysis(used_models=[])
        if person_recognition == {}:
            state_variable = 3
        else:
            state_variable = 4

    elif(state_variable == 3):
        #Registration
        print('State: new user/user not recognized')
        glob.registration()
        state_variable = 6

    elif(state_variable == 4):
        #Known Person
        print('State: user recognized')
        glob.TTSInterface(text="Welcome Back." + person_recognition["Name"] + " Hold still in order to catch your emotions", lang='en')
        emotion = AFERS.analysis(used_models=['Emotion'])
        mood = glob.emotion_check(emotion)
        if mood == 'negative':
            state_variable = 5
        else:
            glob.TTSInterface(text=mood)
            moods= emotion['Emotions']
            dh.DBHDetectionCommit(name=emotion['Name'], surname=emotion['Surname'], mood= mood, angry=moods.loc[moods.Emotion == "Angry", 'Score'].values[0], disgust=moods.loc[moods.Emotion == "Disgust", 'Score'].values[0], fear=moods.loc[moods.Emotion == "Fear", 'Score'].values[0], happiness=moods.loc[moods.Emotion == "Happy", 'Score'].values[0], sad=moods.loc[moods.Emotion == "Sad", 'Score'].values[0], surprise=moods.loc[moods.Emotion == "Surprise", 'Score'].values[0], neutral=moods.loc[moods.Emotion == "Neutral", 'Score'].values[0])
            state_variable = 7

    elif(state_variable == 5):
        #PER
        print('State: neutral emotion stimulation')
        glob.TTSInterface("pronto per il PER", lang="it")
        PER(emotion['Name'], emotion['Surname'], globals_reference=glob , afers=AFERS, dh=dh)
        state_variable = 7

    elif(state_variable == 6):
        #Registration Pre-Idle Operations
        print('State: registration pre-idle')
        glob.preprocessing()
        state_variable = 7

    elif(state_variable == 7):
        #Pre-Idle
        print('State: pre-idle')
        if(glob.idle_recognition()):
            state_variable = 1
