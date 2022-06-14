import random
import cv2
import numpy
import requests

from request_handler import RequestHandler
from globals import Globals
from afers import AFERS
from database_handling import DataBaseHandler


class PER:

    #Initialization
    def __init__(self, name, surname, globals_reference):

        self.globals_reference = globals_reference
        #Call to the database handler
        re = DataBaseHandler()
        #If the elder exists...
        if re.DBHElderExists(name=name, surname=surname):
            #...save their informations into the class properties
            self.name = name
            self.surname = surname
            self.dict, self.variable = re.DBHGetBlobAndVariable(name=name, surname=surname)
            
            #If the variable is not -1
            if self.variable != -1:
                #we are still in the training phase
                self.training()
            else:
                #else
                self.analysis()
        #else, there's an error
        else:
            print("The Elder does not exist")

        #close the connection
        re.DBHClose()


#To finish writing

    #Training method
    def training(self):
        iteration = self.variable + 1
        tag = self.dict(iteration % (len(self.dict)))

        if tag.find('_') == -1:
            retValue = self.show_image(tag)
        elif tag.find('_v') != -1:
            retValue = self.show_video(tag.rpartition('_')[0])
        elif tag.find('_a') != -1:
            retValue = self.play_audio(tag.rpartition('_')[0])

        
        pass
    


    def analysis(self):

        sums = sum(self.dict.values())
        tag = numpy.random.choice(a=self.dict.keys(), size = 1, p=self.dict.values()/sums)

        if tag.find('_') == -1:
            retValue = self.show_image(tag)
        elif tag.find('_v') != -1:
            retValue = self.show_video(tag.rpartition('_')[0])
        elif tag.find('_a') != -1:
            retValue = self.play_audio(tag.rpartition('_')[0])

        
        pass


    def show_image(tag):

        rh = RequestHandler(API='563492ad6f9170000100000192345faa49b644a28763f37812c770a0')
        rh.pexels_request(media='image', query=tag, page=1, per_page='10', size='large')
        image_ret = random.choice(rh.format_request(media='image'))

        response = requests.get(image_ret['Link'], stream=True).raw
        image = numpy.asarray(bytearray(response.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Image", image)

        pass




#To write


    #Update the weights and the variable
    def update(self, name, surname, dictionary, variable):
        up = DataBaseHandler()
        up.DBHUpdateBlobAndVariable(name=name, surname=surname, df= dictionary, variable=variable)
        
        up.DBHClose()


    #Save the results on the emotion analysis just obtained
    def save(self, name, surname, dict):
        sa = DataBaseHandler()
        sa.DBHDetectionCommit(name=name, surname=surname, mood = mood, angry=dict.get['Angry'], disgust=dict.get['Disgust'], fear=dict.get['fear'], happiness=dict.get['happiness'], sad=dict.get['sad'], surprise=dict.get['surprise'], neutral=dict.get['neutral'])
        
        sa.DBHClose()


#For Future Releases

    def show_video(self, tag):
        pass


    def play_audio(self, tag):
        pass


