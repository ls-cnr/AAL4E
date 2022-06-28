import random
import time
import os

import cv2
import numpy
import requests
from pprint import pprint

from afers import AFERS
from database_handling import DataBaseHandler
from request_handler import RequestHandler


class PER:

    #Initialization
    def __init__(self, name, surname, globals_reference, afers, dh):
        self.img_fold = os.getcwd() +  '/AFERS/IMGS/'
        self.globals_reference = globals_reference
        self.afers = afers
        #Call to the database handler
        self.dh = dh
        #If the elder exists...
        if self.dh.DBHElderExists(name=name, surname=surname):
            #...save their informations into the class properties
            self.name = name
            self.surname = surname

            '''self.dict, self.variable = self.dh.DBHGetBlobAndVariable(name=name, surname=surname)

            #If the variable is not -1
            if self.variable != -1:
                #we are still in the training phase
                self.training()
            else:
                #else
                self.analysis()'''

            self.analysis()
        #else, there's an error
        else:
            print("The Elder does not exist")

        #close the connection
        self.dh.DBHClose()


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

        #Da inserire metodo per vedere di aggiornare i pesi
        if self.globals_reference.emotion_check(retValue) == 'positive':
            self.dict[tag] = self.dict[tag] + 1
        self.update(self.name, self.surname, self.dict, self.variable + 1 if self.variable + 1< 3*len(self.dict) else -1)


        self.save(self.name, self.surname, retValue)



    def analysis(self):

        '''sums = sum(self.dict.values())
        tag = numpy.random.choice(a=self.dict.keys(), size = 1, p=self.dict.values()/sums)

        if tag.find('_') == -1:
            retValue = self.show_image(tag)
        elif tag.find('_v') != -1:
            retValue = self.show_video(tag.rpartition('_')[0])
        elif tag.find('_a') != -1:
            retValue = self.play_audio(tag.rpartition('_')[0])
        '''
        self.show_image()

        self.save(self.name, self.surname, retValue)


    def show_image(self, tag= 0):
        if tag != 0:
            rh = RequestHandler(API='563492ad6f9170000100000192345faa49b644a28763f37812c770a0')
            rh.pexels_request(media='image', query=tag, page=1, per_page='10', size='large')
            image_ret = random.choice(rh.format_request(media='image'))

            response = requests.get(image_ret['Link'], stream=True).raw
            image = numpy.asarray(bytearray(response.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            cv2.namedWindow(image_ret['Media Name'] + 'by' + image_ret['Creator'], cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(image_ret['Media Name'] + 'by' + image_ret['Creator'], cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow(image_ret['Media Name'] + 'by' + image_ret['Creator'], image)

            res = []
            for x in range(2):
                res.append(self.afers.analysis(used_models = ['Emotion']))
                time.sleep(3.5)

            cv2.destroyAllWindows()
            meanRes = self.dict_mean(res)

            return meanRes
        else:
            files = os.listdir(self.img_fold)
            img_rel_path = random.choice(files)
            img = cv2.imread(self.img_fold + img_rel_path, cv2.IMREAD_UNCHANGED)
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('image', img)
            cv2.waitKey()
            ress = []
            for x in range(2):
                res = self.afers.analysis(used_models = ['Emotion'])
                ress.append(res)
                print("Sleeping, while my guitar gently weeps")
                time.sleep(3.5)
            cv2.destroyAllWindows()
            
            meanRes = self.dict_mean(ress)

            return meanRes
            pass


    #Update the weights and the variable
    def update(self, name, surname, dictionary, variable):
        self.dh.DBHUpdateBlobAndVariable(name=name, surname=surname, df= dictionary, variable=variable)



    #Save the results on the emotion analysis just obtained
    def save(self, name, surname, dict):
        mood = self.globals_reference.emotion_check(dict)
        self.dh.DBHDetectionCommit(name=name, surname=surname, mood = mood, angry=dict.get['Angry'], disgust=dict.get['Disgust'], fear=dict.get['fear'], happiness=dict.get['happiness'], sad=dict.get['sad'], surprise=dict.get['surprise'], neutral=dict.get['neutral'])

    def dict_mean(self, ddl):
        #Number of dictiionaries in input to compute the mean
        pprint(ddl)
        input()
        n = len(ddl)
        
        dtfr = ddl[0]['Emotions']
        input(dtfr)
        result = {}
        for line in dtfr:
            result[dtfr[line]] = 0
            
        for i in range(n):
            tempdtfr = ddl[i]['Emotions']
            for line in tempdtfr:
                result[tempdtfr[line]] = result[tempdtfr[line]] + line['Score']
        
        
        for row in result:
            result[row] = result[row] / n
            
        pprint(result)
        pass
        return result

#For Future Releases

    def show_video(self, tag):
        pass


    def play_audio(self, tag):
        pass
