import os
import time

import myProcedures as Proc
from DB import DB
from CameraWrapper import CameraWrapper
from ConsoleUX import ConsoleUX

# initialize
app_path = os.getcwd() + '/AFERS'

state = 2

db = DB(app_path+'/DB/')
cam = CameraWrapper(app_path+'/models/')
ux = ConsoleUX()
current_user = None

while state != -1:

    try:
        if state == 1:
            ux.set_state("idle")
            something_appeared = cam.detect_face() #cam.detect_movement()
            if something_appeared:
                state = 2

        # elif state == 10:
        #     image_path = app_path+'/DB/profiles/images/brad_pitt.jpeg'
        #     print(image_path)
        #     image_repr = cam.calculate_representation(image_path)
        #     db.register_user("brad","pitt",image_repr)
        #     state = -1


        elif state == 2:
            state = -1  #TO REMOVE
            ux.set_state("face has entered")
            known_faces = db.get_known_faces()
            #print(known_faces)

            identity = cam.recognize_face_optimized(known_faces)
            print("user_id: "+str(identity))
            # if len(identities) == 0:
            #     state = 1
            # elif len(identities) == 1:
            #     user_id = identities[0]
            #     current_user = db.get_user(user_id)
            #     state = 4
            # else:
            #     time.sleep(2.0)

        elif state == 3:
            ux.set_state("face not recognized")
            Proc.propose_registration(ux, cam)
            state = 6

        elif state == 4:
            ux.set_state("face recognized")
            ux.welcomeback_user(current_user)
            if current_user.need_recognition:
                state = 5
            else:
                state = 6

        elif state == 5:
            ux.set_state("proactive emotion recognition")
            emotion = Proc.proactive_emotion_detection()
            if emotion > 0:
                db.commit_emotion(current_user, emotion)
                gradient = db.evaluate_history(current_user)
                ux.emotion_gradient(current_user,gradient)
                state = 6
            elif emotion == -1:
                state = 6

        elif state == 6:
            ux.set_state("pre idle")
            idle = Proc.check_idle()
            if idle:
                state = 1

    except:
        cam.release()

    time.sleep(0.5)

# terminate
cam.release()