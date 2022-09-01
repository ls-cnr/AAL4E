import os
import sys
import time

import myProcedures as Proc
from User import UserProfile
from DB import DB
from CameraWrapper import CameraWrapper
from ConsoleUX import ConsoleUX

# config
enable_registration = False
show_cam_captures = True
debug_verbosity = False

# initialize
app_path = os.getcwd() + '/AFERS'

ux = ConsoleUX(debug_verbosity)
db = DB(app_path + '/DB/',ux)
cam = CameraWrapper(app_path + '/models/',show_cam_captures,ux)

current_user = UserProfile(-1, "null", "null", [])
state = 1

while state != -1:

    try:
        if state == 1:
            ux.set_state("idle")
            someone_appeared = cam.face_detection()
            if someone_appeared:
                state = 2

        elif state == 2:
            ux.set_state("face has entered")
            known_faces = db.get_known_faces()
            identity = cam.face_recognition(known_faces)

            if identity is None:
                state = 3
            elif identity is not None:
                current_user = db.get_user(int(identity))
                state = 4

        elif state == 3:
            ux.set_state("face not recognized")
            if enable_registration:
                Proc.propose_registration(ux, cam)
            state = 6

        elif state == 4:
            ux.set_state("face recognized")
            ux.welcomeback_user(current_user)
            state = 6
            # if current_user.need_recognition:
            #     state = 5
            # else:
            #     state = 6

        elif state == 5:
            ux.set_state("proactive emotion recognition")
            emotion = Proc.proactive_emotion_detection()
            if emotion > 0:
                db.commit_emotion(current_user, emotion)
                gradient = db.evaluate_history(current_user)
                ux.emotion_gradient(current_user, gradient)
                state = 6
            elif emotion == -1:
                state = 6

        elif state == 6:
            ux.set_state("pre idle")
            idle = cam.face_detection() == 0 #Proc.check_idle()
            if idle:
                state = 1

        # TO REMOVE code used to manually add entries into the database
        # elif state == 10:
        #     image_path = app_path+'/DB/profiles/images/brad_pitt.jpeg'
        #     print(image_path)
        #     image_repr = cam.calculate_representation(image_path)
        #     db.register_user("brad","pitt",image_repr)
        #     state = -1

        time.sleep(0.5)

    except Exception as e:
        print(e)
        cam.release()
        sys.exit(e)

# terminate
cam.release()
