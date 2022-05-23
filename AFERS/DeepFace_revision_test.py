from deepface import DeepFace
import os
from newer import AFERS
used_models = ["Emotion", "Age", "Gender"]

db_path = '/AFERS/DB/'

model = AFERS()

model.analysis(used_models= used_models, db_path=db_path)
