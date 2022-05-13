import pandas

db = pandas.DataFrame(["Name", "Surname", "Age", "Gender"])

class Registrator:
    def registration(name, surname, age, gender):
        db.append([name, surname, age, gender])

    def face_registration(image):
        #salvare l'immagine come nome (se esiste già una entry con lo stesso nome, mettere anche il cognome)
        return