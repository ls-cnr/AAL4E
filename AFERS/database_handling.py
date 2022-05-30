import sqlite3
import os

class DatabaseHandler:

    def __init__(self):
        #Where to store our database
        file = os.getcwd() + '/AFERS/DB/elder.db'

        try:
            #Try creating the Database
            open(file, 'a').close()
        except OSError:
            print('Failed creating the file')

        #Connect to the DB
        self.connection = sqlite3.connect(file)

        #Create a cursor
        self.cursor = self.connection.cursor()


    def DBHFirstInit(self):
        #Table creation
        self.cursor.execute("""
                            CREATE TABLE Elders (
                                name TEXT NOT NULL,
                                surname TEXT NOT NULL,
                                picture_location TEXT NOT NULL,
                                PRIMARY KEY (name, surname)
                            );
                            """)

        self.cursor.execute("""
                            CREATE TABLE Moods (
                                elder INTEGER NOT NULL,
                                acquisition_time TEXT NOT NULL,
                                mood TEXT NOT NULL,
                                angry REAL NOT NULL,
                                disgust REAL NOT NULL,
                                fear REAL NOT NULL,
                                happiness REAL NOT NULL,
                                sad REAL NOT NULL,
                                surprise REAL NOT NULL,
                                neutral REAL NOT NULL,
                                FOREIGN KEY(elder) REFERENCES elders(id),
                                PRIMARY KEY(elder, acquisition_time)
                            )
                            """)

        #Committing
        self.connection.commit()


    def DBHElderlyCommit(self, name, surname, picture):
        self.cursor.execute("INSERT INTO Elders (name, surname, picture_location) VALUES ('{}' , '{}', '{}')".format(name.capitalize(), surname.capitalize(), picture))

        self.connection.commit()


    def DBHRecognitionCommit(self, name, surname, acquisitionTime, mood, angry, disgust, fear, happiness, sad, surprise, neutral):
        elder = self.DBHGetProgressiveID(name, surname)
        self.cursor.execute("INSERT INTO Moods VALUES ({} , '{}', '{}', {}, {}, {}, {}, {}, {},{})".format(elder, acquisitionTime, mood, angry, disgust, fear, happiness, sad, surprise, neutral))

        self.connection.commit()


    def DBHGetProgressiveID(self, name,surname):
        self.cursor.execute("""
                            SELECT rowid FROM Elders
                            WHERE name = "{}" AND surname = "{}"
                        """.format(name.capitalize(), surname.capitalize()))
        
        
        return self.cursor.fetchall()[0][0]


    def DBHGetPicture(self, name, surname):
        self.cursor.execute("""
                            SELECT picture_location FROM Elders
                            WHERE name = "{}" AND surname = "{}"
                        """.format(name.capitalize(), surname.capitalize()))
        
        
        return self.cursor.fetchall()[0][0]


    def DBHGetLastAcquisition(self, name, surname):
        id = self.DBHGetProgressiveID(name, surname)
        self.cursor.execute(""" SELECT acquisition_time, mood, angry, disgust, fear, happiness, sad, surprise, neutral FROM Moods
                                WHERE elder == {}
                                ORDER BY acquisition_time DESC
                            """.format(id))
        
        return self.cursor.fetchone()

    
    def DBHGetLastNAcquisition(self, name, surname, n):
        id = self.DBHGetProgressiveID(name, surname)
        self.cursor.execute(""" SELECT acquisition_time, mood, angry, disgust, fear, happiness, sad, surprise, neutral FROM Moods
                                WHERE elder == {}
                                ORDER BY acquisition_time DESC
                            """.format(id))
        
        return self.cursor.fetchmany(size=n)


    def DBHClose(self):
        #Close connection
        self.connection.close()


d = DatabaseHandler()
d.DBHFirstInit()
d.DBHElderlyCommit("Pierfra","Me", "ciao")
d.DBHRecognitionCommit("Pierfra", "me", "Mai", "Hulk", 5,4,3,2,1,0,8)
d.DBHRecognitionCommit("Pierfra", "me", "Mai più", "Hulk", 5,4,3,2,1,0,8)
d.DBHRecognitionCommit("Pierfra", "me", "er", "Hulk", 5,4,3,2,1,0,8)
d.DBHRecognitionCommit("Pierfra", "me", "kjhgf", "Hulk", 5,4,3,2,1,0,8)
d.DBHRecognitionCommit("Pierfra", "me", "sdfu", "Hulk", 5,4,3,2,1,0,8)
d.DBHRecognitionCommit("Pierfra", "me", "jhgf", "Hulk", 5,4,3,2,1,0,8)
d.DBHRecognitionCommit("Pierfra", "me", "oiuygf", "Hulk", 5,4,3,2,1,0,8)
d.DBHRecognitionCommit("Pierfra", "me", "pojhg", "Hulk", 5,4,3,2,1,0,8)
d.DBHRecognitionCommit("Pierfra", "me", "wertyui", "Hulk", 5,4,3,2,1,0,8)

ret = d.DBHGetLastNAcquisition("Pierfra", "me", 8)
for row in ret:
    print(row)
d.DBHClose()