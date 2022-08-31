import sqlite3
from os.path import exists


class DB:

    # Define where to store the data. If the database does not exist, it gets created and then a connection is made
    def __init__(self, database_path):
        self.path = database_path
        # Where to store our data
        file = database_path + 'user_profiles.db'

        # If the file does not exist, ...
        if not exists(file):
            try:
                # ...try creating it
                open(file, 'a').close()

                # Connect to the DB
                self.connection = sqlite3.connect(file)

                # Create a cursor
                self.cursor = self.connection.cursor()

                # Initialize the database by creating the tables
                self.create_database()

            # if the creation process encounters a problem, print the following error
            except OSError:
                print('Failed creating the file: ' + file)

        # If the file esists, just establish a connection with it
        else:
            # Connect to the DB
            self.connection = sqlite3.connect(file)

            # Create a cursor
            self.cursor = self.connection.cursor()

    # Definitions of the queries to create the tables of our database with their relations
    def create_database(self):

        # Creation of the table Elders
        self.cursor.execute("""
                            CREATE TABLE User (
                                id INTEGER PRIMARY KEY, 
                                name TEXT NOT NULL,
                                surname TEXT NOT NULL,
                                image_representation TEXT NOT NULL
                            );
                            """)

        # Creation of the table Moods
        self.cursor.execute("""
                            CREATE TABLE Emotion (
                                id INTEGER PRIMARY KEY,
                                user INTEGER,
                                acquisition_time TEXT NOT NULL,
                                mood TEXT NOT NULL,
                                angry REAL NOT NULL,
                                disgust REAL NOT NULL,
                                fear REAL NOT NULL,
                                happiness REAL NOT NULL,
                                sad REAL NOT NULL,
                                surprise REAL NOT NULL,
                                neutral REAL NOT NULL,
                                FOREIGN KEY(user) REFERENCES User(id)
                            );
                            """)

        # Committing the changes through the connection
        self.connection.commit()

    def get_known_faces(self):
        dict = {}
        self.cursor.execute("""
                            SELECT id, name,surname,image_representation FROM User
                            """)
        try:
            row = self.cursor.fetchone()
            while row:
                id = str(row[0])
                #name = str(row[1])
                #surname = str(row[2])
                image_rep = self.decode_representation(str(row[3]))
                dict[id]=image_rep
                row = self.cursor.fetchone()
        except:
            print("db error")

        return dict

    def get_user(self, user_id):
        pass

    def commit_emotion(self, current_user, emotion):
        pass

    def register_user(self, name, surname, image_representation):
        image_representation_str = self.encode_representation(image_representation)
        print(image_representation)
        self.cursor.execute("INSERT INTO User (name, surname, image_representation) VALUES (?, ?, ?)",
                            (name, surname, image_representation_str))
        self.connection.commit()


    def encode_representation(self,list_of_float):
        image_representation_str = ""
        for x in list_of_float:
            image_representation_str = image_representation_str + str(x) + " "
        return image_representation_str

    def decode_representation(self,encoded_list_of_float):
        floats_list = []
        for item in encoded_list_of_float.split():
            floats_list.append(float(item))

        return floats_list