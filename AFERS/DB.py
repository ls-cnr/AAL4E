import sqlite3
from os.path import exists


class DB:

    # Define where to store the data. If the database does not exist, it gets created and then a connection is made
    def __init__(self, database_path):
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
                print('Failed creating the file')

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
                                picture_location TEXT NOT NULL,
                                weight BLOB,
                                training_variable INTEGER NOT NULL
                            );
                            """)

        # Creation of the table Moods
        self.cursor.execute("""
                            CREATE TABLE Emotion (
                                id INTEGER PRIMARY KEY,
                                FOREIGN KEY(user) REFERENCES User(id),
                                acquisition_time TEXT NOT NULL,
                                mood TEXT NOT NULL,
                                angry REAL NOT NULL,
                                disgust REAL NOT NULL,
                                fear REAL NOT NULL,
                                happiness REAL NOT NULL,
                                sad REAL NOT NULL,
                                surprise REAL NOT NULL,
                                neutral REAL NOT NULL
                            );
                            """)

        # Committing the changes through the connection
        self.connection.commit()

    def get_known_faces(self):
        pass

    def get_user(self, user_id):
        pass

    def commit_emotion(self, current_user, emotion):
        pass
