

class UserProfile:

    def __init__(self, user_id, name, surname, image_repr):
        self.user_id = user_id
        self.name = name
        self.surname = surname
        self.image_repr = image_repr
        self.need_recognition = False
