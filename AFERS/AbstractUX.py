from abc import abstractmethod


class AbstractUX:

    @abstractmethod
    def set_state(self, state_string):
        pass

    @abstractmethod
    def welcomeback_user(self, current_user):
        pass

    @abstractmethod
    def emotion_gradient(self, current_user, gradient):
        pass

    @abstractmethod
    def get_name_surname(self):
        pass
