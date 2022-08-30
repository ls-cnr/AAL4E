from abc import ABC

from AFERS.AbstractUX import AbstractUX


class ConsoleUX(AbstractUX, ABC):

    def set_state(self, state_string):
        print("State: "+state_string)

    def welcomeback_user(self, current_user):
        print("Welcome back: " + current_user.name)

    def emotion_gradient(self, current_user, gradient):
        if gradient<0:
            print("Today I find you worst than usual")
        elif gradient==0:
            print("Today I find you as usual")
        else:
            print("Today I find you better than usual")

    def get_name_surname(self):
        name = input('Write your name (without surname): ')
        surname = input('Write your surname: ')
        return name,surname