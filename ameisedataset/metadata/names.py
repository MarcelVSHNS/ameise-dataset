class InfoBaseClass(type):
    def __getitem__(cls, key):
        return getattr(cls, key)


class Camera(metaclass=InfoBaseClass):
    MONO_LEFT = 0
    STEREO_LEFT = 1
    STEREO_RIGHT = 2
    MONO_RIGHT = 3

    @classmethod
    def is_type_of(cls, value):
        return value in cls.__dict__


class Lidar(metaclass=InfoBaseClass):
    OS0_LEFT = 0
    OS1_TOP = 1
    OS0_RIGHT = 2

    @classmethod
    def is_type_of(cls, value):
        return value in cls.__dict__
