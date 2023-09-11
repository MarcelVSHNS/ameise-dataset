class InfoBaseClass(type):
    """ Base metaclass to provide dictionary-like access to class attributes.
    Allows retrieval of class attributes using dictionary key access style.
    """
    def __getitem__(cls, key):
        """ Retrieve class attribute value using key access.
        Args:
            key (str): Attribute key.
        Returns:
            Any: Value of the specified class attribute.
        """
        return getattr(cls, key)


class Camera(metaclass=InfoBaseClass):
    """ Defines constants representing different camera types.
    Attributes:
        MONO_LEFT (int): Represents a mono left camera.
        STEREO_LEFT (int): Represents a stereo left camera.
        STEREO_RIGHT (int): Represents a stereo right camera.
        MONO_RIGHT (int): Represents a mono right camera.
    """
    MONO_LEFT = 0
    STEREO_LEFT = 1
    STEREO_RIGHT = 2
    MONO_RIGHT = 3

    @classmethod
    def is_type_of(cls, value):
        """ Check if the provided value corresponds to a defined camera type.
        Args:
            value (Any): Value to be checked.
        Returns:
            bool: True if the value is a defined camera type, False otherwise.
        """
        return value in cls.__dict__


class Lidar(metaclass=InfoBaseClass):
    """ Defines constants representing different Lidar types.
    Attributes:
        OS0_LEFT (int): Represents a OS0 left Lidar.
        OS1_TOP (int): Represents a OS1 top Lidar.
        OS0_RIGHT (int): Represents a OS0 right Lidar.
    """
    OS0_LEFT = 0
    OS1_TOP = 1
    OS0_RIGHT = 2

    @classmethod
    def is_type_of(cls, value):
        """
        Check if the provided value corresponds to a defined lidar type.
        Args:
            value (Any): Value to be checked.
        Returns:
            bool: True if the value is a defined Lidar type, False otherwise.
        """
        return value in cls.__dict__
