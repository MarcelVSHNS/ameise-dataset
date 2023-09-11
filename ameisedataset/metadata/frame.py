import zlib
import dill
import numpy as np
from PIL import Image as PilImage
from typing import List
from datetime import datetime, timedelta, timezone
from ameisedataset.metadata import Camera, Lidar
from ameisedataset.miscellaneous import INT_LENGTH, NUM_CAMERAS, NUM_LIDAR


def _convert_unix_to_utc(unix_timestamp_ns: str, utc_offset_hours: int = 2) -> str:
    """
    Convert a Unix timestamp (in nanoseconds) to a human-readable UTC string with a timezone offset.
    This function also displays milliseconds, microseconds, and nanoseconds.
    Parameters:
    - unix_timestamp_ns: Unix timestamp in nanoseconds as a string.
    - offset_hours: UTC timezone offset in hours.
    Returns:
    - Human-readable UTC string with the given timezone offset and extended precision.
    """
    # Extract the whole seconds and the fractional part
    timestamp_s, fraction_ns = divmod(int(unix_timestamp_ns), int(1e9))
    milliseconds, remainder_ns = divmod(fraction_ns, int(1e6))
    microseconds, nanoseconds = divmod(remainder_ns, int(1e3))
    # Convert to datetime object and apply the offset
    dt = datetime.fromtimestamp(timestamp_s, timezone.utc) + timedelta(hours=utc_offset_hours)
    # Create the formatted string with extended precision
    formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
    extended_precision = f".{milliseconds:03}{microseconds:03}{nanoseconds:03}"
    return formatted_time + extended_precision


class Image:
    """ Represents an image along with its metadata.
    Attributes:
        name (str): Name of the image.
        image (PilImage): The actual image data.
        exif_data (dict): Metadata associated with the image.
            - height (int): Height of the image.
            - width (int): Width of the image.
            - channels (int): Number of channels in the image.
            - encoding (str): Encoding format of the image.
            - timestamp (str): Timestamp associated with the image.
    Methods:
        get_timestamp: Returns the UTC timestamp of the image.
        from_bytes: Class method to create an Image instance from byte data.
    """
    def __init__(self):
        self.name: str = ""
        self.image: PilImage = None
        self.exif_data = {
            "height": 0,
            "width": 0,
            "channels": 0,
            "encoding": "",
            "timestamp": ""
        }

    def get_timestamp(self, utc=2):
        """ Get the UTC timestamp of the image.
        Args:
            utc (int, optional): Timezone offset in hours. Default is 2.
        Returns:
            str: The UTC timestamp of the image.
        """
        return _convert_unix_to_utc(self.exif_data["timestamp"], utc_offset_hours=utc)

    @classmethod
    def from_bytes(cls, name: str, data_bytes: bytes, exif_data):
        """ Create an Image instance from byte data.
        Args:
            name (str): Name of the image.
            data_bytes (bytes): Byte data of the image.
            exif_data (bytes): Serialized exif data associated with the image.
        Returns:
            Image: An instance of the Image class.
        """
        img_instance = cls()
        img_instance.name = name
        img_instance.exif_data = dill.loads(exif_data)
        shape = (img_instance.exif_data["width"], img_instance.exif_data["height"])
        img_instance.image = PilImage.frombytes(img_instance.exif_data["encoding"], shape, data_bytes)
        return img_instance


class Points:
    """ Represents a collection of points.
    Attributes:
        name (str): Name associated with the points.
        points (np.ndarray): An array holding the point data.
    Methods:
        from_bytes: Class method to create a Points instance from byte data.
    """
    def __init__(self):
        self.name: str = ""
        self.points: np.ndarray = np.array([])

    @classmethod
    def from_bytes(cls, name: str, data_bytes: bytes, pts_dtype: np.dtype):
        """ Create a Points instance from byte data.
        Args:
            name (str): Name associated with the points.
            data_bytes (bytes): Byte data representing the points.
            pts_dtype (np.dtype): Data type of the points.
        Returns:
            Points: An instance of the Points class.
        """
        pts_instance = cls()
        pts_instance.name = name
        pts_instance.points = np.frombuffer(data_bytes, dtype=pts_dtype)
        return pts_instance


class Frame:
    """ Represents a frame containing both images and points.
    Attributes:
        frame_id (int): Unique identifier for the frame.
        timestamp (str): Timestamp associated with the frame.
        images (List[Image]): List of images associated with the frame.
        points (List[Points]): List of point data associated with the frame.
    Methods:
        from_bytes: Class method to create a Frame instance from compressed byte data.
    """
    def __init__(self, frame_id: int, timestamp: str):
        self.frame_id: int = frame_id
        self.timestamp: str = timestamp
        self.images: List[Image] = [Image()] * NUM_CAMERAS
        self.points: List[Points] = [Points()] * NUM_LIDAR

    @classmethod
    def from_bytes(cls, compressed_data, pts_dtype):
        """ Create a Frame instance from compressed byte data.
        Args:
            compressed_data (bytes): Compressed byte data representing the frame.
            pts_dtype (np.dtype): Data type of the points.
        Returns:
            Frame: An instance of the Frame class.
        """
        # Decompress the provided data
        decompressed_data = zlib.decompress(compressed_data)
        # Extract frame information length and data
        frame_info_len = int.from_bytes(decompressed_data[:INT_LENGTH], 'big')
        frame_info_bytes = decompressed_data[INT_LENGTH:INT_LENGTH + frame_info_len]
        frame_info = dill.loads(frame_info_bytes)
        frame_instance = cls(frame_info[0], frame_info[1])
        # Initialize offset for further data extraction
        offset = INT_LENGTH + frame_info_len
        for info_name in frame_info[2:]:
            # Check if the info name corresponds to a Camera type
            if Camera.is_type_of(info_name.upper()):
                # Extract image length and data
                img_len = int.from_bytes(decompressed_data[offset:offset + INT_LENGTH], 'big')
                offset += INT_LENGTH
                camera_img_bytes = decompressed_data[offset:offset + img_len]
                offset += img_len
                # Extract Exif data length and data
                exif_len = int.from_bytes(decompressed_data[offset:offset + INT_LENGTH], 'big')
                offset += INT_LENGTH
                exif_data = decompressed_data[offset:offset + exif_len]
                offset += exif_len
                # Create Image instance and store it in the frame instance
                frame_instance.images[Camera[info_name.upper()]] = Image.from_bytes(info_name, camera_img_bytes,
                                                                                    exif_data)
            # Check if the info name corresponds to a Lidar type
            elif Lidar.is_type_of(info_name.upper()):
                # Extract points length and data
                pts_len = int.from_bytes(decompressed_data[offset:offset + INT_LENGTH], 'big')
                offset += INT_LENGTH
                laser_pts_bytes = decompressed_data[offset:offset + pts_len]
                offset += pts_len
                # Create Points instance and store it in the frame instance
                frame_instance.points[Lidar[info_name.upper()]] = Points.from_bytes(info_name, laser_pts_bytes,
                                                                                    pts_dtype)
        # Return the fully populated frame instance
        return frame_instance
