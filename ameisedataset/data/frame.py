import zlib
import dill
import numpy as np
from PIL import Image as PilImage
from typing import List, Tuple
from datetime import datetime, timedelta, timezone
from ameisedataset.data import Camera, Lidar
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
        timestamp (str): timestamp of the image as UNIX.
        image (PilImage): The actual image data.
    Methods:
        get_timestamp: Returns the UTC timestamp of the image.
        from_bytes: Class method to create an Image instance from byte data.
    """
    def __init__(self, image=None, timestamp=""):
        self.image: PilImage = image
        self.timestamp: str = timestamp

    def __getattr__(self, attr) -> PilImage:
        """For a direct call of the variable, it returns the image"""
        if hasattr(self.image, attr):
            return getattr(self.image, attr)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def get_timestamp(self, utc=2):
        """ Get the UTC timestamp of the image.
        Args:
            utc (int, optional): Timezone offset in hours. Default is 2.
        Returns:
            str: The UTC timestamp of the image.
        """
        return _convert_unix_to_utc(self.timestamp, utc_offset_hours=utc)

    @classmethod
    def from_bytes(cls, data_bytes: bytes, ts_data: bytes, shape: Tuple[int, int]):
        """ Create an Image instance from byte data.
        Args:
            data_bytes (bytes): Byte data of the image.
            ts_data (bytes): Serialized timestamp data associated with the image.
            shape (Tuple[int, int]): height and width as Tuple.
        Returns:
            Image: An instance of the Image class.
        """
        img_instance = cls()
        img_instance.timestamp = ts_data.decode('utf-8')
        img_instance.image = PilImage.frombytes("RGB", shape, data_bytes)
        return img_instance


class Frame:
    """ Represents a frame containing both images and points.
    Attributes:
        frame_id (int): Unique identifier for the frame.
        timestamp (str): Timestamp associated with the frame.
        cameras (List[Image]): List of images associated with the frame.
        lidar (List[np.array]): List of point data associated with the frame.
    Methods:
        from_bytes: Class method to create a Frame instance from compressed byte data.
    """
    def __init__(self, frame_id: int, timestamp: str):
        self.frame_id: int = frame_id
        self.timestamp: str = timestamp
        self.cameras: List[Image] = [Image()] * NUM_CAMERAS
        self.lidar: List[np.array] = [np.array([])] * NUM_LIDAR

    @classmethod
    def from_bytes(cls, compressed_data, meta_info):
        """ Create a Frame instance from compressed byte data.
        Args:
            compressed_data (bytes): Compressed byte data representing the frame.
            meta_info (Infos): Data type of the points.
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
                ts_len = int.from_bytes(decompressed_data[offset:offset + INT_LENGTH], 'big')
                offset += INT_LENGTH
                ts_data = decompressed_data[offset:offset + ts_len]
                offset += ts_len
                # Create Image instance and store it in the frame instance
                frame_instance.cameras[Camera[info_name.upper()]] = Image.from_bytes(camera_img_bytes, ts_data,
                                                                                     meta_info.cameras[Camera[info_name.upper()]].shape)
            # Check if the info name corresponds to a Lidar type
            elif Lidar.is_type_of(info_name.upper()):
                # Extract points length and data
                pts_len = int.from_bytes(decompressed_data[offset:offset + INT_LENGTH], 'big')
                offset += INT_LENGTH
                laser_pts_bytes = decompressed_data[offset:offset + pts_len]
                offset += pts_len
                # Create Points instance and store it in the frame instance
                # .lidar[Lidar.OS1_TOP].dtype
                frame_instance.lidar[Lidar[info_name.upper()]] = np.frombuffer(laser_pts_bytes,
                                                                               dtype=meta_info.lidar[Lidar[info_name.upper()]].dtype)
        # Return the fully populated frame instance
        return frame_instance
