import dill
from decimal import Decimal
import numpy as np
from PIL import Image as PilImage
from typing import List, Tuple
from datetime import datetime, timedelta, timezone
from ameisedataset.data import Camera, Lidar
from ameisedataset.miscellaneous import INT_LENGTH, NUM_CAMERAS, NUM_LIDAR, compute_checksum


def _convert_unix_to_utc(unix_timestamp_ns: Decimal, utc_offset_hours: int = 2) -> str:
    """
    Convert a Unix timestamp (in nanoseconds as Decimal) to a human-readable UTC string with a timezone offset.
    This function also displays milliseconds, microseconds, and nanoseconds.
    Parameters:
    - unix_timestamp_ns: Unix timestamp in nanoseconds as a Decimal.
    - offset_hours: UTC timezone offset in hours.
    Returns:
    - Human-readable UTC string with the given timezone offset and extended precision.
    """
    # Convert the Decimal to integer for calculations
    unix_timestamp_ns = int(unix_timestamp_ns)
    # Extract the whole seconds and the fractional part
    timestamp_s, fraction_ns = divmod(unix_timestamp_ns, int(1e9))
    milliseconds, remainder_ns = divmod(fraction_ns, int(1e6))
    microseconds, nanoseconds = divmod(remainder_ns, int(1e3))
    # Convert to datetime object and apply the offset
    dt = datetime.fromtimestamp(timestamp_s, timezone.utc) + timedelta(hours=utc_offset_hours)
    # Create the formatted string with extended precision
    formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
    extended_precision = f".{milliseconds:03}{microseconds:03}{nanoseconds:03}"
    return formatted_time + extended_precision


class Image:
    """
    Represents an image along with its metadata.
    Attributes:
        timestamp (str): Timestamp of the image as UNIX.
        image (PilImage): The actual image data.
    """
    def __init__(self, image: PilImage = None, timestamp: Decimal = '0'):
        """
        Initializes the Image object with the provided image data and timestamp.
        Parameters:
            image (PilImage, optional): The actual image data. Defaults to None.
            timestamp (Decimal, optional): Timestamp of the image as UNIX. Defaults to '0'.
        """
        self.image: PilImage = image
        self.timestamp: Decimal = timestamp

    def __getattr__(self, attr) -> PilImage:
        """
        Enables direct access to attributes of the `image` object.
        Parameters:
            attr (str): Name of the attribute to access.
        Returns:
            PilImage: Attribute value if it exists in the `image` object.
        Raises:
            AttributeError: If the attribute does not exist.
        """
        if hasattr(self.image, attr):
            return getattr(self.image, attr)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    @classmethod
    def from_bytes(cls, data_bytes: bytes, ts_data: bytes, shape: Tuple[int, int]):
        """
        Creates an Image instance from byte data.
        Args:
            data_bytes (bytes): Byte data of the image.
            ts_data (bytes): Serialized timestamp data associated with the image.
            shape (Tuple[int, int]): Dimensions of the image as (width, height).
        Returns:
            Image: An instance of the Image class populated with the provided data.
        """
        img_instance = cls()
        img_instance.timestamp = Decimal(ts_data.decode('utf-8'))
        img_instance.image = PilImage.frombytes("RGB", shape, data_bytes)
        return img_instance

    def get_timestamp(self, utc=2):
        """
        Retrieves the UTC timestamp of the points.
        Args:
            utc (int, optional): Timezone offset in hours. Default is 2.
        Returns:
            str: The UTC timestamp of the points.
        """
        return _convert_unix_to_utc(self.timestamp, utc_offset_hours=utc)


class Points:
    """
    Represents a collection of points with an associated timestamp.
    Attributes:
        points (np.array): Array containing the points.
        timestamp (Decimal): Timestamp associated with the points.
    """
    def __init__(self, points: np.array = np.array([]), timestamp: Decimal = '0'):
        """
        Initializes the Points object with the provided points and timestamp.
        Parameters:
            points (np.array, optional): Array containing the points. Defaults to an empty array.
            timestamp (Decimal, optional): Timestamp associated with the points. Defaults to '0'.
        """
        self.points: np.array = points
        self.timestamp: Decimal = timestamp

    def __getattr__(self, attr) -> np.array:
        """
        Enables direct access to attributes of the `points` object.
        Parameters:
            attr (str): Name of the attribute to access.
        Returns:
            np.array: Attribute value if it exists.
        Raises:
            AttributeError: If the attribute does not exist.
        """
        if hasattr(self.points, attr):
            return getattr(self.points, attr)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    @classmethod
    def from_bytes(cls, data_bytes: bytes, ts_data: bytes, dtype: np.dtype):
        """
        Creates a Points instance from byte data.
        Parameters:
            data_bytes (bytes): Byte data representing the points.
            ts_data (bytes): Byte data representing the timestamp.
            dtype (np.dtype): Data type of the points.
        Returns:
            Points: A Points instance initialized with the provided data.
        """
        img_instance = cls()
        img_instance.timestamp = Decimal(ts_data.decode('utf-8'))
        img_instance.points = np.frombuffer(data_bytes, dtype=dtype)
        return img_instance

    def get_timestamp(self, utc=2):
        """
        Retrieves the UTC timestamp of the points.
        Args:
            utc (int, optional): Timezone offset in hours. Default is 2.
        Returns:
            str: The UTC timestamp of the points.
        """
        return _convert_unix_to_utc(self.timestamp, utc_offset_hours=utc)


class Frame:
    """
    Represents a frame containing both images and points.
    Attributes:
        frame_id (int): Unique identifier for the frame.
        timestamp (str): Timestamp associated with the frame.
        cameras (List[Image]): List of images associated with the frame.
        lidar (List[Points]): List of point data associated with the frame.
    """
    def __init__(self, frame_id: int, timestamp: Decimal):
        """
        Initializes the Frame object with the provided frame ID and timestamp.
        Sets default values for cameras and lidar attributes.
        Parameters:
            frame_id (int): Unique identifier for the frame.
            timestamp (Decimal): Timestamp associated with the frame.
        """
        self.frame_id: int = frame_id
        self.timestamp: Decimal = timestamp
        self.cameras: List[Image] = [Image()] * NUM_CAMERAS
        self.lidar: List[Points] = [Points()] * NUM_LIDAR

    @classmethod
    def from_bytes(cls, data, meta_info):
        """
        Creates a Frame instance from compressed byte data.
        Args:
            data (bytes): Compressed byte data representing the frame.
            meta_info (Infos): Metadata information about the frame's data types.
        Returns:
            Frame: An instance of the Frame class.
        """
        # Extract frame information length and data
        frame_info_len = int.from_bytes(data[:INT_LENGTH], 'big')
        frame_info_bytes = data[INT_LENGTH:INT_LENGTH + frame_info_len]
        frame_info = dill.loads(frame_info_bytes)
        frame_instance = cls(frame_info[0], frame_info[1])
        # Initialize offset for further data extraction
        offset = INT_LENGTH + frame_info_len
        for info_name in frame_info[2:]:
            # Check if the info name corresponds to a Camera type
            if Camera.is_type_of(info_name.upper()):
                # Extract image length and data
                img_len = int.from_bytes(data[offset:offset + INT_LENGTH], 'big')
                offset += INT_LENGTH
                camera_img_bytes = data[offset:offset + img_len]
                offset += img_len
                # Extract timestamp
                ts_len = int.from_bytes(data[offset:offset + INT_LENGTH], 'big')
                offset += INT_LENGTH
                ts_data = data[offset:offset + ts_len]
                offset += ts_len
                # Create Image instance and store it in the frame instance
                frame_instance.cameras[Camera[info_name.upper()]] = Image.from_bytes(camera_img_bytes, ts_data,
                                                                                     meta_info.cameras[Camera[info_name.upper()]].shape)
            # Check if the info name corresponds to a Lidar type
            elif Lidar.is_type_of(info_name.upper()):
                # Extract points length and data
                pts_len = int.from_bytes(data[offset:offset + INT_LENGTH], 'big')
                offset += INT_LENGTH
                laser_pts_bytes = data[offset:offset + pts_len]
                offset += pts_len
                # extract timestamp
                ts_len = int.from_bytes(data[offset:offset + INT_LENGTH], 'big')
                offset += INT_LENGTH
                ts_data = data[offset:offset + ts_len]
                # Create Points instance and store it in the frame instance
                # .lidar[Lidar.OS1_TOP].dtype
                frame_instance.lidar[Lidar[info_name.upper()]] = Points.from_bytes(laser_pts_bytes, ts_data,
                                                                                   dtype=meta_info.lidar[Lidar[info_name.upper()]].dtype)
        # Return the fully populated frame instance
        return frame_instance

    def to_bytes(self):
        """
        Converts the Frame instance to compressed byte data.
        Returns:
            bytes: Compressed byte representation of the Frame.
        """
        # convert data to bytes
        image_bytes = b""
        laser_bytes = b""
        camera_indices, lidar_indices = self.get_data_lists()
        frame_info = [self.frame_id, self.timestamp]
        for data_index in camera_indices:
            frame_info.append(Camera.get_name_by_value(data_index))
        for data_index in lidar_indices:
            frame_info.append(Lidar.get_name_by_value(data_index))
        frame_info_bytes = dill.dumps(frame_info)
        frame_info_len = len(frame_info_bytes).to_bytes(4, 'big')
        # Encode images together with their time
        cam_msgs_to_write = [self.cameras[idx] for idx in camera_indices]
        for img_obj in cam_msgs_to_write:
            encoded_img = img_obj.image.tobytes()
            encoded_ts = str(img_obj.timestamp).encode('utf-8')
            img_len = len(encoded_img).to_bytes(4, 'big')
            ts_len = len(encoded_ts).to_bytes(4, 'big')
            image_bytes += img_len + encoded_img + ts_len + encoded_ts
        # Encode laser points
        lidar_msgs_to_write = [self.lidar[idx] for idx in lidar_indices]
        for laser in lidar_msgs_to_write:
            encoded_pts = laser.points.tobytes()
            encoded_ts = str(laser.timestamp).encode('utf-8')
            pts_len = len(encoded_pts).to_bytes(4, 'big')
            ts_len = len(encoded_ts).to_bytes(4, 'big')
            laser_bytes += pts_len + encoded_pts + ts_len + encoded_ts
        # pack bytebuffer all together and compress them to one package
        combined_data = frame_info_len + frame_info_bytes + image_bytes + laser_bytes
        # compressed_data = combined_data  #zlib.compress(combined_data)  # compress if something is compressable
        # calculate length and checksum
        compressed_data_len = len(combined_data).to_bytes(4, 'big')
        compressed_data_checksum = compute_checksum(combined_data)
        # return a header with the length and byteorder
        return compressed_data_len + compressed_data_checksum + combined_data

    def get_data_lists(self) -> Tuple[List[int], List[int]]:
        """
        Retrieves indices of cameras and lidars based on specific conditions.
        Returns:
            Tuple[List[int], List[int]]:
                - First list contains indices of cameras with non-null images.
                - Second list contains indices of lidar data with non-zero size.
        """
        camera_indices = [idx for idx, image in enumerate(self.cameras) if image.image is not None]
        lidar_indices = [idx for idx, array in enumerate(self.lidar) if array.size != 0]
        return camera_indices, lidar_indices
