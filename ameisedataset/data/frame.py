import dill
from decimal import Decimal
import numpy as np
from PIL import Image as PilImage
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timedelta, timezone
from ameisedataset.data import Camera, Lidar, IMU
from ameisedataset.miscellaneous import INT_LENGTH, NUM_CAMERAS, NUM_LIDAR, NUM_IMU, compute_checksum
from enum import Enum


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


def _read_data_block(data, offset):
    data_len = int.from_bytes(data[offset:offset + INT_LENGTH], 'big')
    offset += INT_LENGTH
    data_bytes = data[offset:offset + data_len]
    offset += data_len
    return data_bytes, offset


class Serializable:

    @staticmethod
    def to_bytes(obj) -> bytes:
        obj_bytes = dill.dumps(obj)
        obj_bytes_len = len(obj_bytes).to_bytes(INT_LENGTH, 'big')
        return obj_bytes_len + obj_bytes

    @staticmethod
    def from_bytes(data: bytes):
        return dill.loads(data)


class Motion:
    def __init__(self,
                 timestamp: Optional[Decimal] = None,
                 orientation: Optional[np.array] = None,
                 orientation_covariance: Optional[np.array] = None,
                 angular_velocity: Optional[np.array] = None,
                 angular_velocity_covariance: Optional[np.array] = None,
                 linear_acceleration: Optional[np.array] = None,
                 linear_acceleration_covariance: Optional[np.array] = None):
        self.timestamp = timestamp
        self.orientation = orientation
        self.orientation_covariance = orientation_covariance
        self.angular_velocity = angular_velocity
        self.angular_velocity_covariance = angular_velocity_covariance
        self.linear_acceleration = linear_acceleration
        self.linear_acceleration_covariance = linear_acceleration_covariance


class Position:
    class NavSatFixStatus(Enum):
        NO_FIX = -1
        FIX = 0
        SBAS_FIX = 1
        GBAS_FIX = 2

    class CovarianceType(Enum):
        UNKNOWN = 0
        APPROXIMATED = 1
        DIAGONAL_KNOWN = 2
        KNOWN = 3

    def __init__(self, timestamp: Optional[Decimal] = None, status: Optional[NavSatFixStatus] = None,
                 services: Optional[Dict[str, Optional[bool]]] = None, latitude: Optional[Decimal] = None,
                 longitude: Optional[Decimal] = None, altitude: Optional[Decimal] = None,
                 covariance: Optional[np.array] = None, covariance_type: Optional[CovarianceType] = None):
        self.timestamp = timestamp
        self.status = status
        self.services = self.init_services(services)
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.covariance = covariance
        self.covariance_type = covariance_type

    def __iter__(self):
        return iter((self.latitude, self.longitude, self.timestamp))

    @staticmethod
    def init_services(services):
        default_services = {'GPS': None, 'Glonass': None, 'Galileo': None, 'Baidou': None}
        if services is None:
            return default_services
        for key in default_services:
            services.setdefault(key, default_services[key])
        return services


# TODO: Umbenennen in Lidar/Camera/GNSS sonst passt logic nicht - cameras variable hält 4 image objekte sollte aber 4 camera objekte halten die jeweils unter anderem images enthalten
class Image:
    """
    Represents an image along with its metadata.
    Attributes:
        timestamp (Optional[Decimal]): Timestamp of the image as UNIX time, can be None.
        image (Optional[PilImage]): The actual image data, can be None.
    """

    def __init__(self, image: PilImage = None, timestamp: Optional[Decimal] = None):
        """
        Initializes the Image object with the provided image data and timestamp.
        Parameters:
            image (Optional[PilImage]): The actual image data. Defaults to None.
            timestamp (Optional[Decimal]): Timestamp of the image as UNIX time. Defaults to None.
        """
        self.image = image
        self.timestamp = timestamp

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

    def to_bytes(self) -> bytes:
        encoded_img = self.image.tobytes()
        encoded_ts = str(self.timestamp).encode('utf-8')
        img_len = len(encoded_img).to_bytes(4, 'big')
        ts_len = len(encoded_ts).to_bytes(4, 'big')
        image_bytes = img_len + encoded_img + ts_len + encoded_ts
        return image_bytes

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

    def __init__(self, points: Optional[np.array] = None, timestamp: Optional[Decimal] = None):
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

    def to_bytes(self) -> bytes:
        encoded_pts = self.points.tobytes()
        encoded_ts = str(self.timestamp).encode('utf-8')
        pts_len = len(encoded_pts).to_bytes(4, 'big')
        ts_len = len(encoded_ts).to_bytes(4, 'big')
        laser_bytes = pts_len + encoded_pts + ts_len + encoded_ts
        return laser_bytes

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
        self.imu: List[List[Motion]] = [[Motion()]] * NUM_IMU
        self.gnss: Position = Position()

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
        # [self.frame_id, self.timestamp]
        frame_instance = cls(frame_info[0], frame_info[1])
        # Initialize offset for further data extraction
        offset = INT_LENGTH + frame_info_len
        for info_name in frame_info[2:]:
            # Check if the info name corresponds to a Camera type
            if Camera.is_type_of(info_name.upper()):
                # Extract image length and data
                camera_img_bytes, offset = _read_data_block(data, offset)
                # Extract timestamp
                ts_img_bytes, offset = _read_data_block(data, offset)
                # Create Image instance and store it in the frame instance
                frame_instance.cameras[Camera[info_name.upper()]] = Image.from_bytes(camera_img_bytes, ts_img_bytes,
                                                                                     meta_info.cameras[Camera[
                                                                                         info_name.upper()]].shape)
            # Check if the info name corresponds to a Lidar type
            elif Lidar.is_type_of(info_name.upper()):
                # Extract points length and data
                laser_pts_bytes, offset = _read_data_block(data, offset)
                # extract timestamp
                ts_laser_bytes, offset = _read_data_block(data, offset)
                # Create Points instance and store it in the frame instance
                # .lidar[Lidar.OS1_TOP].dtype
                frame_instance.lidar[Lidar[info_name.upper()]] = Points.from_bytes(laser_pts_bytes, ts_laser_bytes,
                                                                                   dtype=meta_info.lidar[
                                                                                       Lidar[info_name.upper()]].dtype)
            elif IMU.is_type_of(info_name.upper()):
                imu_bytes, offset = _read_data_block(data, offset)
                frame_instance.imu[IMU[info_name.upper()]] = Serializable.from_bytes(imu_bytes)
            elif info_name == 'GNSS':
                gnss_bytes, offset = _read_data_block(data, offset)
                frame_instance.gnss = Serializable.from_bytes(gnss_bytes)
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
        imu_bytes = b""
        camera_indices, lidar_indices, imu_indices, gnss_available = self.get_data_lists()
        frame_info = [self.frame_id, self.timestamp]
        for data_index in camera_indices:
            frame_info.append(Camera.get_name_by_value(data_index))
        for data_index in lidar_indices:
            frame_info.append(Lidar.get_name_by_value(data_index))
        for data_index in imu_indices:
            frame_info.append(IMU.get_name_by_value(data_index))
        if gnss_available:
            frame_info.append("GNSS")
            gnss_bytes = Serializable.to_bytes(self.gnss)
        else:
            gnss_bytes = b""
        frame_info_bytes = dill.dumps(frame_info)
        frame_info_len = len(frame_info_bytes).to_bytes(4, 'big')
        # Encode images together with their time
        cam_msgs_to_write = [self.cameras[idx] for idx in camera_indices]
        for img_obj in cam_msgs_to_write:
            image_bytes += img_obj.to_bytes()
        # Encode laser points
        lidar_msgs_to_write = [self.lidar[idx] for idx in lidar_indices]
        for laser in lidar_msgs_to_write:
            laser_bytes += laser.to_bytes()
        imu_msgs_to_write = [self.imu[idx] for idx in imu_indices]
        for imu in imu_msgs_to_write:
            imu_bytes += Serializable.to_bytes(imu)

        # pack bytebuffer all together and compress them to one package
        combined_data = frame_info_len + frame_info_bytes + image_bytes + laser_bytes + imu_bytes + gnss_bytes
        # compressed_data = combined_data  #zlib.compress(combined_data)  # compress if something is compressable
        # calculate length and checksum
        combined_data_len = len(combined_data).to_bytes(4, 'big')
        combined_data_checksum = compute_checksum(combined_data)
        # return a header with the length and byteorder
        return combined_data_len + combined_data_checksum + combined_data

    def get_data_lists(self) -> Tuple[List[int], List[int], bool]:
        """
        Retrieves indices of cameras and lidars based on specific conditions.
        Returns:
            Tuple[List[int], List[int]]:
                - First list contains indices of cameras with non-null images.
                - Second list contains indices of lidar data with non-zero size.
        """
        camera_indices = [idx for idx, image in enumerate(self.cameras) if image.image is not None]
        lidar_indices = [idx for idx, points in enumerate(self.lidar) if points.points is not None]
        imu_indices = [idx for idx, imu in enumerate(self.imu) if imu[-1].timestamp is not None]
        gnss_available = False if self.gnss.status is None else True
        return camera_indices, lidar_indices, imu_indices, gnss_available
