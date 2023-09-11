import hashlib
import dill
import json
from typing import List, Tuple

from ameisedataset.miscellaneous import compute_checksum, INT_LENGTH, NUM_CAMERAS, NUM_LIDAR


class Infos:
    """ Represents a collection of metadata information about a dataset.
    Attributes:
        filename (str): Name of the dataset file.
        SHA256 (str): SHA256 checksum of the dataset.
        camera (List[CameraInformation]): List of camera information associated with the dataset.
        lidar (List[LidarInformation]): List of lidar information associated with the dataset.
    """
    def __init__(self, filename: str):
        self.filename: str = filename
        self.SHA256: str = ""
        self.camera: List[CameraInformation] = [CameraInformation()] * NUM_CAMERAS
        self.lidar: List[LidarInformation] = [LidarInformation()] * NUM_LIDAR


class CameraInformation:
    """ Represents detailed information about a camera.
    Attributes:
        name (str): Name of the camera.
        shape (Tuple[int, int]): Dimensions (height, width) of the camera's image.
        camera_mtx: Camera matrix.
        distortion_mtx: Distortion matrix.
        rectification_mtx: Rectification matrix.
        projection_mtx: Projection matrix.
        region_of_interest: Region of interest in the camera's view.
        camera_type (str): Type of the camera.
        focal_length (int): Focal length of the camera in millimeters.
        aperture (int): Aperture size of the camera.
        exposure_time (int): Exposure time of the camera in milliseconds.
    """
    def __init__(self, name: str = "", camera_type: str = "", focal_length: int = 0, aperture: int = 0, exposure_time: int = 0):
        """ Initialize a CameraInformation instance with the specified attributes.
        Args:
            name (str): Name of the camera.
            camera_type (str): Type of the camera.
            focal_length (int): Focal length of the camera.
            aperture (int): Aperture size of the camera.
            exposure_time (int): Exposure time of the camera.
        """
        self.name: str = name
        self.shape: Tuple[int, int] = (0, 0)
        self.camera_mtx = None
        self.distortion_mtx = None
        self.rectification_mtx = None
        self.projection_mtx = None
        self.region_of_interest = None
        self.camera_type: str = camera_type
        self.focal_length: int = focal_length
        self.aperture: int = aperture
        self.exposure_time: int = exposure_time

    def add_from_ros_cam_info(self, camera_info_obj):
        """ Populate the CameraInformation attributes from a ROS (Roboter Operating System) camera info object.
        Args:
            camera_info_obj: ROS camera info msg.
        """
        self.shape = (camera_info_obj.height, camera_info_obj.width)
        self.camera_mtx = camera_info_obj.cam_matrix
        self.distortion_mtx = camera_info_obj.dist_coeff
        self.rectification_mtx = camera_info_obj.rect_matrix
        self.projection_mtx = camera_info_obj.proj_matrix
        # self.region_of_interest = camera_info_obj.roi

    def to_bytes(self) -> bytes:
        """ Serialize the CameraInformation instance to bytes.
        Returns:
            bytes: Serialized byte representation of the CameraInformation instance.
        """
        info_bytes = dill.dumps(self)
        info_bytes_len = len(info_bytes).to_bytes(INT_LENGTH, 'big')
        info_bytes_checksum = compute_checksum(info_bytes)
        return info_bytes_len + info_bytes_checksum + info_bytes

    @classmethod
    def from_bytes(cls, info_data: bytes):
        """ Create a CameraInformation instance from byte data.
        Args:
            info_data (bytes): Serialized byte representation of a CameraInformation instance.
        Returns:
            CameraInformation: A CameraInformation instance populated with the provided byte data.
        """
        return dill.loads(info_data)


class LidarInformation:
    """ Represents detailed information about a LiDAR sensor.
    Attributes:
        name (str): Name of the LiDAR sensor.
        dtype: Data type of the LiDAR points.
        beam_altitude_angles: Altitude angles of the LiDAR beams.
        beam_azimuth_angles: Azimuth angles of the LiDAR beams.
        lidar_origin_to_beam_origin_mm: Distance from the LiDAR origin to the origin of the beams.
        columns_per_frame: Number of columns in each LiDAR frame.
        pixels_per_column: Number of pixels in each LiDAR column.
        phase_lock_offset: Phase lock offset of the LiDAR sensor.
        lidar_to_sensor_transform: Transformation matrix from the LiDAR to the sensor.
        type: Product line or type of the LiDAR sensor.
    """
    def __init__(self, name=""):
        """ Initialize a LidarInformation instance with a given name.
        Args:
            name (str): Name of the LiDAR sensor.
        """
        self.name = name
        self.dtype = None
        self.beam_altitude_angles = None
        self.beam_azimuth_angles = None
        self.lidar_origin_to_beam_origin_mm = None
        self.columns_per_frame = None
        self.pixels_per_column = None
        self.phase_lock_offset = None
        self.lidar_to_sensor_transform = None
        self.type = None

    def add_from_ros_lidar_info(self, laser_info_obj):
        """ Populate the LidarInformation attributes from a ROS (Roboter Operating System) LiDAR info object.
        Args:
            laser_info_obj: ROS LiDAR info object.
        """
        data_dict = json.loads(laser_info_obj.data)
        self.beam_altitude_angles = data_dict["beam_intrinsics"]["beam_altitude_angles"]
        self.beam_azimuth_angles = data_dict["beam_intrinsics"]["beam_azimuth_angles"]
        self.lidar_origin_to_beam_origin_mm = data_dict["beam_intrinsics"]["lidar_origin_to_beam_origin_mm"]
        self.columns_per_frame = data_dict["lidar_data_format"]["columns_per_frame"]
        self.pixels_per_column = data_dict["lidar_data_format"]["pixels_per_column"]
        self.phase_lock_offset = data_dict["config_params"]["phase_lock_offset"]
        self.lidar_to_sensor_transform = data_dict["lidar_intrinsics"]["lidar_to_sensor_transform"]
        self.type = data_dict["sensor_info"]["prod_line"]

    def to_bytes(self) -> bytes:
        """ Serialize the LidarInformation instance to bytes.
        Returns:
            bytes: Serialized byte representation of the LidarInformation instance.
        """
        info_bytes = dill.dumps(self)
        info_bytes_len = len(info_bytes).to_bytes(INT_LENGTH, 'big')
        info_bytes_checksum = compute_checksum(info_bytes)
        return info_bytes_len + info_bytes_checksum + info_bytes

    @classmethod
    def from_bytes(cls, info_data: bytes):
        """ Create a LidarInformation instance from byte data.
        Args:
            info_data (bytes): Serialized byte representation of a LidarInformation instance.
        Returns:
            LidarInformation: A LidarInformation instance populated with the provided byte data.
        """
        return dill.loads(info_data)
