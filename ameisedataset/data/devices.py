import hashlib
import dill
import json
from typing import List, Tuple

import numpy as np

from ameisedataset.miscellaneous import compute_checksum, INT_LENGTH, NUM_CAMERAS, NUM_LIDAR


class Infos:
    """ Represents a collection of metadata information about a dataset.
    Attributes:
        filename (str): Name of the dataset file.
        SHA256 (str): SHA256 checksum of the dataset.
        cameras (List[CameraInformation]): List of camera information associated with the dataset.
        lidar (List[LidarInformation]): List of lidar information associated with the dataset.
    """
    def __init__(self, filename: str = ""):
        self.filename: str = filename
        self.SHA256: str = ""
        self.cameras: List[CameraInformation] = [CameraInformation()] * NUM_CAMERAS
        self.lidar: List[LidarInformation] = [LidarInformation()] * NUM_LIDAR

    def get_info_lists(self) -> Tuple[List[int], List[int]]:
        camera_indices = [idx for idx, item in enumerate(self.cameras) if item.shape[0] != 0]
        lidar_indices = [idx for idx, item in enumerate(self.lidar) if item.dtype is not None]
        return camera_indices, lidar_indices


class CameraInformation:
    """ Represents detailed information about a camera.
    Attributes:
        name (str): Name of the camera.
        shape (Tuple[int, int]): Dimensions (width, height) of the camera's image.
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
    def __init__(self, name: str = ''):
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
        self.distortion_type: str = ''
        self.camera_mtx: np.array = np.array([])
        self.distortion_mtx: np.array = np.array([])
        self.rectification_mtx: np.array = np.array([])
        self.projection_mtx: np.array = np.array([])
        self.region_of_interest: ROI = ROI()
        self.camera_type: str = ''
        self.focal_length: int = 0
        self.aperture: int = 0
        self.exposure_time: int = 0
        self.extrinsic: Pose = Pose()   # Transformation to Top_Lidar
        self.stereo_transform: TransformationMtx = TransformationMtx()

    def add_from_ros_cam_info(self, cam_info_msg):
        """ Populate the CameraInformation attributes from a ROS (Roboter Operating System) camera info object.
        Args:
            cam_info_msg: ROS camera info msg.
        """
        self.shape = (cam_info_msg.width, cam_info_msg.height)
        self.camera_mtx = np.array(cam_info_msg.K).reshape(3, 3)
        self.distortion_mtx = np.array(cam_info_msg.D)
        self.rectification_mtx = np.array(cam_info_msg.R).reshape(3, 3)
        self.projection_mtx = np.array(cam_info_msg.P).reshape(3, 4)
        self.region_of_interest = ROI(x_off=cam_info_msg.roi.x_offset,
                                      y_off=cam_info_msg.roi.y_offset,
                                      height=cam_info_msg.roi.height,
                                      width=cam_info_msg.roi.width)

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
        self.name: str = name
        self.dtype = None
        self.beam_altitude_angles = None
        self.beam_azimuth_angles = None
        self.lidar_origin_to_beam_origin_mm = None
        self.columns_per_frame = None
        self.pixels_per_column = None
        self.phase_lock_offset = None
        self.lidar_to_sensor_transform = None
        self.type = None
        self.extrinsic: Pose = Pose()

    def add_from_json_lidar_info(self, laser_info_obj):
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


class Pose:
    def __init__(self):
        self.xyz: np.array = np.array([0, 0, 0])
        self.rpy: np.array = np.array([0, 0, 0])


class TransformationMtx:
    def __init__(self):
        self.rotation: np.array = np.zeros((3, 3))
        self.translation: np.array = np.zeros((1, 3))


class ROI:
    def __init__(self, x_off=0, y_off=0, width=0, height=0):
        self.x_offset = x_off
        self.y_offset = y_off
        self.width = width
        self.height = height

    def __iter__(self):
        return iter((self.x_offset, self.y_offset, self.width, self.height))
