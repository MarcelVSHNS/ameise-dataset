import hashlib
import dill
import json


def compute_checksum(data):
    # calculates the has value of a given bytestream - SHA256
    return hashlib.sha256(data).digest()


class Infos:
    def __init__(self, filename):
        self.filename = filename
        self.SHA256 = None
        self.camera = [None] * 5
        self.lidar = [None] * 3


class CameraInformation:
    def __init__(self, name, camera_type, focal_length, aperture, exposure_time):
        self.name = name
        self.shape = None
        self.camera_mtx = None
        self.distortion_mtx = None
        self.rectification_mtx = None
        self.projection_mtx = None
        self.region_of_interest = None
        self.camera_type = camera_type
        self.focal_length = focal_length
        self.aperture = aperture
        self.exposure_time = exposure_time

    def add_from_ros_cam_info(self, camera_info_obj):
        self.shape = (camera_info_obj.height, camera_info_obj.width)
        self.camera_mtx = camera_info_obj.cam_matrix
        self.distortion_mtx = camera_info_obj.dist_coeff
        self.rectification_mtx = camera_info_obj.rect_matrix
        self.projection_mtx = camera_info_obj.proj_matrix
        # self.region_of_interest = camera_info_obj.roi

    def to_bytes(self) -> bytes:
        info_bytes = dill.dumps(self)
        info_bytes_len = len(info_bytes).to_bytes(4, 'big')
        info_bytes_checksum = compute_checksum(info_bytes)
        return info_bytes_len + info_bytes_checksum + info_bytes

    @classmethod
    def from_bytes(cls, info_data: bytes):
        return dill.loads(info_data)


class LidarInformation:
    def __init__(self, name):
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
        info_bytes = dill.dumps(self)
        info_bytes_len = len(info_bytes).to_bytes(4, 'big')
        info_bytes_checksum = compute_checksum(info_bytes)
        return info_bytes_len + info_bytes_checksum + info_bytes

    @classmethod
    def from_bytes(cls, info_data: bytes):
        return dill.loads(info_data)
