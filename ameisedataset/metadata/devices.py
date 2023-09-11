import hashlib
import dill
import json


def compute_checksum(data):
    # calculates the has value of a given bytestream - SHA256
    return hashlib.sha256(data).digest()


class Image:
    def __init__(self):
        self.name = None
        self.image = None
        self.timestamp = None

    @classmethod
    def from_bytes(cls, name: str, data_bytes: bytes, timestamp: str):
        img_instance = cls()
        img_instance.name = name
        img_instance.image = data_bytes         # TODO: Enter PIL function or equal here
        img_instance.timestamp = timestamp      # TODO: Enter Datetime function or equal here
        return img_instance


class Points:
    def __init__(self):
        self.name = None
        self.points = None

    @classmethod
    def from_bytes(cls, name: str, data_bytes: bytes):
        pts_instance = cls()
        pts_instance.name = name
        pts_instance.points = data_bytes        # TODO: Enter Numpy function or equal here
        return pts_instance


class CameraInformation:
    def __init__(self, name, camera_type, focal_length, aperture, exposure_time):
        self.name = name
        self.height = None
        self.width = None
        self.camera_mtx = None
        self.distortion_mtx = None
        self.rectification_mtx = None
        self.projection_mtx = None
        self.region_of_interest = None
        self.camera_type = camera_type
        self.focal_length = focal_length
        self.aperture = aperture
        self.exposure_time = exposure_time

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
        self.beam_altitude_angles = None
        self.beam_azimuth_angles = None
        self.lidar_origin_to_beam_origin_mm = None
        self.columns_per_frame = None
        self.pixels_per_column = None
        self.phase_lock_offset = None
        self.lidar_to_sensor_transform = None
        self.type = None

    def to_bytes(self) -> bytes:
        info_bytes = dill.dumps(self)
        info_bytes_len = len(info_bytes).to_bytes(4, 'big')
        info_bytes_checksum = compute_checksum(info_bytes)
        return info_bytes_len + info_bytes_checksum + info_bytes

    @classmethod
    def from_bytes(cls, info_data: bytes):
        return dill.loads(info_data)
