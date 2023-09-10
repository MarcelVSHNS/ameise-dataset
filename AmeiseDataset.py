import dill
import hashlib
import zlib
import json
import numpy as np

from metadata.devices import CameraInformation, Image, LidarInformation, Points
from metadata.names import Camera, Lidar


def compute_checksum(data):
    # calculates the has value of a given bytestream - SHA256
    return hashlib.sha256(data).digest()


def unpack_record(filename) -> []:
    frames = []
    infos = []
    with open(filename, 'rb') as file:
        # 1. Read the length of the header
        header_length = int.from_bytes(file.read(4), 'big')
        # 2. Deserialize the header to get the order of info objects
        header_bytes = file.read(header_length)
        info_names = dill.loads(header_bytes)
        # 3. Read info objects based on the order in header
        for name in info_names:
            info_length = int.from_bytes(file.read(4), 'big')
            info_checksum = file.read(32)  # 32 Bytes für SHA-256
            info_bytes = file.read(info_length)
            # Checksum
            if compute_checksum(info_bytes) != info_checksum:
                raise ValueError(f"Checksum of {name} is not correct! Check file.")
            # Deserialisiere das Info-Objekt
            if Camera.is_type_of(name.upper()):
                infos.append(CameraInformation.from_bytes(info_bytes))
            elif Lidar.is_type_of(name.upper()):
                infos.append(LidarInformation.from_bytes(info_bytes))

        # 4. Read the total length of frames/ payload e.g. 146
        num_frames = int.from_bytes(file.read(4), 'big')
        # 5. Read frame from e.g. 0 to 146
        for _ in range(num_frames):
            # Extract the length of the compressed data
            compressed_data_len = int.from_bytes(file.read(4), 'big')
            # Extract the checksum
            compressed_data_checksum = file.read(32)  # Assuming SHA-256 for checksum (32 bytes)
            # Extract the compressed data
            compressed_data = file.read(compressed_data_len)
            # Verify checksum
            if compute_checksum(compressed_data) != compressed_data_checksum:
                raise ValueError("Checksum mismatch. Data might be corrupted!")
            frames.append(Frame.from_bytes(compressed_data))
    return infos, frames


class Frame:
    def __init__(self, frame_id, timestamp):
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.images = [None] * 5
        self.points = [None] * 3

    @classmethod
    def from_bytes(cls, compressed_data):
        # Decompress the data
        decompressed_data = zlib.decompress(compressed_data)

        frame_info_len = int.from_bytes(decompressed_data[:4], 'big')
        frame_info_bytes = decompressed_data[4:4 + frame_info_len]
        frame_info = dill.loads(frame_info_bytes)
        frame_instance = cls(frame_info[0], frame_info[1])

        offset = 4 + frame_info_len
        for info_name in frame_info[2:]:
            if Camera.is_type_of(info_name.upper()):
                img_len = int.from_bytes(decompressed_data[offset:offset + 4], 'big')
                offset += 4
                camera_img_bytes = np.frombuffer(decompressed_data[offset:offset + img_len], dtype=np.uint8)
                offset += img_len
                time_len = int.from_bytes(decompressed_data[offset:offset + 4], 'big')
                offset += 4
                time = decompressed_data[offset:offset + time_len].decode('utf-8')
                offset += time_len
                frame_instance.images[Camera[info_name.upper()]] = Image.from_bytes(info_name, camera_img_bytes, time)
            elif Lidar.is_type_of(info_name.upper()):
                pts_len = int.from_bytes(decompressed_data[offset:offset + 4], 'big')
                offset += 4
                laser_pts_bytes = np.frombuffer(decompressed_data[offset:offset + pts_len])
                offset += pts_len
                frame_instance.points[Lidar[info_name.upper()]] = Points.from_bytes(info_name, laser_pts_bytes)
        # return an instance of the class
        return frame_instance
