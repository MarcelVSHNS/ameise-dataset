import zlib
import dill
import numpy as np
from ameisedataset.metadata import Camera, Lidar, Image, Points


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