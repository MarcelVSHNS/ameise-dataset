import zlib
import dill
import numpy as np
from PIL import Image as PilImage
from io import BytesIO
from datetime import datetime, timedelta, timezone
from ameisedataset.metadata import Camera, Lidar


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
    def __init__(self):
        self.name = None
        self.image = None
        self.exif_data = {
            "height": None,
            "width": None,
            "channels": None,
            "encoding": None,
            "timestamp": None
        }

    def get_timestamp(self, utc=2):
        return _convert_unix_to_utc(self.exif_data["timestamp"], utc_offset_hours=utc)

    @classmethod
    def from_bytes(cls, name: str, data_bytes, exif_data):
        img_instance = cls()
        img_instance.name = name
        img_instance.exif_data = dill.loads(exif_data)
        shape = (img_instance.exif_data["width"], img_instance.exif_data["height"])
        img_instance.image = PilImage.frombytes(img_instance.exif_data["encoding"], shape, data_bytes)
        return img_instance


class Points:
    def __init__(self):
        self.name = None
        self.points = None

    @classmethod
    def from_bytes(cls, name: str, data_bytes, pts_dtype):
        pts_instance = cls()
        pts_instance.name = name
        pts_instance.points = np.frombuffer(data_bytes, dtype=pts_dtype)
        return pts_instance


class Frame:
    def __init__(self, frame_id, timestamp):
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.images = [None] * 5
        self.points = [None] * 3

    @classmethod
    def from_bytes(cls, compressed_data, img_shape, pts_dtype):
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
                exif_len = int.from_bytes(decompressed_data[offset:offset + 4], 'big')
                offset += 4
                exif_data = decompressed_data[offset:offset + exif_len]
                offset += exif_len
                frame_instance.images[Camera[info_name.upper()]] = Image.from_bytes(info_name, camera_img_bytes, exif_data)
            elif Lidar.is_type_of(info_name.upper()):
                pts_len = int.from_bytes(decompressed_data[offset:offset + 4], 'big')
                offset += 4
                laser_pts_bytes = np.frombuffer(decompressed_data[offset:offset + pts_len])
                offset += pts_len
                frame_instance.points[Lidar[info_name.upper()]] = Points.from_bytes(info_name, laser_pts_bytes, pts_dtype)
        # return an instance of the class
        return frame_instance