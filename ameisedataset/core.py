import dill
import hashlib

from ameisedataset.metadata import CameraInformation, LidarInformation, Camera, Lidar, Frame


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
