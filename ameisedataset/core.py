import os
from typing import List, Tuple

from ameisedataset.data import *
from ameisedataset.miscellaneous import compute_checksum, InvalidFileTypeError, ChecksumError, SHA256_CHECKSUM_LENGTH, INT_LENGTH


def _read_info_object(file):
    """Read and deserialize an info object from the file."""
    info_length = int.from_bytes(file.read(INT_LENGTH), 'big')
    info_checksum = file.read(SHA256_CHECKSUM_LENGTH)  # SHA-256 checksum length
    combined_info = file.read(info_length)

    # Verify checksum
    if compute_checksum(combined_info) != info_checksum:
        raise ChecksumError(f"Checksum of Info is not correct! Check file.")
    return Infos.from_bytes(combined_info)


def _read_frame_object(file, meta_infos):
    """Read and deserialize a frame from the file."""
    combined_data_len = int.from_bytes(file.read(INT_LENGTH), 'big')
    combined_data_checksum = file.read(SHA256_CHECKSUM_LENGTH)  # SHA-256 checksum length
    combined_data = file.read(combined_data_len)
    # Verify checksum
    if compute_checksum(combined_data) != combined_data_checksum:
        raise ChecksumError("Checksum mismatch. Data might be corrupted!")
    return Frame.from_bytes(combined_data, meta_info=meta_infos)


def unpack_record(filename) -> Tuple[Infos, List[Frame]]:
    """Unpack an AMEISE record file and extract meta information and frames.
        Args:
            filename (str): Path to the AMEISE record file.
        Returns:
            Tuple[Infos, List[Frame]]: Meta information and a list of frames.
        """
    # Ensure the provided file has the correct extension
    if os.path.splitext(filename)[1] != ".4mse":
        raise InvalidFileTypeError("This is not a valid AMEISE-Record file.")
    frames: List[Frame] = []

    with open(filename, 'rb') as file:
        chunk_info, meta_info = _read_info_object(file)
        # Read num frames
        num_frames = int.from_bytes(file.read(INT_LENGTH), 'big')
        # Read frames
        for _ in range(num_frames):
            frames.append(_read_frame_object(file, meta_info))
    return meta_info, frames


class AmeiseDataloader:
    # usage: frame_34 = AmeiseDataloader[34]
    # vorteil: es wird nur ausgelesen was benoetigt wird und wann es benoetigt wird
    # es wird trotzdem nur einmal metainfos im .4mse file gespeichert und zur laufzeit ein objekt
    # dass jedesmal auf die metainfos zugreift und ein vollstaendiges z.b. kamera objekt erzeugt mit
    # bild, intrinsic, timestamp etc.
    # ABER AGAIN: WIR SOLLTEN LANGSAM EINEN SDK FREEZE MACHEN UND UNS AUF DIE DATEN BZW. DEN TOWER ETC. KUEMMERN
    def __init__(self):
        self.frame_map = None
        self.name = None
        self.meta_infos = None
        self.frames = None

    def __getitem__(self, item):
        # es wird erst ein objekt ausgelesen und erstellt, wenn man es ueber den index aufruft
        # READ_from_file at pos: x to y and put it into extended frame object
        return None

class AmeiseData:
    # usage: frame_34.STEREO_LEFT.image.show
    # usage: frame_34.STEREO_LEFT.camera_mtx
    # oder direkt: AmeiseDataloader[34].STEREO_LEFT.image.get_timestamp() oder AmeiseDataloader[34].timestamp
    def __init__(self, frame, meta_infos):
        self.timestamp = None
        self.name = None
        self.STEREO_LEFT = None
        self.STEREO_RIGHT = None
        # ... STEREO_LEFT erbt von camera und ergänzt um ein Image
        self.OS1_TOP = None
        self.OS0_LEFT = None
        # ... OS1_top erbt von lidar und ergänzt um Points und IMU
        self.gnss = None

    def get_camera(self, num):
        # usage: frame.get_camera(Camera.STEREO_LEFT) bzw. frame.get_camera(1)
        # damit man auch durchiterieren kann
        pass

    def get_lidar(self, num):
        pass





