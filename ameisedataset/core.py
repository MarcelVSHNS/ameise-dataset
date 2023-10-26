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
        # TODO: Wollen wir den header/chunkinfo? wenn ja sollten wir den hier zurückgeben
        chunk_info, meta_info = _read_info_object(file)
        # Read num frames
        num_frames = int.from_bytes(file.read(INT_LENGTH), 'big')
        # Read frames
        for _ in range(num_frames):
            frames.append(_read_frame_object(file, meta_info))
    return meta_info, frames
