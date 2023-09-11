import hashlib


def compute_checksum(data):
    """Compute the SHA-256 checksum for the provided data.
    Args:
        data (bytes): Data for which the checksum needs to be computed.
    Returns:
        bytes: SHA-256 checksum of the provided data.
    """
    # calculates the has value of a given bytestream - SHA256
    return hashlib.sha256(data).digest()
