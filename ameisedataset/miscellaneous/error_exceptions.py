# Specific exception types
class ChecksumError(Exception):
    """Raised when there's a checksum mismatch."""
    pass


class InvalidFileTypeError(Exception):
    """Raised when the provided file type is not supported."""
    pass
