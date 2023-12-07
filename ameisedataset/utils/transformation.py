from typing import List
from PIL import Image as PilImage
import numpy as np
import cv2
from typing import Tuple, List

from ameisedataset.data import Pose, CameraInformation, LidarInformation, Image


def rectify_image(image: Image, camera_information: CameraInformation):
    """Rectify the provided image using camera information."""
    # Init and calculate rectification matrix
    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix=camera_information.camera_mtx,
                                             distCoeffs=camera_information.distortion_mtx[:-1],
                                             R=camera_information.rectification_mtx,
                                             newCameraMatrix=camera_information.projection_mtx,
                                             size=camera_information.shape,
                                             m1type=cv2.CV_16SC2)
    # Apply matrix
    rectified_image = cv2.remap(np.array(image.image), mapx, mapy, interpolation=cv2.INTER_LINEAR)

    return Image(PilImage.fromarray(rectified_image), image.timestamp)


def euler_to_rotation_matrix(roll, pitch, yaw):
    """Convert Euler angles to rotation matrix."""
    cos_r, sin_r = np.cos(roll), np.sin(roll)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)

    R_x = np.array([[1, 0, 0],
                    [0, cos_r, -sin_r],
                    [0, sin_r, cos_r]])

    R_y = np.array([[cos_p, 0, sin_p],
                    [0, 1, 0],
                    [-sin_p, 0, cos_p]])

    R_z = np.array([[cos_y, -sin_y, 0],
                    [sin_y, cos_y, 0],
                    [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def create_transformation_matrix(translation, rotation):
    """Create a 4x4 homogeneous transformation matrix."""
    T = np.eye(4)
    T[:3, :3] = euler_to_rotation_matrix(rotation[0], rotation[1], rotation[2])
    T[:3, 3] = translation
    return T


def get_points_on_image(pcloud: List[np.ndarray], lidar_info: LidarInformation, cam_info: CameraInformation, get_valid_only=True,
                        dtype_points_return=None) -> Tuple[np.array, List[Tuple]]:
    """Retrieve the projection matrix based on provided parameters."""
    if dtype_points_return is None:
        dtype_points_return = ['x', 'y', 'z', 'intensity', 'range']
    lidar_to_cam_tf_mtx = transform_to_sensor(lidar_info.extrinsic, cam_info.extrinsic)
    rect_mtx = np.eye(4)
    rect_mtx[:3, :3] = cam_info.rectification_mtx
    proj_mtx = cam_info.projection_mtx

    projection = []
    points = []
    for point in pcloud:
        point_vals = np.array(point.tolist()[:3])
        # Transform points to new coordinate system
        point_in_camera = proj_mtx.dot(rect_mtx.dot(lidar_to_cam_tf_mtx.dot(np.append(point_vals[:3], 1))))
        # check if pts are behind the camera
        u = point_in_camera[0] / point_in_camera[2]
        v = point_in_camera[1] / point_in_camera[2]
        if get_valid_only:
            if point_in_camera[2] <= 0:
                continue
            elif 0 <= u < cam_info.shape[0] and 0 <= v < cam_info.shape[1]:
                projection.append((u, v))
                points.append(point[dtype_points_return])
            else:
                continue
        else:
            projection.append((u, v))
    return np.array(points, dtype=points[0].dtype), projection


def transform_to_sensor(sensor1: Pose, sensor2: Pose):
    """Transform the data to the sensor's coordinate frame."""
    # Creating transformation matrices
    t1 = create_transformation_matrix(sensor1.xyz, sensor1.rpy)
    t2 = create_transformation_matrix(sensor2.xyz, sensor2.rpy)

    # Computing the transformation from the second sensor (new origin) to the first sensor
    t2_to_1 = np.dot(np.linalg.inv(t2), t1)
    return t2_to_1


def create_stereo_image(image_left: Image, image_right: Image, cam_right_info: CameraInformation) -> np.ndarray:
    """
    Create a disparity map from two rectified images.
    Parameters:
    - image_left: First image as a PIL Image.
    - image_right: Second image as a PIL Image.
    - cam_right_info: Camera Info object
    Returns:
    - Depth map as a numpy array.
    """
    # Convert PIL images to numpy arrays
    img1 = np.array(image_left.convert('L'))  # Convert to grayscale
    img2 = np.array(image_right.convert('L'))  # Convert to grayscale
    # Create the block matching algorithm with high-quality settings
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,         # Depending on the camera setup, this might need to be increased.
        blockSize=5,                # Smaller block size can detect finer details.
        P1=8 * 3 * 5 ** 2,          # Control smoothness of the disparity. Adjust as needed.
        P2=32 * 3 * 5 ** 2,         # Control smoothness. This is usually larger than P1.
        disp12MaxDiff=1,            # Controls maximum allowed difference in disparity check.
        uniquenessRatio=15,         # Controls uniqueness. Higher can mean more robustness against noise.
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY  # Utilizes 3-way dynamic programming. May provide more robust results.
    )
    # Compute the disparity map
    disparity = stereo.compute(img1, img2)
    # Normalize for better visualization
    disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # To avoid division by zero, set disparity values of 0 to a small value
    safe_disparity = np.where(disparity == 0, 0.000001, disparity)
    f = cam_right_info.focal_length
    b = abs(cam_right_info.stereo_transform.translation[0]) * 10 ** 3
    depth_map = f * b / safe_disparity
    return depth_map
