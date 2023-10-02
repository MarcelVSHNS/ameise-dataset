from typing import List
from PIL import Image as PilImage
import numpy as np
import cv2

from ameisedataset.data import Pose, CameraInformation, LidarInformation, Image


def rectify_image(image: Image, camera_information: CameraInformation, crop=False, resize_to_original_size=True):
    # Init and calculate rectification matrix
    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix=camera_information.camera_mtx,
                                             distCoeffs=camera_information.distortion_mtx[:-1],
                                             R=camera_information.rectification_mtx,
                                             newCameraMatrix=camera_information.projection_mtx,
                                             size=camera_information.shape,
                                             m1type=cv2.CV_16SC2)
    # Apply matrix
    rectified_image = cv2.remap(np.array(image.image), mapx, mapy, interpolation=cv2.INTER_LINEAR)
    # Crop image if wanted
    if crop:
        x, y, h, w = camera_information.region_of_interest
        rectified_image = rectified_image[y:y + h, x:x + w]
        if resize_to_original_size:
            rectified_image = cv2.resize(rectified_image, (1920, 1200))
        else:
            rectified_image = rectified_image
    return Image(PilImage.fromarray(rectified_image), image.timestamp)


def invert_transformation(T):
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -np.dot(R.T, t)
    return T_inv


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


def get_transformation_matrix(pitch, yaw, roll, x, y, z):
    # Erstelle die Rotationsmatrizen
    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    R_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    # Kombiniere die Rotationsmatrizen
    R = np.dot(R_yaw, np.dot(R_pitch, R_roll))

    # Erstelle die 4x4 Transformationsmatrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    print(T)
    T = invert_transformation(T)
    print(T)
    return T


def get_projection_matrix(pcloud: List[np.ndarray], lidar_info: LidarInformation, cam_info: CameraInformation):
    lidar_to_cam_tf_mtx = transform_to_sensor(lidar_info.extrinsic, cam_info.extrinsic)
    projection = []
    for point in pcloud:
        point = np.array(point.tolist()[:3])
        # Transformiere den Punkt in das Kamerakoordinatensystem
        point_in_camera = np.dot(lidar_to_cam_tf_mtx, np.append(point[:3], 1))  # Nehmen Sie nur die ersten 3 Koordinaten
        # Überprüfen Sie, ob der Punkt vor der Kamera liegt
        if point_in_camera[2] <= 0:
            # projection.append((None, None))
            pass
        else:
            # Projiziere den Punkt auf die Bildebene
            pixel = np.dot(cam_info.camera_mtx, point_in_camera[:3])
            projection.append((int(pixel[0] / pixel[2]), int(pixel[1] / pixel[2])))
    return projection


def transform_to_sensor(sensor1: Pose, sensor2: Pose):
    # Creating transformation matrices
    t1 = create_transformation_matrix(sensor1.xyz, sensor1.rpy)
    t2 = create_transformation_matrix(sensor2.xyz, sensor2.rpy)

    # Computing the transformation from the second sensor (new origin) to the first sensor
    t2_to_1 = np.dot(np.linalg.inv(t2), t1)
    return t2_to_1
