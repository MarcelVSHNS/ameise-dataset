import numpy as np
import matplotlib.pyplot as plt
import cv2
from ameisedataset import core as ameise
from ameisedataset.utils import transformation as tf
from ameisedataset.metadata.names import Camera, Lidar

def plot_points_on_image(img, points):
    """
    Zeichnet Punkte auf ein Bild.

    Parameters:
    - image_path: Pfad zur Bilddatei.
    - points: Liste von Punkten der Form [(x1,y1), (x2,y2), ...]
    """
    # Bild anzeigen
    plt.imshow(img)


    # Punkte mit scatter zeichnen
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    plt.scatter(x_coords, y_coords, color='red')

    plt.show()

infos, frames = ameise.unpack_record("samples/frame.4mse")
infos.cameras[ameise.Camera.STEREO_LEFT].extrinsic.xyz = np.array([1.2071032524108887, 0.11899397522211075, -0.8313015103340149])
infos.cameras[ameise.Camera.STEREO_LEFT].extrinsic.rpy = np.array([-1.5636708736419678, 0.0407794751226902, -1.5704575777053833])
infos.lidar[ameise.Lidar.OS1_TOP].extrinsic.xyz = np.array([0, 0, 0])
infos.lidar[ameise.Lidar.OS1_TOP].extrinsic.rpy = np.array([0, 0, 0])

print(tf.transform_to_sensor(infos.lidar[ameise.Lidar.OS1_TOP].extrinsic, infos.cameras[ameise.Camera.STEREO_LEFT].extrinsic))
x = tf.get_projection_matrix(frames[-1].lidar[ameise.Lidar.OS1_TOP], infos.lidar[ameise.Lidar.OS1_TOP], infos.cameras[Camera.STEREO_LEFT])

image = np.array(frames[-1].cameras[Camera.STEREO_LEFT].image)

for pt in x:
    cv2.circle(image, pt, 2, (0, 255, 0), -1)

cv2.imshow("", image)
cv2.waitKey(0)
cv2.destroyWindow()

plot_points_on_image(frames[-1].cameras[Camera.STEREO_LEFT].image, x)


