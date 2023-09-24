import numpy as np
import matplotlib.pyplot as plt
import cv2
from ameisedataset import core as ameise
from ameisedataset.utils import transformation as tf
from ameisedataset.data.names import Camera, Lidar


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

print(tf.transform_to_sensor(infos.lidar[Lidar.OS1_TOP].extrinsic, infos.cameras[Camera.STEREO_LEFT].extrinsic))
x = tf.get_projection_matrix(frames[-1].lidar[ameise.Lidar.OS1_TOP], infos.lidar[ameise.Lidar.OS1_TOP], infos.cameras[Camera.STEREO_LEFT])

image = np.array(frames[-1].cameras[Camera.STEREO_LEFT])
im_rect = tf.rectify_image(image, infos.cameras[Camera.STEREO_LEFT], crop=True)
print(im_rect.size)
im_rect.show()

"""
for pt in x:
    cv2.circle(image, pt, 2, (0, 255, 0), -1)

cv2.imshow("", image)
cv2.waitKey(0)
cv2.destroyWindow()

plot_points_on_image(frames[-1].cameras[Camera.STEREO_LEFT].image, x)
"""
