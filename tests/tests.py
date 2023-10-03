import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import ImageDraw
from ameisedataset import core as ameise
from ameisedataset.utils import transformation as tf
from ameisedataset.data.names import Camera, Lidar


def plot_points_on_image(img, points, farbe="red", radius=2):
    draw = ImageDraw.Draw(img)
    for punkt in points:
        x, y = punkt
        draw.ellipse([(x-radius, y-radius), (x+radius, y+radius)], fill=farbe)
    img.show()


infos, frames = ameise.unpack_record("samples/id00050_1696157166674880156.4mse")

print(tf.transform_to_sensor(infos.lidar[Lidar.OS1_TOP].extrinsic, infos.cameras[Camera.STEREO_LEFT].extrinsic))
x = tf.get_projection_matrix(frames[-1].lidar[ameise.Lidar.OS1_TOP], infos.lidar[ameise.Lidar.OS1_TOP], infos.cameras[Camera.STEREO_LEFT])

# frames[-1].cameras[Camera.STEREO_LEFT].save("stereo_left.png")
# frames[-1].cameras[Camera.STEREO_RIGHT].save("stereo_right.png")

image_left = frames[-1].cameras[Camera.STEREO_LEFT]
image_right = frames[-1].cameras[Camera.STEREO_RIGHT]
#image_right.show()
print(image_right.get_timestamp())
im_rect_l = tf.rectify_image(image_left, infos.cameras[Camera.STEREO_LEFT])
im_rect_r = tf.rectify_image(image_right, infos.cameras[Camera.STEREO_RIGHT])


#plot_points_on_image(image_left, x)

image_array = np.array(im_rect_l)
# Konvertieren Sie RGB zu BGR
cv2_image = image_array[:, :, ::-1]


plot_points_on_image(im_rect_l, x)


