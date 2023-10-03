from PIL import ImageDraw
import numpy as np
from ameisedataset import core as ameise
from ameisedataset.utils import transformation as tf
from ameisedataset.data.names import Camera, Lidar


def plot_points_on_image(img, points, farbe="red", radius=2):
    draw = ImageDraw.Draw(img)
    for punkt in points:
        x, y = punkt
        draw.ellipse([(x-radius, y-radius), (x+radius, y+radius)], fill=farbe)
    img.show()


infos, frames = ameise.unpack_record("samples/frame.4mse")

x = tf.get_projection_matrix(frames[-1].lidar[ameise.Lidar.OS1_TOP].points, infos.lidar[ameise.Lidar.OS1_TOP], infos.cameras[Camera.STEREO_LEFT])

image_left = frames[-1].cameras[Camera.STEREO_LEFT]
image_right = frames[-1].cameras[Camera.STEREO_RIGHT]
points_top = frames[-1].lidar[Lidar.OS1_TOP]
print(points_top.get_timestamp())
print(np.amax(points_top.points['t']))

im_rect_l = tf.rectify_image(image_left, infos.cameras[Camera.STEREO_LEFT])
im_rect_r = tf.rectify_image(image_right, infos.cameras[Camera.STEREO_RIGHT])

#plot_points_on_image(im_rect_l, x)


