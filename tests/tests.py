from PIL import ImageDraw, Image
import numpy as np
import ameisedataset as ad
from ameisedataset.data.names import Camera, Lidar
import matplotlib
import matplotlib.pyplot as plt
import glob
import yaml
import os

infos, frames = ad.unpack_record("/home/marcel/datasets/testing/id00101-id00156_1697468271541681418-1697468276441730096.4mse")
pts, proj = ad.utils.get_projection_matrix(frames[-1].lidar[Lidar.OS1_TOP].points, infos.lidar[Lidar.OS1_TOP], infos.cameras[Camera.STEREO_LEFT])

image_left = frames[-1].cameras[Camera.STEREO_LEFT]
image_right = frames[-1].cameras[Camera.STEREO_RIGHT]
points_top = frames[-1].lidar[Lidar.OS1_TOP]
print(points_top.get_timestamp())
print(np.amax(points_top.points['t']))

im_rect_l = ad.utils.rectify_image(image_left, infos.cameras[Camera.STEREO_LEFT])
im_rect_r = ad.utils.rectify_image(image_right, infos.cameras[Camera.STEREO_RIGHT])
stereo_img = ad.utils.create_stereo_image(im_rect_l, im_rect_r, infos.cameras[Camera.STEREO_RIGHT])

proj_img = ad.utils.plot_points_on_image(im_rect_l, proj, pts['range'], val_min=8, val_max=50)
proj_img.show()
#ad.utils.show_disparity_map(stereo_img, val_min=-5, val_max=100)






