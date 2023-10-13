from PIL import ImageDraw, Image
import numpy as np
import ameisedataset as ad
from ameisedataset.data.names import Camera, Lidar
import matplotlib
import matplotlib.pyplot as plt
import glob
import yaml
import os

with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)
data_dir = config['raw_data_path'] + config['phase']

ameise_record_map = glob.glob(os.path.join(data_dir, '*.4mse'))
print(len(ameise_record_map))

for entry in ameise_record_map:
    try:
        infos, frames = ad.unpack_record(entry)
        print (entry + ' is okay!!')
    except:
        print(entry + " is corrupted...")
        os.remove(entry)

"""
pts, proj = ad.utils.get_projection_matrix(frames[-1].lidar[Lidar.OS1_TOP].points, infos.lidar[Lidar.OS1_TOP], infos.cameras[Camera.STEREO_LEFT])

image_left = frames[-1].cameras[Camera.STEREO_LEFT]
image_right = frames[-1].cameras[Camera.STEREO_RIGHT]
points_top = frames[-1].lidar[Lidar.OS1_TOP]
print(points_top.get_timestamp())
print(np.amax(points_top.points['t']))

im_rect_l = ad.utils.rectify_image(image_left, infos.cameras[Camera.STEREO_LEFT])
im_rect_r = ad.utils.rectify_image(image_right, infos.cameras[Camera.STEREO_RIGHT])
stereo_img = ad.utils.create_stereo_image(im_rect_l, im_rect_r, infos.cameras[Camera.STEREO_RIGHT])

#plot_points_on_image(im_rect_l, proj, pts['range'], val_min=8, val_max=50)
ad.utils.show_disparity_map(stereo_img, val_min=-5, val_max=100)
"""





