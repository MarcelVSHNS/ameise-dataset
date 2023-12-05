from PIL import ImageDraw, Image
import numpy as np
from ameisedataset import core as ameise
from ameisedataset.data.names import Camera, Lidar
import matplotlib
import open3d as o3d
from datetime import datetime, timezone
from decimal import Decimal


def visualize_points(points):
    # Convert structured NumPy array to a regular 3D NumPy array with contiguous memory.
    xyz_points = np.stack((points['x'], points['y'], points['z']), axis=-1)

    # Ensure the data type is float64, which is expected by Open3D.
    xyz_points = xyz_points.astype(np.float64)

    # Create Open3D point cloud.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_points)

    # Estimate normals.
    pcd.estimate_normals()

    # Visualize the point cloud.
    o3d.visualization.draw_geometries([pcd])


def vis_lidar_temporal(points):
    max_timestamp = np.max(points['t'])
    min_timestamp = np.min(points['t'])

    threshold_ts = min_timestamp + 0.05 * (max_timestamp - min_timestamp)
    subset_points = points[points['t'] <= threshold_ts]
    visualize_points(subset_points)

    threshold_ts = min_timestamp + 0.1 * (max_timestamp - min_timestamp)
    subset_points = points[points['t'] <= threshold_ts]
    visualize_points(subset_points)

    threshold_ts = min_timestamp + 0.2 * (max_timestamp - min_timestamp)
    subset_points = points[points['t'] <= threshold_ts]
    visualize_points(subset_points)

    threshold_ts = min_timestamp + 0.3 * (max_timestamp - min_timestamp)
    subset_points = points[points['t'] <= threshold_ts]
    visualize_points(subset_points)

    threshold_ts = min_timestamp + 0.4 * (max_timestamp - min_timestamp)
    subset_points = points[points['t'] <= threshold_ts]
    visualize_points(subset_points)

def show_disparity_map(disparity_map, cmap_name="viridis", val_min=None, val_max=None):
    cmap = matplotlib.colormaps[cmap_name]
    val_min = val_min if val_min is not None else np.min(disparity_map)
    val_max = val_max if val_max is not None else np.max(disparity_map)
    norm_values = (disparity_map - val_min) / (val_max - val_min)
    colored_map = (cmap(norm_values)[:, :, :3] * 255).astype(np.uint8)

    # Konvertieren Sie das NumPy-Array in ein PIL-Bild und zeigen Sie es an
    img = Image.fromarray(colored_map)
    img.show()


def plot_points_on_image(img, points, values, cmap_name="inferno", radius=2, val_min=None, val_max=None):
    draw = ImageDraw.Draw(img)
    cmap = matplotlib.colormaps[cmap_name + "_r"]
    val_min = val_min * 1000 if val_min is not None else np.min(values)
    val_max = val_max * 1000 if val_max is not None else np.max(values)
    norm_values = (values - val_min) / (val_max - val_min)
    for punkt, value in zip(points, norm_values):
        x, y = punkt
        rgba = cmap(value)
        farbe = (int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
        draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill=farbe)
    img.show()

#infos, frames = ameise.unpack_record("/home/ameise/datasets/sequences_packed/2023-11-02_16-08-52_phase150both/id00051-id00100_1698937774641520377-1698937779541846231.4mse")
infos, frames = ameise.unpack_record("/home/ameise/datasets/sequences_packed/2023-11-07_13-45-30_unknown/id00001-id00050_1699361167242136102-1699361172141735036.4mse")

points_top = frames[-1].lidar[Lidar.OS1_TOP].points
points_top_ts = frames[-1].lidar[Lidar.OS1_TOP].timestamp

img_left = frames[-1].cameras[Camera.STEREO_LEFT].image
img_left_ts = frames[-1].cameras[Camera.STEREO_LEFT].timestamp



# Umwandlung in Sekunden (Dezimalnanosekunden / 1 Milliarde)
points_top_ts_seconds = points_top_ts / Decimal(1e9)
img_left_ts_seconds = img_left_ts / Decimal(1e9)

# Umwandlung der Dezimalzahlen in Fließkommazahlen für die Datums-/Zeitumwandlung
points_top_ts_human_readable = datetime.fromtimestamp(float(points_top_ts_seconds), tz=timezone.utc)
img_left_ts_human_readable = datetime.fromtimestamp(float(img_left_ts_seconds), tz=timezone.utc)

print(f'Phase Lock OS1_TOP Offset: {infos.lidar[Lidar.OS1_TOP].phase_lock_offset}')
print(f'Timestamp OS1_TOP: {points_top_ts_human_readable}')
print(f'Phase Lock OS0_Left Offset: NO LOCK')
print(f'Timestamp OS0_Left: {img_left_ts_human_readable}')

'''
pts, proj = tf.get_projection_matrix(frames[-1].lidar[ameise.Lidar.OS1_TOP].points, infos.lidar[ameise.Lidar.OS1_TOP], infos.cameras[Camera.STEREO_LEFT])

image_left = frames[-1].cameras[Camera.STEREO_LEFT]
image_right = frames[-1].cameras[Camera.STEREO_RIGHT]
points_top = frames[-1].lidar[Lidar.OS1_TOP]
print(points_top.get_timestamp())
print(np.amax(points_top.points['t']))

im_rect_l = tf.rectify_image(image_left, infos.cameras[Camera.STEREO_LEFT])
im_rect_r = tf.rectify_image(image_right, infos.cameras[Camera.STEREO_RIGHT])
disparity_map = tf.create_disparity_map(im_rect_l, im_rect_r)

plot_points_on_image(im_rect_l, proj, pts['range'], val_min=8, val_max=50)
show_disparity_map(disparity_map, val_min=0, val_max=100)
'''