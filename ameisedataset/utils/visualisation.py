from PIL import ImageDraw, Image
import numpy as np
import matplotlib


def show_disparity_map(disparity_map, cmap_name="viridis", val_min=None, val_max=None):
    cmap = matplotlib.colormaps[cmap_name]
    val_min = val_min if val_min is not None else np.min(disparity_map)
    val_max = val_max if val_max is not None else np.max(disparity_map)
    mask = disparity_map > 10 * val_max
    norm_values = np.where(mask, 0, (disparity_map - val_min) / (val_max - val_min))
    colored_map = cmap(norm_values)
    colored_map[mask] = [0, 0, 0, 1]  # Set masked values to black
    colored_map = (colored_map[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(colored_map)
    return img


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
    return img
