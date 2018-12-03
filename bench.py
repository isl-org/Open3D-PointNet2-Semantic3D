import numpy as np
import open3d
import multiprocessing
import time

_map_label_to_color = (
    [255, 255, 255],  # white
    [0, 0, 255],  # blue
    [128, 0, 0],  # maroon
    [255, 0, 255],  # fuchisia
    [0, 128, 0],  # green
    [255, 0, 0],  # red
    [128, 0, 128],  # purple
    [0, 0, 128],  # navy
    [128, 128, 0],  # olive
)


def _label_to_color(label):
    global _map_label_to_color
    return _map_label_to_color[label]


def _label_to_colors(labels):
    labels = np.array(labels).astype(np.int32)
    labels = list(labels)
    with multiprocessing.Pool() as pool:
        colors = pool.map(_label_to_color, labels)
    return np.array(colors).astype(np.int32)

labels = list(np.random.randint(0, 9, size=(5000000,), dtype=np.int32))

s = time.time()
with multiprocessing.Pool() as pool:
    colors = pool.map(_label_to_color, labels)
    colors = np.array(colors)
    print(colors.shape)
print("time to map", time.time() - s, flush=True)
