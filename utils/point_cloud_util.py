import numpy as np
import open3d


def _label_to_colors(labels):
    map_label_to_color = {
        0: [255, 255, 255],  # white
        1: [0, 0, 255],  # blue
        2: [128, 0, 0],  # maroon
        3: [255, 0, 255],  # fuchisia
        4: [0, 128, 0],  # green
        5: [255, 0, 0],  # red
        6: [128, 0, 128],  # purple
        7: [0, 0, 128],  # navy
        8: [128, 128, 0],  # olive
    }
    return np.array([map_label_to_color[label] for label in labels]).astype(np.int32)


def colorize_point_cloud(point_cloud, labels):
    if len(point_cloud.points) != len(labels):
        raise ValueError("len(point_cloud.points) != len(labels)")
    point_cloud.colors = open3d.Vector3dVector(_label_to_colors(labels))


def load_labels(label_path):
    # Assuming each line is a valid int
    with open(label_path, "r") as f:
        labels = [int(line) for line in f]
    return labels


def write_labels(label_path, labels):
    with open(label_path, "w") as f:
        for label in labels:
            f.write("%d\n" % label)
