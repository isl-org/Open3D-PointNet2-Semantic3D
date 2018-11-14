map_label_to_color = {
    0: [255, 0, 0, 255],
    1: [255, 165, 0, 255],
    2: [179, 255, 0, 255],
    3: [7, 255, 0, 255],
    4: [0, 255, 157, 255],
    5: [0, 181, 255, 255],
    6: [0, 15, 255, 255],
    7: [155, 0, 255, 255],
    8: [255, 0, 189, 255],
}


def write_ply_color(points, labels, file_name):
    if len(points) != len(labels):
        raise ValueError("# points != # labels")
    labels = labels.astype(int)
    with open(file_name, "w") as f:
        for point, label in zip(points, labels):
            color = map_label_to_color[label]
            f.write("v %f %f %f %d %d %d\n"
                    % (point[0], point[1], point[2], color[0], color[1], color[2]))


def write_pts_label_as_color(points, labels, file_name):
    if len(points) != len(labels):
        raise ValueError("# points != # labels")
    labels = labels.astype(int)
    with open(file_name, "w") as f:
        f.write("%d\n" % len(points))
        for point, label in zip(points, labels):
            color = map_label_to_color[label]
            f.write("%f %f %f 0 %d %d %d\n"
                    % (point[0], point[1], point[2], color[0], color[1], color[2]))


def write_ply_true_color(points, colors, file_name):
    colors = colors.astype(int)
    num_points = points.shape[0]
    fout = open(file_name, "w")
    for i in range(num_points):
        fout.write(
            "v %f %f %f %d %d %d\n"
            % (
                points[i, 0],
                points[i, 1],
                points[i, 2],
                colors[i, 0],
                colors[i, 1],
                colors[i, 2],
            )
        )
    fout.close()
