from __future__ import print_function
import numpy as np
from pprint import pprint
from sklearn.metrics import confusion_matrix as skl_get_confusion_matrix


class ConfusionMatrix:
    def __init__(self, num_classes):
        """
        label must be {0, 1, 2, ..., num_classes - 1}
        """
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )
        self.valid_labels = set(range(self.num_classes))

    def increment(self, gt_label, pd_label):
        if gt_label not in self.valid_labels:
            raise ValueError("Invalid value for gt_label")
        if pd_label not in self.valid_labels:
            raise ValueError("Invalid value for pd_label")
        self.confusion_matrix[gt_label][pd_label] += 1

    def increment_from_list(self, gt_labels, pd_labels):
        increment_cm = skl_get_confusion_matrix(
            gt_labels, pd_labels, labels=list(range(self.num_classes))
        )
        np.testing.assert_array_equal(self.confusion_matrix.shape, increment_cm.shape)
        self.confusion_matrix += increment_cm

    def get_per_class_ious(self):
        """
        Warning: Semantic3D assumes label 0 is not used.
        I.e. 1. if gt == 0, this data point is simply ignored
             2. it's always true that pd != 0

        |        | 0 (pd)      | 1 (pd)      | 2 (pd)      | 3 (pd)      |
        |--------|-------------|-------------|-------------|-------------|
        | 0 (gt) | (must be) 0 | (ignored) 1 | (ignored) 2 | (ignored) 3 |
        | 1 (gt) | (must be) 0 | 4           | 5           | 6           |
        | 2 (gt) | (must be) 0 | 7           | 8           | 9           |
        | 3 (gt) | (must be) 0 | 10          | 11          | 12          |

        Returns a list of num_classes - 1 elements
        """

        # Check that pd != 0
        if any(self.confusion_matrix[:, 0] != 0):
            print("[Warn] Contains prediction of label 0:", self.confusion_matrix[:, 0])

        # Ignore gt == 0
        valid_confusion_matrix = self.confusion_matrix[1:, 1:]
        ious = []
        for c in range(len(valid_confusion_matrix)):
            intersection = valid_confusion_matrix[c, c]
            union = (
                np.sum(valid_confusion_matrix[c, :])
                + np.sum(valid_confusion_matrix[:, c])
                - intersection
            )
            if union == 0:
                union = 1
            ious.append(float(intersection) / union)
        return ious

    def get_mean_iou(self):
        """
        Warning: Semantic3D assumes label 0 is not used.
        E.g. 1. if gt == 0, this data point is simply ignored
             2. assert that pd != 0
        """
        per_class_ious = self.get_per_class_ious()
        return np.sum(per_class_ious) / len(per_class_ious)

    def get_accuracy(self):
        """
        Warning: Semantic3D assumes label 0 is not used.
        E.g. 1. if gt == 0, this data point is simply ignored
             2. assert that pd != 0
        """
        valid_confusion_matrix = self.confusion_matrix[1:, 1:]
        return np.trace(valid_confusion_matrix) / np.sum(valid_confusion_matrix)

    def print_metrics(self, labels=None):
        # 1. Confusion matrix
        print("Confusion matrix:")

        # Fill default labels: ["0", "1", "2", ...]
        if labels == None:
            labels = [str(val) for val in range(self.num_classes)]
        elif len(labels) != self.num_classes:
            raise ValueError("len(labels) != self.num_classes")

        # Formatting helpers
        column_width = max([len(x) for x in labels] + [7])
        empty_cell = " " * column_width

        # Print header
        print("    " + empty_cell, end=" ")
        for label in labels:
            print("%{0}s".format(column_width) % label, end=" ")
        print()

        # Print rows
        for i, label in enumerate(labels):
            print("    %{0}s".format(column_width) % label, end=" ")
            for j in range(len(labels)):
                cell = "%{0}.0f".format(column_width) % self.confusion_matrix[i, j]
                print(cell, end=" ")
            print()

        # 2. IoU per class
        print("IoU per class:")
        pprint(self.get_per_class_ious())

        # 3. Mean IoU
        # Warning: excluding class 0
        print("mIoU (ignoring label 0):")
        print(self.get_mean_iou())

        # 4. Overall accuracy
        print("Overall accuracy")
        print(self.get_accuracy())


if __name__ == "__main__":
    # Test data
    # |        | 0 (pd)      | 1 (pd)      | 2 (pd)      | 3 (pd)      |
    # |--------|-------------|-------------|-------------|-------------|
    # | 0 (gt) | (must be) 0 | (ignored) 1 | (ignored) 2 | (ignored) 3 |
    # | 1 (gt) | (must be) 0 | 4           | 5           | 6           |
    # | 2 (gt) | (must be) 0 | 7           | 8           | 9           |
    # | 3 (gt) | (must be) 0 | 10          | 11          | 12          |
    ref_confusion_matrix = np.array(
        [[0, 1, 2, 3], [0, 4, 5, 6], [0, 7, 8, 9], [0, 10, 11, 12]]
    )

    # Build CM
    cm = ConfusionMatrix(num_classes=4)
    for gt in range(4):
        for pd in range(4):
            for _ in range(ref_confusion_matrix[gt, pd]):
                cm.increment(gt, pd)

    # Check confusion matrix
    np.testing.assert_allclose(ref_confusion_matrix, cm.confusion_matrix)
    print(cm.confusion_matrix)

    # Check IoU
    ref_per_class_ious = np.array(
        [
            4.0 / (4 + 7 + 10 + 5 + 6),
            8.0 / (5 + 8 + 11 + 7 + 9),
            12.0 / (6 + 9 + 12 + 10 + 11),
        ]
    )
    np.testing.assert_allclose(cm.get_per_class_ious(), ref_per_class_ious)
    print(cm.get_per_class_ious())

    ref_mean_iou = np.mean(ref_per_class_ious)
    assert cm.get_mean_iou() == ref_mean_iou
    print(cm.get_mean_iou())

    # Check accuracy
    ref_accuracy = float(4 + 8 + 12) / ((4 + 12) * 9 / 2)
    assert cm.get_accuracy() == ref_accuracy
    print(cm.get_accuracy())
