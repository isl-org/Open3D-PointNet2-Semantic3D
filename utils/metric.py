from __future__ import print_function
import numpy as np
from pprint import pprint


class ConfusionMatrix:
    def __init__(self, num_classes):
        """
        label must be {0, 1, 2, ..., num_classes - 1}
        """
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def increment(self, gt_label, pd_label):
        valid_labels = set(range(self.num_classes))
        if gt_label not in valid_labels:
            raise ValueError("Invalid value for gt_label")
        if pd_label not in valid_labels:
            raise ValueError("Invalid value for pd_label")
        self.confusion_matrix[gt_label][pd_label] += 1

    def increment_from_file(self, gt_file, pd_file):
        """
        For Semantic3D: num_classes == 9, and both gt_file and pd_file only contains
        label 1, 2, ..., 8. Label 0 is not used at all.
        """
        with open(gt_file, "r") as gt_f, open(pd_file, "r") as pd_f:
            for gt_line, pd_line in zip(gt_f, pd_f):
                gt_label = int(float(gt_line.strip()))
                pd_label = int(float(pd_line.strip()))
                self.increment(gt_label, pd_label)

    def get_per_class_iou(self):
        ious = []
        for c in range(self.num_classes):
            intersection = self.confusion_matrix[c, c]
            union = (
                np.sum(self.confusion_matrix[c, :])
                + np.sum(self.confusion_matrix[:, c])
                - intersection
            )
            if union == 0:
                union = 1
            ious.append(float(intersection) / union)
        return ious

    def get_mean_iou(self):
        """
        Warning: Semantic3D assumes label 0 is not used for computing mean
        """
        return np.sum(self.get_per_class_iou()) / (self.num_classes - 1)

    def get_accuracy(self):
        return np.trace(self.confusion_matrix) / np.sum(self.confusion_matrix)

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
        pprint(self.get_per_class_iou())

        # 3. Mean IoU
        # Warning: excluding class 0
        print("mIoU (ignoring label 0):")
        print(self.get_mean_iou())

        # 4. Overall accuracy
        print("Overall accuracy")
        print(self.get_accuracy())
