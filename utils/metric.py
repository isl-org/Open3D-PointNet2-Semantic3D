from __future__ import print_function
import numpy as np


class ConfusionMatrix:
    """Streaming interface to allow for any source of predictions. Initialize it, count
       predictions one by one, then print confusion matrix and intersection-union score"""

    def __init__(self, num_classes):
        """
        label must be {0, 1, 2, ..., num_classes - 1}
        """
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def count_predicted(self, gt_label, pd_label):
        # TODO: add checks
        self.confusion_matrix[gt_label][pd_label] += 1

    def get_count(self, gt_label, pd_label):
        """labels are integers from 0 to num_classes-1"""
        return self.confusion_matrix[gt_label][pd_label]

    def get_confusion_matrix(self):
        """
        result[gt_label][pd_label]: # of gt_label are predicted as pd_label
        """
        return self.confusion_matrix

    def get_intersection_union_per_class(self):
        """returns list of 64-bit floats"""

        matrix_diagonal = [self.confusion_matrix[i][i] for i in range(self.num_classes)]
        errors_summed_by_row = [0] * self.num_classes
        for row in range(self.num_classes):
            for column in range(self.num_classes):
                if row != column:
                    errors_summed_by_row[row] += self.confusion_matrix[row][column]
        errors_summed_by_column = [0] * self.num_classes
        for column in range(self.num_classes):
            for row in range(self.num_classes):
                if row != column:
                    errors_summed_by_column[column] += self.confusion_matrix[row][
                        column
                    ]

        divisor = [0] * self.num_classes
        for i in range(self.num_classes):
            divisor[i] = (
                matrix_diagonal[i]
                + errors_summed_by_row[i]
                + errors_summed_by_column[i]
            )
            if matrix_diagonal[i] == 0:
                divisor[i] = 1

        return [float(matrix_diagonal[i]) / divisor[i] for i in range(self.num_classes)]

    def get_overall_accuracy(self):
        """returns 64-bit float"""
        matrix_diagonal = 0
        all_values = 0
        for row in range(self.num_classes):
            for column in range(self.num_classes):
                all_values += self.confusion_matrix[row][column]
                if row == column:
                    matrix_diagonal += self.confusion_matrix[row][column]
        if all_values == 0:
            all_values = 1
        return float(matrix_diagonal) / all_values

    def get_average_intersection_union(self):
        values = self.get_intersection_union_per_class()
        return sum(values) / len(values)

    def increment_conf_matrix_from_file(self, gt_file, pd_file):
        """
        Typical use case: num_classes == 9, and both gt_file and pd_file only contains
                          label 1, 2, ..., 8. Label 0 is not used at all.
        """
        with open(gt_file, "r") as gt_f, open(pd_file, "r") as pd_f:
            for gt_line, pd_line in zip(gt_f, pd_f):
                gt_label = int(float(gt_line.strip()))
                pd_label = int(float(pd_line.strip()))
                self.count_predicted(gt_label, pd_label)

    def print_metrics(
        self, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None
    ):
        cm = self.get_confusion_matrix()
        columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
        empty_cell = " " * columnwidth
        # Print header
        print("    " + empty_cell, end=" ")
        for label in labels:
            print("%{0}s".format(columnwidth) % label, end=" ")
        print()
        # Print rows
        for i, label1 in enumerate(labels):
            print("    %{0}s".format(columnwidth) % label1, end=" ")
            for j in range(len(labels)):
                cell = "%{0}.0f".format(columnwidth) % cm[i, j]
                if hide_zeroes:
                    cell = cell if float(cm[i, j]) != 0 else empty_cell
                if hide_diagonal:
                    cell = cell if i != j else empty_cell
                if hide_threshold:
                    cell = cell if cm[i, j] > hide_threshold else empty_cell
                print(cell, end=" ")
            print()


if __name__ == "__main__":
    CM = ConfusionMatrix(3)
    CM.count_predicted(0, 0)
    CM.count_predicted(1, 1)
    CM.count_predicted(2, 2)
    CM.count_predicted(0, 1)
    CM.count_predicted(2, 0)
    CM.print_metrics(["test1", "test2", "test2"])
