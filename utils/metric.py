from __future__ import print_function
import numpy as np

class ConfusionMatrix:

  """Streaming interface to allow for any source of predictions. Initialize it, count predictions one by one, then print confusion matrix and intersection-union score"""
  def __init__(self, number_of_labels = 2):
    self.number_of_labels = number_of_labels
    self.confusion_matrix = np.zeros(shape=(self.number_of_labels,self.number_of_labels))

  def count_predicted(self, ground_truth, predicted, number_of_added_elements=1):
    self.confusion_matrix[ground_truth][predicted] += number_of_added_elements

  """labels are integers from 0 to number_of_labels-1"""
  def get_count(self, ground_truth, predicted):
    return self.confusion_matrix[ground_truth][predicted]

  """returns list of lists of integers; use it as result[ground_truth][predicted]
     to know how many samples of class ground_truth were reported as class predicted"""
  def get_confusion_matrix(self):
    return self.confusion_matrix
  
  """returns list of 64-bit floats"""
  def get_intersection_union_per_class(self):
    matrix_diagonal = [self.confusion_matrix[i][i] for i in xrange(self.number_of_labels)]
    errors_summed_by_row = [0] * self.number_of_labels
    for row in xrange(self.number_of_labels):
      for column in xrange(self.number_of_labels):
        if row != column:
          errors_summed_by_row[row] += self.confusion_matrix[row][column]
    errors_summed_by_column = [0] * self.number_of_labels
    for column in xrange(self.number_of_labels):
      for row in xrange(self.number_of_labels):
        if row != column:
          errors_summed_by_column[column] += self.confusion_matrix[row][column]
          
    divisor = [0] * self.number_of_labels
    for i in xrange(self.number_of_labels):
      divisor[i] = matrix_diagonal[i] + errors_summed_by_row[i] + errors_summed_by_column[i]
      if matrix_diagonal[i] == 0:
        divisor[i] = 1
               
    return [float(matrix_diagonal[i]) / divisor[i] for i in xrange(self.number_of_labels)]
  
  """returns 64-bit float"""  
  def get_overall_accuracy(self):
    matrix_diagonal = 0
    all_values = 0
    for row in xrange(self.number_of_labels):
      for column in xrange(self.number_of_labels):
        all_values += self.confusion_matrix[row][column]
        if row == column:
          matrix_diagonal += self.confusion_matrix[row][column]
    if all_values == 0:
      all_values = 1
    return float(matrix_diagonal) / all_values


  def get_average_intersection_union(self):
    values = self.get_intersection_union_per_class()
    return sum(values) / len(values)

  def build_conf_matrix_from_file(self, ground_truth_file, classified_file):
    #read line by line without storing everything in ram
    with open(ground_truth_file, "r") as f_gt, open(classified_file, "r") as f_cl:
      for index, (line_gt, line_cl) in enumerate(izip(f_gt, f_cl)):
         label_gt = int(line_gt)
         label_cl_ = int(line_cl)
         label_cl = max([min([label_cl_, 10000]), 1]) #protection against erroneous submissions: no infinite labels (for instance NaN) or classes smaller 1
         if label_cl_ != label_cl:
             return -1
         max_label = max([label_gt, label_cl])
         if max_label > self.number_of_labels:
            #resize to larger confusion matrix
            b = np.zeros((max_label,max_label))
            for row in xrange(self.number_of_labels):
              for column in xrange(self.number_of_labels):
                 b[row][column] = self.confusion_matrix[row][column]
            self.confusion_matrix = b
            self.number_of_labels = max_label
            
         if label_gt == 0: 
            continue
         self.confusion_matrix[label_gt - 1][label_cl - 1] += 1
         return 0

  def print_cm(self, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
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



if __name__ == '__main__':
  CM = ConfusionMatrix(3)
  CM.count_predicted(0,0,3)
  CM.count_predicted(1,1,4)
  CM.count_predicted(2,2,2)
  CM.count_predicted(0,1,1)
  CM.count_predicted(2,0,1)
  CM.print_cm(["test1","test2","test2"])
