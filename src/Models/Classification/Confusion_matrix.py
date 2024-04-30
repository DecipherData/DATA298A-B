import numpy as np
from tabulate import tabulate

# Provided confusion matrix
confusion_matrix = np.array([
    [219, 0, 0, 2, 1, 2, 2, 1, 2, 1, 1],
    [8, 132, 0, 4, 1, 1, 3, 1, 15, 5, 0],
    [1, 0, 242, 1, 0, 0, 4, 0, 0, 1, 1],
    [1, 1, 1, 194, 0, 1, 11, 3, 1, 1, 0],
    [0, 0, 0, 0, 53, 0, 0, 0, 0, 0, 0],
    [4, 7, 0, 1, 0, 138, 6, 3, 0, 10, 0],
    [7, 0, 0, 4, 0, 3, 119, 0, 1, 4, 1],
    [5, 3, 0, 9, 0, 6, 3, 152, 6, 7, 1],
    [5, 7, 2, 6, 0, 1, 10, 2, 150, 13, 1],
    [5, 2, 1, 5, 0, 2, 5, 2, 8, 159, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 194]
])

# Class headings
class_headings = ["Biowaste", "Cardboard", "Clothes", "E-waste", "Furniture", "Glass", "Medical", "Metal", "Paper", "Plastic", "Shoes"]

# Create a table
table = [[""] + class_headings] + [[class_headings[i]] + list(confusion_matrix[i]) for i in range(len(class_headings))]

# Print the table
print(tabulate(table, headers="firstrow", tablefmt="grid"))
