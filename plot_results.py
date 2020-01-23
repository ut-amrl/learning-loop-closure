from matplotlib import pyplot as plt
import numpy as np

thresholds = np.array([2, 4, 10, 15, 20, 30])
accuracies = np.array([
    0.645588,
    0.683333,
    0.761765,
    0.805882,
    0.822549,
    0.773039
])
precisions = np.array([
    0.983713,
    0.979487,
    0.936275,
    0.891960,
    0.845588,
    0.714396
])
recalls = np.array([
    0.296078,
    0.374510,
    0.561765,
    0.696078,
    0.789216,
    0.909804
])

plt.plot(thresholds, accuracies, color='r', label="Accuracy")
plt.plot(thresholds, precisions, color='g', label="Precision")
plt.plot(thresholds, recalls, color='b', label="Recall")

plt.legend()

plt.show()