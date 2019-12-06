from helpers import compute_overlap
import numpy as np
import matplotlib.pyplot as plt

loc_a = np.array([0, 0, 0])
loc_b = np.array([0, 0, np.pi / 2])

loc_c = np.array([-2, 0, 0])
loc_d = np.array([2, 0, np.pi])
loc_e = np.array([.75 * 4, 0, np.pi])

# print("AA", compute_overlap(loc_a, loc_a))
# print("AB", compute_overlap(loc_a, loc_b))
# print("AC", compute_overlap(loc_a, loc_c))
# print("AD", compute_overlap(loc_a, loc_d))
# print("AE", compute_overlap(loc_a, loc_e))

test_a = [66.32367  , 51.624554 , -1.6080709]
test_b = [66.05289  , 50.675644 , -2.0107317]

print("test", compute_overlap(test_b, test_a))
print("test", compute_overlap(test_a, test_b))

plt.show()