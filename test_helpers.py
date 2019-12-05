from helpers import compute_overlap
import numpy as np

loc_a = np.array([0, 0, 0])
loc_b = np.array([0, 0, np.pi / 2])

loc_c = np.array([-2, 0, 0])
loc_d = np.array([2, 0, np.pi])
loc_e = np.array([.75 * 4, 0, np.pi])

print("AA", compute_overlap(loc_a, loc_a))
print("AB", compute_overlap(loc_a, loc_b))
print("AC", compute_overlap(loc_a, loc_c))
print("AD", compute_overlap(loc_a, loc_d))
print("AE", compute_overlap(loc_a, loc_e))