import math
radius = 7
threshold = math.pi * math.pow(radius, 2) * 0.75
import random

def compute_overlap(loc, locB):
    d = math.sqrt(math.pow((locB[0] - loc[0]), 2) + math.pow((locB[1] - loc[1]), 2))

    if d == 0:
        return abs(locB[2] - loc[2]) < math.pi * 0.75

    if d < 2 * radius:
        sqr = radius * radius

        x = (d * d) / (2 * d)
        z = x * x
        y = math.sqrt(sqr - z)

        overlap = sqr * math.asin(y / radius) + sqr * math.asin(y / radius) - y * (x + math.sqrt(z))
        print("Overlap", overlap, "THRESHOLD", threshold)
        return overlap

    return 0

import matplotlib.pyplot as plt
import matplotlib.patches as patches


while True:
    location = [100 + random.random() * 10, 50 + random.random() * 10, 0.25 + random.random()]
    similar_loc = [100 + random.random() * 10, 50 + random.random() * 10, -1.25 + random.random()]
    print(location, similar_loc)
    compute_overlap(location, similar_loc)
    circle1 = patches.Arc((location[0], location[1]), radius, radius, angle=180 /math.pi * location[2], theta1=-135, theta2=135, color='r')
    circle2 = patches.Arc((similar_loc[0], similar_loc[1]), radius, radius, angle=180 / math.pi * similar_loc[2], theta1=-135, theta2=135, color='b')

    fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot

    plt.xlim((100, 150))
    plt.ylim((50, 100))
    ax.set_aspect('equal')
    ax.add_artist(circle1)
    ax.add_artist(circle2)

    fig.show()
    plt.show()