import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

def draw_map(plt, map_file):
    segments = []
    with open(map_file) as f:
        for line in f:
            segments.append([float(n) for n in line.split(',')])

    line_segments = [[(x1, y1), (x2, y2)] for x1, y1, x2, y2 in segments]

    lc = LineCollection(line_segments, color=["k","black"], lw=1)
    plt.gca().add_collection(lc)
    plt.gca().autoscale()

def visualize_location(plt, location, color='blue'):
    orientation = fix_angle(location[2])
    arc_patch(location[:2], RADIUS, np.rad2deg(orientation - FOV/2), np.rad2deg(orientation + FOV/2), ax=plt.gca(), fill=False, color=color, zorder=1)
    plt.gca().set_aspect('equal')
    plt.gca().autoscale()

def visualize_cloud(plt, cloud, color='blue'):
    bound = max(np.max(cloud[:, 0]), np.max(cloud[:, 1]), -np.min(cloud[:, 0]), -np.min(cloud[:, 1]))
    plt.xlim(-bound, bound)
    plt.ylim(-bound, bound)
    plt.scatter(cloud[:, 0], cloud[:, 1], c=color, marker='.')

def arc_patch(center, radius, theta1, theta2, ax=None, resolution=50, **kwargs):
    # make sure ax is not empty
    if ax is None:
        ax = plt.gca()
    # generate the points
    theta = np.linspace(np.radians(theta1), np.radians(theta2), resolution)
    points = np.vstack((radius*np.cos(theta) + center[0], 
                        radius*np.sin(theta) + center[1]))
    points = np.append(points, [[center[0]], [center[1]]], axis=1)
    # build the polygon and add it to the axes
    poly = mpatches.Polygon(points.T, closed=True, **kwargs)
    ax.add_patch(poly)
    return poly

def embedding_for_scan(model, cloud, batched=False):
    clouds = cloud if batched else [cloud,]
    clouds = torch.tensor(clouds)
    clouds = clouds.transpose(2, 1).cuda()
    result = model(clouds)
    del clouds
    return result

def visualize_localizations_from_bag(plt, bag, localization_topic):
    localizations = []
    print ("Loading Localization from Bag file")
    print("Start time:", bag.get_start_time())
    print("End time:", bag.get_end_time())

    for topic, msg, t in tqdm(bag.read_messages(topics=[localization_topic])):
        localizations.append([msg.x, msg.y, msg.angle])

    localization_xs = [l[0] for l in localizations]
    localization_ys = [l[1] for l in localizations]

    plt.plot(localization_xs, localization_ys)