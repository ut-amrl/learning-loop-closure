import rosbag
import rospy
import argparse
from sensor_msgs.msg import PointCloud2
from helpers import draw_map, visualize_localizations_from_bag

TIMESTEP = 1.5

parser = argparse.ArgumentParser(
    description='Visualize the models classification for given inputs')
parser.add_argument('--bag_file', type=str, help='the bag from which to pull the localizations')
args = parser.parse_args()

bag = rosbag.Bag(args.bag_file)

# TODO do away with matplotlib
import matplotlib.pyplot as plt

visualize_localizations_from_bag(plt, bag, '/Cobot/Localization')
draw_map(plt, '../../cobot/maps/GDC3/GDC3_vector.txt')

plt.show()
