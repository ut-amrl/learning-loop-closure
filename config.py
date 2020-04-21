import numpy as np

execution_config = {
  'BATCH_SIZE': 40,
  'NUM_WORKERS': 8,
}

training_config = {
  'TRAIN_SET': 'train',
  'VALIDATION_SET': 'val',
  'NUM_EPOCH': 90
}

evaluation_config = {
  'THRESHOLD_MIN': 0,
  'THRESHOLD_MAX': 4,
  'EVALUATION_SET': 'val'
}

lidar_config = {
  'FOV': np.pi,
  'MAX_RANGE': 10,
}

data_config = {
  'OVERLAP_RADIUS': 4,
  'OVERLAP_SAMPLE_RESOLUTION': 10,
  'CLOSE_DISTANCE_THRESHOLD': 1.5,
  'FAR_DISTANCE_THRESHOLD': 3,
  'OVERLAP_SIMILARITY_THRESHOLD': 0.7,
  'TIME_IGNORE_THRESHOLD': 1,
  'MATCH_REPEAT_FACTOR': 1,
  'AUGMENTATION_PROBABILITY': 0.8
}

DEV_SPLIT = 0.8
data_generation_config = {
  'TIME_SPACING': 0.001,
  'TRAIN_SPLIT': 0.15,
  'DEV_SPLIT': DEV_SPLIT,
  'VAL_SPLIT': 1 - DEV_SPLIT,
  'LIDAR_TOPIC': '/Cobot/Laser',
  'LOCALIZATION_TOPIC': '/Cobot/Localization',
  'MAX_PARTITION_COUNT': 10,
  'MAX_PARTITION_SIZE': 1200,
  'MIN_PARTITION_SIZE': 50
}

visualization_config = {
  'TIMESTEP': 1.5
}
