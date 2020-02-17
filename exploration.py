import os
import cv2
import glob 
import yaml
import logging
import numpy as np
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', 
                             level=logging.INFO)
logger = logging.getLogger()

data_path = './data/data_in_class_folder/'
# check the number of car classes
if os.path.exists(data_path):
    class_names = os.listdir(data_path)
else:
    raise ValueError('Path not exists.')
logger.info('All classes number: {}'.format(len(class_names)))

# check each class image data
# image_data = {class_name: {class_length: value, image_shapes: 2d-list}}
# try read, if not exits, create data.
try:
    with open("image_data.yaml", "r") as file:
        image_data = yaml.load(file, Loader=yaml.FullLoader)
except OSError:
    image_data = {}
    for name in tqdm(class_names):
        image_shapes = []
        image_folder = os.path.join(data_path, name)
        image_paths = glob.glob(image_folder + '/*')
        for path in image_paths:
            image = cv2.imread(path)
            image_shapes.append(str(image.shape))
        image_data[name] = {'class_length': len(image_paths), 'image_shapes': image_shapes}

    with open('image_data.yaml', 'w') as file:
        file.write(yaml.dump(image_data))

logger.info('All images data information: {}'.format(image_data))

class_length_data = {}
list_length = []
for key in image_data.keys():
    class_length_data[key] = {'class_length': image_data[key]['class_length']}
    list_length.append(image_data[key]['class_length'])

logger.info('All images data length information: {}'.format(list_length))
