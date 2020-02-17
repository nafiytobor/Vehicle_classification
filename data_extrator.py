import os
import glob


def data_extractor(data_path):
    class_names = os.listdir(data_path)
    list_image_paths = []
    label_dict = {}
    for i, name in enumerate(class_names):
        class_path = os.path.join(data_path, name)
        image_paths = glob.glob(class_path + '/*')
        list_image_paths.append(image_paths)
        label_dict[name] = i

    return list_image_paths, label_dict