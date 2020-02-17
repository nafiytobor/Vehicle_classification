import random


def data_preprocessor(list_image_paths, train_ratio):
    # train val split 
    train_list = []
    val_list = []
    for one_class_image_paths in list_image_paths:
        class_length = len(one_class_image_paths)
        train_length = int(train_ratio * class_length)
        train_idxs = random.sample(range(0, class_length-1), train_length)
        for i in range(0,class_length):
            if i in train_idxs:
                train_list.append(one_class_image_paths[i])
            else:
                val_list.append(one_class_image_paths[i])
    return train_list, val_list