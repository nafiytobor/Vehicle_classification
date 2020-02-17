import os
import torch
import torchvision

import data_extrator
import data_preprocessor
import data_generator
import models_struct
import training_processor


def main():
    root_path = os.getcwd()
    data_path = os.path.join(root_path, 'data', 'data_in_class_folder')
    batch_size = 32
    learning_rate = 0.001
    epochs = 10


    cuda_avail = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_avail else "cpu")
    print('[INFO] cuda available state: {}'.format(cuda_avail))
    print('[INFO] cuda current device: {}'.format(device))

    # Data extraction
    list_image_paths, label_dict = data_extrator.data_extractor(data_path=data_path)
    # Data preprocessing
    train_list, val_list = data_preprocessor.data_preprocessor(list_image_paths, train_ratio=0.8)
    # Check Data train/val split
    print('[INFO] Length train list {}, Length validation list {}.'.format(len(train_list), len(val_list)))
    # print('[INFO] Train example : {} .'.format(train_list[0]))
    # print('[INFO] Val example : {} .'.format(val_list[0]))
    print('[INFO] Actually train rate: {} '.format(round(len(train_list)/(len(train_list) + len(val_list)), 2)))

    # Data generation
    train_loader, val_loader = data_generator.data_generator(train_list, val_list, label_dict, batch_size)

    # build model
    model = torchvision.models.resnet18(pretrained=True)
    model.to(device)

    # define compiler 
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    training_processor.training(path=root_path, epochs=epochs, optimizer=optimizer, criterion=criterion, model=model, 
                                            train_loader=train_loader, val_loader=val_loader, cuda_device=device)

    




if __name__ == '__main__':

    main()