import os
import PIL
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

def default_loader(path):
        return PIL.Image.open(path).convert('RGB')

class CarDataset(data.Dataset):
    def __init__(self, lists, label_dict, transform=None, loader=default_loader):
        images = []
        for path in lists:
            cls_name = path.split(os.sep)[-2]
            label = label_dict[cls_name]
            images.append((path, label))
        self.images = images
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path, label = self.images[index]
        image = self.loader(path)

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.Tensor(image)
        return image, label
        

    def __len__(self):
        return len(self.images)

def data_generator(train_list, val_list, label_dict, batch_size):
                
    train_Transformer = transforms.Compose(
        [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.Resize([256, 256]),
        transforms.ToTensor()
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    val_Transformer = transforms.Compose(
        [transforms.Resize([256, 256]),
        transforms.ToTensor()
        ]
    )

    train_data = CarDataset(train_list, label_dict, transform=train_Transformer)
    val_data = CarDataset(val_list, label_dict, transform=val_Transformer)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=2)

    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()
    # print('[INFO] Train images_shape : {}, and labels {}. '.format(images.shape, labels))

    # dataiter = iter(val_loader)
    # images, labels = dataiter.next()
    # print('[INFO] Val images_shape : {}, and labels {}. '.format(images.shape, labels))

    return train_loader, val_loader