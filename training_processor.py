import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def training(path, epochs, optimizer, criterion, model, train_loader, val_loader, cuda_device):

    writer = SummaryWriter(path)
    print('[INFO] Start training .........................')

    for epoch in range(epochs):
        epoch_loss = 0
        print('[INFO] [Epoch %d] .................' %(epoch + 1))
        for data in tqdm(train_loader):
            inputs, labels = data[0].to(cuda_device), data[1].to(cuda_device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)

            ce_loss = criterion(outputs, labels)
            ce_loss.backward()
            epoch_loss += ce_loss.item()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)

        print('[INFO] [Epoch %d] train_loss: %d' % (epoch + 1, round(epoch_loss/len(train_loader),4)))

        correct = 0
        total = 0
        with torch.no_grad():
            for data in train_loader:
                images, labels = data[0].to(cuda_device), data[1].to(cuda_device)
                # predict
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            train_acc = 100 * correct / total

        print('[INFO] [Epoch %d] Accuracy of train images: %d %%' % (epoch + 1, train_acc))

        correct = 0
        total = 0
        epoch_test_loss = 0
        with torch.no_grad():
            for data in tqdm(val_loader):
                images, labels = data[0].to(cuda_device), data[1].to(cuda_device)
                # predict
                outputs = model(images)
                test_loss = criterion(outputs, labels)
                epoch_test_loss += test_loss
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            test_acc = 100 * correct / total

        print('[INFO] [Epoch %d] test_loss: %d' % (epoch + 1, round(epoch_test_loss/len(val_loader),4)))
        print('[INFO] [Epoch %d] Accuracy of val images: %d %%' % (epoch + 1, test_acc))

        writer.add_scalar('Train_loss', epoch_loss/len(train_loader), epoch+1)
        writer.add_scalar('Train_accuracy', train_acc, epoch+1)
        writer.flush()
        writer.add_scalar('Test_loss', epoch_test_loss/len(val_loader), epoch+1)
        writer.add_scalar('Test_accuracy', test_acc, epoch+1)
        writer.flush()

    print('Done Training!')