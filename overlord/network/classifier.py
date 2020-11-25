import torch
import torchvision
from torchsummary import summary
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torchvision
from network.utils import AverageMeter
import os
import numpy as np
from network.utils import NamedTensorDataset, AugmentedDataset

class Classifier:
    def __init__(self, num_classes):
        self.model = torchvision.models.resnet50(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
          nn.Linear(num_ftrs, 2048), nn.Dropout(0.5),
          nn.Linear(2048, num_classes)
        )
        print(num_ftrs)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)

    def train(self, model_dir, imgs, classes, batch_size=64, n_epochs=100, split_size=0.9):
        split_index = int(len(imgs) * split_size)
        np.random.seed(0)
        indexes = np.random.choice(np.arange(imgs.shape[0]), size=int(len(imgs) * split_size), replace=False)
        other_indexes = np.setdiff1d(np.arange(imgs.shape[0]), indexes)
        train_data = dict(
            img=torch.from_numpy(imgs[indexes]).permute(0, 3, 1, 2),
            img_id=torch.from_numpy(np.arange(imgs[indexes].shape[0])),
            class_id=torch.from_numpy(classes[indexes].astype(np.int64))
        )
        train_dataset = NamedTensorDataset(train_data)
                          
        val_data = dict(
            img=torch.from_numpy(imgs[other_indexes]).permute(0, 3, 1, 2),
            img_id=torch.from_numpy(np.arange(imgs[other_indexes].shape[0])),
            class_id=torch.from_numpy(classes[other_indexes].astype(np.int64))
        )
        val_dataset = NamedTensorDataset(val_data)     
        id_criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, pin_memory=True, drop_last=True
        )

        optimizer = Adam(self.model.parameters(), lr=0.001)

        scheduler = CosineAnnealingLR(optimizer,
                                      T_max=n_epochs * len(train_loader),
                                      eta_min=0.000001)
        self.train_(model_dir, id_criterion, optimizer,
                    train_loader, val_dataset, scheduler, n_epochs)

    @staticmethod
    def accuracy_(predictions, labels):
        correct = (predictions.argmax(dim=1).eq(labels)).sum()
        return 100.0 * correct / labels.size(0)

    def train_(self, model_dir, id_criterion, optimizer,
               data_loader, val_dataset, scheduler, n_epochs):
        if os.path.exists(model_dir) and os.path.exists(
                os.path.join(model_dir, 'objs3.pkl')):
            objs = pickle.load(open(os.path.join(model_dir, 'objs3.pkl'), 'rb'))
            epochs = objs['epochs']
            optimizer.load_state_dict(objs['optimizer'])
            self.model.load_state_dict(torch.load(os.path.join(model_dir, 'classifier.pth')))
        else:
            epochs = n_epochs

        for epoch in range(1, epochs + 1):
            pbar = tqdm(iterable=data_loader)
            self.model.train()
            for batch in pbar:
                inputs, labels = batch['img'], batch['class_id']
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(inputs)
                loss = id_criterion(predictions, labels)

                loss.backward()
                optimizer.step()
                scheduler.step()

                train_id_loss =  id_loss.item()
                accuracy = self.accuracy_(predictions, labels).item()

                pbar.set_description_str('epoch #{}'.format(epoch))
                pbar.set_postfix(accuracy=accuracy, id_loss=train_id_loss)
            if epoch % 10 == 0:
                if os.path.exists(model_dir):
                    objs = {'epochs': epochs - epoch,
                            'optimizer': optimizer.state_dict()}
                    with open(os.path.join(model_dir, 'objs3.pkl'), 'wb') as f:
                        pickle.dump(objs, f)
            if epoch % 5 == 0:
                model.eval(val_dataset)
            pbar.close()
            self.save(model_dir)

    def save(self, model_dir):
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        torch.save(self.model.state_dict(), os.path.join(model_dir,
                                                         'classifier.pth'))

    def load(self, model_dir):
        self.model.load_state_dict(torch.load(os.path.join(model_dir,
                                                           'classifier.pth')))

    def eval(self, test_dataset, batch_size=64):
        self.model.eval()
        accuracy, test_loss = AverageMeter(), AverageMeter()
        data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                 shuffle=False, pin_memory=True)
        criterion = nn.CrossEntropyLoss()
        pbar = tqdm(iterable=data_loader)

        for batch in pbar:
            inputs, labels = batch['img'], batch['class_id']
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            predictions = self.model(inputs)
            loss = criterion(predictions, labels)

            test_loss.update(loss.item(), labels.size(0))
            acc = self.accuracy_(predictions, labels)
            accuracy.update(acc.item(), labels.size(0))

            pbar.set_postfix(loss=test_loss.avg, accuracy=accuracy.avg)
        pbar.close()
        print('accuracy on test:', accuracy.avg,
              'loss on test:', test_loss.avg)

