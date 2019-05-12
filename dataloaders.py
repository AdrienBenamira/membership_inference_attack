import torch
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
import numpy as np
from PIL import Image

class custum_CIFAR10(CIFAR10):

    def __init__(self, target, num, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        if self.train:
            if target:
                #target train size 0:config.general.train_target_size
                self.data = self.data[:config.general.train_target_size]
                self.targets = self.targets[:config.general.train_target_size]
            else:
                #shadow train size config.general.train_target_size:-1
                self.data = self.data[config.general.train_target_size:]
                self.targets = self.targets[config.general.train_target_size*(num+1):config.general.train_target_size*(num+2)]
                # TODO: découper en taille config.general.train_target_size
        else:
            if target:
                #target test size 0:config.general.test_target_size
                self.data = self.data[:config.general.test_target_size]
                self.targets = self.targets[:config.general.test_target_size]
            else:
                #shadow test size config.general.test_target_size:-1
                self.data = self.data[config.general.test_target_size:]
                self.targets = self.targets[config.general.test_target_size*(num+1):config.general.test_target_size*(num+2)]
                # TODO: découper en taille config.general.test_target_size

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            index = index % self.config.general.train_target_size
        else:
            index = index % self.config.general.test_target_size
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class custum_CIFAR100(CIFAR100):

    def __init__(self, target, num, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        if self.train:
            if target:
                #target train size 0:config.general.train_target_size
                self.data = self.data[:config.general.train_target_size]
                self.targets = self.targets[:config.general.train_target_size]
            else:
                #shadow train size config.general.train_target_size:-1
                self.data = self.data[config.general.train_target_size:]
                self.targets = self.targets[config.general.train_target_size*(num+1):config.general.train_target_size*(num+2)]
                # TODO: découper en taille config.general.train_target_size
        else:
            if target:
                #target test size 0:config.general.test_target_size
                self.data = self.data[:config.general.test_target_size]
                self.targets = self.targets[:config.general.test_target_size]
            else:
                #shadow test size config.general.test_target_size:-1
                self.data = self.data[config.general.test_target_size:]
                self.targets = self.targets[config.general.test_target_size*(num+1):config.general.test_target_size*(num+2)]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            index = index % self.config.general.train_target_size
        else:
            index = index % self.config.general.test_target_size
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class custum_MNIST(MNIST):

    def __init__(self, target, num, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        if self.train:
            if target:
                #target train size 0:config.general.train_target_size
                self.data = self.data[:config.general.train_target_size]
                self.targets = self.targets[:config.general.train_target_size]
            else:
                #shadow train size config.general.train_target_size:-1
                self.data = self.data[config.general.train_target_size:]
                self.targets = self.targets[config.general.train_target_size*(num+1):config.general.train_target_size*(num+2)]
                # TODO: découper en taille config.general.train_target_size
        else:
            if target:
                #target test size 0:config.general.test_target_size
                self.data = self.data[:config.general.test_target_size]
                self.targets = self.targets[:config.general.test_target_size]
            else:
                #shadow test size config.general.test_target_size:-1
                self.data = self.data[config.general.test_target_size:]
                self.targets = self.targets[config.general.test_target_size*(num+1):config.general.test_target_size*(num+2)]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            index = index % self.config.general.train_target_size
        else:
            index = index % self.config.general.test_target_size
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target



def get_data_for_final_eval(models, all_dataloaders, device):
    Y = []
    X = []
    C = []
    for idx_model, model in enumerate(models):
        model.eval()
        #print(all_dataloaders)
        dataloaders = all_dataloaders[idx_model]
        for phase in ['train', 'val']:
            for batch_idx, (data, target) in enumerate(dataloaders[phase]):
                inputs, labels = data.to(device), target.to(device)
                output = model(inputs)
                for out in output.cpu().detach().numpy():
                    X.append(out)
                    if phase == "train":
                        Y.append(1)
                    else:
                        Y.append(0)
                for cla in labels.cpu().detach().numpy():
                    C.append(cla)
    return (np.array(X), np.array(Y), np.array(C))
