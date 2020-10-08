import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
import torchvision.transforms as transforms

class NonAugmentDataset(Dataset):
    def __init__(self, name, load_function=None):
        import os
        file_name = os.path.join('./data', name + '.pth')
        if os.path.isfile(file_name):
            self.x, self.y = torch.load(file_name)
            self.n = len(self.x)
        else:
            dataset = load_function()
            size = dataset[0][0].shape
            self.n = len(dataset)
            self.x = torch.empty(self.n, *size)
            self.y = torch.empty(self.n, dtype=torch.long)
            for i in range(self.n):
                self.x[i].copy_(dataset[i][0])
                self.y[i] = dataset[i][1]
            torch.save([self.x, self.y], os.path.join('./data', name + '.pth'))
        self.x = self.x.cuda()
        self.y = self.y.cuda()

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n

def mnist(augment):
    from functools import partial
    dataset = []
    for train in (True, False):
        transform = transforms.Compose(
            ([transforms.RandomCrop(28, padding=2)] if augment and train else []) +
            [transforms.ToTensor(),transforms.Normalize(mean=[0.5], std=[1.0])])
        fun = partial(MNIST, root='./data', train=train, transform=transform, download=True)
        if augment:
            dataset.append(fun())
        else:
            dataset.append(NonAugmentDataset('mnist_' + ('train' if train else 'test'), fun))
    return dataset

def cifar10(augment):
    dataset = []
    for train in (True, False):
        transform = transforms.Compose(
            ([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if augment and train else []) +
            [transforms.ToTensor(), transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])])
        dataset.append(CIFAR10('./data/', train=train, transform=transform, download=True))
    return dataset

class NonAugmentDataLoader():
    def __init__(self, dataset, batch_size, shuffle):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.n = (len(self.dataset) -1 ) // self.batch_size + 1

    def __iter__(self):
        self.mini_batch = 0
        if self.shuffle:
            self.ids = torch.randperm(len(self.dataset))
        else:
            self.ids = torch.arange(0, len(self.dataset))
        return self

    def __len__(self):
        return self.n

    def __next__(self):
        self.mini_batch += 1
        if self.mini_batch > self.n:
            raise StopIteration
        if self.mini_batch == self.n:
            s = slice((self.mini_batch - 1) * self.batch_size, len(self.dataset))
        else:
            s = slice((self.mini_batch - 1) * self.batch_size, self.mini_batch * self.batch_size)
        return self.dataset.x[self.ids[s]], self.dataset.y[self.ids[s]]
