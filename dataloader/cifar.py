# -*- coding: utf-8 -*-
"""
@author: QgZhan
@contact: zhanqg@foxmail.com
@file: cifar.py
@time: 2022/4/19 11:19
"""

import os
from torch.utils.data import Dataset
from dataloader.dataloader_utils import *
from torchvision import datasets, transforms
from spikingjelly.datasets import cifar10_dvs
from torch.utils.data.sampler import SubsetRandomSampler


# your own data dir
DIR = {'CIFAR10': '/data/zhan/CV_data/cifar10',
       'CIFAR10DVS': '/data/zhan/Event_Camera_Datasets/CIFAR10DVS',
       'CIFAR10DVS_CATCH': '/data/zhan/Event_Camera_Datasets/CIFAR10DVS_dst_cache'
       }


def get_cifar10(batch_size, train_set_ratio=1.0):
    """
    get the train loader and test loader of cifar10.
    :return: train_loader, test_loader
    """
    trans_train = transforms.Compose([transforms.Resize(48),
                                      transforms.RandomCrop(48, padding=4),
                                      transforms.RandomHorizontalFlip(),  # 随机水平翻转
                                      CIFAR10Policy(),  # TODO: 待注释
                                      transforms.ToTensor(),
                                      # transforms.RandomGrayscale(),  # 随机变为灰度图
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 归一化
                                      # transforms.Normalize((0., 0., 0.), (1, 1, 1)),
                                      # Cutout(n_holes=1, length=16)  # 随机挖n_holes个length * length的洞
                                      ])
    trans_test = transforms.Compose([transforms.Resize(48),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    train_data = datasets.CIFAR10(DIR['CIFAR10'], train=True, transform=trans_train, download=True)
    test_data = datasets.CIFAR10(DIR['CIFAR10'], train=False, transform=trans_test, download=True)

    # take train set by train_set_ratio
    n_train = len(train_data)
    split = int(n_train * train_set_ratio)
    indices = list(range(n_train))
    random.shuffle(indices)
    train_sampler = SubsetRandomSampler(indices[:split])

    if train_set_ratio < 1.0:
        train_dataloader = DataLoaderX(train_data, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True,
                                       sampler=train_sampler, pin_memory=True)
    else:
        train_dataloader = DataLoaderX(train_data, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True,
                                       pin_memory=True)
    test_dataloader = DataLoaderX(test_data, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False,
                                  pin_memory=True)

    return train_dataloader, test_dataloader


def get_cifar10_DVS(batch_size, T, split_ratio=0.9, train_set_ratio=1, size=48, encode_type='TET'):
    """
    get the train loader and test loader of cifar10.
    :param batch_size:
    :param T:
    :param split_ratio: the ratio of train set: test set
    :param train_set_ratio: the real used train set ratio
    :param size:
    :param encode_type:
    :return: train_loader, test_loader
    """
    if encode_type is "spikingjelly":
        trans = DVSResize((size, size), T)

        train_set_pth = os.path.join(DIR['CIFAR10DVS_CATCH'], f'train_set_{T}_{split_ratio}_{size}.pt')
        test_set_pth = os.path.join(DIR['CIFAR10DVS_CATCH'], f'test_set_{T}_{split_ratio}_{size}.pt')

        if os.path.exists(train_set_pth) and os.path.exists(test_set_pth):
            train_set = torch.load(train_set_pth)
            test_set = torch.load(test_set_pth)
        else:
            origin_set = cifar10_dvs.CIFAR10DVS(root=DIR['CIFAR10DVS'], data_type='frame', frames_number=T,
                                                split_by='number', transform=trans)

            train_set, test_set = split_to_train_test_set(split_ratio, origin_set, 10)
            if not os.path.exists(DIR['CIFAR10DVS_CATCH']):
                os.makedirs(DIR['CIFAR10DVS_CATCH'])
            torch.save(train_set, train_set_pth)
            torch.save(test_set, test_set_pth)
    elif encode_type is "TET":
        path = '/data/zhan/Event_Camera_Datasets/CIFAR10DVS/temporal_effecient_training_0.9'
        train_path = path + '/train'
        test_path = path + '/test'
        train_set = DVSCifar10(root=train_path)
        test_set = DVSCifar10(root=test_path)
    elif encode_type is "3_channel":
        path = '/data/zhan/Event_Camera_Datasets/CIFAR10DVS/temporal_effecient_training_0.9'
        train_path = path + '/train'
        test_path = path + '/test'
        train_set = Channel_3_DVSCifar10(root=train_path)
        test_set = Channel_3_DVSCifar10(root=test_path)

    # take train set by train_set_ratio
    n_train = len(train_set)
    split = int(n_train * train_set_ratio)
    indices = list(range(n_train))
    random.shuffle(indices)
    train_sampler = SubsetRandomSampler(indices[:split])
    # valid_sampler = SubsetRandomSampler(indices[split:])

    # generate dataloader
    # train_data_loader = DataLoaderX(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True,
    #                                 num_workers=8, pin_memory=True)
    train_data_loader = DataLoaderX(dataset=train_set, batch_size=batch_size, shuffle=False, drop_last=True,
                                    sampler=train_sampler, num_workers=8,
                                    pin_memory=True)  # SubsetRandomSampler 自带shuffle，不能重复使用
    test_data_loader = DataLoaderX(dataset=test_set, batch_size=batch_size, shuffle=False, drop_last=False,
                                   num_workers=8, pin_memory=True)

    return train_data_loader, test_data_loader


def get_cifar100(batch_size):
    """
    get the train loader and test loader of cifar100.
    :return: train_loader, test_loader
    """
    trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]],
                                                       std=[n / 255. for n in [68.2, 65.4, 70.4]]),
                                  Cutout(n_holes=1, length=16)
                                  ])
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]],
                                                     std=[n / 255. for n in [68.2, 65.4, 70.4]])])

    train_data = datasets.CIFAR100(DIR['CIFAR100'], train=True, transform=trans_t, download=True)
    test_data = datasets.CIFAR100(DIR['CIFAR100'], train=False, transform=trans, download=True)

    train_dataloader = DataLoaderX(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    test_dataloader = DataLoaderX(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    return train_dataloader, test_dataloader


class DVSCifar10(Dataset):
    # This code is form https://github.com/Gus-Lab/temporal_efficient_training
    def __init__(self, root, train=True, transform=True, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.resize = transforms.Resize(size=(48, 48))  # 48 48
        self.tensorx = transforms.ToTensor()
        self.imgx = transforms.ToPILImage()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = torch.load(self.root + '/{}.pt'.format(index))
        # if self.train:
        new_data = []
        for t in range(data.size(0)):
            new_data.append(self.tensorx(self.resize(self.imgx(data[t, ...]))))
        data = torch.stack(new_data, dim=0)

        if self.transform:
            flip = random.random() > 0.5
            if flip:
                data = torch.flip(data, dims=(3,))
            off1 = random.randint(-5, 5)
            off2 = random.randint(-5, 5)
            data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))
        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target.long().squeeze(-1)

    def __len__(self):
        return len(os.listdir(self.root))


class Channel_3_DVSCifar10(Dataset):
    def __init__(self, root, train=True, transform=True, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.resize = transforms.Resize(size=(48, 48))  # 48 48
        self.tensorx = transforms.ToTensor()
        self.imgx = transforms.ToPILImage()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = torch.load(self.root + '/{}.pt'.format(index))
        T, C, H, W = data.shape
        # if self.train:
        new_data = []
        for t in range(T):
            tmp = data[t, ...]  # (2, H, W)
            tmp = torch.cat((tmp, torch.zeros(1, H, W)), dim=0)  # (3, H, W)
            mask = (torch.randn((H, W)) > 0).to(data)
            tmp[2].data = tmp[0].data * mask + tmp[1].data * (1 - mask)
            new_data.append(self.tensorx(self.resize(self.imgx(tmp))))
        data = torch.stack(new_data, dim=0)

        if self.transform:
            flip = random.random() > 0.5
            if flip:
                data = torch.flip(data, dims=(3,))
            off1 = random.randint(-5, 5)
            off2 = random.randint(-5, 5)
            data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))
        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target.long().squeeze(-1)

    def __len__(self):
        return len(os.listdir(self.root))

