from PIL import Image
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
class MyDataset(Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, rawroot, tureroot,falseroot, datatxt, transform=None, target_transform=None):  # 初始化一些需要传入的参数
        fh = open(datatxt, 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []  # 创建一个名为img的空列表，一会儿用来装东西
        for line in fh:  # 按行循环txt文本中的内容
            line = line.rstrip()  # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()  # 通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
            imgs.append((words[0], 0)) # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
            # 很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable
        self.falseroot = falseroot
        self.tureroot = tureroot
        self.rawroot = rawroot
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        fn, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        allimg = cv2.imread(self.rawroot + fn)  # 按照path读入图片from PIL import Image # 按照路径读取图片
        size=allimg.shape[1]//5
        rawimg = allimg[:,0:size,:]
        tureimg1 = allimg[:, 1* size:2 * size,:]
        tureimg2 = allimg[:, 2 * size:3 * size,:]
        tureimg3 = allimg[:, 3 * size:4 * size,:]
        falseimg = allimg[:, 4 * size:5 * size,:]
        rawimg = cv2.resize(rawimg, (224, 224))
        tureimg1 = cv2.resize(tureimg1, (224, 224))
        tureimg2 = cv2.resize(tureimg2, (224, 224))
        tureimg3 = cv2.resize(tureimg3, (224, 224))
        falseimg = cv2.resize(falseimg, (224, 224))
        if self.transform is not None:
            rawimg = self.transform(rawimg)  # 是否进行transform
            falseimg = self.transform(falseimg)
            tureimg1 = self.transform(tureimg1)# 是否进行transform
            tureimg2 = self.transform(tureimg2)  # 是否进行transform
            tureimg3 = self.transform(tureimg3)  # 是否进行transform

        return rawimg, tureimg1,tureimg2,tureimg3, falseimg,1,0  # return很关键，return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):
        return len(self.imgs)

class TestDataset(Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, rawroot, tureroot,falseroot,xmlroot=None, datatxt=None, transform=None, target_transform=None):  # 初始化一些需要传入的参数
        fh = open(datatxt, 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []  # 创建一个名为img的空列表，一会儿用来装东西
        for line in fh:  # 按行循环txt文本中的内容
            line = line.rstrip()  # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()  # 通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
            imgs.append((words[0],0))# 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
            # 很显然，根据我刚才截图所示txt的
        self.tureroot = tureroot
        self.rawroot = rawroot
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.falseroot = falseroot

    def __getitem__(self, index):
        fn, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        allimg = cv2.imread(self.rawroot + fn)  # 按照path读入图片from PIL import Image # 按照路径读取图片
        size = allimg.shape[1] // 5
        rawimg = allimg[:, 0:size, :]
        tureimg1 = allimg[:, 1 * size:2 * size, :]
        tureimg2 = allimg[:, 2 * size:3 * size, :]
        tureimg3 = allimg[:, 3 * size:4 * size, :]
        falseimg = allimg[:, 4 * size:5 * size, :]
        rawimg = cv2.resize(rawimg, (224, 224))
        tureimg1 = cv2.resize(tureimg1, (224, 224))
        tureimg2 = cv2.resize(tureimg2, (224, 224))
        tureimg3 = cv2.resize(tureimg3, (224, 224))
        falseimg = cv2.resize(falseimg, (224, 224))
        if self.transform is not None:
            rawimg = self.transform(rawimg)  # 是否进行transform
            falseimg = self.transform(falseimg)
            tureimg1 = self.transform(tureimg1)  # 是否进行transform
            tureimg2 = self.transform(tureimg2)  # 是否进行transform
            tureimg3 = self.transform(tureimg3)  # 是否进行transform
        return rawimg, tureimg1,tureimg2,tureimg3, falseimg,label,fn  # return很关键，return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):
        return len(self.imgs)
