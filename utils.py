# -*- coding:utf-8 -*-
# Author:Ding
import os
import random

from tqdm import tqdm
import platform
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import Dataset, DataLoader


# set experiment seed
def set_seed(seed):
    # python
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    # torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# 边界拓展：镜像
def mirror_hsi(height, width, band, input_normalize, patch = 5):
    padding = patch // 2
    mirror_hsi = np.zeros((height + 2 * padding, width + 2 * padding, band)).astype(np.float16)
    # 中心区域
    mirror_hsi[padding:(padding + height), padding:(padding + width), :] = input_normalize
    # 左边镜像
    for i in range(padding):
        mirror_hsi[padding:(height + padding), i, :] = input_normalize[:, padding - i - 1, :]
    # 右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height + padding), width + padding + i, :] = input_normalize[:, width - 1 - i, :]
    # 上边镜像
    for i in range(padding):
        mirror_hsi[i, :, :] = mirror_hsi[padding * 2 - i - 1, :, :]
    # 下边镜像
    for i in range(padding):
        mirror_hsi[height + padding + i, :, :] = mirror_hsi[height + padding - 1 - i, :, :]

    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0], mirror_hsi.shape[1], mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi


# -------------------------------------------------------------------------------
# 获取patch的图像数据
def get_patches(data, img_height, img_width, channel, patch_size):
    """get patches"""
    # patch_size:the size of target pixel's neighborhood
    patches = np.empty([img_height * img_width, patch_size, patch_size, channel],
                       dtype = 'float16')  # img_height * img_width
    for i in range(img_height):
        for j in range(img_width):
            patches[i * img_width + j, ...] = data[i:i + patch_size, j:j + patch_size, :]

    # patches = (img_height * img_width, patch_size, patch_size, band_size)
    return patches


def make_data(dataset: str, patch_size = 5, n_class = 2):
    """读取数据——>将数据整理为B×b×N格式——>标准化——>"""
    global path, data_t1, data_t2, y
    if platform.system().lower() == 'windows':
        print("[Info]: Use Windows!")
        path = 'E:'
    elif platform.system().lower() == 'linux':
        print("[Info]: Use Linux!")
        path = '..'
    if dataset == 'China':
        data = sio.loadmat(path + '/DataSet/China&USA/China_Change_Dataset.mat')  # data set X
        data_t1 = data['T1']
        data_t2 = data['T2']
        if n_class == 2:  # the binary ground truth
            y = 1.0 * data['Binary']
        else:  # multiple
            y = sio.loadmat(path + '/DataSet/China&USA/China_multiple_gt.mat')['Multiple']
    elif dataset == 'USA':
        data = sio.loadmat(path + '/DataSet/China&USA/USA_Change_Dataset.mat')  # data set X
        data_t1 = data['T1']
        data_t2 = data['T2']
        if n_class == 2:  # the binary ground truth
            y = 1.0 * data['Binary']
        else:  # multiple
            y = sio.loadmat(path + '/DataSet/China&USA/USA_multiple_gt.mat')['Multiple']
    elif dataset == 'Yellow River':
        data_t1 = sio.loadmat(path + '/DataSet/Yellow River/river_before.mat')['river_before']  # data set X
        data_t2 = sio.loadmat(path + '/DataSet/Yellow River/river_after.mat')['river_after']  # data set X
        y = sio.loadmat(path + '/DataSet/Yellow River/groundtruth.mat')['lakelabel_v1'] / 255  # the binary ground truth
    elif dataset == 'Bay_Area':
        data_t1 = sio.loadmat(path + '/DataSet/three_data/bayArea/Bay_Area_2013.mat')['HypeRvieW']
        data_t2 = sio.loadmat(path + '/DataSet/three_data/bayArea/Bay_Area_2015.mat')['HypeRvieW']
        y = sio.loadmat(path + '/DataSet/three_data/bayArea/bayArea_gtChanges.mat')['HypeRvieW']
    elif dataset == 'Barbara':
        data_t1 = sio.loadmat(path + '/DataSet/three_data/santaBarbara/barbara_2013_170.mat')['HypeRvieW']
        data_t2 = sio.loadmat(path + '/DataSet/three_data/santaBarbara/barbara_2014_170.mat')['HypeRvieW']
        y = sio.loadmat(path + '/DataSet/three_data/santaBarbara/barbara_gtChanges.mat')['HypeRvieW']

    img_height, img_width, channel = data_t1.shape
    # normalize data by band norm
    data_t1_normalize, data_t2_normalize = np.zeros(data_t1.shape), np.zeros(data_t1.shape)
    for i in range(channel):
        x_min = min(data_t1[..., i].min(), data_t2[..., i].min())
        x_max = max(data_t1[..., i].max(), data_t2[..., i].max())
        data_t1_normalize[..., i] = (data_t1[..., i] - x_min) / (x_max - x_min)
        data_t2_normalize[..., i] = (data_t2[..., i] - x_min) / (x_max - x_min)

    # get patches
    # [img_height * img_width, patch_size, patch_size, band_size]
    mirror_t1 = mirror_hsi(img_height, img_width, channel, data_t1_normalize, patch_size)
    mirror_t2 = mirror_hsi(img_height, img_width, channel, data_t2_normalize, patch_size)
    patches_t1 = get_patches(mirror_t1, img_height, img_width, channel, patch_size).astype(np.float16)
    patches_t2 = get_patches(mirror_t2, img_height, img_width, channel, patch_size).astype(np.float16)

    y = np.reshape(y, (-1,))
    if dataset in ['Bay_Area', 'Barbara']:
        labeled = np.argwhere(y != 0)  # 0-black:unlabeled, 1-gray:changed, 2-white:unchanged
        y = 2 - y  # 替换label
    else:
        labeled = np.array([i for i in range(len(y))])

    x = np.transpose(np.concatenate((patches_t1, patches_t2), axis = -1), (0, 3, 1, 2))

    target = y
    del path, data_t1, data_t2, y

    return x, target, labeled


class MyDataset(Dataset):
    def __init__(self, x, y = None, transform = None):
        if transform:
            self.x = x
        else:
            self.x = torch.FloatTensor(x)
        # label是需要LongTensor型
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X


def split_train_val(x, y3, args):
    """ratio: the ratio of train data"""
    if not os.path.exists('./train_index'):
        os.mkdir(f'./train_index')
    ratio_str = "{:.2f}".format(args.ratio).replace('.', '')
    if not os.path.exists(f'./train_index/{args.dataset}-train-index-{ratio_str}-{args.seed}.npy'):
        train_x_set, val_x_set, train_y_set, val_y_set = train_test_split(x, y3, test_size = 1 - args.ratio,
                                                                          random_state = args.seed,
                                                                          stratify = y3)
        train_index = get_train_index(x, train_x_set)
        np.save(f'./train_index/{args.dataset}-train-index-{ratio_str}-{args.seed}.npy', train_index)
    else:
        train_index = np.load(f'./train_index/{args.dataset}-train-index-{ratio_str}-{args.seed}.npy')
        index = np.arange(0, len(y3))
        val_index = np.delete(index, train_index)
        train_x_set = x[train_index]
        train_y_set = y3[train_index]
        val_x_set = x[val_index]
        val_y_set = y3[val_index]

    return train_x_set, train_y_set, val_x_set, val_y_set


def get_train_index(data, data_train):
    print('Start get train index!')
    train_index = []
    pbar = tqdm(total = len(data_train), ncols = 0, desc = f"Processing", unit = "step")
    for i in range(len(data_train)):
        pbar.set_postfix(step = i + 1)
        pbar.update()
        for j in range(len(data)):
            if (data_train[i, :10, :10] == data[j, :10, :10]).all():
                train_index.append(j)
                break
    pbar.close()

    return train_index


# -------------------------------------------------------------------------------
class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n = 1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


# -------------------------------------------------------------------------------
def accuracy(output, target, topk = (1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, target, pred.squeeze()


# -------------------------------------------------------------------------------
def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype = np.float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA


# -------------------------------------------------------------------------------
def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA


# -------------------------------------------------------------------------------
# validate model
def valid_epoch(model, valid_loader, criterion, device):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)

        batch_pred = model(batch_data)

        loss = criterion(batch_pred, batch_target)

        prec1, t, p = accuracy(batch_pred, batch_target, topk = (1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return tar, pre


def test_epoch(model, test_loader, device):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)

        batch_pred = model(batch_data)

        _, pred = batch_pred.topk(1, 1, True, True)
        pp = pred.squeeze()
        pre = np.append(pre, pp.data.cpu().numpy())
    return pre


# -------------------------------------------------------------------------------
# train model
def train_epoch(model, train_loader, criterion, optimizer, device):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)

        optimizer.zero_grad()
        batch_pred = model(batch_data)
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()

        prec1, t, p = accuracy(batch_pred, batch_target, topk = (1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre
# Example
# data, target, labels = make_data('China', patch_size = 3, band_patch = 3)
