# -*- coding:utf-8 -*-
# Author:Ding
import os
import time
import argparse

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import ParameterGrid
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils import set_seed, make_data, MyDataset, split_train_val, output_metric, \
    train_epoch, valid_epoch, test_epoch
from vit_pytorch import ViT


def main(args, search_num):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {device} now!")

    all_x, all_y, labeled_index = make_data(args.dataset, patch_size = args.patches)
    # 创建训练数据集 & 验证数据集
    labeled_x, labeled_y = all_x[labeled_index].squeeze(), all_y[labeled_index].squeeze()
    train_x_set, train_y_set, val_x_set, val_y_set = split_train_val(labeled_x, labeled_y, args)
    # 创建Dataset
    train_set = MyDataset(train_x_set, train_y_set)
    train_loader = DataLoader(train_set, batch_size = args.batch_size,
                              num_workers = args.num_workers, pin_memory = True, shuffle = True)
    val_set = MyDataset(val_x_set, val_y_set)
    val_loader = DataLoader(val_set, batch_size = args.batch_size,
                            num_workers = args.num_workers, pin_memory = True, shuffle = True)
    print(f"[Info]: Finish loading data!", flush = True)
    # -------------------------------------------------------------------------------
    # create model
    model = ViT(
        patch_size = args.patches,
        num_feats = 1 * 4,
        band_size = args.band_size,
        num_classes = args.num_classes,
        dim = args.dim,
        depth = args.depth,
        heads = args.head,
        mlp_dim = 8,
        dropout = 0.1,
        emb_dropout = 0.1,
    )
    model = model.to(device)
    # criterion
    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay = args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = args.epoches // 10, gamma = args.gamma)
    # -------------------------------------------------------------------------------
    if args.flag_test == 'test':
        model.load_state_dict(torch.load(f'./output/{search_num}_model_parameter.pkl'))
        model.eval()
        # tar_v, pre_v = valid_epoch(model, val_loader, criterion, device)
        # OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)

        all_set = MyDataset(all_x, all_y)
        all_loader = DataLoader(all_set, batch_size = args.batch_size,
                                num_workers = args.num_workers, pin_memory = True, shuffle = False)
        img_size = {'China': [420, 140], 'USA': [307, 241], 'Yellow River': [463, 241], 'Bay_Area': [600, 500]}
        height, width = img_size[args.dataset]
        # output classification maps
        tic = time.time()
        pre_u = test_epoch(model, all_loader, device)
        toc = time.time()
        print("Inference Time: {:.2f}".format(toc - tic))

        OA2, AA_mean2, Kappa2, AA2 = output_metric(all_y[labeled_index], pre_u[labeled_index])
        if args.dataset == 'Bay_Area':
            pre_u[np.argwhere(all_y == 2)] = 0.5
        prediction_matrix = np.reshape(pre_u, (height, width))
        plt.imshow(prediction_matrix, 'gray')
        plt.axis('off')
        plt.savefig(f'{args.out_dir}/{search_num}-{args.dataset}-result.jpg', bbox_inches = 'tight', pad_inches = -0.1)
        plt.show()
        sio.savemat(f'{args.out_dir}/{search_num}-{args.dataset}-result.mat', {'CM': prediction_matrix})

        print("Final result:")
        print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2))
        print("**************************************************")
        print("Parameter:")

        return OA2, AA_mean2, Kappa2
    elif args.flag_test == 'train':
        print("start training")
        total_time = 0
        best = -1
        best_state_dict = None
        for epoch in range(args.epoches):
            # train model
            model.train()
            tic = time.time()
            train_acc, train_obj, tar_t, pre_t = train_epoch(model, train_loader, criterion, optimizer, device)
            scheduler.step()
            toc = time.time()
            total_time += (toc - tic)
            OA1, AA_mean1, Kappa1, AA1 = output_metric(tar_t, pre_t)
            print("{:02d}-Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f}"
                  .format(search_num, epoch + 1, train_obj, train_acc))

            # 验证集
            if (epoch + 1) % args.test_freq == 0:
                model.eval()
                tar_v, pre_v = valid_epoch(model, val_loader, criterion, device)
                OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
                if OA2 > best:
                    best = OA2
                    acc, aa, kappa = OA2, AA_mean2, Kappa2
                    best_state_dict = model.state_dict()
                print('Best Acc: {:.4f}'.format(best))
            # 保存模型
            if (epoch + 1) % args.save_epoch == 0 and best_state_dict is not None:
                if not os.path.exists(args.out_dir):
                    os.mkdir(args.out_dir)
                save_path = f"{args.out_dir}/{search_num}_model_parameter.pkl"
                torch.save(best_state_dict, save_path)
                print('best model saved: {:.4f}'.format(best))

        print("Running Time: {:.2f}".format(total_time))
        print("**************************************************")

        print("Final result:")
        print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(acc, aa, kappa))
        print("**************************************************")
        print("Parameter:")

        return acc, aa, kappa


def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k, v))


def get_args_parser():
    parser = argparse.ArgumentParser("HSI")
    parser.add_argument('--dataset', choices = ['China', 'USA'], default = 'China', help = 'dataset to use')
    parser.add_argument('--flag_test', choices = ['test', 'train'], default = 'train', help = 'testing mark')
    parser.add_argument('--device', default = 'cuda:1', help = 'device')
    parser.add_argument('--seed', type = int, default = 0, help = 'number of seed')
    parser.add_argument('--batch_size', type = int, default = 64, help = 'number of batch size')
    parser.add_argument('--test_freq', type = int, default = 5, help = 'number of evaluation')
    parser.add_argument('--patches', type = int, default = 7, help = 'number of patches')
    parser.add_argument('--epoches', type = int, default = 50, help = 'epoch number')
    parser.add_argument('--learning_rate', type = float, default = 5e-4, help = 'learning rate')
    parser.add_argument('--gamma', type = float, default = 0.9, help = 'gamma')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'weight_decay')

    parser.add_argument('--ratio', type = float, default = 0.05, help = 'train ratio')
    parser.add_argument('--num_workers', default = 0, type = int)
    parser.add_argument('--band_size', default = 198, type = int)
    parser.add_argument('--num_classes', default = 2, type = int)  # China: 2\4, USA: 7
    parser.add_argument('--depth', default = 5, type = int, help = 'depth of transformer')
    parser.add_argument('--head', default = 4, type = int, help = 'number of transformer head')
    parser.add_argument('--dim', type = int, default = 128, help = 'feature dimension')
    parser.add_argument('--out_dir', default = './output', help = 'path where to save')
    parser.add_argument('--save_epoch', type = int, default = 50, help = 'epoch number to save model')
    args = parser.parse_args()

    return args


def grid_search(args, num):
    param_grid = {
        'patches': [3],
        'depth': [2],  # 1, 2, 3, 4
        'head': [1],  # 1, 2, 4, 8
        'seed': [2030, 2031, 2032, 2033, 2034, 2035],  # 2030, 2031,
        'ratio': [0.05]  # 0.01, 0.03, 0.05, 0.10, 0.20
    }
    parameters = list(ParameterGrid(param_grid))[num]

    args.patches = parameters['patches']
    args.depth = parameters['depth']
    args.head = parameters['head']
    args.seed = parameters['seed']
    args.ratio = parameters['ratio']

    return args, parameters


if __name__ == '__main__':
    args = get_args_parser()
    set_seed(seed = args.seed)
    results = []  # 保存训练结果
    max_search = 6
    args.dataset = 'Yellow River'  # Yellow River  Bay_Area  China USA
    for i in range(0, max_search):
        args, searched_params = grid_search(args, i)
        args.flag_test = 'test'
        i += 1
        print(f'Start {i}-th random search')
        print(searched_params)
        Acc, AA, Kappa = main(args, search_num = i)
        searched_params['search_num'] = i
        values = list(searched_params.values())
        values.append(Acc)
        values.append(AA)
        values.append(Kappa)
        results.append(values)

        np.savetxt(f'{args.dataset}-{args.flag_test}.txt', np.array(results),
                   fmt = "%i, %i, %i, %.2f, %i, %i, %.6f, %.6f, %.6f")  # %.2f,

        print_args(vars(args))
