import os
# from datetime import datetime
import random
import sys
import numpy as np
import argparse
import torch
from backbones import get_model
from dataset import my_ImageFolder
# from lr_scheduler import PolynomialLRWarmup
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import importlib
import matplotlib.pyplot as plt


import math
from typing import Callable

import torch
from torch import distributed
from torch.nn.functional import linear, normalize


def load_fc_model(model_path,ce_func,loss_func,device,cfg):
    margin_loss = loss_func()
    fc = ce_func(margin_loss, cfg.embedding_size, cfg.num_classes, False)
    fc.load_state_dict(torch.load(model_path,weights_only=True))
    fc.eval().to(device)
    return fc

def cosine_similarity_to_nearest_neighbor(matrix):
    norm_weights = normalize(matrix)
    # Calculate cosine similarity matrix
    # Initialize a list to store the nearest neighbor cosine similarities
    nearest_neighbor_sim_list = []
    batch_size = 1024

    # Process in batches
    for i in range(0, norm_weights.size(0), batch_size):
        end_idx = min(i + batch_size, norm_weights.size(0))
        batch = norm_weights[i:end_idx]

        # Compute cosine similarity between the batch and all vectors
        cos_sim = batch@norm_weights.T

        # Mask the similarities of vectors with themselves
        cos_sim[range(len(batch)), range(i, end_idx)] = 0

        # Find the nearest neighbor's similarity for each vector in the batch
        nearest_neighbor_sim, _ = torch.max(cos_sim, dim=1)

        # Append to the list
        nearest_neighbor_sim_list.extend(nearest_neighbor_sim.tolist())

    return nearest_neighbor_sim_list

def Wmap(all_sim_tensor, data_name, density=False):
    # 计算直方图
    counts, bin_edges = np.histogram(all_sim_tensor.numpy(), bins=300, density=density)
    
    # 找到最高频率的bin
    max_count_idx = np.argmax(counts)
    max_count_bin_center = (bin_edges[max_count_idx] + bin_edges[max_count_idx + 1]) / 2
    
    # 绘制直方图
    plt.hist(all_sim_tensor.numpy(), bins=300, density=density)
    plt.title(f'Nearest sim Distribution({data_name})')
    plt.xlabel('Nearest negative proxy similarity')
    plt.ylabel('Frequency')
    
    # 在最高频率的cos值处画一条纵向虚线
    plt.axvline(x=max_count_bin_center, color='r', linestyle='--', label=f'Max Frequency at {max_count_bin_center:.4f}')
    
    # 显示图例
    plt.legend()
    
    # 显示图形
    plt.show()




def main(config_file):
    # get config
    data_path = r"output/ms1mv3_cosface"
    # 添加包含模块的目录路径到 sys.path
    sys.path.append(data_path)

    # 导入自定义CE 与 lossFunc
    from mycode.losses import SFace
    from mycode.partial_fc_v2 import my_CE
    # get config
    config = importlib.import_module("mycode."+config_file)
    cfg = config.cfg()
    device = torch.device(cfg.device)

    ce_func = my_CE
    loss_func = SFace


    fc_model_path = f"{data_path}\\fc.pt".replace('\\','/')
    data_name = data_path.split("/")[-1]
    fc_model = load_fc_model(fc_model_path,ce_func,loss_func,device,cfg)
    fc_weights = fc_model.weight
    nearest_neighbor_sim_list = cosine_similarity_to_nearest_neighbor(fc_weights)



        
    Wmap(torch.tensor(nearest_neighbor_sim_list),data_name)
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Get configurations')
    parser.add_argument('--config', default="ms1mv3", help='the name of config file')
    args = parser.parse_args()
    main(args.config)
