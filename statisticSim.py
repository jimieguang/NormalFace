# from __future__ import absolute_import

import os
import sys
# from datetime import datetime
import random
import numpy as np
import argparse

import torch
from backbones import get_model
from dataset import my_ImageFolder

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

def cosine_similarity_to_all_neighbor(matrix):
    norm_weights = normalize(matrix)
    # Calculate cosine similarity matrix
    # Initialize a list to store the all neighbor cosine similarities
    all_neighbor_sim_list = []
    batch_size = 512   # 当batch size=1时可粗略认为是单个样本对负类中心相似度的可视化（仍呈正态分布）
    # batch_size = 1

    # Process in batches
    for i in range(0, 1):
        end_idx = min(i + batch_size, norm_weights.size(0))
        batch = norm_weights[i:end_idx]

        # Compute cosine similarity between the batch and all vectors
        cos_sim = batch@norm_weights.T

        # Mask the similarities of vectors with themselves
        cos_sim[range(len(batch)), range(i, end_idx)] = 0

        # Find the all neighbor's similarity for each vector in the batch
        all_neighbor_sim = cos_sim.flatten()

        # Append to the list
        all_neighbor_sim_list.extend(all_neighbor_sim.tolist())

    return all_neighbor_sim_list

def Wmap(all_sim_tensor, data_name, density=False):
    all_sim_tensor = all_sim_tensor.numpy()
    # 计算直方图
    counts, bin_edges = np.histogram(all_sim_tensor, bins=300, density=density)
    
    # 找到最高频率的bin
    max_count_idx = np.argmax(counts)
    max_count_bin_center = (bin_edges[max_count_idx] + bin_edges[max_count_idx + 1]) / 2
    
    # 绘制直方图
    plt.hist(all_sim_tensor, bins=300, density=density)
    plt.title(f'All sim Distribution({data_name})')
    plt.xlabel('Cosine similarity')
    plt.ylabel('Frequency')
    
    # 在最高频率的cos值处画一条纵向虚线
    plt.axvline(x=max_count_bin_center, color='r', linestyle='--', label=f'Max Frequency at {max_count_bin_center:.4f}')
    
    # 显示图例
    plt.legend()
    # 显示图形
    plt.show()

def Wmap_latex(all_sim_tensor, data_name, density=False):
    """专为latex图片引用美化的"""
    # 全局字体配置
    FONT_SIZE = 18  # 通过修改这个值控制所有字体大小
    plt.rcParams.update({
        'font.size': FONT_SIZE,           # 控制常规文本大小
        'axes.titlesize': FONT_SIZE,      # 标题字体大小
        'axes.labelsize': FONT_SIZE,      # 坐标轴标签字体大小
        'xtick.labelsize': FONT_SIZE,     # X轴刻度字体大小
        'ytick.labelsize': FONT_SIZE,     # Y轴刻度字体大小
        'legend.fontsize': FONT_SIZE,     # 图例字体大小
        'pdf.fonttype': 42                # 确保PDF嵌入可编辑字体
    })

    all_sim_tensor = all_sim_tensor.numpy()
    # 为了美化做的数据过滤
    # 生成布尔掩码（True表示在范围内）
    mask = (all_sim_tensor >= -0.3) & (all_sim_tensor <= 0.3)
    all_sim_tensor = all_sim_tensor[mask]  # 仅保留范围内的值
    # 计算直方图
    counts, bin_edges = np.histogram(all_sim_tensor, bins=300, density=density)
    
    # 找到最高频率的bin
    max_count_idx = np.argmax(counts)
    max_count_bin_center = (bin_edges[max_count_idx] + bin_edges[max_count_idx + 1]) / 2
    
    name = "ArcFace"
    # 绘制直方图
    # 创建独立figure并设置尺寸
    plt.figure(figsize=(6,5))
    plt.hist(all_sim_tensor, bins=300, density=density)
    plt.title(name)
    plt.xlabel('Cosine similarity')
    plt.ylabel('Frequency')
    
    # 在最高频率的cos值处画一条纵向虚线
    plt.axvline(x=max_count_bin_center, color='r', linestyle='--', label=f'μ = {max_count_bin_center:.4f}')
    
    # 显示图例
    plt.legend()
    # 保存图像
    plt.savefig(name+".pdf", dpi=300, bbox_inches='tight')


def Compare(all_sim_tensor, data_name, density,mean,std_dev):
    all_sim_tensor = all_sim_tensor.numpy()
    # 计算直方图
    counts, bin_edges = np.histogram(all_sim_tensor, bins=300, density=density)
    
    # 找到最高频率的bin
    max_count_idx = np.argmax(counts)
    max_count_bin_center = (bin_edges[max_count_idx] + bin_edges[max_count_idx + 1]) / 2
    
    # 绘制直方图
    plt.hist(all_sim_tensor, bins=300, density=density)
    plt.title(f'All sim Distribution({data_name})')
    plt.xlabel('Cosine similarity')
    plt.ylabel('Frequency')
    
    # 在最高频率的cos值处画一条纵向虚线
    plt.axvline(x=max_count_bin_center, color='r', linestyle='--', label=f'Max Frequency at {max_count_bin_center:.4f}')
    
    # 显示图例
    plt.legend()

    # 为了绘制正态分布，我们需要指定平均值和方差
    # mean = 0.00030250538839027286  # 示例平均值
    # std_dev = 0.05400579050183296  # 示例标准差

    # 生成正态分布数据
    from scipy.stats import norm
    x = np.linspace(-1, 1, 300)
    normal_distribution = norm.pdf(x, mean, std_dev)
    plt.plot(x, normal_distribution, label=f'Normal Distribution (mean={mean}, std={std_dev})', color='b')
    
    # 显示图形
    plt.show()




def main(config_file):
    data_path = r"output/ms1mv3"
    # 添加包含模块的目录路径到 sys.path
    sys.path.append(data_path)

    # 导入自定义CE 与 lossFunc
    from mycode.losses import SFace,CosFace
    from mycode.partial_fc_v2 import my_CE
    # get config
    config = importlib.import_module("mycode."+config_file)
    cfg = config.cfg()
    device = torch.device(cfg.device)

    ce_func = my_CE
    loss_func = CosFace
    
    fc_model_path = f"{data_path}\\fc.pt".replace('\\','/')
    data_name = data_path.split("/")[-1]
    fc_model = load_fc_model(fc_model_path,ce_func,loss_func,device,cfg)
    fc_weights = fc_model.weight
    all_neighbor_sim_list = cosine_similarity_to_all_neighbor(fc_weights)

    all_neighbor_sim_list_tensor = torch.tensor(all_neighbor_sim_list)
    all_neighbor_sim_list_numpy = all_neighbor_sim_list_tensor.numpy()
    mean = np.mean(all_neighbor_sim_list_numpy)
    std = np.std(all_neighbor_sim_list_numpy)
    from scipy.stats import kstest,skew,kurtosis
    sample_size = 100000
    random_indices = random.sample(range(len(all_neighbor_sim_list_numpy)), sample_size)
    skewness = skew(all_neighbor_sim_list_numpy[random_indices])
    kurt = kurtosis(all_neighbor_sim_list_numpy[random_indices])
    # 执行KS检验
    d_value, p_value = kstest(all_neighbor_sim_list_numpy[random_indices], 
                             'norm', 
                             args=(mean, std))
    
    # 打印新增统计量
    print(f"Skewness: {skewness:.4f}")
    print(f"Kurtosis: {kurt:.4f}") 
    print(f"KS Test D: {d_value:.4f}")
    print(f"KS Test p-value: {p_value:.4f}")
    print(f"Mean:{mean}")
    print(f"std:{std}")
    s = 24
    res = np.exp(s*all_neighbor_sim_list_numpy).mean()
    std_theory = np.sqrt(2*np.log(res))/s
    print(f"std_theory:{std_theory}")
    print(f"std_bias:{(std_theory-std)/std*100:.2f}%")
    # Wmap_latex(all_neighbor_sim_list_tensor,data_name,True)
    # Wmap(all_neighbor_sim_list_tensor,data_name,True)
    Compare(all_neighbor_sim_list_tensor,data_name,True,mean,std)

    

    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Get configurations')
    parser.add_argument('--config', default="ms1mv3", help='the name of config file')
    args = parser.parse_args()
    main(args.config)
