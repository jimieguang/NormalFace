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


def load_embedding_model(model_path, device,cfg):
    backbone = get_model(cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size)
    backbone.load_state_dict(torch.load(model_path,weights_only=True))
    backbone.eval().to(device)
    return backbone

def load_fc_model(model_path,ce_func,loss_func,device,cfg):
    margin_loss = loss_func()
    fc = ce_func(margin_loss, cfg.embedding_size, cfg.num_classes, False)
    fc.load_state_dict(torch.load(model_path,weights_only=True))
    fc.eval().to(device)
    return fc

def calculate_logits(embeddings,fc_weights,labels):
    logits = []
    norm_embeddings = normalize(embeddings)
    norm_weight_activated = normalize(fc_weights)
    logits = linear(norm_embeddings, norm_weight_activated)
    logits = logits.clamp(-1, 1)

    index = torch.where(labels != -1)[0]
    target_logit = logits[index, labels[index].view(-1)]

    return target_logit

def Pmap(all_logits_tensor,data_name,density=False):
    plt.hist(all_logits_tensor.numpy(), bins=300, density=density)
    plt.title(f'logit Distribution({data_name})')
    plt.xlabel('logit')
    plt.ylabel('Frequency')
    plt.show()




def main(config_file):
    # get config
    data_path = r"output/ms1mv3_test1"
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

    
    # Image Folder
    transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),])
    # train_set = ImageFolder(cfg.rec, transform)
    train_set = my_ImageFolder(cfg.rec, transform)
    train_loader = DataLoader(dataset=train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    data_name = data_path.split("/")[-1]
    embedding_model_path = f"./output/{data_name}/model.pt"
    backbone = load_embedding_model(embedding_model_path,device,cfg)

    fc_model_path = f"./output/{data_name}/fc.pt"
    fc_model = load_fc_model(fc_model_path,ce_func,loss_func,device,cfg)
    fc_weights = fc_model.weight


    all_logits = []
    step = 0
    for _, (img, local_labels) in enumerate(train_loader):
        step += 1
        local_embeddings = backbone(img.to(device))
        logits = calculate_logits(local_embeddings, fc_weights,local_labels.to(device)) 
        logits = logits.cpu().detach().numpy().flatten()
        all_logits.extend(logits)
        if step >= 1000:
            break
        
    Pmap(torch.tensor(all_logits),data_name)
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Get configurations')
    # parser.add_argument('--config', default="mnist", help='the name of config file')
    parser.add_argument('--config', default="ms1mv3", help='the name of config file')
    args = parser.parse_args()
    main(args.config)
