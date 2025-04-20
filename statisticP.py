import os
# from datetime import datetime
import random
import numpy as np
import argparse

import torch
from backbones import get_model
from dataset import my_ImageFolder
from losses import CombinedMarginLoss, ArcFace, CosFace, NaiveFace, PFace
# from lr_scheduler import PolynomialLRWarmup
from torchvision import transforms
from torchvision.datasets import ImageFolder
from partial_fc_v2 import PartialFC_V2, my_PFC, LoraFC, my_CE
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

def load_fc_model(model_path, device,cfg):
    margin_loss = PFace()
    fc = my_CE(margin_loss, cfg.embedding_size, cfg.num_classes, False)
    fc.load_state_dict(torch.load(model_path,weights_only=True))
    fc.eval().to(device)
    return fc

def calculate_probability(scale,embeddings,fc_weights,labels):
    probabilities = []
    s = scale
    norm_embeddings = normalize(embeddings)
    norm_weight_activated = normalize(fc_weights)
    logits = linear(norm_embeddings, norm_weight_activated)
    logits = logits.clamp(-1, 1)

    final_logits = s * logits
    index = torch.where(labels != -1)[0]
    target_logit = final_logits[index, labels[index].view(-1)]

    exp_logits = torch.exp(final_logits)
    sigma = torch.sum(exp_logits,dim=1)
    probability = torch.exp(target_logit) / sigma
    return probability

def Pmap(all_probabilities_tensor,data_name,density=False):
    plt.hist(all_probabilities_tensor.numpy(), bins=300, density=density)
    plt.title(f'Probability Distribution({data_name})')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.show()




def main(config_file):
    # get config
    config = importlib.import_module("configs."+config_file)
    cfg = config.cfg()
    
    device = torch.device(cfg.device)


    
    # Image Folder
    transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),])
    train_set = my_ImageFolder(cfg.rec, transform)
    train_loader = DataLoader(dataset=train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    data_name = "ms1mv3"
    scale = 24
    embedding_model_path = f"./output/{data_name}/model.pt"
    backbone = load_embedding_model(embedding_model_path,device,cfg)

    fc_model_path = f"./output/{data_name}/fc.pt"
    fc_model = load_fc_model(fc_model_path, device,cfg)
    fc_weights = fc_model.weight


    all_probabilities = []
    step = 0
    for _, (img, local_labels) in enumerate(train_loader):
        step += 1
        local_embeddings = backbone(img.to(device))
        probabilities = calculate_probability(scale,local_embeddings, fc_weights,local_labels.to(device)) 
        probabilities = probabilities.cpu().detach().numpy().flatten()
        all_probabilities.extend(probabilities)
        if step >= 1000:
            break
        
    Pmap(torch.tensor(all_probabilities),data_name)
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Get configurations')
    # parser.add_argument('--config', default="mnist", help='the name of config file')
    parser.add_argument('--config', default="ms1mv3", help='the name of config file')
    args = parser.parse_args()
    main(args.config)
